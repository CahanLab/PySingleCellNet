import numpy as np
import pandas as pd
import scanpy as sc
import anndata
from anndata import AnnData
from typing import List
from typing import Dict
from sklearn.ensemble import RandomForestClassifier
from scipy.sparse import csr_matrix
import warnings
import matplotlib.pyplot as plt
from alive_progress import alive_bar
from .utils import *
from .tsp_rf import *


def randomize(expDat: pd.DataFrame, num: int = 50) -> pd.DataFrame:
    """
    Randomize the rows and columns of a pandas DataFrame.

    Args:
        expDat (pd.DataFrame): the input DataFrame
        num (int): the number of rows to return (default 50)

    Returns:
        pd.DataFrame: the randomized DataFrame with num rows and the same columns as expDat
    """

    # Convert DataFrame to NumPy array
    temp = expDat.to_numpy()

    # Randomize the rows of the array
    temp = np.array([np.random.choice(x, len(x), replace=False) for x in temp])

    # Transpose the array and randomize the columns
    temp = temp.T
    temp = np.array([np.random.choice(x, len(x), replace=False) for x in temp]).T

    # Convert the array back to a DataFrame and return the first num rows
    return pd.DataFrame(data=temp, columns=expDat.columns).iloc[0:num, :]

def sc_makeClassifier(
    expTrain: pd.DataFrame,
    genes: np.ndarray,
    groups: np.ndarray,
    nRand: int = 70,
    ntrees: int = 2000,
    stratify: bool = False
) -> RandomForestClassifier:
    """
    Train a random forest classifier on gene expression data.

    Args:
        expTrain (pd.DataFrame): the training data as a pandas DataFrame
        genes (np.ndarray): the gene names corresponding to the columns of expTrain
        groups (np.ndarray): the class labels for each sample in expTrain
        nRand (int): the number of randomized samples to generate (default 70)
        ntrees (int): the number of trees in the random forest (default 2000)
        stratify (bool): whether to stratify the random forest by class label (default False)

    Returns:
        RandomForestClassifier: the trained random forest classifier
    """

    # Randomize the training data and concatenate with the original data
    randDat = randomize(expTrain, num=nRand)
    expT = pd.concat([expTrain, randDat])

    # Get the list of all genes and missing genes
    allgenes = expT.columns.values
    missingGenes = np.setdiff1d(np.unique(genes), allgenes)

    # Get the genes that are present in both the training data and the gene list
    ggenes = np.intersect1d(np.unique(genes), allgenes)

    # Train a random forest classifier
    if not stratify:
        clf = RandomForestClassifier(n_estimators=ntrees, random_state=100)
    else:
        clf = RandomForestClassifier(n_estimators=ntrees, class_weight="balanced", random_state=100)
    ggroups = np.append(np.array(groups), np.repeat("rand", nRand)).flatten()
    ## #### ## clf.fit(expT.loc[:, ggenes].to_numpy(), ggroups)
    clf.fit(expT.loc[:, ggenes], ggroups)

    # Return the trained classifier 
    ## #### ## return [expT.loc[:, ggenes], ggroups]
    return clf

def scn_classify(adata: AnnData, rf_tsp, nrand: int = 0):
    """
    Classifies cells in the `adata` object based on the given gene expression and cross-pair information using a
    random forest classifier in rf_tsp trained with the provided xpairs genes.
    
    Parameters:
    -----------
    adata: `AnnData`
        An annotated data matrix containing the gene expression information for cells.
    rf_tsp: List[float]
        A list of random forest classifier parameters used for classification.
    nrand: int
        Number of random permutations for the null distribution. Default is 0.
    
    Returns:
    --------
    Updates adata with classification results 
    """

    # Classify cells using the `scn_predict` function
    classRes = scn_predict(rf_tsp, adata, nrand=nrand)

    # add the classification result as to `obsm`
    # adNew = AnnData(classRes, obs=adata.obs, var=pd.DataFrame(index=categories))
    adata.obsm['SCN_score'] = classRes
    
    # Get the categories (i.e., predicted cell types) from the classification result
    # categories = classRes.columns.values
    # possible_classes = rf_tsp['classifier'].classes_
    possible_classes = pd.Categorical(classRes.columns)
    # Add a new column to `obs` for the predicted cell types
    predicted_classes = classRes.idxmax(axis=1)
    adata.obs['SCN_class'] = pd.Categorical(predicted_classes, categories=possible_classes, ordered=True)

    # store this for consistent coloring
    adata.uns['SCN_class_colors'] = rf_tsp['ctColors']        
    # can return copy if called for
    # return adata if copy else None

def scn_predict(rf_tsp, aDat, nrand = 2):
    
    if isinstance(aDat.X,np.ndarray):
        # in the case of aDat.X is a numpy array 
        aDat.X = anndata._core.views.ArrayView(aDat.X)
###    expDat= pd.DataFrame(data=aDat.X, index= aDat.obs.index.values, columns= aDat.var.index.values)
    expDat = pd.DataFrame(data=aDat.X.toarray(), index= aDat.obs.index.values, columns= aDat.var.index.values)
    expValTrans = query_transform(expDat.reindex(labels=rf_tsp['tpGeneArray'], axis='columns', fill_value=0), rf_tsp['topPairs'])
    classRes_val = rf_classPredict(rf_tsp['classifier'], expValTrans, numRand=nrand)
    return classRes_val

# there is an issue in that making the random profiles here will break later addition of results to original annData object
def rf_classPredict(rfObj,expQuery,numRand=50):
    if numRand > 0 :
        randDat = randomize(expQuery, num=numRand)
        expQuery = pd.concat([expQuery, randDat])
    xpreds = pd.DataFrame(rfObj.predict_proba(expQuery), columns= rfObj.classes_, index=expQuery.index)
    return xpreds

def scn_train(aTrain,
    dLevel,
    nRand = None,
    cell_type_to_color = None,
    nTopGenes = 20,
    nTopGenePairs = 20,
    nTrees = 1000,
    propOther=0.5,
    counts_per_cell_after = 1e4,
    scaleMax = 10,
    limitToHVG = True,
    normalization = True,
    include_all_genes = False
):
    progress_total = 5
    with alive_bar(progress_total, title="Training classifier") as bar:
        warnings.filterwarnings('ignore')

        # auto determine nRand = mean number of cells per type
        if nRand is None:
            nRand = np.floor(np.mean(aTrain.obs[dLevel].value_counts()))

        stTrain= aTrain.obs
        expRaw = aTrain.to_df()
        expRaw = expRaw.loc[stTrain.index.values]
        adNorm = aTrain.copy()
        if normalization:
            sc.pp.normalize_per_cell(adNorm, counts_per_cell_after=counts_per_cell_after)
            sc.pp.log1p(adNorm)
            # print("HVG")
            if limitToHVG:
                try:
                    sc.pp.highly_variable_genes(adNorm, min_mean=0.0125, max_mean=4, min_disp=0.5)
                except Exception as e:
                    raise ValueError(f"PySCN encountered an error when selecting variable genes. This may be avoided if you do not call scale or regress_out on the training data. Original error text: {repr(e)}") 
                adNorm = adNorm[:, adNorm.var.highly_variable]

            sc.pp.scale(adNorm, max_value=scaleMax)
            #print("data normalized")

        expTnorm = adNorm.to_df()
        expTnorm = expTnorm.loc[stTrain.index.values]
        bar() # Bar 1

        ### expTnorm= pd.DataFrame(data=aTrain.X,  index= aTrain.obs.index.values, columns= aTrain.var.index.values)
        ### expTnorm=expTnorm.loc[stTrain.index.values]
        
        ### cgenesA, grps, cgenes_list =findClassyGenes(expTnorm,stTrain, dLevel = dLevel, topX = nTopGenes)
        if include_all_genes == False:
            cgenesA, grps, cgenes_list =findClassyGenes(adNorm, dLevel = dLevel, topX = nTopGenes)
        else: 
            cgenesA = np.array(aTrain.var.index)
            grps = aTrain.obs[dLevel]
            cgenes_list = dict()
            for g in np.unique(grps):
                cgenes_list[g] = cgenesA

        bar() # Bar 2
        # print("There are ", len(cgenesA), " classification genes\n")
        ### xpairs= ptGetTop(expTnorm.loc[:,cgenesA], grps, cgenes_list, topX=nTopGenePairs, sliceSize=5000)
        xpairs= ptGetTop(expTnorm.loc[:,cgenesA], grps, cgenes_list, topX=nTopGenePairs, sliceSize=5000, propOther=propOther)
        bar() # Bar 3
        # print("There are", len(xpairs), "top gene pairs\n")
        pdTrain= query_transform(expRaw.loc[:,cgenesA], xpairs)
        # print("Finished pair transforming the data\n")
       

        tspRF=sc_makeClassifier(pdTrain.loc[:, xpairs], genes=xpairs, groups=grps, nRand = nRand, ntrees = nTrees)
        bar() # Bar 4
    
    ## set celltype colors
        ## Do this here because we add a 'rand' celltype
        
        # Need to add checks that all classes have a color if ct_colors is provided
        if cell_type_to_color is None:
            ## assume this is a Series
            cell_types = stTrain[dLevel].cat.categories.to_list()
            cell_types.append('rand')
            unique_colors = get_unique_colors(len(cell_types))
            cell_type_to_color = {cell_type: color for cell_type, color in zip(cell_types, unique_colors)}
        bar() # Bar 5

    return {'tpGeneArray': cgenesA, 'topPairs':xpairs, 'classifier': tspRF, 'diffExpGenes':cgenes_list, 'ctColors':cell_type_to_color}


def add_training_dlevel(adata, dlevel):
    adata.obs['SCN_class'] = adata.obs[dlevel]
    return adata

def check_adX(adata: AnnData) -> AnnData:
    from scipy import sparse
    if( isinstance(adata.X, np.ndarray)):
        adata.X = sparse.csr_matrix(adata.X)

def add_classRes(adata: AnnData, adClassRes, copy=False) -> AnnData:
    cNames = adClassRes.var_names
    for cname in cNames:
        adata.obs[cname] = adClassRes[:,cname].X.toarray()
    # adata.obs['category'] = adClassRes.obs['category']
    adata.obs['SCN_class'] = adClassRes.obs['SCN_class']
    return adata if copy else None


def compute_celltype_proportions(adata, celltype_col='celltype', stage_col='stage', exclude=None):
    """
    Compute the proportion of cell types per stage in the provided AnnData object.

    Parameters:
        adata (AnnData): The AnnData object containing scRNA-seq data.
        celltype_col (str): The name of the .obs column containing cell type information. Default is 'celltype'.
        stage_col (str): The name of the .obs column containing stage information. Default is 'stage'.
        exclude (list or None): A list of cell types to exclude from the calculation. Default is None.

    Returns:
        pd.DataFrame: A DataFrame with stages as rows, cell types as columns, and proportions as values.
    """
    # Filter out excluded cell types if any
    if exclude:
        adata = adata[~adata.obs[celltype_col].isin(exclude)]
    
    # Compute cell counts per stage and cell type
    stage_celltype_counts = adata.obs.groupby([stage_col, celltype_col]).size().unstack(fill_value=0)
    
    # Normalize counts to proportions within each stage
    stage_proportions = stage_celltype_counts.div(stage_celltype_counts.sum(axis=1), axis=0)
    
    return stage_proportions


def correlate_proportions(reference_proportions, query_adata, celltype_col='SCN_class'):
    """
    Compute the correlation between cell type proportions in the query AnnData object and
    the reference proportions for each stage.

    Parameters:
        reference_proportions (pd.DataFrame): The output of the compute_celltype_proportions function, representing the reference proportions.
        query_adata (AnnData): The AnnData object to compute proportions on for correlation analysis.
        celltype_col (str): The name of the .obs column containing cell type information in the query AnnData object. Default is 'SCN_class'.

    Returns:
        pd.Series: Correlations between the query proportions and each stage in the reference proportions.
    """
    # Compute the cell type proportions in the query data
    query_counts = query_adata.obs[celltype_col].value_counts(normalize=True)
    
    # Align with the reference data
    query_proportions = query_counts.reindex(reference_proportions.columns, fill_value=0)
    
    # Compute correlation with each stage in the reference proportions
    correlations = reference_proportions.apply(lambda x: x.corr(query_proportions), axis=1)
    
    return correlations




