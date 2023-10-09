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

def sc_makeClassifier(expTrain: pd.DataFrame, genes: np.ndarray, groups: np.ndarray, nRand: int = 70,
                      ntrees: int = 2000, stratify: bool = False) -> RandomForestClassifier:
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

def scn_classify(adata: AnnData, rf_tsp, nrand: int = 0, copy=False) -> AnnData:
    """
    Classifies cells in the `adata` object based on the given gene expression and cross-pair information using a
    random forest classifier in rf_tsp trained with the provided xpairs genes.
    
    Parameters:
    -----------
    adata: `AnnData`
        An annotated data matrix containing the gene expression information for cells.
    cgenes: List[str]
        A list of gene names used for classification.
    xpairs: List[str]
        A list of cross-pair features used for classification.
    rf_tsp: List[float]
        A list of random forest classifier parameters used for classification.
    nrand: int
        Number of random permutations for the null distribution. Default is 0.
    
    Returns:
    --------
    An annotated data matrix containing the classified cells as a new object.
    """
    
    # Classify cells using the `scn_predict` function
    # classRes = scn_predict(cgenes, xpairs, rf_tsp, adata, nrand=nrand)
    classRes = scn_predict(rf_tsp, adata, nrand=nrand)
    
    # Get the categories (i.e., predicted cell types) from the classification result
    categories = classRes.columns.values
    
    # add the classification result as to `obsm`
#    adNew = AnnData(classRes, obs=adata.obs, var=pd.DataFrame(index=categories))
    adata.obsm['SCN_score'] = classRes

    # Add a new column to `obs` for the predicted cell types
    adata.obs['SCN_class'] = classRes.idxmax(axis=1)

    ##
    ## adNew.obs['SCN_class'] = classRes.idxmax(axis=1)
    
    # return copy if called for
    return adata if copy else None

def scn_predict(rf_tsp, aDat, nrand = 2):
    
    if isinstance(aDat.X,np.ndarray):
        # in the case of aDat.X is a numpy array 
        aDat.X = anndata._core.views.ArrayView(aDat.X)
###    expDat= pd.DataFrame(data=aDat.X, index= aDat.obs.index.values, columns= aDat.var.index.values)
    expDat= pd.DataFrame(data=aDat.X.toarray(), index= aDat.obs.index.values, columns= aDat.var.index.values)
    expValTrans=query_transform(expDat.reindex(labels=rf_tsp['tpGeneArray'], axis='columns', fill_value=0), rf_tsp['topPairs'])
    classRes_val=rf_classPredict(rf_tsp['classifier'], expValTrans, numRand=nrand)
    return classRes_val

# there is an issue in that making the random profiles here will break later addition of results to original annData object
def rf_classPredict(rfObj,expQuery,numRand=50):
    if numRand > 0 :
        randDat=randomize(expQuery, num=numRand)
        expQuery=pd.concat([expQuery, randDat])
    xpreds= pd.DataFrame(rfObj.predict_proba(expQuery), columns= rfObj.classes_, index=expQuery.index)
    return xpreds

def scn_train(aTrain,dLevel,nTopGenes = 100,nTopGenePairs = 100,nRand = 100, nTrees = 1000,stratify=False,counts_per_cell_after=1e4, scaleMax=10, limitToHVG=True, normalization = True, include_all_genes = False, propOther=0.5):
    warnings.filterwarnings('ignore')
    stTrain= aTrain.obs
    
    expRaw = aTrain.to_df()
    expRaw = expRaw.loc[stTrain.index.values]

    adNorm = aTrain.copy()
    if normalization == True:
        sc.pp.normalize_per_cell(adNorm, counts_per_cell_after=counts_per_cell_after)
        sc.pp.log1p(adNorm)

        print("HVG")
        if limitToHVG:
            try:
                sc.pp.highly_variable_genes(adNorm, min_mean=0.0125, max_mean=4, min_disp=0.5)
            except Exception as e:
                raise ValueError(f"PySCN encountered an error when selecting variable genes. This may be avoided if you do not call scale or regress_out on the training data. Original error text: {repr(e)}") 
            adNorm = adNorm[:, adNorm.var.highly_variable]

        sc.pp.scale(adNorm, max_value=scaleMax)

    expTnorm = adNorm.to_df()
    expTnorm = expTnorm.loc[stTrain.index.values]

    ### expTnorm= pd.DataFrame(data=aTrain.X,  index= aTrain.obs.index.values, columns= aTrain.var.index.values)
    ### expTnorm=expTnorm.loc[stTrain.index.values]
    print("Matrix normalized")
    ### cgenesA, grps, cgenes_list =findClassyGenes(expTnorm,stTrain, dLevel = dLevel, topX = nTopGenes)
    if include_all_genes == False:
        cgenesA, grps, cgenes_list =findClassyGenes(adNorm, dLevel = dLevel, topX = nTopGenes)
    else: 
        cgenesA = np.array(aTrain.var.index)
        grps = aTrain.obs[dLevel]
        cgenes_list = dict()
        for g in np.unique(grps):
            cgenes_list[g] = cgenesA

    print("There are ", len(cgenesA), " classification genes\n")
    ### xpairs= ptGetTop(expTnorm.loc[:,cgenesA], grps, cgenes_list, topX=nTopGenePairs, sliceSize=5000)
    xpairs= ptGetTop(expTnorm.loc[:,cgenesA], grps, cgenes_list, topX=nTopGenePairs, sliceSize=5000, propOther=propOther)

    print("There are", len(xpairs), "top gene pairs\n")
    pdTrain= query_transform(expRaw.loc[:,cgenesA], xpairs)
    print("Finished pair transforming the data\n")
   

    tspRF=sc_makeClassifier(pdTrain.loc[:, xpairs], genes=xpairs, groups=grps, nRand = nRand, ntrees = nTrees, stratify=stratify)
    # return [cgenesA, xpairs, tspRF, cgenes_list]
    
    ## #### ## tmpAns = dict(preds = pdTrain.loc[:, xpairs], y_s = grps)
    ## #### ## return tmpAns
    ## #### ## return tspRF
    return {'tpGeneArray': cgenesA, 'topPairs':xpairs, 'classifier': tspRF, 'diffExpGenes':cgenes_list}


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





