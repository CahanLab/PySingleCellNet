import numpy as np
import pandas as pd
import scanpy as sc
import anndata
from anndata import AnnData
from typing import List, Dict
from sklearn.ensemble import RandomForestClassifier
from scipy.sparse import csr_matrix
import warnings
import matplotlib.pyplot as plt
from alive_progress import alive_bar
from ..utils import build_knn_graph, get_unique_colors, split_adata_indices
from sklearn.metrics import classification_report
from pySingleCellNet.config import SCN_DIFFEXP_KEY, SCN_CATEGORY_COLOR_DICT
import random as rand 

def _rank_genes_subsets(
    adata,
    groupby,
    grpA,
    grpB,
    pval = 0.01,
    layer=None
):
    """
    Subset an AnnData object to specified groups, create a new .obs column labeling cells
    as group A or B, and run rank_genes_groups for differential expression analysis. Necessary because the scanpy reference does not seem to work
    
    Parameters:
        adata (AnnData): The AnnData object.
        groupby (str): The .obs column to group cells by.
        grpA (list): Values used to subset cells into group A.
        grpB (list): Values used to subset cells into group B.
        layer (str, optional): Layer to use for expression values.
        
    Returns:
        AnnData: Subsetted and labeled AnnData object after running rank_genes_groups.
    """
    # Subset the data to cells in either grpA or grpB
    subset = adata[adata.obs[groupby].isin(grpA + grpB)].copy()
    # Create a new .obs column labeling cells as 'grpA' or 'grpB'
    subset.obs["comparison_group"] = subset.obs[groupby].apply(
        lambda x: "grpA" if x in grpA else "grpB"
    )
    # Run rank_genes_groups
    sc.tl.rank_genes_groups(
        subset,
        groupby="comparison_group",
        layer=layer,
        pts = True,
        use_raw=False
    )
    # return subset
    ans = sc.get.rank_genes_groups_df(subset, group='grpA', pval_cutoff=pval)
    return ans


def _query_transform(expMat, genePairs):
    npairs = len(genePairs)
    ans = pd.DataFrame(0, index = expMat.index, columns = np.arange(npairs))
    genes1=[]
    genes2=[]
    for g in genePairs:
        sp=g.split("_")
        genes1.append(sp[0])
        genes2.append(sp[1])
    expTemp=expMat.loc[:,np.unique(np.concatenate([genes1,genes2]))]
    ans = pd.DataFrame(0, index = expTemp.index, columns = np.arange(npairs))
    ans = ans.astype(pd.SparseDtype("int", 0))
    temp1= expTemp.loc[:,genes1]
    temp2= expTemp.loc[:,genes2]
    temp1.columns=np.arange(npairs)
    temp2.columns=np.arange(npairs)
    boolArray = temp1 > temp2
    ans = boolArray.astype(int)
    ans.columns = genePairs
    return(ans)

def create_classifier_report(adata: AnnData,
    ground_truth: str,
    prediction: str) -> pd.DataFrame:
    """
    Generate a classification report as a pandas DataFrame from an AnnData object.
    
    This function computes a classification report using ground truth and prediction
    columns in `adata.obs`. It supports both string and dictionary outputs from
    `sklearn.metrics.classification_report` and transforms them into a standardized
    DataFrame format.
    
    Args:
        adata (AnnData): An annotated data matrix containing observations with
            categorical truth and prediction labels.
        ground_truth (str): The column name in `adata.obs` containing the true
            class labels.
        prediction (str): The column name in `adata.obs` containing the predicted
            class labels.
    
    Returns:
        pd.DataFrame: A DataFrame with columns ["Label", "Precision", "Recall",
        "F1-Score", "Support"] summarizing classification metrics for each class.
    
    Raises:
        ValueError: If the classification report is neither a string nor a dictionary.
    """

    report = classification_report(adata.obs[ground_truth], adata.obs[prediction],labels=adata.obs[ground_truth].cat.categories, output_dict = True)
    # Parse the sklearn classification report into a DataFrame
    if isinstance(report, str):
        lines = report.split('\n')
        rows = []
        for line in lines[2:]:
            if line.strip() == '':
                continue
            row = line.split()
            if row[0] == 'micro' or row[0] == 'macro' or row[0] == 'weighted':
                row[0] = ' '.join(row[:2])
                row = [row[0]] + row[2:]
            elif len(row) > 5:
                row[0] = ' '.join(row[:2])
                row = [row[0]] + row[2:]
            rows.append(row)
        
        df = pd.DataFrame(rows, columns=["Label", "Precision", "Recall", "F1-Score", "Support"])
        df["Precision"] = pd.to_numeric(df["Precision"], errors='coerce')
        df["Recall"] = pd.to_numeric(df["Recall"], errors='coerce')
        df["F1-Score"] = pd.to_numeric(df["F1-Score"], errors='coerce')
        df["Support"] = pd.to_numeric(df["Support"], errors='coerce')
    elif isinstance(report, dict):
        df = pd.DataFrame(report).T.reset_index()
        df.columns = ["Label", "Precision", "Recall", "F1-Score", "Support"]
    else:
        raise ValueError("Report must be a string or a dictionary.")
    return df


def get_top_genes_from_df(
    df,
    by_score = True,
    topX = 10,
    min_in = 0.15,
    max_out = 0.10,
    proportion_top = 1.0
):
    """
    Get the top genes from a DataFrame based on upregulation.
    
    Parameters:
        df (DataFrame): Input DataFrame with gene data.
        by_score (bool): Whether to sort by 'scores' or 'pct_nz_group'.
        topX (int): Number of top genes to return.
        min_in (float): Minimum pct_nz_group for filtering when by_score is False.
        max_out (float): Maximum pct_nz_reference for filtering when by_score is False.
        proportion_top (float): Proportion of topX genes to take from the top (0-1).
    
    Returns:
        list: List of top gene names.
    """
    # Calculate number of genes from the top and bottom
    num_top = int(round(topX * proportion_top))
    num_bottom = topX - num_top
    
    gene_list = []
    
    if by_score:
        # Sort by scores
        dfTemp = df.sort_values(by=['scores'], ascending=False)
        
        # Get top genes
        top_genes = dfTemp['names'].head(num_top)
        
        # Get bottom genes (most negative scores)
        bottom_genes = dfTemp['names'].tail(num_bottom)
    
    else:
        # Sort by pct_nz_group
        dfTemp = df.sort_values(by=['pct_nz_group'], ascending=False)
        dfTempUp = dfTemp[dfTemp['pct_nz_group'] > min_in]
        dfTempUp = dfTempUp[dfTempUp['pct_nz_reference'] < max_out]
        
        # Get top genes
        top_genes = dfTempUp['names'].head(num_top)
        
        # Get bottom genes (those specifically not expressed in grpA)
        dfTempDn = dfTemp[dfTemp['pct_nz_group'] < max_out]
        dfTempDn = dfTempDn[dfTempDn['pct_nz_reference'] > min_in]
        bottom_genes = dfTempDn['names'].tail(num_bottom)
    
    # Combine top and bottom genes into final list
    gene_list = list(top_genes) + list(bottom_genes)
    
    return gene_list


def _get_classy_genes_3(
    adata,
    groupby,
    key_name="rank_genes_groups",  # Default differential expression key
    topX_per_diff_type=10,
    pval=0.01,
    bottom_min_in=0.15,
    bottom_max_out=0.1,
    top_min_in=0.4,
    top_max_out=0.25,
    proportion_top=1,
    k_of_knn=1,
    layer=None, #"lognorm",
    min_genes=20,  # NEW: minimum number of genes required per cluster
):
    """
    Identifies cell-type-specific genes using differential expression analysis 
    and kNN graph-based comparisons. If a cluster doesn't meet the min_genes 
    requirement, it uses a fallback strategy to find additional genes.
    
    Args:
        adata (AnnData): The AnnData object containing single-cell data.
        groupby (str): The .obs column to group cells by.
        key_name (str, optional): Key in `.uns` to use for rank_genes_groups results.
        topX_per_diff_type (int, optional): Number of top genes to select per DE comparison type.
        pval (float, optional): P-value cutoff for differential expression.
        bottom_min_in (float, optional): Min proportion of cells expressing a gene in the target group (for "bottom" genes).
        bottom_max_out (float, optional): Max proportion of cells expressing a gene in reference groups (for "bottom" genes).
        top_min_in (float, optional): Min proportion of cells expressing a gene in the target group (for "top" genes).
        top_max_out (float, optional): Max proportion of cells expressing a gene in the reference groups (for "top" genes).
        proportion_top (float, optional): Fraction of genes selected from the top vs bottom categories.
        k_of_knn (int, optional): Number of neighbors in the kNN cell-type graph.
        layer (str, optional): Which data layer to use for differential expression.
        min_genes (int, optional): Minimum number of genes that must be identified for each cluster.
    
    Returns:
        tuple: A tuple containing:
            - cgenes2 (list): Combined list of all unique genes identified.
            - grps (pandas.Series): Group labels from the specified `.obs` column.
            - gene_dict (dict): Dictionary of genes specific to each group.
    """
    
    # Copy the AnnData object to avoid modifying the original
    adTemp = adata.copy()
    grps = adata.obs[groupby]
    groups = np.unique(grps)
    
    # Retrieve the general differential expression table
    diff_tab_general = sc.get.rank_genes_groups_df(
        adTemp, None, pval_cutoff=pval, key=key_name
    )
    diff_tab_general = diff_tab_general.sort_values(
        by=["group", "pct_nz_group"], ascending=[True, False]
    )
    
    gene_list = []
    gene_dict = {}
    
    # Build kNN graph from dendrogram correlation matrix
    celltype_graph = build_knn_graph(
        adata.uns["dendrogram_" + groupby]["correlation_matrix"],
        list(adata.obs[groupby].cat.categories),
        k_of_knn,
    )
    
    for g in groups:
        # Filter the differential expression table for the current group
        tempTab = diff_tab_general[diff_tab_general["group"] == g]
        
        # ----------------------------------------------------------------------
        # Main strategy: gather top genes from the general DE table
        # ----------------------------------------------------------------------
        xlist = get_top_genes_from_df(
            tempTab, topX=topX_per_diff_type, proportion_top=proportion_top
        )
        xlist += get_top_genes_from_df(
            tempTab,
            by_score=False,
            topX=topX_per_diff_type,
            min_in=bottom_min_in,
            max_out=bottom_max_out,
            proportion_top=proportion_top,
        )
        xlist += get_top_genes_from_df(
            tempTab,
            by_score=False,
            topX=topX_per_diff_type,
            min_in=top_min_in,
            max_out=top_max_out,
            proportion_top=proportion_top,
        )
        
        # ----------------------------------------------------------------------
        # Additional DE analysis: compare group g to its kNN neighbors
        # ----------------------------------------------------------------------
        other_groups = celltype_graph.vs[
            celltype_graph.neighbors(celltype_graph.vs.find(name=g).index)
        ]["name"]
        
        xdata = adata.copy()
        subsetDF = _rank_genes_subsets(
            xdata, groupby=groupby, grpA=[g], grpB=other_groups, layer=layer, pval = pval
        )
        
        # Add genes from the subset analysis
        xlist += get_top_genes_from_df(
            subsetDF, topX=topX_per_diff_type, proportion_top=proportion_top
        )
        xlist += get_top_genes_from_df(
            subsetDF,
            by_score=False,
            topX=topX_per_diff_type,
            min_in=bottom_min_in,
            max_out=bottom_max_out,
            proportion_top=proportion_top,
        )
        xlist += get_top_genes_from_df(
            subsetDF,
            by_score=False,
            topX=topX_per_diff_type,
            min_in=top_min_in,
            max_out=top_max_out,
            proportion_top=proportion_top,
        )
        
        # De-duplicate and store results for this cell type
        final_list = list(set(xlist))
        gene_dict[g] = final_list
        gene_list += final_list
        
        # ----------------------------------------------------------------------
        # FALLBACK LOGIC: Check if we reached min_genes for cluster g
        # ----------------------------------------------------------------------
        if len(gene_dict[g]) < min_genes:
            # For example, let's do a fallback comparison: group g vs all other cells
            # You can change the DE method, or compare with all clusters, or relax thresholds, etc.
            xdata_fallback = adata.copy()
            all_others = [x for x in groups if x != g]
            
            # E.g., run a fallback differential expression
            fallbackDF = _rank_genes_subsets(
                xdata_fallback, groupby=groupby, grpA=[g], grpB=all_others, layer=layer, pval=1
            )
            fallback_genes = get_top_genes_from_df(
                fallbackDF, topX=(min_genes * 2),  # e.g. more generous
                proportion_top=proportion_top
            )
            
            # Maybe also relax min_in / max_out conditions in fallback or skip them.
            # Combine fallback results
            fallback_final = list(set(gene_dict[g] + fallback_genes))
            
            # If we *still* don't meet min_genes, just take the top N from fallback
            if len(fallback_final) < min_genes:
                # force it to the top min_genes from fallback
                fallback_final = fallback_final[:min_genes]
            
            gene_dict[g] = fallback_final
            # Merge into overall gene_list
            gene_list += fallback_final
    
    # Combine all unique genes across all groups
    cgenes2 = list(set(gene_list))
    
    return cgenes2, grps, gene_dict



def _randomize(expDat: pd.DataFrame, num: int = 50) -> pd.DataFrame:
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

def _sc_makeClassifier(
    expTrain: pd.DataFrame,
    genes: np.ndarray,
    groups: np.ndarray,
    nRand: int = 70,
    ntrees: int = 2000,
    stratify: bool = False
) -> RandomForestClassifier:
    """
    Train a random forest classifier on gene expression data. Private

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
    randDat = _randomize(expTrain, num=nRand)
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

def classify_anndata(adata: AnnData, rf_tsp, nrand: int = 0):
    """Classifies cells in the `adata` object based on the given gene expression and cross-pair information using a
    random forest classifier in rf_tsp trained with the provided xpairs genes.

    Args:
        adata: An annotated data matrix containing the gene expression information for cells.
        rf_tsp: A list of random forest classifier parameters used for classification.
        nrand: Number of random permutations for the null distribution. Default is 0.

    Returns:
        Updates adata with classification results.
    """

    # Classify cells using the `_scn_predict` function
    classRes = _scn_predict(rf_tsp, adata, nrand=nrand)

    # add the classification result as to `obsm`
    # adNew = AnnData(classRes, obs=adata.obs, var=pd.DataFrame(index=categories))
    adata.obsm['SCN_score'] = classRes
    
    # Get the categories (i.e., predicted cell types) from the classification result
    # categories = classRes.columns.values
    # possible_classes = rf_tsp['classifier'].classes_
    possible_classes = pd.Categorical(classRes.columns)
    # Add a new column to `obs` for the predicted cell types
    predicted_classes = classRes.idxmax(axis=1)
    adata.obs['SCN_class_argmax'] = pd.Categorical(predicted_classes, categories=possible_classes, ordered=True)

    # store this for consistent coloring
    # adata.uns['SCN_class_colors'] = rf_tsp['ctColors']        

    # import matplotlib.colors as mcolors
    # celltype_colors = rf_tsp['ctColors']
    # mycolors = [celltype_colors[ct] for ct in adata.obs['SCN_class_argmax'].cat.categories]
    # cmap = mcolors.ListedColormap(mycolors)
    adata.uns['SCN_class_argmax_colors'] = rf_tsp['ctColors']

    # can return copy if called for
    # return adata if copy else None

def _scn_predict(rf_tsp, aDat, nrand = 2):
    
    if isinstance(aDat.X,np.ndarray):
        # in the case of aDat.X is a numpy array 
        aDat.X = anndata._core.views.ArrayView(aDat.X)
###    expDat= pd.DataFrame(data=aDat.X, index= aDat.obs.index.values, columns= aDat.var.index.values)
    expDat = pd.DataFrame(data=aDat.X.toarray(), index= aDat.obs.index.values, columns= aDat.var.index.values)
    expValTrans = _query_transform(expDat.reindex(labels=rf_tsp['tpGeneArray'], axis='columns', fill_value=0), rf_tsp['topPairs'])
    classRes_val = _rf_classPredict(rf_tsp['classifier'], expValTrans, numRand=nrand)
    return classRes_val

# there is an issue in that making the random profiles here will break later addition of results to original annData object
def _rf_classPredict(rfObj,expQuery,numRand=50):
    if numRand > 0 :
        randDat = _randomize(expQuery, num=numRand)
        expQuery = pd.concat([expQuery, randDat])
    xpreds = pd.DataFrame(rfObj.predict_proba(expQuery), columns= rfObj.classes_, index=expQuery.index)
    return xpreds


def train_classifier(
    adata: AnnData = None,
    groupby: str = None,
    n_rand: int = None,
    cell_type_to_color: dict = None,
    n_top_genes: int = 30,
    n_top_gene_pairs: int = 40,
    n_trees: int = 1000,
    prop_other: float = 0.5,
    layer: str = None,
    n_comps: int = 50,
    **kwargs
) -> dict:
    """Train a cell type classifier on labeled single-cell data.

    Trains a random forest classifier using gene pair features to classify
    cells by type. Assumes data is normalized and highly variable genes are defined.

    Args:
        adata: AnnData object with training data. Must have .var['highly_variable'] set.
        groupby: Column in .obs containing cell type labels.
        n_rand: Number of random samples to generate. Defaults to mean cells per type.
        cell_type_to_color: Optional dict mapping cell types to colors.
        n_top_genes: Number of top genes per differential type. Defaults to 30.
        n_top_gene_pairs: Number of top gene pairs to use. Defaults to 40.
        n_trees: Number of trees in random forest. Defaults to 1000.
        prop_other: Proportion of pairs from other cell types. Defaults to 0.5.
        layer: Layer to use for expression values. Defaults to None (uses .X).
        n_comps: Number of PCA components. Defaults to 50.

    Returns:
        Dictionary containing trained classifier and associated metadata.

    Example:
        >>> clf = train_classifier(adata, groupby='celltype', n_top_genes=50)
    """
    # Handle deprecated parameter names
    if 'aTrain' in kwargs:
        warnings.warn("aTrain is deprecated, use adata instead", DeprecationWarning, stacklevel=2)
        adata = kwargs.pop('aTrain')
    if 'dLevel' in kwargs:
        warnings.warn("dLevel is deprecated, use groupby instead", DeprecationWarning, stacklevel=2)
        groupby = kwargs.pop('dLevel')
    if 'nRand' in kwargs:
        warnings.warn("nRand is deprecated, use n_rand instead", DeprecationWarning, stacklevel=2)
        n_rand = kwargs.pop('nRand')
    if 'nTopGenes' in kwargs:
        warnings.warn("nTopGenes is deprecated, use n_top_genes instead", DeprecationWarning, stacklevel=2)
        n_top_genes = kwargs.pop('nTopGenes')
    if 'nTopGenePairs' in kwargs:
        warnings.warn("nTopGenePairs is deprecated, use n_top_gene_pairs instead", DeprecationWarning, stacklevel=2)
        n_top_gene_pairs = kwargs.pop('nTopGenePairs')
    if 'nTrees' in kwargs:
        warnings.warn("nTrees is deprecated, use n_trees instead", DeprecationWarning, stacklevel=2)
        n_trees = kwargs.pop('nTrees')
    if 'propOther' in kwargs:
        warnings.warn("propOther is deprecated, use prop_other instead", DeprecationWarning, stacklevel=2)
        prop_other = kwargs.pop('propOther')

    progress_total = 5
    with alive_bar(progress_total, title="Training classifier") as bar:
        warnings.filterwarnings('ignore')

        # auto determine n_rand = mean number of cells per type
        if n_rand is None:
            n_rand = int(np.floor(np.mean(adata.obs[groupby].value_counts())))

        n_comps = n_comps if 0 < n_comps < min(adata.shape) else min(adata.shape) - 1

        stTrain = adata.obs
        expRaw = adata.to_df()
        expRaw = expRaw.loc[stTrain.index.values]
        adNorm = adata.copy()
        # cluster for comparison to closest celltype neighbors

        sc.pp.pca(adNorm, n_comps=n_comps, mask_var='highly_variable')
        sc.tl.dendrogram(adNorm, groupby=groupby, linkage_method='average', use_rep='X_pca', n_pcs=n_comps)
        sc.tl.rank_genes_groups(adNorm, use_raw=False, layer=layer, groupby=groupby, mask_var='highly_variable', key_added=SCN_DIFFEXP_KEY, pts=True)

        expTnorm = adNorm.to_df()
        expTnorm = expTnorm.loc[stTrain.index.values]
        bar()  # Bar 1
        topX_per_diff_type = np.ceil(n_top_genes / 3)
        cgenesA, grps, cgenes_list = _get_classy_genes_3(adNorm, groupby=groupby, key_name=SCN_DIFFEXP_KEY, topX_per_diff_type=topX_per_diff_type, layer=layer)
        bar()  # Bar 2
        xpairs = _generate_gene_pairs(cgenes_list, npairs=n_top_gene_pairs)
        bar()  # Bar 3
        pdTrain = _query_transform(expRaw.loc[:, cgenesA], xpairs)

        tspRF = _sc_makeClassifier(pdTrain.loc[:, xpairs], genes=xpairs, groups=grps, nRand=n_rand, ntrees=n_trees)
        bar()  # Bar 4

        # set celltype colors - add 'rand' celltype
        if cell_type_to_color is None:
            cell_types = stTrain[groupby].cat.categories.to_list()
            cell_types.append('rand')
            unique_colors = get_unique_colors(len(cell_types))
            cell_type_to_color = {cell_type: color for cell_type, color in zip(cell_types, unique_colors)}
            cell_type_to_color['rand'] = np.array(SCN_CATEGORY_COLOR_DICT['Rand'])
        bar()  # Bar 5

        argList = {'n_rand': n_rand, 'n_top_genes': n_top_genes, 'n_top_gene_pairs': n_top_gene_pairs, 'n_trees': n_trees, 'prop_other': prop_other}

    return {'tpGeneArray': cgenesA, 'topPairs':xpairs, 'classifier': tspRF, 'diffExpGenes':cgenes_list, 'ctColors':cell_type_to_color, 'argList': argList}


def compute_celltype_proportions(adata, celltype_col='celltype', stage_col='stage', exclude=None):
    """Compute the proportion of cell types per stage in the provided AnnData object.

    Args:
        adata: The AnnData object containing scRNA-seq data.
        celltype_col: The name of the .obs column containing cell type information. Default is 'celltype'.
        stage_col: The name of the .obs column containing stage information. Default is 'stage'.
        exclude: A list of cell types to exclude from the calculation. Default is None.

    Returns:
        A DataFrame with stages as rows, cell types as columns, and proportions as values.
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
    """Compute the correlation between cell type proportions in the query AnnData object and
    the reference proportions for each stage.

    Args:
        reference_proportions: The output of the compute_celltype_proportions function, representing the reference proportions.
        query_adata: The AnnData object to compute proportions on for correlation analysis.
        celltype_col: The name of the .obs column containing cell type information in the query AnnData object. Default is 'SCN_class'.

    Returns:
        Correlations between the query proportions and each stage in the reference proportions.
    """
    # Compute the cell type proportions in the query data
    query_counts = query_adata.obs[celltype_col].value_counts(normalize=True)
    
    # Align with the reference data
    query_proportions = query_counts.reindex(reference_proportions.columns, fill_value=0)
    
    # Compute correlation with each stage in the reference proportions
    correlations = reference_proportions.apply(lambda x: x.corr(query_proportions), axis=1)
    
    return correlations


def _generate_gene_pairs(
    cgenes_list: dict,
    npairs: int = 50,
    genes_const: list = None,
    prop_other: float = 0.4,
    prop_same: float = 0.4
) -> np.ndarray:
    """
    Generate a random selection of gene pairs for each list in cgenes_list and 
    return them as a flattened NumPy array of unique pair strings.
    
    :param cgenes_list: A dict where each value is a list of genes. 
                       Example: {'A': ['geneA1', 'geneA2'], 'B': ['geneB1', 'geneB2', ...], ...}
    :param npairs: Total number of gene pairs to generate for each key in cgenes_list.
    :param genes_const: Optional list of genes to draw from for the 'const' category.
    :param prop_other: Proportion of pairs (out of npairs) whose second gene will be from OTHER lists.
    :param prop_same: Proportion of pairs whose second gene will be from the SAME list as the first gene.
    :return: A 1D NumPy array of unique gene pairs (e.g. ["gene1_gene2", "geneA_geneB", ...]).
    
    The function ensures that:
      1) prop_other + prop_same <= 1.0
      2) Each list in cgenes_list has length > 0
      3) If genes_const is provided (and non-empty), then prop_other + prop_same < 1.0
      
    If genes_const is used, then prop_const = 1 - (prop_other + prop_same). 
    This will be the proportion of pairs whose second gene is drawn from genes_const.
    
    Additionally, no gene pair will be made of the same two genes (i.e. g1 != g2).
    """

    def pick_different_gene(current_gene, gene_pool):
        """
        Return a random gene from gene_pool that is not equal to current_gene.
        If no such gene is found, return None.
        """
        valid_genes = [g for g in gene_pool if g != current_gene]
        if not valid_genes:
            return None
        return rand.choice(valid_genes)
    
    # 1) Basic checks
    if (prop_other + prop_same) > 1.0:
        raise ValueError("prop_other + prop_same cannot exceed 1.0.")
    
    # 2) Check that each list in cgenes_list has length > 0
    for key, gene_list in cgenes_list.items():
        if len(gene_list) == 0:
            raise ValueError(f"The gene list for key '{key}' is empty. Each list must have length > 0.")
    
    # 3) If genes_const is given, check prop_other + prop_same < 1 and compute prop_const
    prop_const = 0.0
    if genes_const is not None and len(genes_const) > 0:
        if (prop_other + prop_same) >= 1.0:
            raise ValueError("If genes_const is used, prop_other + prop_same must be < 1.0.")
        prop_const = 1.0 - (prop_other + prop_same)
    else:
        # If genes_const is None or empty, we don't use the const category at all
        genes_const = []
    
    # This will hold *all* pairs (flattened) across all keys
    all_pairs = []
    
    # For each group in cgenes_list, generate npairs of gene pairs
    for key, gene_list in cgenes_list.items():
        # Determine how many pairs come from 'same', 'other', and 'const'
        same_count = int(round(npairs * prop_same))
        other_count = int(round(npairs * prop_other))
        const_count = int(round(npairs * prop_const))
        
        # Fix any rounding discrepancy
        total_assigned = same_count + other_count + const_count
        leftover = npairs - total_assigned
        
        # Distribute leftover to one of the categories (simplest: add to 'same')
        same_count += leftover
        
        # --- SAME pairs (both genes from same list, g2 != g1) ---
        for _ in range(same_count):
            g1 = rand.choice(gene_list)
            g2 = pick_different_gene(g1, gene_list)
            if g2 is not None:
                all_pairs.append(f"{g1}_{g2}")
        
        # --- OTHER pairs (second gene from other lists, g2 != g1) ---
        other_genes = []
        for other_key, other_list in cgenes_list.items():
            if other_key != key:
                other_genes.extend(other_list)
        
        for _ in range(other_count):
            g1 = rand.choice(gene_list)
            g2 = pick_different_gene(g1, other_genes)
            if g2 is not None:
                all_pairs.append(f"{g1}_{g2}")
        
        # --- CONST pairs (second gene from genes_const, g2 != g1) ---
        for _ in range(const_count):
            g1 = rand.choice(gene_list)
            g2 = pick_different_gene(g1, genes_const)
            if g2 is not None:
                all_pairs.append(f"{g1}_{g2}")
    
    # Convert to a NumPy array of unique values
    return np.unique(all_pairs)








