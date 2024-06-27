import pandas as pd
import scanpy as sc
import numpy as np
import pySingleCellNet as pySCN
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from scipy.sparse import issparse
from scipy.stats import rankdata
from sklearn.decomposition import FastICA
from .utils import *

def train_rank_classifier(adata, dLevel, nRand: int = 200, n_trees = 1000):
    """
    convert adata to ranked values and pass to sc_makeClassifier, reformat result to make compatible with 
    scn_classify
    """
    labels = adata.obs[dLevel].to_list()
    xgenes = adata.var_names.to_list()

    adTrainRank = pySCN.rank_genes_fast2(adata)

    stTrain= adTrainRank.obs.copy()
    expTrain = adTrainRank.to_df()
    expTrain = expTrain.loc[stTrain.index.values]
    
    clf = pySCN.sc_makeClassifier(expTrain, xgenes, labels, nRand=nRand, ntrees = n_trees)
    return {'tpGeneArray':  None, 'topPairs':None, 'classifier': clf, 'diffExpGenes':None}


def rank_classify(adata: AnnData, rf_rank, nrand: int = 0, copy=False) -> AnnData:

    adQuery =  pySCN.rank_genes_fast2(adata)
    stQuery = adata.obs.copy()
    expQuery = adQuery.to_df()
    expQuery = expQuery.loc[stQuery.index.values]

    # adHO_1r.X = anndata._core.views.ArrayView(adHO_1r.X)
    #### newx = pd.DataFrame(data=adQuery.X.toarray(), index = adQuery.obs.index.values, columns= adQuery.var.index.values)
    clf = rf_rank['classifier']
    feature_names = clf.feature_names_in_
    #### newx = newx.reindex(labels=feature_names, axis='columns', fill_value=0)
    #### xans = pySCN.rf_classPredict(clf, newx, numRand=nrand)
    expQuery = expQuery.reindex(labels=feature_names, axis='columns', fill_value=0)
    xans = pySCN.rf_classPredict(clf, expQuery, numRand=nrand)
    adata.obsm['SCN_score'] = xans.copy()
    adata.obs['SCN_class'] = xans.idxmax(axis=1)
    return adata if copy else None


def rank_dense_submatrix(submatrix):
    # Operate on a dense submatrix to get the ranks
    # return np.apply_along_axis(lambda x: np.argsort(np.argsort(x)), 1, submatrix)
    return np.apply_along_axis(lambda x: rankdata(-x, method='average') - 1, 1, submatrix)

def rank_genes_fast2(adOrig):
    """
    Efficiently replace each gene's expression value with its rank for each cell such that the highest values get the highest ranks
    genes that are not detected are assigned rank of 0
    ties are resolved by averaging
    Parameters:
    -----------
    adata : AnnData
        The annotated data matrix.

    Returns:
    --------
    AnnData
        an AnnData object with ranked expression values as described above
    """
    adata = adOrig.copy()
    # Check if matrix is sparse and get total number of rows
    is_sparse_mtx = issparse(adata.X)
    n_rows = adata.X.shape[0]
    if is_sparse_mtx:
        ranked_matrix = rank_dense_submatrix(adata.X.toarray())
    else:
        ranked_matrix = rank_dense_submatrix(adata.X)

    # handle undetected genes
    ranked_matrix = zero_out_b_where_a_is_zero(adata.X, ranked_matrix)
    
    # Update the AnnData object's expression matrix with ranks
    # return ranked_matrix
    adata.X = ranked_matrix
    return adata
    

def zero_out_b_where_a_is_zero(a, b):
    """
    Set entries in matrix `b` to 0 where the corresponding entries in matrix `a` are 0.
    
    Parameters:
    a (csr_matrix or np.ndarray): Sparse matrix `a`.
    b (csr_matrix or np.ndarray): Sparse matrix `b`, which will be modified in-place.
    
    Returns:
    csr_matrix: Modified sparse matrix `b`.
    """
    # Convert to csr_matrix if necessary
    if not sp.isspmatrix_csr(a):
        a = csr_matrix(a)
    if not sp.isspmatrix_csr(b):
        b = csr_matrix(b)
    # Create a mask where `a` is non-zero
    a_nonzero_mask = a.copy()
    a_nonzero_mask.data = np.ones_like(a_nonzero_mask.data)
    # Element-wise multiply b with the mask
    b = b.multiply(a_nonzero_mask)
    return b


def rank_genes_fast(adOrig):
    """
    Efficiently replace each gene's expression value with its rank for each cell.
    
    Parameters:
    -----------
    adOrig : AnnData
        The annotated data matrix.

    Returns:
    --------
    AnnData
        A new AnnData object with ranked expression values.
    """
    
    adata = adOrig.copy()
    # Check if matrix is sparse and get total number of rows
    is_sparse_mtx = issparse(adata.X)
    n_rows = adata.X.shape[0]
        
    if is_sparse_mtx:
        ranked_matrix = rank_dense_submatrix(adata.X.toarray())
    else:
        ranked_matrix = rank_dense_submatrix(adata.X)
    
    # Update the AnnData object's expression matrix with ranks
    adata.X = ranked_matrix
    return adata


def top_genes_pca(adata, n_pcs, top_x_genes):
    """
    Get the top contributing genes for the first n_pcs principal components.

    Parameters:
    -----------
    adata : AnnData
        The annotated data matrix.
    n_pcs : int
        Number of top principal components to consider.
    top_x_genes : int
        Number of top contributing genes to fetch for each principal component.

    Returns:
    --------
    numpy.ndarray
        Array of top contributing gene names.
    """
    # Run PCA if not already present in adata
    if 'X_pca' not in adata.obsm.keys():
        sc.tl.pca(adata, n_comps=n_pcs)
    # Extract loadings
    loadings = adata.varm['PCs'][:, :n_pcs]
    # Identify top genes for each PC
    top_genes_idx = np.argpartition(np.abs(loadings), -top_x_genes, axis=0)[-top_x_genes:, :]
    # Flatten indices array and take unique values
    unique_top_genes_idx = np.unique(top_genes_idx)
    # Map indices to gene names
    top_genes = adata.var_names[unique_top_genes_idx].values
    return top_genes

def extract_top_genes(
    adata,
    dLevel,
    topX=25
):
    # Get the list of clusters
    clusters = adata.obs[dLevel].cat.categories.tolist()
    # Dictionary to store top genes for each cluster
    cluster_genes_dict = {}
    for cluster in clusters:
        # Extract ranked genes for the specific cluster
        gene_names = adata.uns['rank_genes_groups_filtered']['names'][cluster]
        # Filter out 'nan' values
        valid_genes = [gene for gene in gene_names if str(gene) != 'nan']
        # Get topX genes from the valid genes list
        cluster_genes = valid_genes[:topX] if len(valid_genes) >= topX else valid_genes
        cluster_genes_dict[cluster] = cluster_genes
    return cluster_genes_dict


def score_clusters_to_obsm(adx, gene_dict):
    adata = adx.copy()
    # Number of cells and clusters
    n_cells = adata.shape[0]
    # Initialize an empty matrix for scores
    # scores_matrix = np.zeros((n_cells, n_clusters))
    scores_df = pd.DataFrame(index=adata.obs_names)
    # For each cluster, calculate the gene scores and store in the DataFrame
    for cluster, genes in gene_dict.items():
        score_name = f"score_{cluster}"
        sc.tl.score_genes(adata, gene_list=genes, score_name=score_name)        
        # Store the scores in the DataFrame
        scores_df[score_name] = adata.obs[score_name].values
        del(adata.obs[score_name])
    # Assign the scores DataFrame to adata.obsm
    adata.obsm['gene_scores'] = scores_df
    return adata

# Usage:
# scored_adata = score_clusters_to_obsm(adata, signature_lists)

def findSigGenes(
    adDat,
    dLevel,
    topX = 25,
    min_fold_change = 1,
    min_in_group_fraction = 0.30,
    max_out_group_fraction = 0.20,
    method='wilcoxon'
):
    adTemp = adDat.copy()
    grps = adDat.obs[dLevel]
    groups = np.unique(grps)
    sc.tl.rank_genes_groups(adTemp, dLevel, use_raw=False, method=method)
    sc.tl.filter_rank_genes_groups(adTemp, min_fold_change=min_fold_change, min_in_group_fraction=min_in_group_fraction, max_out_group_fraction=max_out_group_fraction)
    tempTab = pd.DataFrame(adTemp.uns['rank_genes_groups_filtered']['names']).head(topX)
    cgenes = {}
    for g in groups:
        temp = tempTab[g] 
        cgenes[g] = temp.to_numpy()
    return cgenes


"""
    Attempts to find gene modules, or sets of genes that have similar expression patterns in adata

    Parameters:
    -----------
    adata : AnnData
        The annotated data matrix.
    use_hvg : Boolean
        Whether to limit search to highly variable genes
    knn : int
        number of nearest neighbors to consider
    n_pcs : int
        Number of top principal components to consider.
    prefix : string
        prepended to gene module index

    Returns:
    --------
    adds the following to the passed adata:
    
    .uns['gene_modules'] : dict
        dictionary of cluster name : gene list

    """
def find_knn_modules(
    adata,
    mean_cluster = True,
    dLevel = 'leiden',
    use_hvg = True,
    knn = 5,
    leiden_resolution=0.5,
    prefix='gmod_',
    npcs_adjust = 1
):
    adOps = adata.copy()
    if use_hvg:
        # add test that hvg is set
        hvg_names = adata.var[adata.var['highly_variable']].index.tolist()
        adOps = adOps[:,hvg_names].copy()
    if mean_cluster:
        adtemp = adOps.copy()
        if dLevel not in adtemp.obs.columns:
            raise ValueError(dLevel + " not in obs.")
        compute_mean_expression_per_cluster(adtemp, dLevel)
        adOps = adtemp.uns['mean_expression'].copy()        
    adata_T = adOps.T
    sc.tl.pca(adata_T)
    elbow = find_elbow(adata_T)
    n_pcs = elbow + npcs_adjust 
    sc.pp.neighbors(adata_T, n_neighbors=knn, n_pcs=n_pcs, metric='correlation')
    sc.tl.leiden(adata_T, leiden_resolution)
    adf = adata_T.obs.copy()
    clusters = adf.groupby('leiden', observed=True).apply(lambda x: x.index.tolist()).to_dict()
    clusters = {prefix + k: v for k, v in clusters.items()}
    adata.uns['knn_modules'] = clusters
    pySCN.score_gene_modules(adata, method='knn')


def score_gene_modules(
    adata,
    method = 'knn'
):
    uns_name = method + "_modules"
    gene_dict = adata.uns[uns_name]
    # Number of cells and clusters
    n_cells = adata.shape[0]
    # Initialize an empty matrix for scores
    # scores_matrix = np.zeros((n_cells, n_clusters))
    scores_df = pd.DataFrame(index=adata.obs_names)
    # For each cluster, calculate the gene scores and store in the DataFrame
    for cluster, genes in gene_dict.items():
        score_name = cluster
        sc.tl.score_genes(adata, gene_list=genes, score_name=score_name, use_raw=False)        
        # Store the scores in the DataFrame
        scores_df[score_name] = adata.obs[score_name].values
        del(adata.obs[score_name])
    # Assign the scores DataFrame to adata.obsm
    obsm_name = method + "_module_scores"
    adata.obsm[obsm_name] = scores_df


def identify_ica_gene_modules(adata, k=10, max_iter=3):
    """
    Identify gene modules using FastICA and store in adata.uns['ica_modules'].

    Parameters:
    - adata : anndata.AnnData
        The input AnnData object with computed HVGs.

    Returns:
    - anndata.AnnData
        The modified AnnData object with ICA modules stored in uns['ica_modules'].
    """
    # Check if HVGs have been computed
    if 'highly_variable' not in adata.var.columns:
        raise ValueError("HVGs have not been computed for the provided AnnData object.")
    # Extract expression matrix of HVGs
    hvg_data = adata[:, adata.var['highly_variable']].X
    if issparse(hvg_data):
        hvg_data = hvg_data.toarray()
    # Perform FastICA
    ica = FastICA(n_components = k, max_iter=max_iter)
    ica_components = ica.fit_transform(hvg_data)
    # Identify gene modules based on FastICA components
    # Here, I'm assuming that genes in the same component are considered a module.
    # You may need a more sophisticated approach for this step.
    gene_modules = {}
    for i, component in enumerate(ica.components_):
        # Get the names of the genes in the current component/module
        gene_names = adata.var_names[adata.var['highly_variable']][component.argsort()[-10:][::-1]]  # Top 10 genes as an example
        gene_modules[f"gmod_{i}"] = gene_names.tolist()
    # Store gene modules in adata.uns
    adata.uns['ica_modules'] = gene_modules
    #return adata



