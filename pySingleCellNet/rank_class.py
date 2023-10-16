import scanpy as sc
import numpy as np
import pySingleCellNet as pySCN
import scanpy as sc
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import is_sparse
from sklearn.decomposition import FastICA
import scanpy as sc
import numpy as np

def rank_dense_submatrix(submatrix):
    # Operate on a dense submatrix to get the ranks
    return np.apply_along_axis(lambda x: np.argsort(np.argsort(x)), 1, submatrix)

def rank_genes_fast(adata):
    """
    Efficiently replace each gene's expression value with its rank for each cell.
    
    Parameters:
    -----------
    adata : AnnData
        The annotated data matrix.

    Returns:
    --------
    AnnData
        The modified AnnData object with ranked expression values.
    """
    
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
    sc.tl.filter_rank_genes_groups(adNorm, min_fold_change=min_fold_change, min_in_group_fraction=min_in_group_fraction, max_out_group_fraction=max_out_group_fraction)
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
def find_gene_modules(
    adata,
    use_hvg = True,
    knn = 10,
    n_pcs = 50,
    prefix='gmod_'
):
    adtemp = adata.copy()
    if use_hvg:
        # add test that hvg is set
        hvg_names = adata.var[adata.var['highly_variable']].index.tolist()
        adtemp = adtemp[:,hvg_names].copy()
    adata_T = adtemp.T
    sc.tl.pca(adata_T)
    sc.pp.neighbors(adata_T, n_neighbors=knn, n_pcs=n_pcs)
    # sc.tl.umap(adata_T)
    sc.tl.leiden(adata_T)
    adf = adata_T.obs.copy()
    clusters = adf.groupby('leiden').apply(lambda x: x.index.tolist()).to_dict()
    clusters = {prefix + k: v for k, v in clusters.items()}
    adata.uns['gene_modules'] = clusters


def score_gene_modules(
    adata
):
    gene_dict = adata.uns['gene_modules']
    # Number of cells and clusters
    n_cells = adata.shape[0]
    # Initialize an empty matrix for scores
    # scores_matrix = np.zeros((n_cells, n_clusters))
    scores_df = pd.DataFrame(index=adata.obs_names)
    # For each cluster, calculate the gene scores and store in the DataFrame
    for cluster, genes in gene_dict.items():
        score_name = cluster
        sc.tl.score_genes(adata, gene_list=genes, score_name=score_name)        
        # Store the scores in the DataFrame
        scores_df[score_name] = adata.obs[score_name].values
        del(adata.obs[score_name])
    # Assign the scores DataFrame to adata.obsm
    adata.obsm['module_scores'] = scores_df


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
        gene_modules[f"module_{i}"] = gene_names.tolist()
    # Store gene modules in adata.uns
    adata.uns['ica_modules'] = gene_modules
    #return adata



