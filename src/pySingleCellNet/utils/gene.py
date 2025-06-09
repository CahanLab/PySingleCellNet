import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy import sparse
# import string
import igraph as ig
from typing import Dict, List
from .adataTools import find_elbow

from typing import Union
from anndata import AnnData

import scanpy as sc
import numpy as np
from scipy import sparse

def build_gene_knn_graph(
    adata,
    mask_var: str = None,
    mean_cluster: bool = True,
    groupby: str = 'leiden',
    knn: int = 5,
    use_knn: bool = True,
    metric: str = "euclidean",
    key: str = "gene"
):
    """
    Compute a gene–gene kNN graph (hard or Gaussian‑weighted) and store sparse connectivities & distances in adata.uns.

    Parameters
    ----------
    adata
        AnnData object (cells × genes). Internally transposed to (genes × cells).
    mask_var
        If not None, must be a column name in adata.var of boolean values.
        Only genes where adata.var[mask_var] == True are included. If None, use all genes.
    mean_cluster
        If True, aggregate cells by cluster defined in adata.obs[groupby].
        The kNN graph is computed on the mean‑expression profiles of each cluster
        (genes × n_clusters) rather than genes × n_cells.
    groupby
        Column in adata.obs holding cluster labels. Only used if mean_cluster=True.
    knn
        Integer: how many neighbors per gene to consider.
        Passed as n_neighbors=knn to sc.pp.neighbors.
    use_knn
        Boolean: passed to sc.pp.neighbors as knn=use_knn. 
        - If True, builds a hard kNN graph (only k nearest neighbors).  
        - If False, uses a Gaussian kernel to weight up to the k-th neighbor.
    metric
        Distance metric for kNN computation (e.g. "euclidean", "manhattan", "correlation", etc.).
        If metric=="correlation" and the gene‑expression matrix is sparse, it will be converted to dense.
    key
        Prefix under which to store results in adata.uns. The function sets:
          - adata.uns[f"{key}_gene_index"]
          - adata.uns[f"{key}_connectivities"]
          - adata.uns[f"{key}_distances"]
    """
    # 1) Work on a shallow copy so we don’t overwrite adata.X prematurely
    adata_work = adata.copy()

    # 2) If mask_var is provided, subset to only those genes first
    if mask_var is not None:
        if mask_var not in adata_work.var.columns:
            raise ValueError(f"Column '{mask_var}' not found in adata.var.")
        gene_mask = adata_work.var[mask_var].astype(bool)
        selected_genes = adata_work.var.index[gene_mask].tolist()
        if len(selected_genes) == 0:
            raise ValueError(f"No genes found where var['{mask_var}'] is True.")
        adata_work = adata_work[:, selected_genes].copy()

    # 3) If mean_cluster=True, aggregate by cluster label in `groupby`
    if mean_cluster:
        if groupby not in adata_work.obs.columns:
            raise ValueError(f"Column '{groupby}' not found in adata.obs.")
        # Aggregate each cluster to its mean expression; stored in .layers['mean']
        adata_work = sc.get.aggregate(adata_work, by=groupby, func='mean')
        # Overwrite .X with the mean‑expression matrix
        adata_work.X = adata_work.layers['mean']

    # 4) Transpose so that each gene (or cluster‑mean) is one “observation”
    adata_genes = adata_work.T.copy()

    # 5) If metric=="correlation" and X is sparse, convert to dense
    if metric == "correlation" and sparse.issparse(adata_genes.X):
        adata_genes.X = adata_genes.X.toarray()

    # 6) Compute neighbors on the (genes × [cells or clusters]) matrix.
    #    Pass n_neighbors=knn and knn=use_knn. Default method selection in Scanpy will
    #    use 'umap' if use_knn=True, and 'gauss' if use_knn=False.
    sc.pp.neighbors(
        adata_genes,
        n_neighbors=knn,
        knn=use_knn,
        metric=metric,
        use_rep="X"
    )

    # 7) Extract the two sparse matrices from adata_genes.obsp:
    conn = adata_genes.obsp["connectivities"].copy()  # CSR: gene–gene adjacency weights
    dist = adata_genes.obsp["distances"].copy()       # CSR: gene–gene distances

    # 8) Record the gene‑order (after masking + optional aggregation)
    gene_index = np.array(adata_genes.obs_names)

    adata.uns[f"{key}_gene_index"]      = gene_index
    adata.uns[f"{key}_connectivities"] = conn
    adata.uns[f"{key}_distances"]      = dist


def query_gene_neighbors(
    adata,
    gene: str,
    n_neighbors: int = 5,
    key: str = "gene",
    use: str = "connectivities"
):
    """
    Retrieve the top `n_neighbors` nearest genes to `gene`, using a precomputed gene–gene kNN graph
    stored in adata.uns (as produced by build_gene_knn_graph).

    This version handles both sparse‐CSR matrices and dense NumPy arrays in adata.uns.

    Parameters
    ----------
    adata
        AnnData that has the following keys in adata.uns:
          - adata.uns[f"{key}_gene_index"]      (np.ndarray of gene names, in order)
          - adata.uns[f"{key}_connectivities"]  (CSR sparse matrix or dense ndarray)
          - adata.uns[f"{key}_distances"]       (CSR sparse matrix or dense ndarray)
    gene
        Gene name (must appear in `adata.uns[f"{key}_gene_index"]`).
    n_neighbors
        Number of neighbors to return.
    key
        Prefix under which the kNN graph was stored. For example, if build_gene_knn_graph(...)
        was called with `key="gene"`, the function will look for:
          - adata.uns["gene_gene_index"]
          - adata.uns["gene_connectivities"]
          - adata.uns["gene_distances"]
    use
        One of {"connectivities", "distances"}.  
        - If "connectivities", neighbors are ranked by descending connectivity weight.  
        - If "distances", neighbors are ranked by ascending distance (only among nonzero entries).

    Returns
    -------
    neighbors : List[str]
        A list of gene names (length ≤ n_neighbors) that are closest to `gene`.
    """
    if use not in ("connectivities", "distances"):
        raise ValueError("`use` must be either 'connectivities' or 'distances'.")

    idx_key = f"{key}_gene_index"
    conn_key = f"{key}_connectivities"
    dist_key = f"{key}_distances"

    if idx_key not in adata.uns:
        raise ValueError(f"Could not find `{idx_key}` in adata.uns.")
    if conn_key not in adata.uns or dist_key not in adata.uns:
        raise ValueError(f"Could not find `{conn_key}` or `{dist_key}` in adata.uns.")

    gene_index = np.array(adata.uns[idx_key])
    if gene not in gene_index:
        raise KeyError(f"Gene '{gene}' not found in {idx_key}.")
    i = int(np.where(gene_index == gene)[0][0])

    # Select the appropriate stored matrix (could be sparse CSR or dense ndarray)
    mat_key = conn_key if use == "connectivities" else dist_key
    stored = adata.uns[mat_key]

    # If stored is a NumPy array, treat it as a dense full matrix:
    if isinstance(stored, np.ndarray):
        row_vec = stored[i].copy()
        # Exclude self
        if use == "connectivities":
            row_vec[i] = -np.inf
            order = np.argsort(-row_vec)  # descending
        else:
            row_vec[i] = np.inf
            order = np.argsort(row_vec)   # ascending
        topk = order[:n_neighbors]
        return [gene_index[j] for j in topk]

    # Otherwise, assume stored is a sparse matrix (CSR or similar):
    if not sparse.issparse(stored):
        raise TypeError(f"Expected CSR or ndarray for `{mat_key}`, got {type(stored)}.")

    row = stored.getrow(i)
    # For connectivities: sort nonzero entries by descending weight
    if use == "connectivities":
        cols = row.indices
        weights = row.data
        mask = cols != i
        cols = cols[mask]
        weights = weights[mask]
        if weights.size == 0:
            return []
        order = np.argsort(-weights)
        topk = cols[order][:n_neighbors]
        return [gene_index[j] for j in topk]

    # For distances: sort nonzero entries by ascending distance
    else:  # use == "distances"
        cols = row.indices
        dists = row.data
        mask = cols != i
        cols = cols[mask]
        dists = dists[mask]
        if dists.size == 0:
            return []
        order = np.argsort(dists)
        topk = cols[order][:n_neighbors]
        return [gene_index[j] for j in topk]




def score_gene_modules(
    adata,
    gene_dict: dict,
    key_added: str = "module_scores"
):

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
    obsm_name = key_added
    adata.obsm[obsm_name] = scores_df



def find_knn_modules(
    adata,
    mean_cluster: bool = True,
    groupby: str = 'leiden',
    mask_var: str = None,
    knn: int = 5,
    leiden_resolution: float = 0.5,
    prefix: str = 'gmod_',
    metric='euclidean'
):
    """
    Finds gene modules by building a kNN graph on raw (or aggregated) expression profiles 
    and clustering with Leiden. Results are written to adata.uns['knn_modules'] in-place.

    Parameters:
    -----------
    adata
        AnnData object to process.
    mean_cluster
        If True, compute the mean expression per cluster defined in `groupby`, otherwise use all cells.
    groupby
        Column in adata.obs to use for clustering when mean_cluster is True.
    mask_var
        Column name in adata.var used to select a subset of genes (boolean mask). If None, use all genes.
    knn
        Number of neighbors to use in the kNN graph.
    leiden_resolution
        Resolution parameter for the Leiden clustering.
    prefix
        Prefix to add to each module name.
    metric
        Distance metric for kNN computation (e.g. 'euclidean', 'manhattan', 'correlation', etc.).
        If metric=='correlation', we force a dense array when the data are sparse.
    """
    # 1) Work on a copy so we don’t modify the user’s adata.X before we’re ready
    adata_subset = adata.copy()

    # 2) If mask_var is provided, subset on that boolean column in adata.var
    if mask_var is not None:
        if mask_var not in adata_subset.var.columns:
            raise ValueError(f"Column '{mask_var}' not found in adata.var.")
        gene_mask = adata_subset.var[mask_var].astype(bool)
        selected = adata_subset.var.index[gene_mask].tolist()
        if len(selected) == 0:
            raise ValueError(f"No genes found where var['{mask_var}'] is True.")
        adata_subset = adata_subset[:, selected].copy()

    # 3) If mean_cluster=True, aggregate cells by cluster and replace X with the mean 
    if mean_cluster:
        if groupby not in adata_subset.obs.columns:
            raise ValueError(f"Column '{groupby}' not found in adata.obs.")
        adata_subset = sc.get.aggregate(adata_subset, by=groupby, func='mean')
        adata_subset.X = adata_subset.layers['mean']

    # 4) Transpose so that genes (or cluster‐means) become observations
    adata_transposed = adata_subset.T.copy()

    # 5) If the metric is 'correlation' and X is sparse, convert to dense
    if metric == 'correlation' and sparse.issparse(adata_transposed.X):
        adata_transposed.X = adata_transposed.X.toarray()

    # 6) Build the kNN graph directly on .X (no PCA)
    sc.pp.neighbors(adata_transposed, n_neighbors=knn, metric=metric, n_pcs=0)

    # 7) Leiden clustering on that graph
    sc.tl.leiden(adata_transposed, resolution=leiden_resolution)

    # 8) Group by Leiden label to collect modules
    clusters = (
        adata_transposed.obs
        .groupby('leiden', observed=True)['leiden']
        .apply(lambda ser: ser.index.tolist())
        .to_dict()
    )
    modules = {f"{prefix}{cluster_id}": gene_list for cluster_id, gene_list in clusters.items()}

    # 9) Write modules back to the original AnnData (in-place)
    adata.uns['knn_modules'] = modules




def what_module_has_gene(
    adata,
    target_gene,
    mod_slot='knn_modules'
) -> list: 
    if mod_slot not in adata.uns.keys():
        raise ValueError(mod_slot + " have not been identified.")
    genemodules = adata.uns[mod_slot]
    return [key for key, genes in genemodules.items() if target_gene in genes]




def extract_top_bottom_genes(
    deg_res: dict,
    ngenes: int,
    sort_by: str = 'scores',
    extraction_map: Dict[str, str] = None
) -> Dict[str, List[str]]:
    """
    Extracts top and bottom ngenes from each gene table in deg_res and organizes them
    into a dictionary with combined keys of group and sample names.

    Parameters:
    -----------
    deg_res : dict
        A dictionary containing differential expression results with keys:
            - 'sample_names': List of sample names (e.g., ['Singular', 'None'])
            - 'geneTab_dict': Dictionary where each key is a group name and each value is
                              a Pandas DataFrame with gene information.
    ngenes : int
        The number of top or bottom genes to extract from each gene table.
    sort_by : str, optional (default='scores')
        The column name in the gene tables to sort by.
    extraction_map : dict, optional
        A dictionary mapping sample names to extraction behavior ('top' or 'bottom').
        If not provided, defaults to:
            - First sample name: 'top'
            - Second sample name: 'bottom'
            - Additional sample names: 'top'

    Returns:
    --------
    result_dict : dict
        A dictionary where each key is a combination of group and sample name
        (e.g., 'Meso.Nascent_Singular') and each value is a list of gene names.
        - For 'top', the list contains the top ngenes based on sort_by.
        - For 'bottom', the list contains the bottom ngenes based on sort_by.

    Raises:
    -------
    KeyError:
        If 'sample_names' or 'geneTab_dict' keys are missing in deg_res,
        or if 'sort_by' is not a column in the gene tables.
    ValueError:
        If ngenes is not a positive integer or if 'sample_names' does not contain at least one entry.
    TypeError:
        If the input types are incorrect.
    """
    # Input Validation
    required_keys = ['sample_names', 'geneTab_dict']
    for key in required_keys:
        if key not in deg_res:
            raise KeyError(f"Key '{key}' not found in deg_res.")

    sample_names = deg_res['sample_names']
    geneTab_dict = deg_res['geneTab_dict']

    if not isinstance(sample_names, list):
        raise TypeError(f"'sample_names' should be a list, got {type(sample_names)}.")

    if not isinstance(geneTab_dict, dict):
        raise TypeError(f"'geneTab_dict' should be a dict, got {type(geneTab_dict)}.")

    if not isinstance(ngenes, int) or ngenes <= 0:
        raise ValueError(f"'ngenes' must be a positive integer, got {ngenes}.")

    if len(sample_names) < 1:
        raise ValueError(f"'sample_names' should contain at least one entry, got {len(sample_names)}.")

    result_dict = {}

    # Define default extraction behavior if extraction_map is not provided
    if extraction_map is None:
        extraction_map = {}
        for idx, sample in enumerate(sample_names):
            if idx == 0:
                extraction_map[sample] = 'top'
            elif idx == 1:
                extraction_map[sample] = 'bottom'
            else:
                extraction_map[sample] = 'top'  # Default behavior for additional samples

    else:
        # Validate extraction_map
        if not isinstance(extraction_map, dict):
            raise TypeError(f"'extraction_map' should be a dict, got {type(extraction_map)}.")
        for sample, behavior in extraction_map.items():
            if behavior not in ['top', 'bottom']:
                raise ValueError(f"Invalid extraction behavior '{behavior}' for sample '{sample}'. Must be 'top' or 'bottom'.")

    # Iterate over each group in geneTab_dict
    for group, df in geneTab_dict.items():
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected DataFrame for group '{group}', got {type(df)}.")

        # Ensure required columns exist
        required_columns = ['names', sort_by]
        for col in required_columns:
            if col not in df.columns:
                raise KeyError(f"Column '{col}' not found in gene table for group '{group}'.")

        # Sort the DataFrame by 'sort_by' column
        sorted_df = df.sort_values(by=sort_by, ascending=False).reset_index(drop=True)

        # Iterate over sample names to determine extraction behavior
        for sample in sample_names:
            combined_key = f"{group}_{sample}"
            behavior = extraction_map.get(sample, 'top')  # Default to 'top' if not specified

            if behavior == 'top':
                # Extract top ngenes
                top_genes = sorted_df['names'].head(ngenes).tolist()
                result_dict[combined_key] = top_genes
            elif behavior == 'bottom':
                # Extract bottom ngenes
                bottom_genes = sorted_df['names'].tail(ngenes).tolist()
                result_dict[combined_key] = bottom_genes

    return result_dict


def rank_genes_subsets(
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



def pull_out_genes(
    diff_genes_dict: dict, 
    cell_type: str,
    category: str, 
    num_genes: int = 0,
    order_by = "logfoldchanges", 
    threshold = 2) -> list:

    ans = []
    #### xdat = diff_genes_dict[cell_type]
    xdat = diff_genes_dict['geneTab_dict'][cell_type]
    xdat = xdat[xdat['pvals_adj'] < threshold].copy()

    category_names = diff_genes_dict['category_names']
    category_index = category_names.index(category)
    
    # any genes left?
    if xdat.shape[0] > 0:

        if num_genes == 0:
            num_genes = xdat.shape[0]

        if category_index == 0:
            xdat.sort_values(by=[order_by], inplace=True, ascending=False)
        else:
            xdat.sort_values(by=[order_by], inplace=True, ascending=True)

        ans = list(xdat.iloc[0:num_genes]["names"])

    return ans


def pull_out_genes_v2(
    diff_genes_dict: dict, 
    cell_type: str,
    category: str, 
    num_genes: int = 0,
    order_by = "logfoldchanges", 
    threshold = 2) -> list:

    ans = []
    #### xdat = diff_genes_dict[cell_type]
    xdat = diff_genes_dict['geneTab_dict'][cell_type]
    xdat = xdat[xdat['pvals_adj'] < threshold].copy()

    dictkey = list(diff_genes_dict.keys())[0]
    category_names = diff_genes_dict[dictkey]
    category_index = category_names.index(category)
    
    # any genes left?
    if xdat.shape[0] > 0:

        if num_genes == 0:
            num_genes = xdat.shape[0]

        if category_index == 0:
            xdat.sort_values(by=[order_by], inplace=True, ascending=False)
        else:
            xdat.sort_values(by=[order_by], inplace=True, ascending=True)

        ans = list(xdat.iloc[0:num_genes]["names"])

    return ans



def remove_genes(adata, genes_to_exclude=None):
    adnew = adata[:,~adata.var_names.isin(genes_to_exclude)].copy()
    return adnew



def limit_anndata_to_common_genes(anndata_list):
    # Find the set of common genes across all anndata objects
    common_genes = set(anndata_list[0].var_names)
    for adata in anndata_list[1:]:
        common_genes.intersection_update(set(adata.var_names))
    
    # Limit the anndata objects to the common genes
    # latest anndata update broke this:
    if common_genes:
         for adata in anndata_list:
            adata._inplace_subset_var(list(common_genes))

    #return anndata_list
    #return common_genes


def add_ambient_rna(
    adata,
    obs_key: str,
    obs_val: str,
    n_cells_to_sample: int = 10,
    weight_of_ambient: float = 0.05
) -> ad.AnnData:

    # What cells will be updated?
    non_cluster_cells = adata[adata.obs[obs_key] != obs_val, :].copy()
    n_mean_cells = non_cluster_cells.n_obs

    # Generate the sample means
    sample_means_adata = create_mean_cells(adata, obs_key, obs_val, n_mean_cells, n_cells_to_sample)
    
    # If there are more non-cluster cells than samples, raise an error
    # if non_cluster_cells.shape[0] > nSamples:
    #    raise ValueError("The number of samples is less than the number of non-cluster cells.")
    
    # Update the non-cluster cells' expression states to be the weighted mean of their current states and the sample means
    non_cluster_cells.X = (1 - weight_of_ambient) * non_cluster_cells.X + weight_of_ambient * sample_means_adata.X
    
    # Update the original adata object with the new expression states
    adata[adata.obs[obs_key] != obs_val, :] = non_cluster_cells
    
    return adata


# to do: params for other species; error handling
def get_Y_chr_genes(adata):
    gene_chrs = sc.queries.biomart_annotations("mmusculus",["mgi_symbol", "ensembl_gene_id", "chromosome_name"],).set_index("mgi_symbol")
    ygenes = gene_chrs[gene_chrs["chromosome_name"]=='Y']
    ygenes = ygenes[ygenes.index.isin(adata.var.index)]
    return ygenes.index.tolist()


def score_sex(
    adata, 
    y_genes=['Eif2s3y', 'Ddx3y', 'Uty'], 
    x_inactivation_genes=['Xist', 'Tsix']
):
    """
    Adds sex chromosome expression scores to an AnnData object.
    
    This function calculates two scores for each cell in a scRNA-seq AnnData object:
      - Y_score: the sum of expression values for a set of Y-chromosome specific genes.
      - X_inact_score: the sum of expression values for genes involved in X-chromosome inactivation.
      
    The scores are added to the AnnData object's `.obs` DataFrame with the keys 'Y_score' and 'X_inact_score'.
    
    Parameters
    ----------
    adata : AnnData
        An AnnData object containing scRNA-seq data, with gene names in `adata.var_names`.
    y_genes : list of str, optional
        List of Y-chromosome specific marker genes (default is ['Eif2s3y', 'Ddx3y', 'Uty']).
    x_inactivation_genes : list of str, optional
        List of genes involved in X-chromosome inactivation (default is ['Xist', 'Tsix']).
        
    Raises
    ------
    ValueError
        If none of the Y-specific or X inactivation genes are found in `adata.var_names`.
    
    Returns
    -------
    None
        The function modifies the AnnData object in place by adding the score columns to `adata.obs`.
    """
    # Filter for genes that are available in the dataset.
    available_y_genes = [gene for gene in y_genes if gene in adata.var_names]
    available_x_genes = [gene for gene in x_inactivation_genes if gene in adata.var_names]
    
    if not available_y_genes:
        raise ValueError("None of the Y-specific genes were found in the dataset.")
    if not available_x_genes:
        raise ValueError("None of the X inactivation genes were found in the dataset.")
    
    # Compute the sum of expression for the Y-specific genes.
    y_expression = adata[:, available_y_genes].X
    if hasattr(y_expression, "toarray"):
        y_expression = y_expression.toarray()
    adata.obs['Y_score'] = np.sum(y_expression, axis=1)
    
    # Compute the sum of expression for the X inactivation genes.
    x_expression = adata[:, available_x_genes].X
    if hasattr(x_expression, "toarray"):
        x_expression = x_expression.toarray()
    adata.obs['X_inact_score'] = np.sum(x_expression, axis=1)
    
    # Optionally, you could log some output:
    print("Added 'Y_score' and 'X_inact_score' to adata.obs for {} cells.".format(adata.n_obs))
    
# Example usage:
# Assuming 'adata' is your AnnData object:
# add_sex_scores(adata)
# print(adata.obs[['Y_score', 'X_inact_score']].head())




