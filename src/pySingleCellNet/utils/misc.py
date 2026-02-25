from __future__ import annotations
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from typing import Dict, List, Optional, Set, Tuple, Union
import re
import igraph as ig
from scipy.sparse import issparse, coo_matrix, csr_matrix, hstack
import scipy.sparse as sp
from scipy.stats import median_abs_deviation, ttest_ind
from scipy import sparse
from sklearn.decomposition import PCA
import math
from scipy.sparse.csgraph import connected_components



def combine_pca_scores(adata, n_pcs=50, score_key='SCN_score'):
    """Combine principal components and gene set scores into a single matrix.

    This function merges the top principal components (PCs) and gene set scores 
    into a combined matrix stored in `.obsm`.

    Args:
        adata (AnnData): 
            AnnData object containing PCA results and gene set scores in `.obsm`.
        n_pcs (int, optional): 
            Number of top PCs to include. Default is 50.
        score_key (str, optional): 
            Key in `.obsm` where gene set scores are stored. Default is `'SCN_score'`.

    Raises:
        ValueError: If `'X_pca'` is not found in `.obsm`.  
        ValueError: If `score_key` is missing in `.obsm`.

    Returns:
        None: Updates `adata` by adding the combined matrix to `.obsm['X_pca_scores_combined']`.

    Example:
        >>> combine_pca_scores(adata, n_pcs=30, score_key='GeneSet_Score')
    """

    # Ensure that the required data exists in .obsm
    if 'X_pca' not in adata.obsm:
        raise ValueError("X_pca not found in .obsm. Perform PCA before combining.")
    
    if score_key not in adata.obsm:
        raise ValueError(f"{score_key} not found in .obsm. Please provide valid gene set scores.")
    
    # Extract the top n_pcs from .obsm['X_pca']
    pca_matrix = adata.obsm['X_pca'][:, :n_pcs]
    
    # Extract the gene set scores from .obsm
    score_matrix = adata.obsm[score_key]
    
    # Combine PCA matrix and score matrix horizontally (along columns)
    combined_matrix = np.hstack([pca_matrix, score_matrix])
    
    # Add the combined matrix back into .obsm with a new key
    adata.obsm['X_pca_scores_combined'] = combined_matrix

    print(f"Combined matrix with {n_pcs} PCs and {score_matrix.shape[1]} gene set scores added to .obsm['X_pca_scores_combined'].")








def impute_knn_dropout(
    adata,
    knn_key: str = "neighbors",
    layer_name: str = None
):
    """Impute zero-expression values using kNN weighted means.

    Replaces each zero in `adata.X` (or `adata.raw.X`) with the weighted mean
    of that gene over its kNN, where weights come from
    `adata.obsp[f"{knn_key}_connectivities"]`.

    Args:
        adata: Annotated data matrix. Reads from `adata.raw.X` if it exists;
            otherwise from `adata.X`.
        knn_key: Prefix for the two sparse matrices in `adata.obsp`:
            `adata.obsp[f"{knn_key}_connectivities"]` and
            `adata.obsp[f"{knn_key}_distances"]`. Defaults to "neighbors".
        layer_name: Name for the new layer to which the imputed expression
            matrix will be saved. Defaults to `f"{knn_key}_imputed"`.

    Returns:
        The same AnnData, with an extra entry
        `adata.layers[layer_name]` containing the imputed expression matrix
        (sparse if original was sparse).
    """

    # 1) Extract the “raw” or primary X matrix
    if hasattr(adata, "raw") and adata.raw is not None and adata.raw.X is not None:
        Xorig = adata.raw.X
    else:
        Xorig = adata.X

    # Convert to dense numpy for easy indexing/arithmetic
    was_sparse = sp.issparse(Xorig)
    X = Xorig.toarray() if was_sparse else Xorig.copy()

    # 2) Grab the connectivities matrix from adata.obsp
    conn_key = f"{knn_key}_connectivities"
    if conn_key not in adata.obsp:
        raise KeyError(
            f"No key '{conn_key}' found in adata.obsp. "
            f"Did you run `sc.pp.neighbors(adata, key_added='{knn_key}')`?"
        )

    C = adata.obsp[conn_key].tocsr()  # shape = (n_cells, n_cells), sparse

    # 3) Row‐normalize C so that each row sums to 1 (for weighted averaging)
    #    If a row already sums to zero (no neighbors—rare after pp.neighbors), we leave it zero.
    row_sums = np.array(C.sum(axis=1)).flatten()  # length = n_cells
    # Avoid division‐by‐zero:
    inv_row = np.zeros_like(row_sums)
    nonzero_mask = row_sums > 0
    inv_row[nonzero_mask] = 1.0 / row_sums[nonzero_mask]
    D_inv = sp.diags(inv_row)          # diagonal matrix of 1/(row_sums)

    W = D_inv.dot(C)  # now each row of W sums to 1 (except rows that were originally all zero)

    # 4) Compute the kNN‐weighted mean for every cell+gene:
    #    X_knn[i, g] = sum_j W[i, j] * X[j, g]
    X_knn = W.dot(X)  # shape = (n_cells, n_genes)

    # 5) Replace zeros in X with weighted neighbor means
    mask_zero = (X == 0)
    X_imputed = X.copy()
    X_imputed[mask_zero] = X_knn[mask_zero]

    # 6) Write the result into a new layer
    if layer_name is None:
        layer_name = f"{knn_key}_imputed"

    if was_sparse:
        adata.layers[layer_name] = sp.csr_matrix(X_imputed)
    else:
        adata.layers[layer_name] = X_imputed

    return adata



def read_broken_geo_mtx(path: str, prefix: str) -> ad.AnnData:
    """Read a GEO-deposited MTX file where obs and var may be swapped.

    Determines the correct orientation by comparing the dimensions of the
    matrix against the sizes of genes.tsv and barcodes.tsv, transposing
    if necessary.

    Args:
        path: Directory path containing the matrix files.
        prefix: Filename prefix prepended to 'matrix.mtx', 'barcodes.tsv',
            and 'genes.tsv'.

    Returns:
        An AnnData object with cells as observations and genes as variables.
    """
    adata = sc.read_mtx(path + prefix + "matrix.mtx")
    cell_anno = pd.read_csv(path + prefix + "barcodes.tsv", delimiter='\t', header=None)
    n_cells = cell_anno.shape[0]
    cell_anno.rename(columns={0:'cell_id'},inplace=True)

    gene_anno = pd.read_csv(path + prefix + "genes.tsv", header=None, delimiter='\t')
    n_genes = gene_anno.shape[0]
    gene_anno.rename(columns={0:'gene'},inplace=True)

    if adata.shape[0] == n_genes:
        adata = adata.T

    adata.obs = cell_anno.copy()
    adata.obs_names = adata.obs['cell_id']
    adata.var = gene_anno.copy()
    adata.var_names = adata.var['gene']
    return adata


def find_elbow(
    adata
):
    """Find the elbow index in the variance explained by principal components.

    Args:
        adata: AnnData object containing PCA results in
            `adata.uns['pca']['variance_ratio']`.

    Returns:
        The index corresponding to the elbow in the variance explained plot.
    """
    variance_explained = adata.uns['pca']['variance_ratio']
    # Coordinates of all points
    n_points = len(variance_explained)
    all_coords = np.vstack((range(n_points), variance_explained)).T
    # Line vector from first to last point
    line_vec = all_coords[-1] - all_coords[0]
    line_vec_norm = line_vec / np.sqrt(np.sum(line_vec**2))
    # Vector being orthogonal to the line
    vec_from_first = all_coords - all_coords[0]
    scalar_prod = np.sum(vec_from_first * np.tile(line_vec_norm, (n_points, 1)), axis=1)
    vec_from_first_parallel = np.outer(scalar_prod, line_vec_norm)
    vec_to_line = vec_from_first - vec_from_first_parallel
    # Distance to the line
    dist_to_line = np.sqrt(np.sum(vec_to_line ** 2, axis=1))
    # Index of the point with max distance to the line
    elbow_idx = np.argmax(dist_to_line)
    return elbow_idx


def assign_optimal_cluster(adata, cluster_reports, new_col="optimal_cluster"):
    """Determine the optimal cluster label per cell by comparing F1-scores.

    For each cell, selects the cluster assignment with the highest F1-score
    across multiple clustering schemes and prepends the chosen label with
    the name of the .obs column that provided it.

    Args:
        adata: The annotated single-cell dataset.
        cluster_reports: A dictionary where keys are column names in
            `adata.obs` (each corresponding to one clustering scheme),
            and values are DataFrames with classification metrics
            including 'Label' and 'F1-Score'.
        new_col: The name of the new `.obs` column in which the optimal
            cluster labels will be stored. Defaults to "optimal_cluster".

    Returns:
        None. Adds a new column to `adata.obs`.
    """
    # Prepare a list to hold the chosen cluster label (prepended with obs_col name) per cell
    optimal_labels = np.empty(adata.n_obs, dtype=object)
    
    # Convert each cluster report into a dictionary for faster F1 lookups:
    # For each clustering key, map cluster_label -> F1_score
    f1_lookup_dict = {}
    for obs_col, df in cluster_reports.items():
        # Convert the "Label" -> "F1-Score" DataFrame to a dictionary for quick lookups
        f1_lookup_dict[obs_col] = dict(zip(df["Label"], df["F1-Score"]))
    
    # Iterate over each cell in adata
    for i in range(adata.n_obs):
        best_f1 = -1
        best_label_full = None  # Will store "<obs_col>_<cluster_label>"
        
        # Check each cluster assignment
        for obs_col, label_to_f1 in f1_lookup_dict.items():
            # Current cell's cluster label in this assignment
            cell_label = adata.obs[obs_col].iloc[i]
            # Lookup F1 score (if label doesn't exist in the classification report, default to -1)
            f1 = label_to_f1.get(cell_label, -1)
            # Update if this is a higher F1
            if f1 > best_f1:
                best_f1 = f1
                # Prepend the obs_col to ensure uniqueness across different clustering schemes
                best_label_full = f"{obs_col}_{cell_label}"
            
        optimal_labels[i] = best_label_full
    
    # Store the new labels in an adata.obs column
    adata.obs[new_col] = optimal_labels
    # convert to categorical
    adata.obs[new_col] = adata.obs[new_col].astype('category')








def reassign_selected_clusters(
    adata,
    dendro_key,
    current_label,
    new_label,
    clusters_to_clean=None
):
    """Reassign selected clusters to their highest-correlation neighbor.

    For cells whose cluster is in `clusters_to_clean`, picks the
    highest-correlation cluster from the dendrogram correlation matrix.
    Accounts for Scanpy's behavior where 'categories_ordered' (leaf order)
    differs from the row order in 'correlation_matrix'.

    Args:
        adata: Must contain `adata.obs[current_label]` with current cluster
            assignments and `adata.uns[dendro_key]` with keys
            'categories_ordered', 'categories_idx_ordered', and
            'correlation_matrix'.
        dendro_key: Key in `adata.uns` that has the dendrogram data.
        current_label: Column in `adata.obs` containing the current cluster
            assignments.
        new_label: Column name in `adata.obs` where reassigned clusters
            will be stored.
        clusters_to_clean: Labels that should be reassigned. Defaults to None.

    Returns:
        None. Adds a new column `adata.obs[new_label]` with updated assignments.
    """
    
    if clusters_to_clean is None:
        clusters_to_clean = []
    clusters_to_clean_set = set(clusters_to_clean)
    
    # Ensure the column is string (not categorical) to avoid assignment issues
    if pd.api.types.is_categorical_dtype(adata.obs[current_label]):
        adata.obs[current_label] = adata.obs[current_label].astype(str)
    
    # Pull out original assignments
    original_assignments = adata.obs[current_label].values
    new_assignments = original_assignments.copy()
    
    # Retrieve dendrogram data
    if dendro_key not in adata.uns:
        raise KeyError(f"{dendro_key} not found in adata.uns.")
    dendro_data = adata.uns[dendro_key]
    
    categories_ordered = dendro_data.get("categories_ordered", None)      # Leaf labels
    leaves = dendro_data.get("categories_idx_ordered", None)             # Leaf indices
    corr_matrix = dendro_data.get("correlation_matrix", None)
    
    if (categories_ordered is None or leaves is None or corr_matrix is None):
        raise ValueError(
            f"adata.uns['{dendro_key}'] must contain "
            "'categories_ordered', 'categories_idx_ordered', and 'correlation_matrix'."
        )
    
    n_cats = len(categories_ordered)
    if n_cats != len(leaves):
        raise ValueError("Mismatch: categories_ordered and categories_idx_ordered differ in length.")
    if corr_matrix.shape != (n_cats, n_cats):
        raise ValueError("Mismatch: correlation_matrix shape does not match number of categories.")
    
    # --------------------------------------------------------
    # 1) Reconstruct the "original" category order used in corr_matrix
    #    Because Scanpy does not reorder corr_matrix to the dendrogram's leaf order;
    #    instead it stores the "dendrogram order" in leaves + categories_ordered.
    #
    #    categories_ordered[i] = label at leaf i
    #    leaves[i] = index in the original order for that leaf
    #
    #    So if leaves = [2, 0, 1], it means:
    #      - leaf 0 is originally row 2,
    #      - leaf 1 is originally row 0,
    #      - leaf 2 is originally row 1.
    #
    #    We'll invert that so original_categories[row_idx] = label for that row.
    # --------------------------------------------------------
    original_categories = [None]*n_cats
    for leaf_pos, row_idx in enumerate(leaves):
        # categories_ordered[leaf_pos] is the label at leaf_pos
        label = categories_ordered[leaf_pos]
        original_categories[row_idx] = label
    
    # Build a lookup from label -> row index in corr_matrix
    label_to_idx = {lbl: i for i, lbl in enumerate(original_categories)}
    
    def find_closest_cluster(label):
        """
        Return the label whose correlation is highest with `label`, 
        skipping the label itself and any in clusters_to_clean_set.
                
        'corr_matrix' is in the "original" order, 
        so we find its row via 'label_to_idx[label]'.
        """
        if label not in label_to_idx:
            return None  # no data for this label
        row_idx = label_to_idx[label]
        row = corr_matrix[row_idx]
        
        # Sort indices by descending correlation
        sorted_idx = np.argsort(row)[::-1]  # highest corr first
        
        for idx_ in sorted_idx:
            # skip itself
            if idx_ == row_idx:
                continue
            candidate_label = original_categories[idx_]
            if candidate_label in clusters_to_clean_set:
                continue  # skip "clean" labels
            return candidate_label
        return None
    
    # Reassign if needed
    for i in range(len(new_assignments)):
        c_label = new_assignments[i]
        if c_label in clusters_to_clean_set:
            fallback = find_closest_cluster(c_label)
            if fallback is not None:
                new_assignments[i] = fallback
            # else remain in the same cluster
    
    adata.obs[new_label] = new_assignments
    # make sure type is category, seems to be needed for sc.tl.dendrogram
    adata.obs[new_label] = adata.obs[new_label].astype('category')




# To do: parameterize .obs column names
def sort_obs_table(adata):
    """
    Sorts the observation table of an AnnData object by 'celltype' and the numeric part of 'stage'.

    This function takes an AnnData object as input, extracts the 'celltype' and 'stage' columns 
    from its observation (obs) DataFrame, counts the occurrences of each unique pair, and sorts 
    these counts first by 'celltype' and then by the numeric value extracted from 'stage'.

    Args:
        adata (AnnData): An AnnData object containing the single-cell dataset.

    Returns:
        pandas.DataFrame: A DataFrame with sorted counts of cell types and stages.

    Notes:
        The 'stage' column is expected to contain string values with a numeric part that can be 
        extracted and sorted numerically. The function does not modify the original AnnData object.
    """
    # Count occurrences of each unique 'celltype' and 'stage' pair
    counts = adata.obs[['celltype', 'stage']].value_counts()
    counts_df = counts.reset_index()
    counts_df.columns = ['celltype', 'stage', 'count']

    # Add a temporary column 'stage_num' for numeric sorting of 'stage'
    # Then sort by 'celltype' and the numeric part of 'stage'
    # Finally, drop the temporary 'stage_num' column
    counts_df = (
        counts_df
        .assign(stage_num=lambda df: df['stage'].str.extract(r'(\d+\.\d+|\d+)')[0].astype(float))
        .sort_values(by=['celltype', 'stage_num'])
        .drop(columns='stage_num')
    )

    return counts_df




def reduce_cells(
    adata: ad.AnnData,
    n_cells: int = 5,
    cluster_key: str = "cluster",
    use_raw: bool = True
) -> ad.AnnData:
    """Reduce cell count by combining transcript counts into meta-cells.

    Groups cells by cluster and sums transcript counts across randomly
    selected sets of `n_cells` cells to create meta-cells.

    Args:
        adata: Annotated data matrix with observations (cells) and
            variables (features).
        n_cells: The number of cells to combine into a meta-cell.
            Defaults to 5.
        cluster_key: The key in `adata.obs` that specifies the cluster
            identity of each cell. Defaults to "cluster".
        use_raw: Whether to use the raw count matrix in `adata.raw`
            instead of `adata.X`. Defaults to True.

    Returns:
        Annotated data matrix with reduced number of cells.
    """
    # Create a copy of the input data
    adata = adata.copy()

    # Use the raw count matrix if specified
    if use_raw:
        adata.X = adata.raw.X

    # Get the k-nearest neighbor graph
    knn_graph = adata.uns['neighbors']['connectivities']

    # Get the indices of the cells in each cluster
    clusters = np.unique(adata.obs[cluster_key])
    cluster_indices = {c: np.where(adata.obs[cluster_key] == c)[0] for c in clusters}

    # Calculate the number of meta-cells to make
    n_metacells = int(sum(np.ceil(adata.obs[cluster_key].value_counts() / n_cells)))

    # Create a list of new AnnData objects to store the combined transcript counts
    ad_list = []

    # Loop over each cluster
    for cluster in clusters:
        # Get the indices of the cells in this cluster
        indices = cluster_indices[cluster]

        # If there are fewer than n_cells cells in the cluster, skip it
        if len(indices) < n_cells:
            continue

        # Compute the total transcript count across n_cells cells in the cluster
        num_summaries = int(np.ceil(len(indices) / n_cells))
        combined = []
        used_indices = set()
        for i in range(num_summaries):
            # Select n_cells cells at random from the remaining unused cells
            unused_indices = list(set(indices) - used_indices)
            np.random.shuffle(unused_indices)
            selected_indices = unused_indices[:n_cells]

            # Add the transcript counts for the selected cells to the running total
            combined.append(np.sum(adata.X[selected_indices,:], axis=0))

            # Add the selected indices to the set of used indices
            used_indices.update(selected_indices)

        # Create a new AnnData object to store the combined transcript counts for this cluster
        tmp_adata = AnnData(X=np.array(combined), var=adata.var)

        # Add the cluster identity to the `.obs` attribute of the new AnnData object
        tmp_adata.obs[cluster_key] = cluster

        # Append the new AnnData object to the list
        ad_list.append(tmp_adata)

    # Concatenate the new AnnData objects into a single AnnData object
    adata2 = anndata.concat(ad_list, join='inner')

    return adata2

def create_mean_cells(adata, obs_key, obs_value, n_mean_cells, n_cells):
    """Create meta-cells by averaging randomly sampled cells.

    Repeatedly samples `n_cells` cells where `obs_key == obs_value` and
    averages their expression, producing `n_mean_cells` meta-cells.

    Args:
        adata: The annotated data matrix of shape n_obs x n_vars.
        obs_key: The key in `adata.obs` used to select cells.
        obs_value: The value in `adata.obs[obs_key]` to filter on.
        n_mean_cells: The number of meta-cells to create.
        n_cells: The number of cells to randomly select for each
            meta-cell.

    Returns:
        Annotated data matrix containing the meta-cells.
    """
    if obs_key not in adata.obs:
        raise ValueError(f"Key '{obs_key}' not found in adata.obs")

    # Create a new anndata object to store the mean-cells
    mean_cells = None
    
    # Get cells in this cluster
    cluster_cells = adata[adata.obs[obs_key] == obs_value].copy()
    seeds = np.random.choice(n_mean_cells, n_mean_cells, replace=False)
    for i in range(n_mean_cells):
        # Randomly select cells
        selected_cells = sc.pp.subsample(cluster_cells, n_obs = n_cells, random_state = seeds[i], copy=True)

        # Average their gene expression
        avg_expression = selected_cells.X.mean(axis=0)

        # Create a new cell in the mean_cells anndata object
        new_cell = sc.AnnData(avg_expression.reshape(1, -1),
            var=adata.var, 
            obs=pd.DataFrame(index=[f'{obs_value}_mean_{i}']))

        # Append new cell to the meta_cells anndata object
        if mean_cells is None:
            mean_cells = new_cell
        else:
            mean_cells = mean_cells.concatenate(new_cell)

    return mean_cells



def create_hybrid_cells(
    adata: ad.AnnData,
    celltype_counts: dict,
    groupby: str,
    n_hybrid_cells: int
) -> ad.AnnData:
    """Generate hybrid cells by averaging randomly selected cells from specified groups.

    Takes the mean of transcript counts for cells sampled from groups
    in proportions indicated by `celltype_counts`.

    Args:
        adata: Annotated data matrix with observations (cells) and
            variables (genes).
        celltype_counts: Keys indicate the subset of cells and values
            are the number of cells to sample from each cell type.
        groupby: The name of the column in `adata.obs` to group cells
            by before selecting cells at random.
        n_hybrid_cells: The number of hybrid cells to generate.

    Returns:
        Annotated data matrix containing only the hybrid cells.
    """
    hybrid_list = []
    for i in range(n_hybrid_cells):
        # Randomly select cells from the specified clusters
        r_cells = sample_cells(adata, celltype_counts, groupby)

        # Calculate the average transcript counts for the selected cells
        ### x_counts = np.average(r_cells.X, axis=0)
        x_counts = np.mean(r_cells.X, axis=0)

        # Create an AnnData object for the hybrid cell
        hybrid = AnnData(
            x_counts.reshape(1, -1),
            obs={'hybrid': [f'Hybrid Cell {i+1}']},
            var=adata.var
        )

        # Append the hybrid cell to the hybrid list
        hybrid_list.append(hybrid)

    # Concatenate the hybrid cells into a single AnnData object
    hybrid = ad.concat(hybrid_list, join='outer')

    return hybrid



def sample_cells(
    adata: ad.AnnData,
    celltype_counts: dict,
    groupby: str
)-> ad.AnnData:
    """Sample cells as specified by celltype_counts and groupby.

    Args:
        adata: Annotated data matrix with observations (cells) and
            variables (genes).
        celltype_counts: Keys indicate the subset of cells and values
            are the number of cells to sample from each cell type.
        groupby: The name of the column in `adata.obs` to group cells
            by before selecting cells at random.

    Returns:
        Annotated data matrix containing only the selected cells.
    """

    sampled_cells = []

    for celltype, count in celltype_counts.items():
        # subset the AnnData object by .obs[groupby], most often will be something like cluster, leiden, SCN_class, celltype
        subset = adata[adata.obs[groupby] == celltype]
        
        # sample cells from the subset
        # sampled_subset = subset.sample(n=count, random_state=1)
        cell_ids = np.random.choice(subset.obs.index, count, replace = False)
        adTmp = subset[np.isin(subset.obs.index, cell_ids, assume_unique = True),:].copy()
        
        # append the sampled cells to the list
        sampled_cells.append(adTmp)

    # concatenate the sampled cells into a single AnnData object
    sampled_adata = sampled_cells[0].concatenate(sampled_cells[1:])

    return sampled_adata




def find_cells_that(adata, genes_thr_pres, all_conditions=True, layer=None):
    """
    Return cell IDs for cells that meet specified gene expression conditions.
    
    This function evaluates an AnnData object for a set of genes and checks, for each gene,
    whether its expression satisfies a user-specified condition (either above a threshold for
    presence or below a threshold for absence). The gene conditions are provided via the
    `genes_thr_pres` parameter. The parameter `all_conditions` specifies whether a cell must
    meet all of the gene conditions (if True) or at least a certain number of them (if an int
    between 1 and the number of conditions). The function supports selecting data from a
    specified layer (defaulting to adata.X if not provided).
    
    Args:
        adata: AnnData object containing expression data. It is expected that adata.var_names 
            contains the gene names and adata.obs_names contains the cell IDs.
        genes_thr_pres (dict): A dictionary mapping gene symbols (str) to a tuple (threshold, present)
            where threshold is a numeric value and present is a boolean. If present is True, the cell must have
            expression > threshold for that gene; if False, the cell must have expression < threshold.
        all_conditions (bool or int, optional): If True, all gene conditions must be met.
            Alternatively, an integer between 1 and the number of genes in genes_thr_pres may be provided
            to require that at least that many conditions are met. Defaults to True.
        layer (str, optional): The name of the layer in adata to use for the expression data.
            If None, adata.X is used. Defaults to None.
    
    Returns:
        array-like: An array of cell IDs for cells that satisfy the specified gene conditions.
    
    Raises:
        ValueError: If any gene in genes_thr_pres is not found in adata.var_names.
        ValueError: If all_conditions is an integer not between 1 and the number of gene conditions.
        ValueError: If all_conditions is neither a boolean nor an integer.
        KeyError: If the specified layer does not exist in adata.layers.
    """
    import numpy as np
    from scipy.sparse import issparse
    
    # Ensure every gene in genes_thr_pres exists in adata.var_names.
    missing_genes = [gene for gene in genes_thr_pres if gene not in adata.var_names]
    if missing_genes:
        raise ValueError("The following genes are not found in adata.var_names: " + ", ".join(missing_genes))
    
    # Determine the number of required conditions.
    n_conditions = len(genes_thr_pres)
    if isinstance(all_conditions, bool):
        n_required = n_conditions if all_conditions else 1
    elif isinstance(all_conditions, int):
        if 1 <= all_conditions <= n_conditions:
            n_required = all_conditions
        else:
            raise ValueError("When provided as an int, all_conditions must be between 1 and the number of gene conditions")
    else:
        raise ValueError("all_conditions must be a boolean or an integer")
    
    # Extract the data for the genes.
    gene_names = list(genes_thr_pres.keys())
    if layer is None:
        data_subset = adata[:, gene_names].X
    else:
        if layer not in adata.layers:
            raise KeyError(f"Layer '{layer}' not found in adata.layers")
        data_subset = adata[:, gene_names].layers[layer]
    
    # Initialize list to store boolean condition for each gene.
    # Handle both sparse and dense matrices.
    bool_conditions = []
    if issparse(data_subset):
        # Iterate over columns in the sparse matrix.
        for i, gene in enumerate(gene_names):
            threshold, present = genes_thr_pres[gene]
            # Extract column as a sparse matrix and convert to dense array.
            col_data = np.array(data_subset[:, i].todense()).flatten()
            if present:
                condition = col_data >= threshold
            else:
                condition = col_data < threshold
            bool_conditions.append(condition)
    else:
        # If the data is dense.
        # Ensure data_subset is 2D (this is relevant if only one gene is provided).
        if data_subset.ndim == 1:
            data_subset = data_subset.reshape(-1, 1)
        for i, gene in enumerate(gene_names):
            threshold, present = genes_thr_pres[gene]
            col_data = data_subset[:, i]
            if present:
                condition = col_data >= threshold
            else:
                condition = col_data < threshold
            bool_conditions.append(condition)
    
    # Combine the individual gene conditions into a boolean matrix of shape (n_cells, n_conditions).
    bool_matrix = np.column_stack(bool_conditions)
    # For each cell, count the number of conditions met.
    conditions_met = np.sum(bool_matrix, axis=1)
    cell_condition = conditions_met >= n_required
    
    # Return the cell IDs for cells that meet the overall condition.
    selected_cell_ids = adata.obs_names[cell_condition]
    return selected_cell_ids





def group_cells_by_lists(adata, groups_dict, handle_overlap="new_group"):
    """
    Create a new AnnData object that includes only cells from the provided groups,
    and add a new column in .obs indicating the group assignment for each cell.
    
    Each key in groups_dict corresponds to a group name and the associated value
    is an array or list of cell IDs. A cell that appears in exactly one group is assigned
    that group name. For cells that appear in more than one group, if handle_overlap is set to
    "new_group" (the default), these cells are labeled as "overlap"; if set to "exclude",
    these cells are not included in the output.
    
    Args:
        adata: AnnData object with cell identifiers in adata.obs_names.
        groups_dict (dict): Dictionary where keys are group names (str) and values are lists
            or arrays of cell IDs.
        handle_overlap (str, optional): How to handle cells that appear in more than one group.
            Options are "new_group" (default) to assign them the label "overlap", or "exclude" 
            to remove them. Defaults to "new_group".
    
    Returns:
        AnnData: A new AnnData object containing only the cells that are present in one or more groups,
            with an added column in .obs called "group" that indicates the group assignment.
    
    Raises:
        ValueError: If any cell ID in groups_dict is not found in adata.obs_names.
        ValueError: If handle_overlap is not "new_group" or "exclude".
    """
    # Validate handle_overlap parameter.
    if handle_overlap not in ["new_group", "exclude"]:
        raise ValueError("handle_overlap must be either 'new_group' or 'exclude'.")
    
    # Ensure that all cell IDs in groups_dict are found in adata.obs_names.
    adata_cells = set(adata.obs_names)
    for group_name, cell_list in groups_dict.items():
        missing_cells = set(cell_list) - adata_cells
        if missing_cells:
            raise ValueError(
                f"The following cells in group '{group_name}' are not found in adata.obs_names: {missing_cells}"
            )
    
    # Build a mapping from cell ID to list of groups it belongs to.
    cell_to_groups = {}
    for group_name, cell_list in groups_dict.items():
        for cell in cell_list:
            cell_to_groups.setdefault(cell, []).append(group_name)
    
    # Determine group assignment for each cell.
    # - If the cell appears in only one group, assign that group.
    # - If it appears in multiple groups:
    #      - if handle_overlap=="new_group", assign the label "overlap"
    #      - if handle_overlap=="exclude", do not include that cell.
    group_assignment = {}
    for cell, group_list in cell_to_groups.items():
        if len(group_list) == 1:
            group_assignment[cell] = group_list[0]
        else:
            if handle_overlap == "new_group":
                group_assignment[cell] = "overlap"
            elif handle_overlap == "exclude":
                # Exclude the cell by not assigning any group.
                continue
    
    # Subset adata to only include cells that were assigned a group.
    included_cells = list(group_assignment.keys())
    new_adata = adata[included_cells].copy()
    
    # Add the group assignment as a new column in .obs.
    new_adata.obs = new_adata.obs.copy()  # ensure a writable copy
    new_adata.obs["group"] = [group_assignment[cell] for cell in new_adata.obs_names]
    
    return new_adata











def add_ambient_rna(
    adata,
    obs_key: str,
    obs_val: str,
    n_cells_to_sample: int = 10,
    weight_of_ambient: float = 0.05
) -> ad.AnnData:
    """Add simulated ambient RNA to non-target cells.

    For cells not matching `obs_key == obs_val`, blends their expression
    with the mean expression of randomly sampled target cells, simulating
    ambient RNA contamination.

    Args:
        adata: AnnData object containing expression data.
        obs_key: Column name in `adata.obs` used to identify target cells.
        obs_val: Value in `adata.obs[obs_key]` that defines the target
            (ambient source) cells.
        n_cells_to_sample: Number of target cells to sample when computing
            each ambient mean. Defaults to 10.
        weight_of_ambient: Fraction of the final expression contributed by
            the ambient signal. Defaults to 0.05.

    Returns:
        The modified AnnData object with updated expression for non-target
        cells.
    """
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
    """Get Y chromosome gene names present in the AnnData object.

    Queries BioMart for mouse gene annotations and returns the subset of
    Y chromosome genes that are found in `adata.var.index`.

    Args:
        adata: AnnData object whose `var.index` contains gene symbols.

    Returns:
        A list of Y chromosome gene names present in the dataset.
    """
    gene_chrs = sc.queries.biomart_annotations("mmusculus",["mgi_symbol", "ensembl_gene_id", "chromosome_name"],).set_index("mgi_symbol")
    ygenes = gene_chrs[gene_chrs["chromosome_name"]=='Y']
    ygenes = ygenes[ygenes.index.isin(adata.var.index)]
    return ygenes.index.tolist()



def extract_top_bottom_genes(
    deg_res: dict,
    ngenes: int,
    sort_by: str = 'scores',
    extraction_map: Dict[str, str] = None
) -> Dict[str, List[str]]:
    """Extract top and bottom genes from differential expression results.

    Extracts the top and bottom `ngenes` from each gene table in `deg_res`
    and organizes them into a dictionary with combined keys of group and
    sample names.

    Args:
        deg_res: A dictionary containing differential expression results
            with keys 'sample_names' (list of sample names) and
            'geneTab_dict' (dict mapping group names to DataFrames with
            gene information).
        ngenes: The number of top or bottom genes to extract from each
            gene table.
        sort_by: The column name in the gene tables to sort by.
            Defaults to 'scores'.
        extraction_map: A dictionary mapping sample names to extraction
            behavior ('top' or 'bottom'). If not provided, defaults to
            'top' for the first sample, 'bottom' for the second, and
            'top' for any additional samples. Defaults to None.

    Returns:
        A dictionary where each key is a combination of group and sample
        name (e.g., 'Meso.Nascent_Singular') and each value is a list of
        gene names.

    Raises:
        KeyError: If 'sample_names' or 'geneTab_dict' keys are missing in
            deg_res, or if 'sort_by' is not a column in the gene tables.
        ValueError: If ngenes is not a positive integer or if
            'sample_names' does not contain at least one entry.
        TypeError: If the input types are incorrect.
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



def pull_out_genes(
    diff_genes_dict: dict,
    cell_type: str,
    category: str,
    num_genes: int = 0,
    order_by = "logfoldchanges",
    threshold = 2) -> list:
    """Extract significant genes for a cell type from differential expression results.

    Filters genes by adjusted p-value threshold, then sorts by the specified
    column. Sort direction depends on the position of `category` in
    `diff_genes_dict['category_names']`: descending for the first category,
    ascending otherwise.

    Args:
        diff_genes_dict: Dictionary containing differential expression results
            with keys 'geneTab_dict' (dict of DataFrames) and
            'category_names' (list of category names).
        cell_type: Key into 'geneTab_dict' identifying the cell type.
        category: Category name used to determine sort direction.
        num_genes: Number of top genes to return. If 0, returns all
            significant genes. Defaults to 0.
        order_by: Column name to sort genes by. Defaults to
            "logfoldchanges".
        threshold: Adjusted p-value threshold for filtering genes.
            Defaults to 2.

    Returns:
        A list of gene names passing the significance filter.
    """
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
    """Extract significant genes for a cell type from differential expression results.

    Similar to `pull_out_genes`, but infers category names from the first
    key in `diff_genes_dict` rather than using a fixed 'category_names' key.

    Args:
        diff_genes_dict: Dictionary containing differential expression results
            with 'geneTab_dict' (dict of DataFrames) and a first key whose
            value is a list of category names.
        cell_type: Key into 'geneTab_dict' identifying the cell type.
        category: Category name used to determine sort direction.
        num_genes: Number of top genes to return. If 0, returns all
            significant genes. Defaults to 0.
        order_by: Column name to sort genes by. Defaults to
            "logfoldchanges".
        threshold: Adjusted p-value threshold for filtering genes.
            Defaults to 2.

    Returns:
        A list of gene names passing the significance filter.
    """
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











