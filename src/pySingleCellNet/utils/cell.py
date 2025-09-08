from __future__ import annotations

import numpy as np
import pandas as pd
# from anndata import AnnData
# import matplotlib.pyplot as plt
import scanpy as sc
import anndata as ad
# from scipy.sparse import issparse
# from alive_progress import alive_bar
from scipy.stats import median_abs_deviation, ttest_ind
# import string
import igraph as ig
from scipy import sparse
from sklearn.decomposition import PCA

def clustering_quality_vs_nn(
    adata,
    label_col: str,
    n_genes: int = 5,
    naive: dict = {"p_val": 1e-2, "fold_change": 0.25},
    strict: dict = {"minpercentin": 0.20, "maxpercentout": 0.10},
    n_pcs_for_nn: int = 40,
):
    """
    For each cluster, find its nearest neighbor cluster and count genes that meet
    (1) naive DE criteria: p-value <= naive['p_val'] AND log2FC >= naive['fold_change']
    (2) strict DE criteria: pct_in >= strict['minpercentin'] AND pct_out <= strict['maxpercentout'].
    
    Parameters
    ----------
    adata : AnnData
        Single-cell object. Uses adata.X for expression (dense or sparse).
    label_col : str
        .obs column with cluster labels.
    n_genes : int, default 5
        For reporting convenience: include top n gene names (by effect size) for each rule.
        Does not affect the counts.
    naive : dict
        {'p_val': float, 'fold_change': float} ; fold_change is on log2 scale.
    strict : dict
        {'minpercentin': float, 'maxpercentout': float}
    n_pcs_for_nn : int, default 30
        Number of PCs (if available or computed) for nearest-neighbor cluster search.

    Returns
    -------
    pandas.DataFrame
        Columns: ['cluster', 'nn_cluster', 'n_genes_naive', 'n_genes_strict',
                  'top_naive_genes', 'top_strict_genes']
    """
    # ----- helpers -----
    def _to_dense(X):
        return X.A if sparse.issparse(X) else np.asarray(X)

    def _get_representation(adata, n_pcs_for_nn):
        # Prefer existing PCA; otherwise compute a compact representation
        if "X_pca" in adata.obsm and adata.obsm["X_pca"].shape[1] >= min(2, n_pcs_for_nn):
            rep = adata.obsm["X_pca"][:, :n_pcs_for_nn]
            return np.asarray(rep)
        # else: compute PCA on log1p(counts), on a subset of genes to keep things light
        X = _to_dense(adata.X)
        # choose up to 2000 HVGs if available; else top-1000 variable genes
        if "highly_variable" in adata.var.columns and adata.var["highly_variable"].any():
            genes_mask = adata.var["highly_variable"].values
        else:
            # compute variance per gene quickly
            var = X.var(axis=0)
            topk = min(1000, X.shape[1])
            genes_mask = np.zeros(X.shape[1], dtype=bool)
            genes_mask[np.argsort(var)[-topk:]] = True
        Xg = np.log1p(X[:, genes_mask])
        # center per gene
        Xg = Xg - Xg.mean(axis=0, keepdims=True)
        pca = PCA(n_components=min(n_pcs_for_nn, Xg.shape[1], Xg.shape[0]-1))
        return pca.fit_transform(Xg)

    def _cluster_centroids(rep, labels):
        centroids = {}
        for c in labels.unique():
            idx = (labels == c).values
            if idx.sum() == 0:
                continue
            centroids[c] = rep[idx].mean(axis=0)
        return centroids

    def _nearest_neighbors(centroids):
        # For each cluster, find the nearest other cluster (Euclidean)
        keys = list(centroids.keys())
        arr = np.stack([centroids[k] for k in keys], axis=0)
        # pairwise distances
        d2 = np.sum((arr[:, None, :] - arr[None, :, :])**2, axis=2)
        np.fill_diagonal(d2, np.inf)
        nn_idx = np.argmin(d2, axis=1)
        return {keys[i]: keys[j] for i, j in enumerate(nn_idx)}

    # ----- main -----
    if label_col not in adata.obs.columns:
        raise ValueError(f"'{label_col}' not found in adata.obs")

    labels = adata.obs[label_col].astype("category")
    clusters = pd.Index(labels.cat.categories)

    rep = _get_representation(adata, n_pcs_for_nn)
    centroids = _cluster_centroids(rep, labels)
    if len(centroids) < 2:
        raise ValueError("Need at least two clusters to compute nearest neighbors.")
    nn_map = _nearest_neighbors(centroids)

    # Expression matrix (cells x genes), dense for vectorized ops
    X = _to_dense(adata.X)
    genes = adata.var_names.to_numpy()

    rows = []
    for c in clusters:
        if c not in nn_map:
            continue
        nnc = nn_map[c]
        in_mask = (labels == c).to_numpy()
        out_mask = (labels == nnc).to_numpy()

        Xin = X[in_mask, :]
        Xout = X[out_mask, :]

        # log1p for t-test stability; raw means for FC with small epsilon
        logXin = np.log1p(Xin)
        logXout = np.log1p(Xout)

        # Welch t-test per gene
        # (scipy vectorizes if axis=0; handle potential NaNs for constant columns)
        t_stat, p_vals = ttest_ind(logXin, logXout, equal_var=False, axis=0, nan_policy="omit")
        p_vals = np.nan_to_num(p_vals, nan=1.0)

        # log2 fold-change on raw means with small epsilon
        eps = 1e-9
        mu_in = Xin.mean(axis=0) + eps
        mu_out = Xout.mean(axis=0) + eps
        log2fc = np.log2(mu_in / mu_out)

        # pct expressed
        pct_in = (Xin > 0).mean(axis=0)
        pct_out = (Xout > 0).mean(axis=0)

        # criteria
        naive_mask = (p_vals <= float(naive["p_val"])) & (log2fc >= float(naive["fold_change"]))
        strict_mask = (pct_in >= float(strict["minpercentin"])) & (pct_out <= float(strict["maxpercentout"]))

        n_naive = int(naive_mask.sum())
        n_strict = int(strict_mask.sum())

        # top-gene reporting (up to n_genes), sorted by effect size
        top_naive = genes[naive_mask]
        if top_naive.size:
            ord_ix = np.argsort(-log2fc[naive_mask])
            top_naive = top_naive[ord_ix][:n_genes]
        top_strict = genes[strict_mask]
        if top_strict.size:
            ord_ix = np.argsort(-log2fc[strict_mask])
            top_strict = top_strict[ord_ix][:n_genes]

        rows.append({
            "cluster": c,
            "nn_cluster": nnc,
            "n_genes_naive": n_naive,
            "n_genes_strict": n_strict,
            "top_naive_genes": ";".join(map(str, top_naive)) if top_naive.size else "",
            "top_strict_genes": ";".join(map(str, top_strict)) if top_strict.size else "",
        })

    out = pd.DataFrame(rows).sort_values(["cluster"]).reset_index(drop=True)
    return out





def cluster_subclusters(
    adata: ad.AnnData,
    cluster_column: str = 'leiden',
    to_subcluster: list[str] = None,
    layer: str = 'counts',
    n_hvg: int = 2000,
    n_pcs: int = 40,
    n_neighbors: int = 10,
    leiden_resolution: float = 0.25,
    subcluster_col_name: str = 'subcluster'
) -> None:
    """
    Subcluster selected clusters (or all clusters) within an AnnData object by recomputing HVGs, PCA,
    kNN graph, and Leiden clustering. Updates the AnnData object in-place, adding or updating
    the `subcluster_col_name` column in `.obs` with new labels prefixed by the original cluster.

    Cells in clusters not listed in `to_subcluster` retain their original cluster label as their "subcluster".

    Args:
        adata: AnnData
            The AnnData object containing precomputed clusters in `.obs[cluster_column]`.
        cluster_column: str, optional
            Name of the `.obs` column holding the original cluster assignments. Default is 'leiden'.
        to_subcluster: list of str, optional
            List of cluster labels (as strings) to subcluster. If `None`, subclusters *all* clusters.
        layer: str, optional
            Layer name in `adata.layers` to use for HVG detection. Default is 'counts'.
        n_hvg: int, optional
            Number of highly variable genes to select per cluster. Default is 2000.
        n_pcs: int, optional
            Number of principal components to compute. Default is 40.
        n_neighbors: int, optional
            Number of neighbors for the kNN graph. Default is 10.
        leiden_resolution: float, optional
            Resolution parameter for Leiden clustering. Default is 0.25.
        subcluster_col_name: str, optional
            Name of the `.obs` column to store subcluster labels. Default is 'subcluster'.

    Raises:
        ValueError: If `cluster_column` not in `adata.obs`.
        ValueError: If `layer` not in `adata.layers`.
        ValueError: If any entry in `to_subcluster` is not found in `adata.obs[cluster_column]`.
    """
    # Error checking
    if cluster_column not in adata.obs:
        raise ValueError(f"Cluster column '{cluster_column}' not found in adata.obs")
    if layer not in adata.layers:
        raise ValueError(f"Layer '{layer}' not found in adata.layers")

    # Cast original clusters to string
    adata.obs['original_cluster'] = adata.obs[cluster_column].astype(str)
    adata.obs[subcluster_col_name] = adata.obs['original_cluster']

    # Determine clusters to process
    unique_clusters = set(adata.obs['original_cluster'])
    if to_subcluster is None:
        clusters_to_process = sorted(unique_clusters)
    else:
        # ensure strings
        requested = {str(c) for c in to_subcluster}
        missing = requested - unique_clusters
        if missing:
            raise ValueError(f"Clusters not found: {missing}")
        clusters_to_process = sorted(requested)

    # Iterate and subcluster each requested cluster
    for orig in clusters_to_process:
        mask = adata.obs['original_cluster'] == orig
        sub = adata[mask].copy()

        # 1) Compute HVGs
        sc.pp.highly_variable_genes(
            sub,
            flavor='seurat_v3',
            n_top_genes=n_hvg,
            layer=layer
        )

        # 2) PCA
        sc.pp.pca(sub, n_comps=n_pcs, use_highly_variable=True)

        # 3) kNN
        sc.pp.neighbors(sub, n_neighbors=n_neighbors, use_rep='X_pca')

        # 4) Leiden
        sc.tl.leiden(
            sub,
            resolution=leiden_resolution,
            flavor='igraph',
            n_iterations=2,
            key_added='leiden_sub'
        )

        # Prefix subcluster labels and write back
        new_labels = orig + "_" + sub.obs['leiden_sub'].astype(str)
        adata.obs.loc[mask, subcluster_col_name] = new_labels.values



def detect_outliers(adata, metric = ["total_counts"], nmads = 5):
    """
    determines whether obs[metric] exceeds nmads 
     
    Parameters:
    -----------
    adata : AnnData
        The input AnnData object containing single-cell data.
    metric : str
        The column name in `adata.obs` holding cell metric 
    nmads : int, optional (default=5)
        The number of median abs deviations to define a cell as an outlier

    Returns
    -------
    None
        The function adds a new column to `adata.obs` named "outlier_" + metric, but does not return anything.
    """
    M = adata.obs[metric]
    outlier = (M < np.median(M) - nmads * median_abs_deviation(M)) | (
        np.median(M) + nmads * median_abs_deviation(M) < M
    )

    new_col = "outlier_" + nmads + "_" + metric
    adata.obs[new_col] = outlier
    


def filter_adata_by_group_size(adata: ad.AnnData, groupby: str, ncells: int = 20) -> ad.AnnData:
    """
    Filters an AnnData object to retain only cells from groups with at least 'ncells' cells.
    
    Parameters:
    -----------
    adata : AnnData
        The input AnnData object containing single-cell data.
    groupby : str
        The column name in `adata.obs` used to define groups (e.g., cluster labels).
    ncells : int, optional (default=20)
        The minimum number of cells a group must have to be retained.
    
    Returns:
    --------
    filtered_adata : AnnData
        A new AnnData object containing only cells from groups with at least 'ncells' cells.
    
    Raises:
    -------
    ValueError:
        - If `groupby` is not a column in `adata.obs`.
        - If `ncells` is not a positive integer.
    """
    # Input Validation
    if not isinstance(adata, ad.AnnData):
        raise TypeError(f"'adata' must be an AnnData object, but got {type(adata)}.")
    
    if not isinstance(groupby, str):
        raise TypeError(f"'groupby' must be a string, but got {type(groupby)}.")
    
    if groupby not in adata.obs.columns:
        raise ValueError(f"'{groupby}' is not a column in adata.obs. Available columns are: {adata.obs.columns.tolist()}")
    
    if not isinstance(ncells, int) or ncells <= 0:
        raise ValueError(f"'ncells' must be a positive integer, but got {ncells}.")
    
    # Compute the size of each group
    group_sizes = adata.obs[groupby].value_counts()
    
    # Identify groups that meet or exceed the minimum cell threshold
    valid_groups = group_sizes[group_sizes >= ncells].index.tolist()
    
    if not valid_groups:
        raise ValueError(f"No groups found in '{groupby}' with at least {ncells} cells.")
    
    # Optionally, inform the user about the filtering
    total_groups = adata.obs[groupby].nunique()
    retained_groups = len(valid_groups)
    excluded_groups = total_groups - retained_groups
    print(f"Filtering AnnData object based on group sizes in '{groupby}':")
    print(f" - Total groups: {total_groups}")
    print(f" - Groups retained (≥ {ncells} cells): {retained_groups}")
    print(f" - Groups excluded (< {ncells} cells): {excluded_groups}")
    
    # Create a boolean mask for cells belonging to valid groups
    mask = adata.obs[groupby].isin(valid_groups)
    
    # Apply the mask to filter the AnnData object
    filtered_adata = adata[mask].copy()
    
    # Optionally, reset indices if necessary
    # filtered_adata.obs_names = range(filtered_adata.n_obs)
    
    print(f"Filtered AnnData object contains {filtered_adata.n_obs} cells from {filtered_adata.obs[groupby].nunique()} groups.")
    
    return filtered_adata



def rename_cluster_labels(
    adata: ad.AnnData,
    old_col: str = "cluster",
    new_col: str = "short_cluster"
) -> None:
    """
    Renames cluster labels in the specified .obs column with multi-letter codes.
    
    - All unique labels (including NaN) are mapped in order of appearance to 
      a base-26 style ID: 'A', 'B', ..., 'Z', 'AA', 'AB', etc.
    - The new labels are stored as a categorical column in `adata.obs[new_col]`.
    
    Args:
        adata (AnnData):
            The AnnData object containing the cluster labels.
        old_col (str, optional):
            The name of the .obs column that has the original cluster labels.
            Defaults to "cluster".
        new_col (str, optional):
            The name of the new .obs column that will store the shortened labels.
            Defaults to "short_cluster".
    
    Returns:
        None: The function adds a new column to `adata.obs` in place.
    """
    
    # 1. Extract unique labels (including NaN), in the order they appear
    unique_labels = adata.obs[old_col].unique()
    
    # 2. Helper function for base-26 labeling
    def index_to_label(idx: int) -> str:
        """
        Convert a zero-based index to a base-26 letter code:
        0 -> A
        1 -> B
        ...
        25 -> Z
        26 -> AA
        27 -> AB
        ...
        """
        letters = []
        while True:
            remainder = idx % 26
            letter = chr(ord('A') + remainder)
            letters.append(letter)
            idx = idx // 26 - 1
            if idx < 0:
                break
        return ''.join(letters[::-1])
    
    # 3. Build the mapping (including NaN -> next code)
    label_map = {}
    for i, lbl in enumerate(unique_labels):
        label_map[lbl] = index_to_label(i)
    
    # 4. Apply the mapping to create the new column
    adata.obs[new_col] = adata.obs[old_col].map(label_map)
    adata.obs[new_col] = adata.obs[new_col].astype("category")



def assign_optimal_cluster(adata, cluster_reports, new_col="optimal_cluster"):
    """
    Determine the optimal cluster label per cell across multiple cluster assignments
    by comparing F1-scores, then prepend the chosen label with the name of the .obs
    column that provided it.
    
    Parameters
    ----------
    adata : anndata.AnnData
        The annotated single-cell dataset.
    cluster_reports : dict[str, pd.DataFrame]
        A dictionary where keys are column names in `adata.obs` (each key 
        corresponds to one clustering scheme), and values are DataFrames 
        with classification metrics including 'Label' and 'F1-Score'.
    new_col : str, optional
        The name of the new `.obs` column in which the optimal cluster labels 
        will be stored. Default is "optimal_cluster".
    
    Returns
    -------
    None
        The function adds a new column to `adata.obs` but does not return anything.
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
    """
    Reassign cells whose cluster is in `clusters_to_clean` by picking the 
    highest-correlation cluster from the dendrogram correlation matrix.

    We fix Scanpy's default behavior where:
      - 'categories_ordered' (leaf order) != the row order in 'correlation_matrix'.
      - Instead, 'categories_idx_ordered' is the permutation that maps leaf positions 
        to row indices in the original correlation matrix.
    
    Parameters
    ----------
    adata : anndata.AnnData
        Must contain:
          - adata.obs[current_label]: the current cluster assignments (strings).
          - adata.uns[dendro_key]: a dict with:
             * "categories_ordered": list of cluster labels in dendrogram (leaf) order
             * "categories_idx_ordered": list of row indices corresponding to the above
             * "correlation_matrix": the NxN matrix of correlations in the original order
    dendro_key : str
        Key in adata.uns that has the dendrogram data.
    current_label : str
        Column in adata.obs containing the current cluster assignments.
    new_label : str
        Column name in adata.obs where we store the reassigned clusters.
    clusters_to_clean : list or set of str, optional
        Labels that should be reassigned. If None, nothing will be cleaned.
    
    Returns
    -------
    None
        Adds a new column `adata.obs[new_label]` with updated assignments.
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





def split_adata_indices(
    adata: ad.AnnData,
    n_cells: int = 100,
    groupby: str = "cell_ontology_class",
    cellid: str = None,
    strata_col: str  = None
) -> tuple:
    """
    Splits an AnnData object into training and validation indices based on stratification by cell type
    and optionally by another categorical variable.
    
    Args:
        adata (AnnData): The annotated data matrix to split.
        n_cells (int): The number of cells to sample per cell type.
        groupby (str, optional): The column name in adata.obs that specifies the cell type.
                                 Defaults to "cell_ontology_class".
        cellid (str, optional): The column in adata.obs to use as a unique identifier for cells.
                                If None, it defaults to using the index.
        strata_col (str, optional): The column name in adata.obs used for secondary stratification,
                                    such as developmental stage, gender, or disease status.
    
    Returns:
        tuple: A tuple containing two lists:
            - training_indices (list): List of indices for the training set.
            - validation_indices (list): List of indices for the validation set.
    
    Raises:
        ValueError: If any specified column names do not exist in the DataFrame.
    """
    if cellid is None:
        adata.obs["cellid"] = adata.obs.index
        cellid = "cellid"
    if groupby not in adata.obs.columns or (strata_col and strata_col not in adata.obs.columns):
        raise ValueError("Specified column names do not exist in the DataFrame.")
    
    cts = set(adata.obs[groupby])
    trainingids = []
    
    for ct in cts:
        subset = adata[adata.obs[groupby] == ct]
        
        if strata_col:
            stratified_ids = []
            strata_groups = subset.obs[strata_col].unique()
            n_strata = len(strata_groups)
            
            # Initialize desired count and structure to store samples per strata
            desired_per_group = n_cells // n_strata
            samples_per_group = {}
            remaining = 0
            
            # First pass: allocate base quota or maximum available if less than base
            for group in strata_groups:
                group_subset = subset[subset.obs[strata_col] == group]
                available = group_subset.n_obs
                if available < desired_per_group:
                    samples_per_group[group] = available
                    remaining += desired_per_group - available
                else:
                    samples_per_group[group] = desired_per_group
                
            # Second pass: redistribute remaining quota among groups that can supply more
            # Continue redistributing until either there's no remaining quota or no group can supply more.
            groups_can_supply = True
            while remaining > 0 and groups_can_supply:
                groups_can_supply = False
                for group in strata_groups:
                    group_subset = subset[subset.obs[strata_col] == group]
                    available = group_subset.n_obs
                    # Check if this group can supply an extra cell beyond what we've allocated so far
                    if samples_per_group[group] < available:
                        samples_per_group[group] += 1
                        remaining -= 1
                        groups_can_supply = True
                        if remaining == 0:
                            break
                        
            # Sample cells for each strata group based on the determined counts
            for group in strata_groups:
                group_subset = subset[subset.obs[strata_col] == group]
                count_to_sample = samples_per_group.get(group, 0)
                if count_to_sample > 0:
                    sampled_ids = np.random.choice(
                        group_subset.obs[cellid].values, 
                        count_to_sample, 
                        replace=False
                    )
                    stratified_ids.extend(sampled_ids)
                
            trainingids.extend(stratified_ids)
        else:
            ccount = min(subset.n_obs, n_cells)
            sampled_ids = np.random.choice(subset.obs[cellid].values, ccount, replace=False)
            trainingids.extend(sampled_ids)
        
    # Get all unique IDs
    all_ids = adata.obs[cellid].values
    # Determine validation IDs
    assume_unique = adata.obs_names.is_unique
    val_ids = np.setdiff1d(all_ids, trainingids, assume_unique=assume_unique)
    
    return trainingids, val_ids



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


def remove_singleton_groups(adata: ad.AnnData, groupby: str) -> ad.AnnData:
    """
    Remove groups with only a single cell from an AnnData object.

    Args:
        adata (anndata.AnnData): An AnnData object.
        groupby (str): The column in `.obs` to group by.

    Returns:
        anndata.AnnData: A new AnnData object with singleton groups removed.
    """
    # Get the group sizes
    group_sizes = adata.obs[groupby].value_counts()
    # Filter out groups with only one cell
    non_singleton_groups = group_sizes[group_sizes > 1].index
    # Subset the AnnData object to exclude singleton groups
    filtered_adata = adata[adata.obs[groupby].isin(non_singleton_groups)].copy()
    
    return filtered_adata



def reduce_cells(
    adata: ad.AnnData,
    n_cells: int = 5,
    cluster_key: str = "cluster",
    use_raw: bool = True
) -> ad.AnnData:
    """
    Reduce the number of cells in an AnnData object by combining transcript counts across clusters.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with observations (cells) and variables (features).
    n_cells : int, optional (default: 5)
        The number of cells to combine into a meta-cell.
    cluster_key : str, optional (default: "cluster")
        The key in `adata.obs` that specifies the cluster identity of each cell.
    use_raw : bool, optional (default: True)
        Whether to use the raw count matrix in `adata.raw` instead of `adata.X`.

    Returns
    -------
    AnnData
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
    """
    Average n_cells randomly sampled cells from obs_key == obs_value obs, n_mean_cells times

    Parameters:
    adata: anndata.AnnData
        The annotated data matrix of shape n_obs x n_vars. Rows correspond to cells and columns to genes.
    cluster_key: str
        The key in the obs field that identifies the clusters.
    n_mean_cells: int
        The number of meta-cells to create.
    n_cells: int
        The number of cells to randomly select from each cluster.

    Returns:
    anndata.AnnData
        Annotated data matrix with meta-cells.
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
    """
    Generate hybrid cells by taking the mean of transcript counts for randomly selected cells
    from the specified groups in proportions as indicated by celltypes

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with observations (cells) and variables (genes).
    celltype_counts : dict
        keys indicate the subset of cells and the values are the number of cells to sample from each cell type.
    groupby : str
        The name of the column in `adata.obs` to group cells by before selecting cells at random.
    n_hybrid_cells : int
        The number of hybrid cells to generate.

    Returns
    -------
    AnnData
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
    """
    Sample cells as specified by celltype_counts and groub_by

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with observations (cells) and variables (genes).
    celltype_counts: dict
        keys indicate the subset of cells and the values are the number of cells to sample from each cell type.
    groupby : str
        The name of the column in `adata.obs` to group cells by before selecting cells at random.

    Returns
    -------
    AnnData
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



def find_knn_cells(adata, cell_ids, knn_key="connectivities", return_mode="union"):
    """
    Return an array of cell IDs that are among the k-nearest neighbors for the input cells.
    
    This function leverages the precomputed kNN graph stored in adata.obsp (e.g. from sc.pp.neighbors)
    to retrieve the neighbors of each cell in cell_ids. By default, the function returns the union
    of the neighbor sets for the input cells. Alternatively, if return_mode is set to "intersection",
    only cells that appear in every input cell's kNN list are returned.
    
    Args:
        adata: AnnData object that includes a kNN graph computed by Scanpy. The kNN graph should be stored in 
            adata.obsp under the key specified by knn_key (default: "connectivities").
        cell_ids (list or array-like): A list of cell IDs (matching adata.obs_names) for which to retrieve kNN cells.
        knn_key (str, optional): The key in adata.obsp that holds the kNN graph (typically "connectivities" or "distances").
            Defaults to "connectivities".
        return_mode (str, optional): Mode for combining kNN sets from each input cell. Use "union" to return all cells 
            that appear in at least one kNN list (default) or "intersection" to return only those cells that are present 
            in every kNN list.
    
    Returns:
        array-like: An array of cell IDs corresponding to the combined kNN cells based on the specified mode.
    
    Raises:
        ValueError: If any cell in cell_ids is not found in adata.obs_names or if the kNN matrix is not sparse.
        KeyError: If knn_key is not present in adata.obsp.
    """
    import numpy as np
    from scipy.sparse import issparse
    
    # Check that each provided cell_id exists in adata.obs_names.
    missing_cells = [cell for cell in cell_ids if cell not in adata.obs_names]
    if missing_cells:
        raise ValueError("The following cell ids are not in adata.obs_names: " + ", ".join(missing_cells))
    
    # Check that the specified knn_key exists in adata.obsp.
    if knn_key not in adata.obsp:
        raise KeyError(f"The key '{knn_key}' is not present in adata.obsp.")
    
    # Retrieve the kNN matrix.
    knn_matrix = adata.obsp[knn_key]
    if not issparse(knn_matrix):
        raise ValueError(f"The kNN matrix at adata.obsp['{knn_key}'] is not a sparse matrix.")
    
    # Map cell_ids to their corresponding indices.
    cell_indices = adata.obs_names.get_indexer(cell_ids)
    
    # For each input cell, get its kNN indices (nonzero entries in the corresponding row).
    neighbor_sets = []
    for idx in cell_indices:
        row = knn_matrix.getrow(idx)
        # row.indices holds the column indices with nonzero entries.
        neighbor_sets.append(set(row.indices))
    
    # Combine neighbor sets according to the specified return_mode.
    if return_mode == "union":
        combined_indices = set().union(*neighbor_sets)
    elif return_mode == "intersection":
        # If neighbor_sets is empty, return an empty set.
        combined_indices = set.intersection(*neighbor_sets) if neighbor_sets else set()
    else:
        raise ValueError("return_mode must be 'union' or 'intersection'")
    
    # Convert indices back to cell IDs.
    result_cell_ids = adata.obs_names[list(combined_indices)]
    
    # add the seed cells, too
    result_cell_ids = list(set(result_cell_ids).union(set(cell_ids)))
    return result_cell_ids


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



