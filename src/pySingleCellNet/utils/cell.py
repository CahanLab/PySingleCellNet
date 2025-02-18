import numpy as np
import pandas as pd
# from anndata import AnnData
# import matplotlib.pyplot as plt
import scanpy as sc
import anndata as ad
# from scipy.sparse import issparse
# from alive_progress import alive_bar
# import string
import igraph as ig



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






