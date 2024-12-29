# import datetime
import numpy as np
import pandas as pd
from anndata import AnnData
import matplotlib.pyplot as plt
import scanpy as sc
import anndata as ad
from scipy.sparse import issparse
# import re
from alive_progress import alive_bar
import string



def rename_cluster_labels(
    adata: AnnData,
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
    adata.obs[new_label] = adata.obs[new_labe].astype('category')


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



def build_knn_graph(correlation_matrix, labels, k=5):
    """
    Build a k-nearest neighbors (kNN) graph from a correlation matrix.
    
    Parameters:
        correlation_matrix (ndarray): Square correlation matrix.
        labels (list): Node labels corresponding to the rows/columns of the correlation matrix.
        k (int): Number of nearest neighbors to connect each node to.
    
    Returns:
        igraph.Graph: kNN graph.
    """

    import igraph as ig

    # Ensure the correlation matrix is square
    assert correlation_matrix.shape[0] == correlation_matrix.shape[1], "Matrix must be square."
    
    # Initialize the graph
    n = len(labels)
    g = ig.Graph()
    g.add_vertices(n)
    g.vs["name"] = labels  # Add node labels
    
    # Build kNN edges
    for i in range(n):
        # Get k largest correlations (excluding self-correlation)
        neighbors = np.argsort(correlation_matrix[i, :])[-(k + 1):-1]  # Exclude the node itself
        for j in neighbors:
            g.add_edge(i, j, weight=correlation_matrix[i, j])
    
    return g


def split_adata_indices(adata, ncells, dLevel="cell_ontology_class", cellid=None, strata_col=None):
    """
    Splits an AnnData object into training and validation indices based on stratification by cell type
    and optionally by another categorical variable.

    Args:
        adata (AnnData): The annotated data matrix to split.
        ncells (int): The number of cells to sample per cell type.
        dLevel (str, optional): The column name in adata.obs that specifies the cell type. Defaults to "cell_ontology_class".
        cellid (str, optional): The column in adata.obs to use as a unique identifier for cells. If None, it defaults to using the index.
        strata_col (str, optional): The column name in adata.obs used for secondary stratification, such as developmental stage, gender, or disease status.

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
    if dLevel not in adata.obs.columns or (strata_col and strata_col not in adata.obs.columns):
        raise ValueError("Specified column names do not exist in the DataFrame.")

    cts = set(adata.obs[dLevel])
    trainingids = []

    for ct in cts:
        # print(ct, ": ")
        subset = adata[adata.obs[dLevel] == ct]

        if strata_col:
            stratified_ids = []
            strata_groups = subset.obs[strata_col].unique()
            
            for group in strata_groups:
                group_subset = subset[subset.obs[strata_col] == group]
                ccount = min(group_subset.n_obs, ncells // len(strata_groups))
                if ccount > 0:
                    sampled_ids = np.random.choice(group_subset.obs[cellid].values, ccount, replace=False)
                    stratified_ids.extend(sampled_ids)

            trainingids.extend(stratified_ids)
        else:
            ccount = min(subset.n_obs, ncells)
            sampled_ids = np.random.choice(subset.obs[cellid].values, ccount, replace=False)
            trainingids.extend(sampled_ids)

        # print(subset.n_obs)

    # Get all unique IDs
    all_ids = adata.obs[cellid].values
    # Determine validation IDs
    assume_unique = adata.obs_names.is_unique
    val_ids = np.setdiff1d(all_ids, trainingids, assume_unique=assume_unique)

    # Return indices instead of actual subsets
    return trainingids, val_ids


def filter_anndata_slots(adata, slots_to_keep):
    """
    Creates a copy of an AnnData object and filters it to retain only the specified 
    slots and elements within those slots. Unspecified slots or elements are removed from the copy.
    
    The function operates on a copy of the provided AnnData object, ensuring that the original
    data remains unchanged. This approach allows users to maintain data integrity while
    exploring different subsets or representations of their dataset.
    
    Args:
        adata (AnnData): The AnnData object to be copied and filtered. This object
            represents a single-cell dataset with various annotations and embeddings.
        slots_to_keep (dict): A dictionary specifying which slots and elements within 
            those slots to keep. The keys should be the slot names ('obs', 'var', 'obsm',
            'obsp', 'varm', 'varp'), and the values should be lists of the names within 
            those slots to preserve. If a slot is not mentioned or its value is None, 
            all its contents are removed in the copy. Example format:
            {'obs': ['cluster'], 'var': ['gene_id'], 'obsm': ['X_pca']}
            
    Returns:
        AnnData: A copy of the original AnnData object filtered according to the specified
        slots to keep. This copy contains only the data and annotations specified by the
        `slots_to_keep` dictionary, with all other data and annotations removed.

    Example:
        adata = sc.datasets.pbmc68k_reduced()
        slots_to_keep = {
            'obs': ['n_genes', 'percent_mito'],
            'var': ['n_cells'],
            # Assuming we want to clear these unless specified to keep
            'obsm': None,
            'obsp': None,
            'varm': None,
            'varp': None,
        }
        filtered_adata = filter_anndata_slots(adata, slots_to_keep)
        # `filtered_adata` is the modified copy, `adata` remains unchanged.
    """
    
    # Create a copy of the AnnData object to work on
    adata_copy = adata.copy()
    
    # Define all possible slots
    all_slots = ['obs', 'var', 'obsm', 'obsp', 'varm', 'varp']
    
    for slot in all_slots:
        if slot not in slots_to_keep or slots_to_keep[slot] is None:
            # If slot is not mentioned or is None, remove all its contents
            if slot in ['obs', 'var']:
                setattr(adata_copy, slot, pd.DataFrame(index=getattr(adata_copy, slot).index))
            else:
                setattr(adata_copy, slot, {})
        else:
            # Specific elements within the slot are specified to be kept
            elements_to_keep = slots_to_keep[slot]
            
            if slot in ['obs', 'var']:
                # Filter columns for 'obs' and 'var'
                df = getattr(adata_copy, slot)
                columns_to_drop = [col for col in df.columns if col not in elements_to_keep]
                df.drop(columns=columns_to_drop, inplace=True)
                
            elif slot in ['obsm', 'obsp', 'varm', 'varp']:
                # Filter keys for 'obsm', 'obsp', 'varm', 'varp'
                mapping = getattr(adata_copy, slot)
                keys_to_drop = [key for key in mapping.keys() if key not in elements_to_keep]
                for key in keys_to_drop:
                    del mapping[key]
    
    return adata_copy



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


def remove_singleton_groups(adata: AnnData, groupby: str) -> AnnData:
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


def read_broken_geo_mtx(path: str, prefix: str) -> AnnData:
    # assumes that obs and var in .mtx _could_ be switched
    # determines which is correct by size of genes.tsv and barcodes.tsv

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


def reduce_cells(
    adata: AnnData,
    n_cells: int = 5,
    cluster_key: str = "cluster",
    use_raw: bool = True
) -> AnnData:
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


def add_ambient_rna(
    adata,
    obs_key: str,
    obs_val: str,
    n_cells_to_sample: int = 10,
    weight_of_ambient: float = 0.05
) -> AnnData:

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



def create_hybrid_cells(
    adata: AnnData,
    celltype_counts: dict,
    groupby: str,
    n_hybrid_cells: int
) -> AnnData:
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
    adata: AnnData,
    celltype_counts: dict,
    groupby: str
)-> AnnData:
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


def compute_mean_expression_per_cluster(
    adata,
    cluster_key
):
    """
    Compute mean gene expression for each gene in each cluster, create a new anndata object, and store it in adata.uns.

    Parameters:
    - adata : anndata.AnnData
        The input AnnData object with labeled cell clusters.
    - cluster_key : str
        The key in adata.obs where the cluster labels are stored.

    Returns:
    - anndata.AnnData
        The modified AnnData object with the mean expression anndata stored in uns['mean_expression'].
    """
    if cluster_key not in adata.obs.columns:
        raise ValueError(f"{cluster_key} not found in adata.obs")

    # Extract unique cluster labels
    clusters = adata.obs[cluster_key].unique().tolist()

    # Compute mean expression for each cluster
    mean_expressions = []
    for cluster in clusters:
        cluster_cells = adata[adata.obs[cluster_key] == cluster, :]
        mean_expression = np.mean(cluster_cells.X, axis=0).A1 if issparse(cluster_cells.X) else np.mean(cluster_cells.X, axis=0)
        mean_expressions.append(mean_expression)

    # Convert to matrix
    mean_expression_matrix = np.vstack(mean_expressions)
    
    # Create a new anndata object
    mean_expression_adata = sc.AnnData(X=mean_expression_matrix, 
                                       var=pd.DataFrame(index=adata.var_names), 
                                       obs=pd.DataFrame(index=clusters))
    
    # Store this new anndata object in adata.uns
    adata.uns['mean_expression'] = mean_expression_adata
    #return adata



def split_common_anndata(adata, ncells, dLevel="cell_ontology_class", cellid = None, cells_reserved = 3):
    if cellid == None: 
         adata.obs["cellid"] = adata.obs.index
         cellid = "cellid"
    cts = set(adata.obs[dLevel])

    trainingids = np.empty(0)
    
    n_cts = len(cts)
    with alive_bar(n_cts, title="Splitting data") as bar:
        for ct in cts:
            # print(ct, ": ")
            aX = adata[adata.obs[dLevel] == ct, :]
            ccount = aX.n_obs - cells_reserved
            ccount = min([ccount, ncells])
            # print(aX.n_obs)
            trainingids = np.append(trainingids, np.random.choice(aX.obs[cellid].values, ccount, replace = False))
            bar()

    val_ids = np.setdiff1d(adata.obs[cellid].values, trainingids, assume_unique = True)
    aTrain = adata[np.isin(adata.obs[cellid], trainingids, assume_unique = True),:]
    aTest = adata[np.isin(adata.obs[cellid], val_ids, assume_unique = True),:]
    return([aTrain, aTest])

def splitCommon(expData, ncells,sampTab, dLevel="cell_ontology_class", cells_reserved = 3):
    cts = set(sampTab[dLevel])
    trainingids = np.empty(0)
    for ct in cts:
        aX = expData.loc[sampTab[dLevel] == ct, :]
        # print(ct, ": ")
        ccount = len(aX.index) - cells_reserved
        ccount = min([ccount, ncells])
        # print(ccount)
        trainingids = np.append(trainingids, np.random.choice(aX.index.values, ccount, replace = False))
    val_ids = np.setdiff1d(sampTab.index, trainingids, assume_unique = True)
    aTrain = expData.loc[np.isin(sampTab.index.values, trainingids, assume_unique = True),:]
    aTest = expData.loc[np.isin(sampTab.index.values, val_ids, assume_unique = True),:]
    return([aTrain, aTest])



def find_elbow(
    adata
):
    """
    Find the "elbow" index in the variance explained by principal components.

    Parameters:
    - variance_explained : list or array
        Variance explained by each principal component, typically in decreasing order.

    Returns:
    - int
        The index corresponding to the "elbow" in the variance explained plot.
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


# convert adata.uns[x] into a dict  data.frames
#### SEE postclass_analysis.py convert_rank_genes_groups_to_dict()
def convert_diffExp_to_dict(
    adata,
    uns_name: str = 'rank_genes_groups'
):
    
    # sc.tl.rank_genes_groups(adTemp, dLevel, use_raw=False, method=test_name)
    # tempTab = pd.DataFrame(adata.uns[uns_name]['names']).head(topX)
    tempTab = sc.get.rank_genes_groups_df(adata, group = None, key=uns_name)
    tempTab = tempTab.dropna()
    groups = tempTab['group'].cat.categories.to_list()
    #groups = np.unique(grps)

    ans = {}
    for g in groups:
        ans[g] = tempTab[tempTab['group']==g].copy()
    return ans




