from __future__ import annotations
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union
import re
from scipy.sparse import issparse, coo_matrix, csr_matrix, hstack
import igraph as ig
import scipy.sparse as sp
from scipy.stats import median_abs_deviation, ttest_ind
from scipy import sparse
from sklearn.decomposition import PCA
import math
from scipy.sparse.csgraph import connected_components


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

def remove_genes(adata, genes_to_exclude=None):
    adnew = adata[:,~adata.var_names.isin(genes_to_exclude)].copy()
    return adnew


def drop_pcs_from_embedding(
    adata: ad.AnnData,
    pc_indices: Sequence[int],
    *,
    base_obsm_key: str = "X_pca",
    pca_uns_key: str = "pca",
    base_varm_key: str = "PCs",
    new_suffix: Optional[str] = None,
    assume_one_indexed: bool = True,
) -> Dict[str, str]:
    """Create PCA-derived slots that exclude specific principal components.

    Parameters
    ----------
    adata
        AnnData containing PCA results (``.obsm[base_obsm_key]``).
    pc_indices
        Iterable of PC identifiers to drop. By default interpreted as 1-indexed
        (PC1 == 1); set ``assume_one_indexed=False`` for zero-based indices.
    base_obsm_key
        Name of the PCA embedding in ``adata.obsm`` (default ``"X_pca"``).
    pca_uns_key
        ``adata.uns`` entry that stores PCA metadata (default ``"pca"``).
    base_varm_key
        ``adata.varm`` key with per-gene loadings (default ``"PCs"``).
    new_suffix
        Optional suffix used when naming the derived slots. If omitted, a suffix
        like ``"noPC1_2"`` is generated from the requested indices.
    assume_one_indexed
        Whether ``pc_indices`` are 1-indexed (default ``True``). If ``False``,
        indices are treated as zero-based column positions.

    Returns
    -------
    dict
        Mapping describing the new keys that were created:
        ``{"obsm": new_obsm_key, "varm": new_varm_key, "variance_ratio": new_var_ratio_key}``.
    """

    if base_obsm_key not in adata.obsm:
        raise ValueError(f"'{base_obsm_key}' not found in adata.obsm; run PCA first.")
    if not pc_indices:
        raise ValueError("pc_indices must contain at least one component to drop.")

    base_embed = adata.obsm[base_obsm_key]
    n_cells, n_pcs = base_embed.shape

    idx = np.array(pc_indices, dtype=int)
    if assume_one_indexed:
        idx = idx - 1
    if (idx < 0).any():
        raise ValueError("pc_indices must be positive when assume_one_indexed=True.")
    if (idx >= n_pcs).any():
        raise ValueError("pc_indices contain entries >= total number of PCs available.")
    idx = np.unique(idx)
    keep = np.setdiff1d(np.arange(n_pcs), idx, assume_unique=False)
    if keep.size == 0:
        raise ValueError("Removing all PCs is not allowed.")

    suffix = new_suffix
    if suffix is None:
        pcs_str = "_".join([str(int(i + (1 if assume_one_indexed else 0))) for i in idx])
        suffix = f"noPC{pcs_str}"

    new_obsm_key = f"{base_obsm_key}_{suffix}"
    adata.obsm[new_obsm_key] = base_embed[:, keep]

    var_ratio_key = None
    if pca_uns_key in adata.uns:
        uns_entry = adata.uns[pca_uns_key]
        if isinstance(uns_entry, dict) and "variance_ratio" in uns_entry:
            var_ratio_key = f"variance_ratio_{suffix}"
            adata.uns[pca_uns_key][var_ratio_key] = np.asarray(uns_entry["variance_ratio"])[keep]

    varm_key = None
    if base_varm_key in adata.varm:
        varm_key = f"{base_varm_key}_{suffix}"
        adata.varm[varm_key] = np.asarray(adata.varm[base_varm_key])[:, keep]

    return {
        "obsm": new_obsm_key,
        "varm": varm_key,
        "variance_ratio": var_ratio_key,
    }


def filter_anndata_slots(
    adata,
    slots_to_keep: Dict[str, Optional[List[str]]],
    *,
    keep_dependencies: bool = True,
):
    """
    Return a filtered COPY of `adata` that only keeps requested slots/keys.
    Unspecified slots (or with value None) are cleared.

    Parameters
    ----------
    adata : AnnData
    slots_to_keep : dict
        Keys among {'obs','var','obsm','obsp','varm','varp','uns'}.
        Values are lists of names to keep within that slot; if a slot is not
        present in the dict or is None, all contents of that slot are removed.
        Example:
            {'obs': ['leiden','sample'],
             'obsm': ['X_pca','X_umap'],
             'uns':  ['neighbors', 'pca', 'umap']}
    keep_dependencies : bool, default True
        If True, automatically keep cross-slot items that are commonly required:
          - For each neighbors block in `.uns[<key>]` with
            'connectivities_key' / 'distances_key', also keep those in `.obsp`.
          - If an `.obsp` key ends with '_connectivities'/'_distances', also keep
            the matching `.uns[<prefix>]` if present.
          - If keeping 'X_pca' in `.obsm`, also keep `.uns['pca']` and `.varm['PCs']` if present.
          - If keeping 'X_umap' in `.obsm`, also keep `.uns['umap']` if present.

    Returns
    -------
    AnnData
        A copy with filtered slots.
    """
    ad = adata.copy()

    # Normalize user intent -> sets (and include absent slots as None)
    wanted: Dict[str, Optional[Set[str]]] = {}
    for slot in ['obs','var','obsm','obsp','varm','varp','uns']:
        v = slots_to_keep.get(slot, None)
        if v is None:
            wanted[slot] = None
        else:
            wanted[slot] = set(v)

    if keep_dependencies:
        # Start with the user's desired keeps; we may add to them
        for slot in ['obsm','obsp','varm','varp','uns']:
            if wanted[slot] is None:
                wanted[slot] = set()
        # --- neighbors dependencies ---
        # From kept UNS neighbors -> add OBSP matrices
        for k in list(wanted['uns']):
            if k in ad.uns and isinstance(ad.uns[k], dict):
                ck = ad.uns[k].get('connectivities_key', None)
                dk = ad.uns[k].get('distances_key', None)
                if ck is not None:
                    wanted['obsp'].add(ck)
                if dk is not None:
                    wanted['obsp'].add(dk)
        # From kept OBSP matrices -> add UNS neighbors blocks (by prefix)
        for m in list(wanted['obsp']):
            # match "<prefix>_connectivities" or "<prefix>_distances"
            m_str = str(m)
            m0 = re.sub(r'_(connectivities|distances)$', '', m_str)
            if m0 != m_str and (m0 in ad.uns):
                wanted['uns'].add(m0)

        # --- PCA/UMAP niceties ---
        if wanted['obsm'] and 'X_pca' in ad.obsm:
            if 'X_pca' in wanted['obsm']:
                if 'pca' in ad.uns:
                    wanted['uns'].add('pca')
                if 'PCs' in ad.varm:
                    wanted['varm'].add('PCs')
        if wanted['obsm'] and 'X_umap' in ad.obsm and 'X_umap' in wanted['obsm']:
            if 'umap' in ad.uns:
                wanted['uns'].add('umap')

        # If the user explicitly set a slot to None, restore None (means "clear all")
        for slot in ['obsm','obsp','varm','varp','uns']:
            if slots_to_keep.get(slot, '___SENTINEL___') is None:
                wanted[slot] = None

    # ---------- Apply filtering ----------
    # obs / var: keep only requested columns (preserve indices & dtypes)
    for slot in ['obs','var']:
        cols_keep = wanted[slot]
        if cols_keep is None:
            # drop all columns, preserve index
            empty = pd.DataFrame(index=getattr(ad, slot).index)
            setattr(ad, slot, empty)
        else:
            df = getattr(ad, slot)
            cols_exist = [c for c in df.columns if c in cols_keep]
            setattr(ad, slot, df.loc[:, cols_exist])

    # Mapping-like slots: operate in place to preserve AnnData's aligned mappings
    def _filter_mapping(mapping, keys_keep: Optional[Set[str]]):
        if keys_keep is None:
            mapping.clear()
            return
        # Remove any key not in keep set
        for k in list(mapping.keys()):
            if k not in keys_keep:
                del mapping[k]

    _filter_mapping(ad.obsm, wanted['obsm'])
    _filter_mapping(ad.obsp, wanted['obsp'])
    _filter_mapping(ad.varm, wanted['varm'])
    _filter_mapping(ad.varp, wanted['varp'])

    # .uns is a plain dict (but can be nested); keep only top-level keys
    if wanted['uns'] is None:
        ad.uns.clear()
    else:
        for k in list(ad.uns.keys()):
            if k not in wanted['uns']:
                del ad.uns[k]

    return ad

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



