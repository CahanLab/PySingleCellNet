from __future__ import annotations
from typing import List, Optional, Tuple, Union
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
import math
from scipy.sparse import coo_matrix, csr_matrix, hstack
from scipy.sparse.csgraph import connected_components



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

