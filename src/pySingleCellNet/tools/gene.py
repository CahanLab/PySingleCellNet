from __future__ import annotations
from typing import Optional, Dict, List, Union
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy import sparse
import igraph as ig
from anndata import AnnData

def build_gene_knn(
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


def find_gene_modules(
    adata,
    mean_cluster: bool = True,
    groupby: str = 'leiden',
    mask_var: Optional[str] = None,
    knn: int = 5,
    leiden_resolution: float = 0.5,
    prefix: str = 'gmod_',
    metric: str = 'euclidean',
    *,
    # NEW:
    uns_key: str = 'knn_modules',                # where to store the dict of modules
    layer: Optional[str] = None,                 # use adata.layers[layer] instead of .X
    min_module_size: int = 2,                    # drop tiny modules (set to 1 to keep all)
    order_genes_by_within_module_connectivity: bool = True,
    random_state: Optional[int] = 0,             # for reproducible Leiden
) -> Dict[str, List[str]]:
    """
    Find gene modules by building a kNN graph over genes (or cluster-mean profiles)
    and clustering with Leiden.

    Writes a dict {f"{prefix}{cluster_id}": [gene names]} to `adata.uns[uns_key]`
    and returns the same dict.

    Parameters
    ----------
    mean_cluster
        If True, aggregate cells by `groupby` before building the gene kNN graph.
    groupby
        Column in adata.obs used for aggregation when `mean_cluster=True`.
    mask_var
        Boolean column in adata.var used to select a subset of genes. If None, use all genes.
    knn
        Number of neighbors for the kNN graph on genes.
    leiden_resolution
        Resolution for Leiden clustering.
    prefix
        Prefix for module names.
    metric
        Distance metric for kNN (e.g. 'euclidean', 'manhattan', 'cosine', 'correlation').
        NOTE: If `metric=='correlation'` and the data are sparse, we densify for stability.
    uns_key
        Top-level .uns key to store the resulting dict of modules (default 'knn_modules').
    layer
        If provided, use `adata.layers[layer]` as expression, otherwise `adata.X`.
        (Aggregation honors this choice.)
    min_module_size
        Remove modules smaller than this size after clustering.
    order_genes_by_within_module_connectivity
        If True, sort each module's genes by their within-module connectivity (descending).
    random_state
        Random seed passed to Leiden for reproducibility.
    """
    # ----------------- 1) Choose expression matrix via a copy -----------------
    adata_subset = adata.copy()
    if layer is not None:
        if layer not in adata.layers:
            raise ValueError(f"Layer '{layer}' not found in adata.layers.")
        adata_subset.X = adata.layers[layer].copy()

    # ----------------- 2) Optional gene mask -----------------
    if mask_var is not None:
        if mask_var not in adata_subset.var.columns:
            raise ValueError(f"Column '{mask_var}' not found in adata.var.")
        gene_mask = adata_subset.var[mask_var].astype(bool).to_numpy()
        if gene_mask.sum() == 0:
            raise ValueError(f"No genes where var['{mask_var}'] is True.")
        adata_subset = adata_subset[:, gene_mask].copy()

    # ----------------- 3) Optional per-cluster aggregation -----------------
    if mean_cluster:
        if groupby not in adata_subset.obs.columns:
            raise ValueError(f"Column '{groupby}' not found in adata.obs.")
        # Prefer Scanpy's aggregation helper if available (Scanpy ≥1.10):
        if hasattr(sc.get, "aggregate"):
            ad_agg = sc.get.aggregate(adata_subset, by=groupby, func='mean')
            adata_subset = ad_agg.copy()
            adata_subset.X = ad_agg.layers['mean']  # make means the working matrix
        else:
            # Fallback: manual aggregation (sparse-aware)
            groups = adata_subset.obs[groupby].astype("category")
            cat = groups.cat.codes.to_numpy()
            n_groups = groups.cat.categories.size
            # Build group indicator sparse matrix G (cells x groups), then G^T * X / counts
            rows = np.arange(adata_subset.n_obs)
            G = sparse.csr_matrix((np.ones_like(rows), (rows, cat)), shape=(adata_subset.n_obs, n_groups))
            if sparse.issparse(adata_subset.X):
                sums = G.T @ adata_subset.X
            else:
                sums = (G.T @ sparse.csr_matrix(adata_subset.X)).toarray()
            counts = np.asarray(G.sum(axis=0)).ravel() + 1e-12
            means = sums / counts[:, None]
            # Build a new AnnData with groups as observations and genes as variables
            adata_subset = sc.AnnData(
                X=means,
                obs=pd.DataFrame(index=groups.cat.categories),
                var=adata_subset.var.copy()
            )

    # ----------------- 4) Transpose: genes become observations -----------------
    adt = adata_subset.T.copy()

    # ----------------- 5) Correlation metric stability (densify if needed) ----
    if metric == 'correlation' and sparse.issparse(adt.X):
        adt.X = adt.X.toarray()

    # ----------------- 6) Build kNN on genes (no PCA) ------------------------
    sc.pp.neighbors(
        adt,
        n_neighbors=int(knn),
        metric=metric,
        n_pcs=0,                # work directly in the expression space
        key_added="gene_neighbors"
    )

    # ----------------- 7) Leiden on that graph --------------------------------
    sc.tl.leiden(
        adt,
        resolution=float(leiden_resolution),
        key_added="gene_modules",
        neighbors_key="gene_neighbors",
        random_state=random_state,
    )

    # ----------------- 8) Collect modules (optionally filter & order) ----------
    # Base groups: leiden label -> list of gene names
    base_groups = (
        adt.obs
        .groupby('gene_modules', observed=True)['gene_modules']
        .apply(lambda s: s.index.tolist())
        .to_dict()
    )

    # Filter tiny modules
    if min_module_size > 1:
        base_groups = {k: v for k, v in base_groups.items() if len(v) >= min_module_size}

    # Optionally order by within-module connectivity
    modules: Dict[str, List[str]] = {}
    if order_genes_by_within_module_connectivity and 'gene_neighbors_connectivities' in adt.obsp:
        C = adt.obsp['gene_neighbors_connectivities']  # sparse CSR
        name_to_idx = {g: i for i, g in enumerate(adt.obs_names)}
        for cluster_id, genes in base_groups.items():
            idx = np.array([name_to_idx[g] for g in genes], dtype=int)
            # sum of weights within the subgraph
            w = np.asarray(C[idx, :][:, idx].sum(axis=1)).ravel()
            order = np.argsort(-w)  # descending
            mod_name = f"{prefix}{cluster_id}"
            modules[mod_name] = [genes[i] for i in order]
    else:
        # keep the original (arbitrary) order
        modules = {f"{prefix}{cluster_id}": gene_list for cluster_id, gene_list in base_groups.items()}

    # ----------------- 9) Store results & return ------------------------------
    adata.uns[uns_key] = modules
    # (optional lightweight metadata alongside; keeps backward-compat for adata.uns[uns_key])
    meta_key = f"{uns_key}__meta"
    adata.uns[meta_key] = {
        "mean_cluster": bool(mean_cluster),
        "groupby": groupby,
        "mask_var": mask_var,
        "knn": int(knn),
        "leiden_resolution": float(leiden_resolution),
        "prefix": prefix,
        "metric": metric,
        "layer": layer,
        "min_module_size": int(min_module_size),
        "ordered_by_within_module_connectivity": bool(order_genes_by_within_module_connectivity),
        "random_state": random_state,
        "n_modules": len(modules),
        "module_sizes": {k: len(v) for k, v in modules.items()},
    }

    return modules


def whoare_genes_neighbors(
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


def what_module_has_gene(
    adata,
    target_gene,
    mod_slot='knn_modules'
) -> list: 
    if mod_slot not in adata.uns.keys():
        raise ValueError(mod_slot + " have not been identified.")
    genemodules = adata.uns[mod_slot]
    return [key for key, genes in genemodules.items() if target_gene in genes]






