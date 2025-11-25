from __future__ import annotations
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union, Callable
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



GeneSetInput = Union[
    Mapping[str, Sequence[str]],   # {"setA": [...], "setB": [...]}
    Sequence[Sequence[str]],       # [[...], [...]] -> auto-named set_1, set_2, ...
    str,                           # name of an adata.uns key mapping to dict[str, list[str]]
]

def score_gene_sets(
    adata,
    gene_sets: GeneSetInput,
    *,
    layer: Optional[str] = None,
    # ---- value-based (existing) options ----
    log_transform: bool = False,
    clip_percentiles: Tuple[float, float] = (1.0, 99.0),
    agg: Union[str, Callable[[np.ndarray], np.ndarray]] = "mean",
    top_p: Optional[float] = 0.5,           # for agg="top_p_mean" (0<p<=1)
    top_k: Optional[int] = None,            # for agg="top_k_mean"
    # ---- rank-based (new) options ----
    rank_method: Optional[str] = None,      # None | "auc" | "ucell"
    rank_universe: Optional[Union[str, Sequence[str]]] = None,  # None | var column | list of genes
    auc_max_rank: Union[int, float] = 0.05, # AUCell window: int = L, float=(0,1] fraction of universe
    batch_size: int = 2048,                 # batch size for ranking
    use_average_ranks: bool = False,        # use scipy.stats.rankdata (average ties); slower
    # ---- misc ----
    min_genes_per_set: int = 1,
    case_insensitive: bool = False,
    obs_prefix: Optional[str] = None,
    return_dataframe: bool = True,
) -> pd.DataFrame:
    """Compute per-cell gene-set scores with both value-based and rank-based (AUCell/UCell) modes.

    Value-based pipeline (when `rank_method is None`):
      1) Optional log1p.
      2) Per-gene percentile clipping (`clip_percentiles`).
      3) Per-gene min–max scaling to [0, 1].
      4) Aggregate across genes in each set per cell with
         'mean' | 'median' | 'sum' | 'nonzero_mean' | 'top_p_mean' | 'top_k_mean' | callable.

    Rank-based pipeline (when `rank_method in {'auc','ucell'}`):
      • For each cell, rank genes within a chosen universe (`rank_universe`).
      • 'auc'  : AUCell-style AUC in the top L ranks (L = `auc_max_rank`).
      • 'ucell': normalized Mann–Whitney U statistic in [0,1].
      • Ranks are computed in batches (`batch_size`) for memory efficiency.

    Args:
        adata: AnnData object.
        gene_sets: Dict[name -> genes], list of gene lists (auto-named), or name of `adata.uns` key.
        layer: Use `adata.layers[layer]` instead of `.X`.
        log_transform: Apply `np.log1p` before scoring (safe monotone transform).
        clip_percentiles: (low, high) clipping percentiles for value-based mode.
        agg: Aggregation for value-based mode or a callable: (cells×genes) -> (cells,).
        top_p: Fraction for 'top_p_mean' (0<p<=1).
        top_k: Count for 'top_k_mean' (>=1).
        rank_method: None | 'auc' | 'ucell' to switch to rank-based scoring.
        rank_universe: None=all genes; or a boolean var column name (e.g. 'highly_variable');
                       or an explicit list of gene names defining the ranking universe.
        auc_max_rank: AUCell top window (int L) or fraction (0,1].
        batch_size: Row batch size for rank computation.
        use_average_ranks: If True, uses average-tie ranks (scipy.stats.rankdata); slower.
        min_genes_per_set: Require at least this many present genes to score a set (else NaN).
        case_insensitive: Case-insensitive gene matching against `var_names`.
        obs_prefix: If provided, also writes scores to `adata.obs[f"{obs_prefix}{name}"]`.
        return_dataframe: If True, return a DataFrame; else return ndarray.

    Returns:
        DataFrame (cells × sets) of scores (and optionally writes to `adata.obs`).

    Notes:
        • Rank-based scores ignore clipping/min–max (ranks are invariant to monotone transforms).
        • AUCell output here is normalized to [0,1] within the top-L window.
        • UCell output is the normalized U statistic in [0,1].
    """
    # ------- resolve gene_sets -> dict[name] -> list[str] -------
    if isinstance(gene_sets, str):
        if gene_sets not in adata.uns:
            raise ValueError(f"gene_sets='{gene_sets}' not found in adata.uns")
        gs_map = dict(adata.uns[gene_sets])
    elif isinstance(gene_sets, Mapping):
        gs_map = {str(k): list(v) for k, v in gene_sets.items()}
    else:
        gs_map = {f"set_{i+1}": list(v) for i, v in enumerate(gene_sets)}
    if not gs_map:
        raise ValueError("No gene sets provided.")

    X = adata.layers[layer] if layer is not None else adata.X
    n_cells, n_genes = X.shape
    var_names = adata.var_names.astype(str)

    # name lookup
    if case_insensitive:
        lut = {g.lower(): i for i, g in enumerate(var_names)}
        def _loc(g: str) -> int: return lut.get(g.lower(), -1)
    else:
        lut = {g: i for i, g in enumerate(var_names)}
        def _loc(g: str) -> int: return lut.get(g, -1)

    # map each set to present indices (deduped)
    present_idx: Dict[str, np.ndarray] = {}
    for name, genes in gs_map.items():
        idx = sorted({_loc(str(g)) for g in genes if _loc(str(g)) >= 0})
        present_idx[name] = np.array(idx, dtype=int)

    # ======================= RANK-BASED BRANCH =======================
    if rank_method is not None:
        method = rank_method.lower()
        if method not in {"auc", "ucell"}:
            raise ValueError("rank_method must be one of {None, 'auc', 'ucell'}.")

        # pick universe
        if rank_universe is None:
            U_idx = np.arange(n_genes, dtype=int)
        elif isinstance(rank_universe, str) and rank_universe in adata.var.columns:
            mask = adata.var[rank_universe].astype(bool).to_numpy()
            U_idx = np.where(mask)[0]
        else:
            # list-like of gene names
            names = pd.Index(rank_universe)  # raises if not list-like; OK
            U_idx = var_names.get_indexer(names)
            U_idx = U_idx[U_idx >= 0]
        if U_idx.size == 0:
            raise ValueError("rank_universe resolved to 0 genes.")

        # restrict sets to universe; build compact col map
        pos_in_U = {j: k for k, j in enumerate(U_idx)}
        set_cols_in_U: Dict[str, np.ndarray] = {}
        for name, idx in present_idx.items():
            idxU = idx[np.isin(idx, U_idx)]
            if idxU.size < min_genes_per_set:
                set_cols_in_U[name] = np.array([], dtype=int)
            else:
                set_cols_in_U[name] = np.array([pos_in_U[j] for j in idxU], dtype=int)

        # slice universe matrix (cells × |U|)
        Xu = X[:, U_idx].toarray() if sparse.issparse(X) else np.asarray(X)[:, U_idx]
        if log_transform:
            Xu = np.log1p(Xu)  # monotone; safe for ranks

        # AUCell window
        if method == "auc":
            if isinstance(auc_max_rank, float):
                if not (0 < auc_max_rank <= 1):
                    raise ValueError("If auc_max_rank is float, it must be in (0,1].")
                L = max(1, int(np.ceil(auc_max_rank * Xu.shape[1])))
            else:
                L = int(auc_max_rank)
                if L < 1 or L > Xu.shape[1]:
                    raise ValueError("auc_max_rank (int) must be in [1, n_universe].")

        # prepare output
        scores = {name: np.full(n_cells, np.nan, float) for name in gs_map.keys()}

        # optional average-tie ranks
        if use_average_ranks:
            from scipy.stats import rankdata  # local import; slower but exact ties

        # rank batches
        for start in range(0, n_cells, batch_size):
            end = min(n_cells, start + batch_size)
            A = Xu[start:end, :]  # (b × nU)

            if use_average_ranks:
                # ranks ascending: 1..nU (average ties). Loop rows for stability.
                ranks_asc = np.vstack([rankdata(row, method="average") for row in A]).astype(np.float64)
                ranks_desc = A.shape[1] + 1 - ranks_asc
            else:
                # fast ordinal ranks via double argsort (stable)
                order = np.argsort(A, axis=1, kind="mergesort")
                ranks_asc = np.empty_like(order, dtype=np.int32)
                row_indices = np.arange(order.shape[0])[:, None]
                ranks_asc[row_indices, order] = np.arange(1, A.shape[1] + 1, dtype=np.int32)
                ranks_desc = A.shape[1] - ranks_asc + 1

            if method == "ucell":
                nU = A.shape[1]
                for name, cols in set_cols_in_U.items():
                    m = cols.size
                    if m < min_genes_per_set:
                        continue
                    r = ranks_asc[:, cols].astype(np.float64)         # (b × m)
                    U = r.sum(axis=1) - (m * (m + 1) / 2.0)          # Mann–Whitney U
                    denom = m * (nU - m)
                    out = np.zeros(U.shape[0], float)
                    np.divide(U, denom, out=out, where=denom > 0)    # normalized to [0,1]
                    scores[name][start:end] = out

            else:  # AUCell
                Lloc = L
                for name, cols in set_cols_in_U.items():
                    m_all = cols.size
                    if m_all < min_genes_per_set:
                        continue
                    r = ranks_desc[:, cols]                           # (b × m)
                    mask = (r <= Lloc)
                    contrib = (Lloc - r + 1) * mask                   # triangular weights
                    raw = contrib.sum(axis=1)
                    m_prime = min(m_all, Lloc)
                    max_raw = m_prime * Lloc - (m_prime * (m_prime - 1)) / 2.0
                    out = np.zeros(raw.shape[0], float)
                    np.divide(raw, max_raw, out=out, where=max_raw > 0)  # normalize to [0,1]
                    scores[name][start:end] = out

        df = pd.DataFrame(scores, index=adata.obs_names)
        if obs_prefix:
            for k in df.columns:
                adata.obs[f"{obs_prefix}{k}"] = df[k].values
        return df if return_dataframe else df.values

    # ======================= VALUE-BASED BRANCH =======================
    # collect unique indices across all sets
    all_idx: List[int] = []
    for idx in present_idx.values():
        all_idx.extend(idx.tolist())
    uniq_idx = np.array(sorted(set(all_idx)), dtype=int)
    if uniq_idx.size == 0:
        raise ValueError("None of the provided genes are present in adata.var_names.")

    # slice (cells × uniq_genes), densify for percentiles
    Xu = X[:, uniq_idx].toarray() if sparse.issparse(X) else np.asarray(X)[:, uniq_idx]

    # optional log1p
    if log_transform:
        Xu = np.log1p(Xu)

    # per-gene clip + scale to [0,1]
    lo_p, hi_p = float(clip_percentiles[0]), float(clip_percentiles[1])
    if not (0.0 <= lo_p < hi_p <= 100.0):
        raise ValueError("clip_percentiles must satisfy 0 <= low < high <= 100.")
    lo = np.percentile(Xu, lo_p, axis=0)
    hi = np.percentile(Xu, hi_p, axis=0)
    Xu = np.clip(Xu, lo[None, :], hi[None, :])
    denom = (hi - lo)
    denom[denom <= 0] = np.inf
    Xu = (Xu - lo[None, :]) / denom[None, :]
    Xu = np.where(np.isfinite(Xu), Xu, 0.0)

    # compact column map
    compact = {j: k for k, j in enumerate(uniq_idx)}

    # row-wise helpers
    def _row_topk_mean(A: np.ndarray, k: int) -> np.ndarray:
        if k <= 0: return np.zeros(A.shape[0], dtype=float)
        k = min(k, A.shape[1])
        idx = A.shape[1] - k
        part = np.partition(A, idx, axis=1)
        return part[:, -k:].mean(axis=1)

    def _row_nonzero_mean(A: np.ndarray) -> np.ndarray:
        mask = (A > 0)
        num = A.sum(axis=1)
        den = mask.sum(axis=1)
        out = np.zeros(A.shape[0], float)
        np.divide(num, den, out=out, where=den > 0)
        return out

    # pick aggregator
    if isinstance(agg, str):
        agg_l = agg.lower()
        if agg_l == "mean":
            agg_fn = lambda A: A.mean(axis=1)
        elif agg_l == "median":
            agg_fn = lambda A: np.median(A, axis=1)
        elif agg_l == "sum":
            agg_fn = lambda A: A.sum(axis=1)
        elif agg_l == "nonzero_mean":
            agg_fn = _row_nonzero_mean
        elif agg_l == "top_p_mean":
            if top_p is None or not (0 < float(top_p) <= 1):
                raise ValueError("For agg='top_p_mean', provide 0 < top_p <= 1.")
            def agg_fn(A, _p=float(top_p)):
                k = max(1, int(np.ceil(_p * A.shape[1])))
                return _row_topk_mean(A, k)
        elif agg_l == "top_k_mean":
            if top_k is None or int(top_k) < 1:
                raise ValueError("For agg='top_k_mean', provide top_k >= 1.")
            agg_fn = lambda A, _k=int(top_k): _row_topk_mean(A, _k)
        else:
            raise ValueError("agg must be 'mean','median','sum','nonzero_mean','top_p_mean','top_k_mean' or a callable.")
    elif callable(agg):
        agg_fn = lambda A: agg(A)
    else:
        raise ValueError("Invalid 'agg' argument.")

    # aggregate per set
    out = {}
    for name, idx in present_idx.items():
        if idx.size < min_genes_per_set:
            out[name] = np.full(n_cells, np.nan, dtype=float)
            continue
        cols = [compact[j] for j in idx]
        A = Xu[:, cols]  # (cells × genes_in_set)
        out[name] = agg_fn(A)

    df = pd.DataFrame(out, index=adata.obs_names)
    if obs_prefix:
        for k in df.columns:
            adata.obs[f"{obs_prefix}{k}"] = df[k].values
    return df if return_dataframe else df.values



def subset_modules_top_genes(
    adata,
    top_n: int = 20,
    uns_key: str = "knn_modules",
    *,
    method: str = "auto",              # "auto", "corr"
    use_abs: bool = True,              # use |r| when scoring by correlation
    layer: Optional[str] = None,       # override: expression layer to use
    mean_cluster: Optional[bool] = None,
    groupby: Optional[str] = None,
    max_profiles: Optional[int] = 2000,  # downsample rows (cells) for speed if not mean_cluster
    random_state: Optional[int] = 0,
    return_scores: bool = False,
) -> Union[Dict[str, List[str]], Tuple[Dict[str, List[str]], pd.DataFrame]]:
    """Select the top-N most similar genes per module for compact visualization.

    This function takes the gene modules saved by :func:`find_gene_modules`
    (typically under ``adata.uns['knn_modules']``) and returns, for each module,
    the top ``top_n`` genes ranked by within-module similarity.

    Similarity is defined as the **mean correlation** of a gene to the other genes
    in its module, computed across profiles (cells or group means). If your modules
    were produced with ``order_genes_by_within_module_connectivity=True``, then
    ``method="auto"`` will simply slice the first ``top_n`` genes from each module
    (fast path). Otherwise, a correlation-based score is computed.

    Args:
        adata (AnnData): The annotated data matrix.
        top_n (int): Number of genes to keep per module. Defaults to 20.
        uns_key (str): ``.uns`` key holding the modules dict (``{module: [genes...]}``).
            Defaults to ``"knn_modules"``.
        method (str, optional): Similarity method.
            - ``"auto"``: If modules are already ordered by connectivity (as saved by
              :func:`find_gene_modules`), slice the first ``top_n`` genes; otherwise
              compute correlation-based scores.
            - ``"corr"``: Always compute mean (absolute) Pearson correlation within
              each module.
            Defaults to ``"auto"``.
        use_abs (bool, optional): Use absolute correlation (|r|) when scoring.
            Defaults to True.
        layer (str, optional): Expression layer to use. If not provided, this function
            will try to use the layer recorded in ``adata.uns[f"{uns_key}__meta"]["layer"]``,
            otherwise falls back to ``adata.X``.
        mean_cluster (bool, optional): If True, compute similarity on group means
            (matching how modules were built when ``mean_cluster=True``). If None,
            attempts to read from ``{uns_key}__meta``; defaults to False otherwise.
        groupby (str, optional): Column in ``.obs`` used to group cells when
            ``mean_cluster=True``. If None, attempts to read from ``{uns_key}__meta``.
        max_profiles (int, optional): When ``mean_cluster=False`` and the number of rows
            (cells) is very large, downsample to this many profiles for speed.
            Set to ``None`` to disable downsampling. Defaults to 2000.
        random_state (int, optional): Seed for row downsampling. Defaults to 0.
        return_scores (bool, optional): If True, also return a long-form DataFrame with
            per-gene scores and ranks. Defaults to False.

    Returns:
        Dict[str, List[str]]: Mapping of module name to the selected top genes (ordered).
        If ``return_scores=True``, also returns a ``pandas.DataFrame`` with columns:
        ``['module', 'gene', 'score', 'rank']``.

    Raises:
        ValueError: If ``uns_key`` is not found or not a dict, if required metadata
            is missing for mean-clustered scoring, or if no genes are found.

    Notes:
        * Correlation-based scoring operates on a **dense** slice of the expression
          matrix restricted to the genes of each module; the work is done per module
          to keep memory use reasonable.
        * If your modules were created with ``mean_cluster=True``, correlation is
          computed on the **group means** for stability and speed.

    Examples:
        >>> top = subset_modules_top_genes(adata, top_n=15, uns_key="knn_modules")
        >>> # Or, get scores as well:
        >>> top, scores = subset_modules_top_genes(adata, top_n=10, return_scores=True)
        >>> scores.head()
    """
    # --- fetch modules & metadata ---
    if uns_key not in adata.uns or not isinstance(adata.uns[uns_key], dict):
        raise ValueError(f"Expected adata.uns['{uns_key}'] to be a dict of modules.")
    modules: Dict[str, List[str]] = adata.uns[uns_key]
    meta = adata.uns.get(f"{uns_key}__meta", {})

    # Defaults from metadata (if present)
    if layer is None:
        layer = meta.get("layer", None)
    if mean_cluster is None:
        mean_cluster = bool(meta.get("mean_cluster", False))
    if groupby is None:
        groupby = meta.get("groupby", None)

    ordered_by_conn = bool(meta.get("ordered_by_within_module_connectivity", False))

    # Fast path: already ordered by connectivity & method='auto'
    if method == "auto" and ordered_by_conn and not return_scores:
        return {m: genes[:top_n] for m, genes in modules.items()}

    # Determine expression matrix
    X = adata.layers[layer] if layer is not None else adata.X

    # Build the profile-by-gene matrix we'll use for correlation
    if mean_cluster:
        if groupby is None or groupby not in adata.obs.columns:
            raise ValueError(
                "mean_cluster=True but 'groupby' is not provided and not found in metadata."
            )
        if hasattr(sc.get, "aggregate"):
            ad_agg = sc.get.aggregate(adata, by=groupby, func="mean")
            M = ad_agg.layers["mean"]  # (n_groups x n_genes)
        else:
            # Manual group means (sparse-aware)
            groups = adata.obs[groupby].astype("category")
            codes = groups.cat.codes.to_numpy()
            n_groups = groups.cat.categories.size
            G = sparse.csr_matrix((np.ones(adata.n_obs), (np.arange(adata.n_obs), codes)),
                                  shape=(adata.n_obs, n_groups))
            if sparse.issparse(X):
                sums = G.T @ X
            else:
                sums = (G.T @ sparse.csr_matrix(X)).toarray()
            counts = np.asarray(G.sum(axis=0)).ravel() + 1e-12
            M = sums / counts[:, None]  # (n_groups x n_genes)
    else:
        M = X
        # Optionally downsample rows (cells) for speed
        if (max_profiles is not None) and (M.shape[0] > max_profiles):
            rng = np.random.default_rng(random_state)
            idx = rng.choice(M.shape[0], size=max_profiles, replace=False)
            M = M[idx, :]

    var_index = pd.Index(adata.var_names)
    rng = np.random.default_rng(random_state)

    top_dict: Dict[str, List[str]] = {}
    score_rows: List[Dict[str, Union[str, float, int]]] = []

    # Helper: compute per-gene mean correlation to others inside a module
    def _scores_from_corr(Y: np.ndarray) -> np.ndarray:
        if Y.shape[1] == 1:
            return np.array([0.0], dtype=float)
        C = np.corrcoef(Y, rowvar=False)
        if use_abs:
            C = np.abs(C)
        np.fill_diagonal(C, np.nan)
        s = np.nanmean(C, axis=1)
        return np.nan_to_num(s, nan=0.0)

    # Compute scores module-by-module
    for mod, gene_list in modules.items():
        # Only keep genes present in adata
        genes_present = [g for g in gene_list if g in var_index]
        if not genes_present:
            top_dict[mod] = []
            continue

        idx = var_index.get_indexer(genes_present)

        # Extract submatrix (profiles x module_genes) and densify
        if sparse.issparse(M):
            Y = M[:, idx].toarray()
        else:
            Y = np.asarray(M)[:, idx]

        # Decide method (currently 'corr' or the auto fallback)
        # (Connectivity-based scoring is not attempted here because the gene graph
        #  isn't stored on the parent AnnData; modules may already be in that order.)
        scores = _scores_from_corr(Y)

        order = np.argsort(-scores)
        k = min(top_n, len(genes_present))
        keep = [genes_present[i] for i in order[:k]]
        top_dict[mod] = keep

        if return_scores:
            for rank, i in enumerate(order, start=1):
                score_rows.append({
                    "module": mod,
                    "gene": genes_present[i],
                    "score": float(scores[i]),
                    "rank": int(rank),
                })

    if return_scores:
        score_df = pd.DataFrame(score_rows)
        return top_dict, score_df
    return top_dict




def what_module_has_gene(
    adata,
    target_gene,
    mod_slot='knn_modules'
) -> list: 
    if mod_slot not in adata.uns.keys():
        raise ValueError(mod_slot + " have not been identified.")
    genemodules = adata.uns[mod_slot]
    return [key for key, genes in genemodules.items() if target_gene in genes]


def correlate_module_scores_with_pcs(
    adata: AnnData,
    score_key: Union[str, Sequence[float], np.ndarray, pd.Series],
    *,
    pca_key: str = "X_pca",
    variance_key: Optional[str] = "pca",
    method: str = "pearson",
    min_abs_corr: Optional[float] = 0.3,
    drop_na: bool = True,
    sort: bool = True,
) -> pd.DataFrame:
    """Quantify the association between a module score and individual PCs.

    Parameters
    ----------
    adata
        AnnData object containing PCs in ``adata.obsm`` and per-cell module scores.
    score_key
        Either the name of an ``adata.obs`` column holding module scores (e.g., the
        output of :func:`score_gene_sets`) or an explicit array-like of shape
        ``(n_cells,)``.
    pca_key
        Key of the embedding in ``adata.obsm`` to correlate against (defaults to
        ``"X_pca"``).
    variance_key
        Optional ``adata.uns`` key that stores ``"variance_ratio"`` for the chosen
        PCA run (defaults to ``"pca"`` when using ``sc.tl.pca``).
    method
        Correlation metric: ``"pearson"`` (default) or ``"spearman"``.
    min_abs_corr
        Absolute-correlation threshold used to flag PCs that strongly follow the
        module score. Set to ``None`` to skip flagging.
    drop_na
        If ``True`` (default), silently drop cells with missing scores/PC values.
        Otherwise raise when NaNs are detected.
    sort
        If ``True`` (default), sort the output by descending absolute correlation.

    Returns
    -------
    pandas.DataFrame
        Table with one row per PC containing the correlation, absolute correlation,
        two-sided p-value, variance ratio (when available), and a boolean flag
        indicating whether the PC exceeds ``min_abs_corr``.
    """

    if pca_key not in adata.obsm:
        raise ValueError(f"'{pca_key}' not found in adata.obsm. Run PCA first.")
    pcs = np.asarray(adata.obsm[pca_key], dtype=np.float64)
    if pcs.ndim != 2:
        raise ValueError(f"adata.obsm['{pca_key}'] must be 2-D (cells × PCs).")
    if pcs.shape[0] != adata.n_obs:
        raise ValueError("Number of rows in the PCA embedding does not match n_obs.")

    # Resolve the module scores vector
    score_label = None
    if isinstance(score_key, str):
        if score_key not in adata.obs:
            raise ValueError(f"score_key='{score_key}' not present in adata.obs.")
        scores = adata.obs[score_key].to_numpy(dtype=np.float64)
        score_label = score_key
    else:
        scores = np.asarray(score_key, dtype=np.float64).reshape(-1)
        if scores.shape[0] != adata.n_obs:
            raise ValueError("score_key array must have length equal to adata.n_obs.")

    # Handle missing data
    finite_scores = np.isfinite(scores)
    finite_pcs = np.all(np.isfinite(pcs), axis=1)
    if drop_na:
        mask = finite_scores & finite_pcs
    else:
        if not (finite_scores.all() and finite_pcs.all()):
            raise ValueError("NaN/inf detected in scores or PCs; set drop_na=True to filter them.")
        mask = np.ones_like(finite_scores, dtype=bool)

    n_valid = int(mask.sum())
    if n_valid < 3:
        raise ValueError("Need at least 3 valid cells to compute correlations.")

    y = scores[mask]
    X = pcs[mask]

    method_lc = method.lower()
    if method_lc not in {"pearson", "spearman"}:
        raise ValueError("method must be either 'pearson' or 'spearman'.")

    if method_lc == "spearman":
        from scipy.stats import rankdata  # local import to avoid global dependency
        y = rankdata(y)
        # Rank each PC separately
        X = np.apply_along_axis(rankdata, 0, X)

    # Center data
    y = y.astype(np.float64)
    y_centered = y - y.mean()
    y_norm = np.sqrt(np.sum(y_centered ** 2))
    if y_norm == 0:
        raise ValueError("Module score has zero variance; correlation undefined.")

    X_centered = X - X.mean(axis=0)
    X_norm = np.sqrt(np.sum(X_centered ** 2, axis=0))

    with np.errstate(divide="ignore", invalid="ignore"):
        corr = (y_centered @ X_centered) / (y_norm * X_norm)
    corr = corr.astype(np.float64)

    n_pcs = corr.size
    dof = n_valid - 2
    if dof < 1:
        raise ValueError("Not enough cells to compute correlation p-values (need >= 3).")

    # Compute two-sided Pearson p-values (valid for Spearman ranks as an approximation)
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = np.clip(1.0 - corr**2, 1e-12, None)
        t_stat = corr * np.sqrt(dof / denom)
    from scipy import stats as _stats  # local import
    p_values = 2.0 * _stats.t.sf(np.abs(t_stat), df=dof)

    # Variance ratios (if available)
    var_ratio = np.full(n_pcs, np.nan)
    if variance_key is not None and variance_key in adata.uns:
        uns_entry = adata.uns[variance_key]
        if isinstance(uns_entry, Mapping) and "variance_ratio" in uns_entry:
            vr = np.asarray(uns_entry["variance_ratio"], dtype=np.float64).ravel()
            if vr.size:
                var_ratio[: min(n_pcs, vr.size)] = vr[:n_pcs]

    result = pd.DataFrame({
        "pc": [f"PC{i}" for i in range(1, n_pcs + 1)],
        "pc_index": np.arange(1, n_pcs + 1, dtype=int),
        "correlation": corr,
        "abs_correlation": np.abs(corr),
        "p_value": p_values,
        "variance_ratio": var_ratio,
        "n_cells": n_valid,
        "score_key": score_label or "array",
    })

    if min_abs_corr is not None:
        threshold = float(min_abs_corr)
        result["flag_high_corr"] = result["abs_correlation"] >= threshold
        result.attrs["min_abs_corr"] = threshold
    else:
        result["flag_high_corr"] = False

    if sort:
        result = result.sort_values("abs_correlation", ascending=False).reset_index(drop=True)

    return result

