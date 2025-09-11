from __future__ import annotations
from typing import List, Dict, Optional, Sequence, Tuple, Any, Union
import itertools
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import igraph as ig

def cluster_alot(
    adata,
    leiden_resolutions: Sequence[float],
    prefix: str = "autoc",
    pca_params: Optional[Dict[str, Any]] = None,
    knn_params: Optional[Dict[str, Any]] = None,
    random_state: Optional[int] = None,   # seed for PC subsampling
    overwrite: bool = True,               # overwrite .obs keys if they already exist
    verbose: bool = True,                 # print progress lines
) -> pd.DataFrame:
    """
    Grid-search Leiden clusterings over (n_pcs, n_neighbors, resolution), with optional
    random PC subsampling for KNN construction.

    Assumptions:
      - adata.X is already log-transformed
      - PCA has been run and `adata.obsm['X_pca']` exists (use this as base)

    Parameters
    ----------
    adata : AnnData
    leiden_resolutions : list of float
        Resolutions to pass to sc.tl.leiden.
    prefix : str, default "autoc"
        Prefix for .obs key names that store cluster labels.
    pca_params : dict
        {
          'top_n_pcs': List[int] (default [40]),
          'percent_of_pcs': Optional[float] (default None; if set, 0<val<=1),
          'n_random_samples': Optional[int] (default None; # repeats per (n_pcs, n_neighbors))
        }
        If percent_of_pcs is set and 0<val<1 and n_random_samples>=1, for each top_n_pcs=N
        we randomly select round(percent_of_pcs * N) PCs from the first N and build the KNN
        on that subset, repeating n_random_samples times.
        If percent_of_pcs is None (or ==1), we simply use the first N PCs.
    knn_params : dict
        {
          'n_neighbors': List[int] (default [10])
        }
    random_state : int or None
        Seed for PC subsampling reproducibility.
    overwrite : bool
        If False and a would-be .obs key exists, that run is skipped.
    verbose : bool
        If True, prints progress.

    Returns
    -------
    pd.DataFrame
        One row per clustering run with columns:
        ['obs_key','neighbors_key','resolution','top_n_pcs','pct_pcs','sample_idx',
         'n_neighbors','n_clusters','pcs_used_count']
    """
    # ---- Validate prerequisites ----
    if "X_pca" not in adata.obsm:
        raise ValueError("`adata.obsm['X_pca']` not found. Please run PCA first.")
    Xpca = adata.obsm["X_pca"]
    n_pcs_available = Xpca.shape[1]
    if n_pcs_available < 2:
        raise ValueError(f"Not enough PCs ({n_pcs_available}) in `X_pca`.")

    # ---- Normalize params ----
    pca_params = dict(pca_params or {})
    knn_params = dict(knn_params or {})
    top_n_pcs: List[int] = pca_params.get("top_n_pcs", [40])
    percent_of_pcs: Optional[float] = pca_params.get("percent_of_pcs", None)
    n_random_samples: Optional[int] = pca_params.get("n_random_samples", None)
    n_neighbors_list: List[int] = knn_params.get("n_neighbors", [10])

    # sanitize lists
    if isinstance(top_n_pcs, (int, np.integer)): top_n_pcs = [int(top_n_pcs)]
    if isinstance(n_neighbors_list, (int, np.integer)): n_neighbors_list = [int(n_neighbors_list)]
    top_n_pcs = [int(x) for x in top_n_pcs]
    n_neighbors_list = [int(x) for x in n_neighbors_list]

    # sanity checks
    if percent_of_pcs is not None:
        if not (0 < float(percent_of_pcs) <= 1.0):
            raise ValueError("`percent_of_pcs` must be in (0, 1] when provided.")
        if (n_random_samples is None) or (int(n_random_samples) < 1):
            raise ValueError("When using `percent_of_pcs`, set `n_random_samples` >= 1.")
        n_random_samples = int(n_random_samples)

    rng = np.random.default_rng(random_state)

    # ---- Helper: build neighbors from a given PC subspace ----
    def _neighbors_from_pc_indices(pc_idx: np.ndarray, n_neighbors: int, neighbors_key: str):
        """Create a neighbors graph using the given PC column indices."""
        # Create a temporary representation name
        temp_rep_key = f"X_pca_sub_{neighbors_key}"
        adata.obsm[temp_rep_key] = Xpca[:, pc_idx]

        # Build neighbors; store under unique keys (in uns & obsp)
        sc.pp.neighbors(
            adata,
            n_neighbors=n_neighbors,
            use_rep=temp_rep_key,
            key_added=neighbors_key,
        )

        # Record which PCs were used (for provenance)
        if neighbors_key in adata.uns:
            adata.uns[neighbors_key]["pcs_indices"] = pc_idx.astype(int)

        # Clean up the temporary representation to save memory
        del adata.obsm[temp_rep_key]

    # ---- Iterate over parameter combinations ----
    rows = []

    for N, kn, res in itertools.product(top_n_pcs, n_neighbors_list, leiden_resolutions):
        if N > n_pcs_available:
            if verbose:
                print(f"[skip] top_n_pcs={N} > available={n_pcs_available}")
            continue

        # Decide how many runs per (N, kn): either 1 (no subsample) or n_random_samples
        do_subsample = (percent_of_pcs is not None) and (percent_of_pcs < 1.0)
        repeats = n_random_samples if do_subsample else 1
        pcs_target_count = int(round((percent_of_pcs or 1.0) * N))

        # guards
        pcs_target_count = max(1, min(pcs_target_count, N))

        for rep_idx in range(repeats):
            # Choose PC indices
            if do_subsample:
                chosen = rng.choice(N, size=pcs_target_count, replace=False)
                chosen.sort()
                pct_str = f"{percent_of_pcs:.2f}"
                sample_tag = f"s{rep_idx+1:02d}"
            else:
                chosen = np.arange(N, dtype=int)
                pct_str = "1.00"
                sample_tag = "s01"

            # Construct unique keys
            neighbors_key = f"{prefix}_nbrs_pc{N}_pct{pct_str}_{sample_tag}_k{kn}"
            obs_key      = f"{prefix}_pc{N}_pct{pct_str}_{sample_tag}_k{kn}_res{res:g}"

            # Skip or overwrite?
            if (not overwrite) and (obs_key in adata.obs.columns):
                if verbose:
                    print(f"[skip-existing] {obs_key}")
                # we still record an entry (marked as skipped) to keep accounting stable
                rows.append({
                    "obs_key": obs_key,
                    "neighbors_key": neighbors_key,
                    "resolution": res,
                    "top_n_pcs": N,
                    "pct_pcs": float(pct_str),
                    "sample_idx": rep_idx + 1,
                    "n_neighbors": kn,
                    "n_clusters": np.nan,
                    "pcs_used_count": int(chosen.size),
                    "status": "skipped_exists",
                })
                continue

            # Build neighbors
            _neighbors_from_pc_indices(chosen, n_neighbors=kn, neighbors_key=neighbors_key)

            # Cluster using THIS neighbors graph (very important: neighbors_key=...)
            if verbose:
                print(f"[leiden] res={res} | N={N} | pct={pct_str} | {sample_tag} | k={kn} -> {obs_key}")

            sc.tl.leiden(
                adata,
                resolution=float(res),
                flavor="igraph",
                n_iterations=2,
                directed=False,
                key_added=obs_key,
                neighbors_key=neighbors_key,
            )

            # Summaries
            n_clusters = int(pd.Series(adata.obs[obs_key]).nunique())
            rows.append({
                "obs_key": obs_key,
                "neighbors_key": neighbors_key,
                "resolution": float(res),
                "top_n_pcs": int(N),
                "pct_pcs": float(pct_str),
                "sample_idx": int(rep_idx + 1),
                "n_neighbors": int(kn),
                "n_clusters": n_clusters,
                "pcs_used_count": int(chosen.size),
                "status": "ok",
            })

    summary_df = pd.DataFrame(rows)
    # nice ordering
    cols = ["obs_key","neighbors_key","resolution","top_n_pcs","pct_pcs","sample_idx",
            "n_neighbors","pcs_used_count","n_clusters","status"]
    summary_df = summary_df[cols]

    return summary_df


def cluster_subclusters(
    adata: ad.AnnData,
    cluster_column: str = 'leiden',
    to_subcluster: list[str] = None,
    layer: str = 'counts',
    n_hvg: int = 2000,
    hvg_flavor: str  = 'cell_ranger',
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
            flavor=hvg_flavor,
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












