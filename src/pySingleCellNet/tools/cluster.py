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
    random_state: Optional[int] = None,
    overwrite: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Grid-search Leiden clusterings over (n_pcs, n_neighbors, resolution).

    Runs a parameter sweep that combines different numbers of principal components,
    k-nearest-neighbor sizes, and Leiden resolutions. Optionally performs random
    *PC subsampling* (within the first ``N`` PCs) when constructing the KNN graph,
    repeating each configuration multiple times for robustness. Cluster labels
    are written to ``adata.obs`` under keys derived from ``prefix`` and the
    parameter settings.

    Assumptions:
        * ``adata.X`` is **already log-transformed**.
        * A base embedding is stored in ``adata.obsm``. By default this is
          ``adata.obsm['X_pca']``, but you can override it via
          ``knn_params['use_rep']`` to leverage an alternative representation.

    Args:
        adata: AnnData object containing the log-transformed expression matrix.
            Must include the embedding referenced by ``knn_params['use_rep']``
            (defaults to ``obsm['X_pca']``).
        leiden_resolutions: Leiden resolution values to evaluate (passed to
            ``sc.tl.leiden``). Each resolution is combined with every KNN/PC
            configuration in the sweep.
        prefix: String prefix used to construct output keys for cluster labels in
            ``adata.obs`` (e.g., ``"{prefix}_pc{N}_k{K}_res{R}"``). Defaults to
            ``"autoc"``.
        pca_params: Configuration for PC selection and optional subsampling.
            Supported keys:
            * ``"top_n_pcs"`` (List[int], default ``[40]``): Candidate values
              for the maximum PC index ``N`` (i.e., use the first ``N`` PCs).
            * ``"percent_of_pcs"`` (Optional[float], default ``None``): If set
              with ``0 < value <= 1``, randomly select
              ``round(value * N)`` PCs **from the first ``N``** for KNN
              construction. If ``None`` or ``1``, use the first ``N`` PCs
              without subsampling.
            * ``"n_random_samples"`` (Optional[int], default ``None``): Number
              of random PC subsets to draw **per (N, K)** when
              ``percent_of_pcs`` is set in ``(0, 1)``. If ``None`` or less than
              1, no repeated subsampling is performed.
        knn_params: KNN graph parameters. Supported keys:
            * ``"n_neighbors"`` (List[int], default ``[10]``): Candidate values
              for ``K`` used in ``sc.pp.neighbors``.
            * ``"use_rep"`` (str, default ``"X_pca"``): Name of the
              ``adata.obsm`` representation to use as the base embedding (e.g.,
              ``"X_pca_noPC1"``). PC subsampling operates on this matrix.
        random_state: Random seed for PC subset sampling (when
            ``percent_of_pcs`` is used). Pass ``None`` for non-deterministic
            sampling. Defaults to ``None``.
        overwrite: If ``True`` (default), overwrite existing ``adata.obs`` keys
            produced by previous runs that match the constructed names. If
            ``False``, skip runs whose target keys already exist.
        verbose: If ``True`` (default), print progress messages for each run.

    Returns:
        pd.DataFrame:  

        * **runs** (``pd.DataFrame``): One row per clustering run with metadata columns such as:
          - ``obs_key``: Name of the column in ``adata.obs`` that stores cluster labels.
          - ``neighbors_key``: Name of the neighbors graph key used/created.
          - ``use_rep``: Embedding key that served as the base representation.
          - ``resolution``: Leiden resolution value used for the run.
          - ``top_n_pcs``: Number of leading PCs considered.
          - ``pct_pcs``: Fraction of PCs used when subsampling (``percent_of_pcs``), or ``1.0`` if all were used.
          - ``sample_idx``: Index of the PC subsampling repeat (``0..n-1``) or ``0`` if no subsampling.
          - ``n_neighbors``: Number of neighbors (``K``) used in KNN construction.
          - ``n_clusters``: Number of clusters returned by Leiden for that run.
          - ``pcs_used_count``: Actual number of PCs used to build the KNN graph
            (``round(pct_pcs * top_n_pcs)`` or ``top_n_pcs`` if no subsampling).

    Raises:
        ValueError: If the requested ``knn_params['use_rep']`` embedding is
            missing from ``adata.obsm`` or if any provided parameter is out of
            range (e.g., ``percent_of_pcs`` not in ``(0, 1]``; empty lists;
            non-positive ``n_neighbors``).
        RuntimeError: If neighbor graph construction or Leiden clustering fails.

    Notes:
        * This function **modifies** ``adata`` in place by adding cluster label
          columns to ``adata.obs`` (and potentially adding or reusing neighbor
          graphs in ``adata.obsp`` / ``adata.uns`` with a constructed
          ``neighbors_key``).
        * To ensure reproducibility when using PC subsampling, set
          ``random_state`` and keep other sources of randomness (e.g., parallel
          BLAS) controlled in your environment.

    Examples:
        >>> runs = cluster_alot(
        ...     adata,
        ...     leiden_resolutions=[0.1, 0.25, 0.5],
        ...     pca_params={"top_n_pcs": [20, 40],
        ...                 "percent_of_pcs": 0.5,
        ...                 "n_random_samples": 3},
        ...     knn_params={"n_neighbors": [10, 20]},
        ...     random_state=42,
        ... )
        >>> runs[["obs_key", "n_clusters"]].head()
    """

    # ---- Normalize params ----
    pca_params = dict(pca_params or {})
    knn_params = dict(knn_params or {})

    use_rep_key = knn_params.get("use_rep", "X_pca")
    if use_rep_key is None:
        use_rep_key = "X_pca"
    if not isinstance(use_rep_key, str):
        raise ValueError("`knn_params['use_rep']` must be a string key in `adata.obsm`.")

    # ---- Validate prerequisites ----
    if use_rep_key not in adata.obsm:
        raise ValueError(
            f"`adata.obsm['{use_rep_key}']` not found. Please compute that representation first."
        )
    X_rep = adata.obsm[use_rep_key]
    n_pcs_available = X_rep.shape[1]
    if n_pcs_available < 2:
        raise ValueError(
            f"Not enough components ({n_pcs_available}) in `adata.obsm['{use_rep_key}']`."
        )
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
        # Create a temporary representation name derived from the requested embedding
        temp_rep_key = f"{use_rep_key}_sub_{neighbors_key}"
        adata.obsm[temp_rep_key] = X_rep[:, pc_idx]

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
            adata.uns[neighbors_key]["base_representation"] = use_rep_key

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
                    "use_rep": use_rep_key,
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
                "use_rep": use_rep_key,
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
    cols = [
        "obs_key",
        "neighbors_key",
        "use_rep",
        "resolution",
        "top_n_pcs",
        "pct_pcs",
        "sample_idx",
        "n_neighbors",
        "pcs_used_count",
        "n_clusters",
        "status",
    ]
    summary_df = summary_df[cols]

    return summary_df


def cluster_subclusters(
    adata: ad.AnnData,
    cluster_column: str = 'leiden',
    to_subcluster: list[str] = None,
    layer = None,
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
        layer: 
            Layer name in `adata.layers` to use for HVG detection. Default is None (and so uses .X)
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
        sc.pp.pca(sub, n_comps=n_pcs, mask_var='highly_variable')

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
