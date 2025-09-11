from __future__ import annotations
from typing import List, Optional, Tuple, Union, Sequence, Dict
import itertools
import numpy as np
import pandas as pd

from scipy.cluster.hierarchy import linkage, fcluster
# (No large dense pairwise matrices; we only compute distances within lenient components)


# ------------------------------
# Small, fast disjoint-set union
# ------------------------------
class _DSU:
    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=np.int64)
        self.size = np.ones(n, dtype=np.int32)

    def find(self, x: int) -> int:
        # Path compression
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        # Union by size
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


def _masked_hamming(u: np.ndarray, v: np.ndarray, missing_val: int = -1) -> float:
    """
    Hamming-like distance ignoring positions where either is missing.
    If no valid positions remain, returns 1.0 (maximally different).
    """
    valid = (u != missing_val) & (v != missing_val)
    n = valid.sum()
    if n == 0:
        return 1.0
    return float((u[valid] != v[valid]).sum()) / float(n)


def _pairwise_masked_hamming(X: np.ndarray, missing_val: int = -1) -> np.ndarray:
    """
    Condensed-distance vector over rows of X using _masked_hamming.
    """
    m = X.shape[0]
    d = np.empty(m * (m - 1) // 2, dtype=float)
    k = 0
    for i in range(m - 1):
        ui = X[i]
        for j in range(i + 1, m):
            d[k] = _masked_hamming(ui, X[j], missing_val=missing_val)
            k += 1
    return d


def discover_cell_cliques(
    adata,
    cluster_cols: Union[List[str], str],
    k: Optional[int] = None,           # number of runs that must agree; None => all runs
    mode: str = "lenient",             # 'lenient' (DSU over k-run signatures) or 'strict'
    out_col: str = "core_cluster",
    min_size: int = 1,
    allow_missing: bool = False,
    max_combinations: Optional[int] = None,   # safety for very large #runs
    return_details: bool = False,
) -> Union[pd.Series, Tuple[pd.Series, Dict]]:
    """
    Define 'core' clusters across multiple clustering runs.

    Parameters
    ----------
    adata : AnnData
        The data object with clustering labels in .obs columns.
    cluster_cols : list[str] or str
        One or more .obs columns, each containing a clustering.
    k : int or None, default None
        Cells must be in the same cluster in at least k runs to be grouped.
        If None, uses all runs (k = n_runs), i.e., exact tuple agreement.
    mode : {'lenient','strict'}, default 'lenient'
        'lenient' uses DSU over all k-run combinations (fast, transitive).
        'strict' refines lenient components so every pair inside a final cluster
        agrees in >= k runs (complete-linkage on masked Hamming).
    out_col : str, default 'core_cluster'
        Name of the output categorical column added to adata.obs.
    min_size : int, default 1
        Minimum size to keep a core cluster; smaller groups get 'core_-1'.
    allow_missing : bool, default False
        If False, raises if any clustering column has missing labels.
        If True, combinations containing missing labels for a cell are either skipped
        (lenient) or ignored in distance computations (strict).
    max_combinations : int or None
        If set and number of k-run combinations exceeds this, raises with guidance.
    return_details : bool, default False
        If True, also returns a dict with bookkeeping info.

    Returns
    -------
    core_labels : pandas.Series (categorical)
    details : dict (optional)
    """
    # ---------------- Validate & prep ----------------
    if isinstance(cluster_cols, str):
        cluster_cols = [cluster_cols]
    if not cluster_cols:
        raise ValueError("Provide at least one column in `cluster_cols`.")
    for c in cluster_cols:
        if c not in adata.obs.columns:
            raise ValueError(f"'{c}' not found in adata.obs")

    n = adata.n_obs
    n_runs = len(cluster_cols)
    if k is None:
        k = n_runs
    if not (1 <= k <= n_runs):
        raise ValueError(f"`k` must be between 1 and {n_runs} (inclusive).")
    if mode not in ("lenient", "strict"):
        raise ValueError("`mode` must be 'lenient' or 'strict'.")

    labels_df = adata.obs[cluster_cols].copy()
    if not allow_missing and labels_df.isna().any().any():
        missing_cols = labels_df.columns[labels_df.isna().any(axis=0)].tolist()
        raise ValueError(
            f"Missing labels detected in columns: {missing_cols}. "
            "Pass allow_missing=True to proceed."
        )
    labels_df = labels_df.astype("category")
    # Codes: (n_cells, n_runs); NaN -> -1 sentinel
    codes = np.vstack([labels_df[c].cat.codes.to_numpy() for c in cluster_cols]).T.astype(np.int64)
    codes[codes < 0] = -1

    # ---- Fast path: exact consensus (k == n_runs) => tuple equality ----
    if k == n_runs:
        key = pd.MultiIndex.from_frame(labels_df.apply(lambda s: s.astype(str))).to_numpy()
        grp_ids, _ = pd.factorize(key, sort=False)
        core = pd.Series([f"core_{i}" for i in grp_ids], index=adata.obs_names, name=out_col)

        sizes = core.value_counts()
        small_mask = core.map(sizes) < min_size
        core = core.mask(small_mask, "core_-1")
        adata.obs[out_col] = pd.Categorical(core)
        details = {
            'n_runs': n_runs, 'k': k, 'mode': mode,
            'component_sizes_before_strict': sizes.to_dict(),
            'component_sizes_after_strict': sizes.to_dict()
        }
        return (adata.obs[out_col], details) if return_details else adata.obs[out_col]

    # --------------- LENIENT (DSU over k-combos) ---------------
    dsu = _DSU(n)
    run_indices = list(range(n_runs))
    combos = list(itertools.combinations(run_indices, k))
    if (max_combinations is not None) and (len(combos) > max_combinations):
        raise RuntimeError(
            f"Number of combinations C({n_runs},{k}) = {len(combos)} exceeds "
            f"max_combinations={max_combinations}. Increase k, reduce runs, or raise max_combinations."
        )

    for comb in combos:
        cols = np.array(comb, dtype=int)
        sub = codes[:, cols]  # (n_cells, k)
        # Valid rows for this combination:
        # - If allow_missing=False: require all k labels present
        # - If allow_missing=True: require at least one present (others may be -1)
        valid = (sub != -1).all(axis=1) if not allow_missing else (sub != -1).any(axis=1)
        if not valid.any():
            continue

        # Build signatures as tuples for hashing (missing stays as -1 if allow_missing=True)
        sigs = np.full(n, None, dtype=object)
        for i, v in enumerate(valid):
            if v:
                sigs[i] = tuple(sub[i].tolist())

        # Group equal signatures and union their members
        bucket: Dict[Tuple[int, ...], List[int]] = {}
        for i, sig in enumerate(sigs):
            if sig is None:
                continue
            lst = bucket.get(sig)
            if lst is None:
                bucket[sig] = [i]
            else:
                lst.append(i)
        for members in bucket.values():
            if len(members) >= 2:
                base = members[0]
                for other in members[1:]:
                    dsu.union(base, other)

    # Extract lenient components
    roots = np.fromiter((dsu.find(i) for i in range(n)), dtype=np.int64, count=n)
    _, comp_codes = np.unique(roots, return_inverse=True)
    core_lenient = pd.Series([f"core_{c}" for c in comp_codes], index=adata.obs_names)

    # --------------- STRICT refinement (fixed block) ---------------
    if mode == "strict":
        # Require every pair inside a final cluster to agree in >= k runs.
        # This is equivalent to complete-linkage with cutoff d_max = 1 - (k/n_runs)
        d_max = 1.0 - (float(k) / float(n_runs))
        refined = pd.Series(index=adata.obs_names, dtype=object)

        comp_sizes_before: Dict[str, int] = {}
        comp_sizes_after: Dict[str, int] = {}

        groups = core_lenient.groupby(core_lenient).groups  # dict: {comp_id: Index(obs_names)}

        for comp_id, obs_keys in groups.items():
            # Map obs-name labels -> integer positions (FIX)
            idxs = adata.obs_names.get_indexer(pd.Index(obs_keys))
            if (idxs < 0).any():
                raise RuntimeError("Encountered unknown obs_names while mapping to positions.")

            comp_sizes_before[comp_id] = idxs.size

            if idxs.size <= 1:
                refined.iloc[idxs] = comp_id + "_0"
                comp_sizes_after[comp_id + "_0"] = idxs.size
                continue

            X = codes[idxs, :]  # (s, n_runs)
            D = _pairwise_masked_hamming(X, missing_val=-1)

            # If already satisfies the strict criterion, keep as one block
            if np.all(D <= d_max):
                refined.iloc[idxs] = comp_id + "_0"
                comp_sizes_after[comp_id + "_0"] = idxs.size
                continue

            Z = linkage(D, method="complete")
            labs = fcluster(Z, t=d_max, criterion="distance")  # 1..K within this component

            for lab in np.unique(labs):
                sel = idxs[labs == lab]  # integer positions
                out_name = f"{comp_id}_{int(lab)-1}"
                refined.iloc[sel] = out_name
                comp_sizes_after[out_name] = sel.size

        core = refined
    else:
        core = core_lenient
        vc = core.value_counts()
        comp_sizes_before = vc.to_dict()
        comp_sizes_after = vc.to_dict()

    # Enforce min_size
    sizes = core.value_counts()
    small_mask = core.map(sizes) < min_size
    core = core.mask(small_mask, "core_-1")
    core.name = out_col

    adata.obs[out_col] = pd.Categorical(core)

    details = {
        'n_runs': n_runs,
        'k': k,
        'mode': mode,
        'component_sizes_before_strict': comp_sizes_before,
        'component_sizes_after_strict': comp_sizes_after,
    }
    return (adata.obs[out_col], details) if return_details else adata.obs[out_col]




