from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List, Sequence, Union
from scipy import sparse
from scipy.stats import ttest_ind
from sklearn.decomposition import PCA
import hashlib

# ---------- simple, dependency-free p-value adjusters ----------
def _p_adjust(p: np.ndarray, method: str = "fdr_bh") -> np.ndarray:
    p = np.asarray(p, dtype=float)
    m = p.size
    if m == 0:
        return p
    if method == "bonferroni":
        return np.minimum(1.0, p * m)
    if method == "holm":
        order = np.argsort(p)
        adj = np.empty_like(p)
        for rank, i in enumerate(order, start=1):
            adj[i] = (m - rank + 1) * p[i]
        adj_sorted = adj[order]
        adj_sorted = np.maximum.accumulate(adj_sorted[::-1])[::-1]
        adj[order] = np.minimum(1.0, adj_sorted)
        return adj
    # Benjamini–Hochberg
    order = np.argsort(p)
    p_sorted = p[order]
    ranks = np.arange(1, m + 1, dtype=float)
    q_sorted = p_sorted * m / ranks
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q = np.empty_like(q_sorted)
    q[order] = np.minimum(1.0, q_sorted)
    return q

# ---------- helpers ----------
def _to_dense(X):
    return X.toarray() if sparse.issparse(X) else np.asarray(X)

def _gini(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.size == 0 or np.all(x == 0):
        return 0.0
    x = np.sort(x)
    n = x.size
    return 1 + 1/n - 2 * np.sum((np.arange(1, n+1) * x)) / (n * x.sum())

def _canonical_codes(series: pd.Series) -> np.ndarray:
    """
    Map cluster labels to stable integer IDs by *first appearance order* over cells.
    This yields the same signature for identical partitions even if label strings differ.
    """
    codes = np.empty(series.shape[0], dtype=np.int32)
    seen = {}
    nxt = 0
    # cast to object to avoid category code idiosyncrasies across columns
    for i, lab in enumerate(series.astype("object").to_numpy()):
        # treat None/NaN as its own label
        if lab not in seen:
            seen[lab] = nxt
            nxt += 1
        codes[i] = seen[lab]
    return codes

def _hash_codes(codes: np.ndarray) -> str:
    return hashlib.sha1(codes.tobytes()).hexdigest()

def _compute_representation(adata, n_pcs_for_nn: int, layer: Optional[str], has_log1p: bool):
    # Prefer existing PCA
    if "X_pca" in adata.obsm and adata.obsm["X_pca"].shape[1] >= max(2, n_pcs_for_nn):
        return np.asarray(adata.obsm["X_pca"][:, :n_pcs_for_nn])
    # Otherwise compute a compact PCA on (optionally log1p) expression
    X = adata.layers[layer] if layer is not None else adata.X
    # use HVGs if present
    if "highly_variable" in adata.var.columns and adata.var["highly_variable"].any():
        mask = adata.var["highly_variable"].to_numpy().astype(bool)
    else:
        Xd = _to_dense(X)
        v = Xd.var(axis=0)
        topk = min(1000, X.shape[1])
        mask = np.zeros(X.shape[1], dtype=bool)
        mask[np.argsort(v)[-topk:]] = True
    Xg = _to_dense(adata[:, mask].X if layer is None else adata[:, mask].layers[layer])
    Z = Xg if has_log1p else np.log1p(Xg)
    Z -= Z.mean(axis=0, keepdims=True)
    pca = PCA(n_components=min(n_pcs_for_nn, Z.shape[1], Z.shape[0]-1))
    return pca.fit_transform(Z)

def _nearest_neighbor_by_centroid(rep: np.ndarray, codes: np.ndarray) -> Dict[int, int]:
    """Return dict cluster_code -> nearest cluster_code using Euclidean on cluster centroids."""
    uniq = np.unique(codes)
    centroids = {}
    for c in uniq:
        idx = (codes == c)
        centroids[c] = rep[idx].mean(axis=0)
    keys = list(centroids.keys())
    arr = np.stack([centroids[k] for k in keys], axis=0)
    d2 = np.sum((arr[:, None, :] - arr[None, :, :])**2, axis=2)
    np.fill_diagonal(d2, np.inf)
    nn_idx = np.argmin(d2, axis=1)
    return {keys[i]: keys[j] for i, j in enumerate(nn_idx)}

# ---------- core per-clustering evaluator ----------
def _evaluate_one_partition(
    adata,
    codes: np.ndarray,              # canonical/stable cluster IDs per cell (length n)
    rep: np.ndarray,                # (n_cells x n_pcs) embedding for NN detection
    expr,                           # expression matrix (layer or X)
    genes: np.ndarray,              # var_names
    gene_mask: np.ndarray,          # boolean mask over genes
    n_genes: int,
    naive: dict,
    strict: dict,
    has_log1p: bool,
    p_adjust_method: str,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Compute per-pair table and summary stats for a single clustering (defined by codes).
    """
    # nearest neighbor map (by centroids in rep)
    nn_map = _nearest_neighbor_by_centroid(rep, codes)

    rows = []
    naive_sets: List[set] = []
    strict_sets: List[set] = []

    uniq = np.unique(codes)
    for c in uniq:
        nnc = nn_map.get(c, None)
        if nnc is None:
            continue
        in_mask = (codes == c)
        out_mask = (codes == nnc)
        if not in_mask.any() or not out_mask.any():
            continue

        Xin = expr[in_mask, :]
        Xout = expr[out_mask, :]

        # restrict to masked genes once
        Xin_g = Xin[:, gene_mask]
        Xout_g = Xout[:, gene_mask]
        Gnames = genes[gene_mask]

        # prevalence
        n_in = Xin_g.shape[0]; n_out = Xout_g.shape[0]
        if sparse.issparse(Xin_g):
            pct_in = Xin_g.getnnz(axis=0) / float(n_in)
            pct_out = Xout_g.getnnz(axis=0) / float(n_out)
        else:
            pct_in = (Xin_g > 0).mean(axis=0)
            pct_out = (Xout_g > 0).mean(axis=0)
        pct_in = np.asarray(pct_in).ravel()
        pct_out = np.asarray(pct_out).ravel()

        # FC (log2 of linear means)
        eps = 1e-9
        if has_log1p:
            Xin_dense = _to_dense(Xin_g)
            Xout_dense = _to_dense(Xout_g)
            mu_in = np.expm1(Xin_dense).mean(axis=0) + eps
            mu_out = np.expm1(Xout_dense).mean(axis=0) + eps
            log2fc = np.log2(np.asarray(mu_in).ravel() / np.asarray(mu_out).ravel())
            logXin = Xin_dense
            logXout = Xout_dense
        else:
            if sparse.issparse(Xin_g):
                mu_in = Xin_g.mean(axis=0).A1 + eps
                mu_out = Xout_g.mean(axis=0).A1 + eps
            else:
                mu_in = np.asarray(Xin_g.mean(axis=0)).ravel() + eps
                mu_out = np.asarray(Xout_g.mean(axis=0)).ravel() + eps
            log2fc = np.log2(mu_in / mu_out)
            logXin = np.log1p(_to_dense(Xin_g))
            logXout = np.log1p(_to_dense(Xout_g))

        # Welch t-test + per-pair MTC
        _t, p_vals = ttest_ind(logXin, logXout, equal_var=False, axis=0, nan_policy="omit")
        p_vals = np.nan_to_num(p_vals, nan=1.0)
        p_adj = _p_adjust(p_vals, method=p_adjust_method)

        # criteria
        naive_mask = (p_adj <= float(naive["p_val"])) & (log2fc >= float(naive["fold_change"]))
        strict_mask = (
            (pct_in >= float(strict["minpercentin"])) &
            (pct_out <= float(strict["maxpercentout"])) &
            (p_adj <= float(strict.get("p_val", 0.01)))
        )

        naive_genes = set(Gnames[naive_mask])
        strict_genes = set(Gnames[strict_mask])
        naive_sets.append(naive_genes)
        strict_sets.append(strict_genes)

        rows.append({
            "cluster_code": int(c),
            "nn_cluster_code": int(nnc),
            "n_naive": len(naive_genes),
            "n_strict": len(strict_genes),
            "tested_genes": int(gene_mask.sum()),
        })

    # per-partition summary
    if not rows:
        pair_df = pd.DataFrame(columns=["cluster_code","nn_cluster_code","n_naive","n_strict","tested_genes"])
        summary = {
            "n_clusters": int(np.unique(codes).size),
            "n_pairs": 0,
            "tested_genes": int(gene_mask.sum()),
            "unique_naive_genes": 0,
            "unique_strict_genes": 0,
            "min_naive_per_pair": 0, "min_strict_per_pair": 0,
            "max_naive_per_pair": 0, "max_strict_per_pair": 0,
            "mean_naive_per_pair": 0.0, "mean_strict_per_pair": 0.0,
            "median_naive_per_pair": 0.0, "median_strict_per_pair": 0.0,
            "gini_naive_per_pair": 0.0, "gini_strict_per_pair": 0.0,
            "frac_pairs_with_at_least_n_naive": 0.0,
            "frac_pairs_with_at_least_n_strict": 0.0,
            "min_naive_exclusive_per_pair": 0, "min_strict_exclusive_per_pair": 0,
            "max_naive_exclusive_per_pair": 0, "max_strict_exclusive_per_pair": 0,
        }
        return summary, pair_df

    pair_df = pd.DataFrame(rows)
    naive_union = set().union(*naive_sets) if naive_sets else set()
    strict_union = set().union(*strict_sets) if strict_sets else set()
    naive_counts = pair_df["n_naive"].to_numpy()
    strict_counts = pair_df["n_strict"].to_numpy()

    def _exclusive_counts(sets: List[set]) -> np.ndarray:
        if not sets:
            return np.array([], dtype=int)
        from collections import Counter
        freq = Counter()
        for s in sets:
            freq.update(s)
        return np.array([sum(1 for g in s if freq[g] == 1) for s in sets], dtype=int)

    naive_excl = _exclusive_counts(naive_sets)
    strict_excl = _exclusive_counts(strict_sets)

    summary = {
        "n_clusters": int(np.unique(codes).size),
        "n_pairs": int(pair_df.shape[0]),
        "tested_genes": int(gene_mask.sum()),
        "unique_naive_genes": int(len(naive_union)),
        "unique_strict_genes": int(len(strict_union)),
        "min_naive_per_pair": int(naive_counts.min(initial=0)),
        "min_strict_per_pair": int(strict_counts.min(initial=0)),
        "max_naive_per_pair": int(naive_counts.max(initial=0)),
        "max_strict_per_pair": int(strict_counts.max(initial=0)),
        "mean_naive_per_pair": float(naive_counts.mean() if naive_counts.size else 0.0),
        "mean_strict_per_pair": float(strict_counts.mean() if strict_counts.size else 0.0),
        "median_naive_per_pair": float(np.median(naive_counts) if naive_counts.size else 0.0),
        "median_strict_per_pair": float(np.median(strict_counts) if strict_counts.size else 0.0),
        "gini_naive_per_pair": float(_gini(naive_counts)),
        "gini_strict_per_pair": float(_gini(strict_counts)),
        "frac_pairs_with_at_least_n_naive": float((naive_counts >= n_genes).mean()),
        "frac_pairs_with_at_least_n_strict": float((strict_counts >= n_genes).mean()),
        "min_naive_exclusive_per_pair": int(naive_excl.min(initial=0)),
        "min_strict_exclusive_per_pair": int(strict_excl.min(initial=0)),
        "max_naive_exclusive_per_pair": int(naive_excl.max(initial=0)),
        "max_strict_exclusive_per_pair": int(strict_excl.max(initial=0)),
    }
    return summary, pair_df

# ---------- public multi-run function ----------
def clustering_quality_vs_nn_summary(
    adata,
    label_cols: Sequence[str],
    n_genes: int = 5,
    naive: dict = {"p_val": 1e-2, "fold_change": 0.5},
    strict: dict = {"minpercentin": 0.20, "maxpercentout": 0.10, "p_val": 0.01},
    n_pcs_for_nn: int = 30,
    has_log1p: bool = True,
    gene_mask_col: Optional[str] = None,
    layer: Optional[str] = None,
    p_adjust_method: str = "fdr_bh",
    deduplicate_partitions: bool = True,
    return_pairs: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]]:
    """Summarize clustering quality across multiple label columns.

    Computes clustering-quality metrics for each `.obs` label column in
    ``label_cols`` and returns a single summary table (one row per labeling).
    Optionally returns per–cluster-pair differential-expression tables for each
    labeling. A single PCA/neighbor graph (using ``n_pcs_for_nn`` PCs) is
    reused across runs, and identical partitions (up to relabeling) can be
    deduplicated for speed.

    The method evaluates per-cluster marker genes under two regimes:

    * **Naive:** rank by test statistic and select the top ``n_genes`` that meet
      the naive thresholds (e.g., unadjusted ``p_val`` and minimum ``fold_change``).
    * **Strict:** apply stricter filters on expression prevalence inside vs.
      outside the cluster (``minpercentin`` / ``maxpercentout``) and an adjusted
      p-value cutoff (``p_val`` after ``p_adjust_method``), then count genes.

    Args:
        adata: AnnData object containing count/expression data. Uses
            ``adata.X`` or the specified ``layer``; cluster labels must be in
            ``adata.obs``.
        label_cols: Names of ``adata.obs`` columns whose clusterings will be
            evaluated (e.g., ``["leiden_0.2", "leiden_0.5"]``).
        n_genes: Number of top genes to consider per cluster in the naive regime
            (after applying naive thresholds). Defaults to ``5``.
        naive: Thresholds for the naive regime. Expected keys:
            - ``"p_val"`` (float): Maximum unadjusted p-value.
            - ``"fold_change"`` (float): Minimum log2 fold-change.
            Defaults to ``{"p_val": 1e-2, "fold_change": 0.5}``.
        strict: Thresholds for the strict regime. Expected keys:
            - ``"minpercentin"`` (float): Minimum fraction of cells within the
              cluster expressing the gene.
            - ``"maxpercentout"`` (float): Maximum fraction of cells outside the
              cluster expressing the gene.
            - ``"p_val"`` (float): Maximum **adjusted** p-value (per
              ``p_adjust_method``).
            Defaults to
            ``{"minpercentin": 0.20, "maxpercentout": 0.10, "p_val": 0.01}``.
        n_pcs_for_nn: Number of principal components to use when building the
            neighbor graph used for nearest-neighbor detection. Defaults to ``30``.
        has_log1p: Whether the data are already log1p-transformed. If ``False``,
            the implementation may log1p-transform counts before testing.
            Defaults to ``True``.
        gene_mask_col: Optional name of a boolean column in ``adata.var`` used to
            mask genes prior to testing (e.g., to restrict to HVGs or exclude
            mitochondrial genes). If ``None``, no mask is applied. Defaults to
            ``None``.
        layer: Name of an ``adata.layers`` matrix to use instead of ``adata.X``.
            For example, ``"log1p"`` or ``"counts"``. Defaults to ``None``.
        p_adjust_method: Method for multiple testing correction (e.g., ``"fdr_bh"``).
            Passed to the underlying p-value adjustment routine. Defaults to ``"fdr_bh"``.
        deduplicate_partitions: If ``True``, detect and skip evaluations for
            labelings that produce the same partition (up to label renaming),
            reusing the computed result. Defaults to ``True``.
        return_pairs: If ``True``, also return a dict of per–cluster-pair result
            tables keyed by the label column. Each value is a ``pd.DataFrame``
            with pairwise statistics for that labeling. Defaults to ``False``.

    Returns:
        Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]]:  

        * **summary** (``pd.DataFrame``): One row per labeling with columns such as:
          - ``label_col``: The label column name.
          - ``n_clusters``: Number of clusters in the labeling.
          - ``n_pairs``: Number of cluster pairs evaluated.
          - ``tested_genes``: Number of genes tested after masking.
          - ``unique_naive_genes`` / ``unique_strict_genes``: Count of genes
            uniquely satisfying naive/strict criteria.
          - ``frac_pairs_with_at_least_n_strict``: Fraction of cluster pairs with
            ≥ *n* strict marker genes (exact column name may reflect *n*).
          - Additional min/max/median summaries for naive/strict exclusivity per pair.
        * **pairs_by_label** (``Dict[str, pd.DataFrame]``, optional): Returned
          only when ``return_pairs=True``. For each labeling, a DataFrame of
          per–cluster-pair statistics and gene sets.

    Raises:
        KeyError: If any entry in ``label_cols`` is not found in ``adata.obs``,
            or if ``gene_mask_col`` is provided but not found in ``adata.var``.
        ValueError: If required keys are missing from ``naive`` or ``strict``,
            if ``n_genes`` < 1, or if ``p_adjust_method`` is unsupported.
        RuntimeError: If neighbor graph construction or differential testing fails.

    Notes:
        * The function does not modify ``adata`` in place (beyond any cached
          neighbor graph/PCs if your implementation chooses to store them).
        * For reproducibility, set any random seeds used by the nearest-neighbor
          or clustering components upstream.

    Examples:
        >>> summary = clustering_quality_vs_nn_summary(
        ...     adata,
        ...     label_cols=["leiden_0.2", "leiden_0.5"],
        ...     n_genes=10,
        ...     strict={"minpercentin": 0.25, "maxpercentout": 0.05, "p_val": 0.01},
        ... )
        >>> summary[["label_col", "n_clusters", "unique_strict_genes"]].head()

        >>> summary, pairs = clustering_quality_vs_nn_summary(
        ...     adata,
        ...     label_cols=["leiden_0.5"],
        ...     return_pairs=True,
        ... )
        >>> pairs["leiden_0.5"].head()
    """

    if not label_cols:
        raise ValueError("Provide at least one column in `label_cols`.")

    # expression matrix & gene mask
    expr = adata.layers[layer] if layer is not None else adata.X
    if expr.shape[0] != adata.n_obs:
        raise ValueError("Selected expression matrix has wrong shape.")
    genes = adata.var_names.to_numpy()

    if gene_mask_col is None:
        gene_mask = np.ones(adata.n_vars, dtype=bool)
    else:
        if gene_mask_col not in adata.var.columns:
            raise ValueError(f"'{gene_mask_col}' not found in adata.var")
        gene_mask = adata.var[gene_mask_col].to_numpy().astype(bool)
        if gene_mask.sum() == 0:
            raise ValueError(f"Gene mask '{gene_mask_col}' selects 0 genes.")

    # shared embedding for NN detection
    rep = _compute_representation(adata, n_pcs_for_nn, layer=layer, has_log1p=has_log1p)

    # dedupe bookkeeping
    sig_to_result: Dict[str, Tuple[Dict[str, Any], pd.DataFrame]] = {}
    sig_to_example_col: Dict[str, str] = {}
    per_run_pairs: Dict[str, pd.DataFrame] = {}
    rows = []

    for col in label_cols:
        if col not in adata.obs.columns:
            raise ValueError(f"'{col}' not found in adata.obs")
        series = adata.obs[col]

        # build canonical signature to detect identical partitions (up to relabeling)
        codes = _canonical_codes(series)
        sig = _hash_codes(codes) if deduplicate_partitions else None

        if deduplicate_partitions and (sig in sig_to_result):
            # reuse
            summary, pair_df = sig_to_result[sig]
            row = {"label_col": col, **summary}
            rows.append(row)
            if return_pairs:
                per_run_pairs[col] = pair_df.copy()
            continue

        # evaluate once
        summary, pair_df = _evaluate_one_partition(
            adata=adata,
            codes=codes,
            rep=rep,
            expr=expr,
            genes=genes,
            gene_mask=gene_mask,
            n_genes=n_genes,
            naive=naive,
            strict=strict,
            has_log1p=has_log1p,
            p_adjust_method=p_adjust_method,
        )
        row = {"label_col": col, **summary}
        rows.append(row)

        if deduplicate_partitions:
            sig_to_result[sig] = (summary, pair_df)
            sig_to_example_col[sig] = col
        if return_pairs:
            per_run_pairs[col] = pair_df

    out_df = pd.DataFrame(rows)
    # nice column order
    ordered_cols = ["label_col","n_clusters","n_pairs","tested_genes",
                    "unique_naive_genes","unique_strict_genes",
                    "min_naive_per_pair","min_strict_per_pair",
                    "max_naive_per_pair","max_strict_per_pair",
                    "mean_naive_per_pair","mean_strict_per_pair",
                    "median_naive_per_pair","median_strict_per_pair",
                    "gini_naive_per_pair","gini_strict_per_pair",
                    "frac_pairs_with_at_least_n_naive","frac_pairs_with_at_least_n_strict",
                    "min_naive_exclusive_per_pair","min_strict_exclusive_per_pair",
                    "max_naive_exclusive_per_pair","max_strict_exclusive_per_pair"]
    out_df = out_df[ordered_cols]

    return (out_df, per_run_pairs) if return_pairs else out_df


