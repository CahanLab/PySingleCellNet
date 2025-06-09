import numpy as np
import matplotlib.pyplot as plt
from anndata import AnnData
from .helpers import _smooth_contour
from matplotlib.colors import to_hex, ListedColormap
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from typing import Union, Sequence, Callable, Optional, Dict

import math

def scatter_genes_oneper(
    adata: AnnData,
    genes: Sequence[str],
    embedding_key: str = "X_spatial",
    spot_size: float = 2,
    alpha: float = 0.9,
    clip_percentiles: tuple = (0, 99.5),
    log_transform: bool = True,
    cmap: Union[str, plt.Colormap] = 'Reds',
    figsize: Optional[tuple] = None,
    panel_width: float = 4.0,
    n_rows: int = 1
) -> None:
    """Plot expression of multiple genes on a 2D embedding arranged in a grid.

    Each gene is optionally log-transformed, percentile-clipped, and rescaled to [0,1].
    Cells are plotted on the embedding, colored by expression, with highest values
    drawn on top. A single colorbar is placed to the right of the grid.
    If `figsize` is None, each panel has width `panel_width` and height
    proportional to the embedding's aspect ratio; total figure dims reflect
    `n_rows` and computed columns.

    Args:
        adata: AnnData containing the embedding in `adata.obsm[embedding_key]`.
        embedding_key: Key in `.obsm` for an (n_obs, 2) coordinate array.
        genes: List of gene names to plot (must be in `adata.var_names`).
        spot_size: Marker size for scatter plots. Default 2.
        alpha: Transparency for markers. Default 0.9.
        clip_percentiles: (low_pct, high_pct) to clip expression before rescaling.
        log_transform: If True, apply `np.log1p` to raw expression.
        cmap: Colormap or name for all plots.
        figsize: (width, height) of entire figure. If None, computed from
            `panel_width`, `n_rows`, and embedding aspect ratio.
        panel_width: Width (in inches) of each panel when `figsize` is None.
        n_rows: Number of rows in the grid. Default 1.

    Raises:
        ValueError: If embedding is missing/malformed or genes not found.
    """
    # Helper to extract array
    def _get_array(x):
        return x.toarray().flatten() if hasattr(x, 'toarray') else x.flatten()

    coords = adata.obsm.get(embedding_key)
    if coords is None or coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"adata.obsm['{embedding_key}'] must be an (n_obs, 2) array.")
    x_vals, y_vals = coords[:, 0], coords[:, 1]

    n_genes = len(genes)
    cols = math.ceil(n_genes / n_rows)
    # Compute figsize if not provided
    if figsize is None:
        x_range = x_vals.max() - x_vals.min()
        y_range = y_vals.max() - y_vals.min()
        aspect = x_range / y_range if y_range > 0 else 1.0
        panel_height = panel_width / aspect
        fig_width = panel_width * cols
        fig_height = panel_height * n_rows
    else:
        fig_width, fig_height = figsize

    fig, axes = plt.subplots(n_rows, cols, figsize=(fig_width, fig_height), squeeze=False)
    axes_flat = axes.flatten()

    scatters = []
    for idx, gene in enumerate(genes):
        ax = axes_flat[idx]
        if gene not in adata.var_names:
            raise ValueError(f"Gene '{gene}' not found in adata.var_names.")
        vals = _get_array(adata[:, gene].X)
        if log_transform:
            vals = np.log1p(vals)
        lo, hi = np.percentile(vals, clip_percentiles)
        clipped = np.clip(vals, lo, hi)
        norm = (clipped - lo) / (hi - lo) if hi > lo else np.zeros_like(clipped)

        order = np.argsort(norm)
        sc = ax.scatter(
            x_vals[order],
            y_vals[order],
            c=norm[order],
            cmap=cmap,
            s=spot_size,
            alpha=alpha,
            vmin=0, vmax=1
        )
        ax.set_title(gene)
        ax.set_xticks([]); ax.set_yticks([])
        scatters.append(sc)

    # Turn off unused axes
    for j in range(len(genes), n_rows*cols):
        axes_flat[j].axis('off')

    # Adjust subplots to make room for colorbar
    fig.subplots_adjust(right=0.85)

    # Colorbar axis on the right, spanning full height (15% margin)
    cbar_ax = fig.add_axes([0.88, 0.05, 0.02, 0.9])
    cb = fig.colorbar(scatters[0], cax=cbar_ax)
    cb.set_label('normalized expression')

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()


def spatial_two_genes(
    adata: AnnData,
    gene1: str,
    gene2: str,
    cmap: ListedColormap,
    spot_size: float = 2,
    alpha: float = 0.9,
    spatial_key: str = 'X_spatial',
    log_transform: bool = True,
    clip_percentiles: tuple = (0, 99.5),
    priority_metric: str = 'sum'
) -> None:
    """Plot two‐gene spatial expression with a bivariate colormap.

    Scales, optionally log‐transforms and percentile‐clips each gene,
    normalizes them to [0,1], then uses bilinear interpolation onto
    a bivariate colormap LUT. Cells with high expression are on top (if overlapping).

    Args:
        adata: AnnData with spatial coords in `adata.obsm[spatial_key]`.
        gene1: First gene name (must be in `adata.var_names`).
        gene2: Second gene name.
        cmap: Bivariate colormap from `make_bivariate_cmap` (n×n LUT).
        spot_size: Scatter point size.
        alpha: Point alpha transparency.
        spatial_key: Key in `adata.obsm` for an (n_obs, 2) coords array.
        log_transform: If True, apply `np.log1p` to raw expression.
        clip_percentiles: Tuple `(low_pct, high_pct)` to clip each gene.
        priority_metric: Which metric to sort drawing order by:
            - 'sum': u + v (default)
            - 'gene1': u only
            - 'gene2': v only

    Raises:
        ValueError: If spatial coords are missing/malformed or
                    if `priority_metric` is invalid.
    """
    # 1) extract raw arrays
    def _get_array(x):
        return x.toarray().flatten() if hasattr(x, 'toarray') else x.flatten()
    X1 = _get_array(adata[:, gene1].X)
    X2 = _get_array(adata[:, gene2].X)

    # 2) optional log1p
    if log_transform:
        X1 = np.log1p(X1)
        X2 = np.log1p(X2)

    # 3) percentile‐clip
    lo1, hi1 = np.percentile(X1, clip_percentiles)
    lo2, hi2 = np.percentile(X2, clip_percentiles)
    X1 = np.clip(X1, lo1, hi1)
    X2 = np.clip(X2, lo2, hi2)

    # 4) normalize to [0,1]
    u = (X1 - lo1) / (hi1 - lo1) if hi1 > lo1 else np.zeros_like(X1)
    v = (X2 - lo2) / (hi2 - lo2) if hi2 > lo2 else np.zeros_like(X2)

    # 5) prepare LUT
    m = len(cmap.colors)
    n = int(np.sqrt(m))
    C = np.array(cmap.colors).reshape(n, n, 3)

    # 6) bilinear interpolate per‐cell
    gu = u * (n - 1); gv = v * (n - 1)
    i0 = np.floor(gu).astype(int); j0 = np.floor(gv).astype(int)
    i1 = np.minimum(i0 + 1, n - 1); j1 = np.minimum(j0 + 1, n - 1)
    du = gu - i0; dv = gv - j0

    wa = (1 - du) * (1 - dv)
    wb = du * (1 - dv)
    wc = (1 - du) * dv
    wd = du * dv

    c00 = C[j0, i0]; c10 = C[j0, i1]
    c01 = C[j1, i0]; c11 = C[j1, i1]

    cols_rgb = (
        c00 * wa[:, None] +
        c10 * wb[:, None] +
        c01 * wc[:, None] +
        c11 * wd[:, None]
    )
    hex_colors = [to_hex(c) for c in cols_rgb]

    # 7) determine draw order
    if priority_metric == 'sum':
        priority = u + v
    elif priority_metric == 'gene1':
        priority = u
    elif priority_metric == 'gene2':
        priority = v
    else:
        raise ValueError("priority_metric must be 'sum', 'gene1', or 'gene2'")
    order = np.argsort(priority)

    # 8) fetch and sort coords/colors
    coords = adata.obsm.get(spatial_key)
    if coords is None or coords.shape[1] < 2:
        raise ValueError(f"adata.obsm['{spatial_key}'] must be an (n_obs, 2) array")
    coords_sorted = coords[order]
    colors_sorted = [hex_colors[i] for i in order]

    # 9) plot scatter + legend grid
    fig, (ax_sc, ax_cb) = plt.subplots(
        1, 2, figsize=(8, 4),
        gridspec_kw={'width_ratios': [3, 1], 'wspace': 0.3}
    )
    ax_sc.scatter(
        coords_sorted[:, 0],
        coords_sorted[:, 1],
        c=colors_sorted,
        s=spot_size,
        alpha=alpha
    )
    ax_sc.set_aspect('equal')
    ax_sc.set_title(f"{gene1} vs {gene2}")

    lut_img = C  # shape (n,n,3)
    ax_cb.imshow(lut_img, origin='lower', extent=[0, 1, 0, 1])
    ax_cb.set_xlabel(f"{gene1}\nlow → high")
    ax_cb.set_ylabel(f"{gene2}\nlow → high")
    ax_cb.set_xticks([0, 1]); ax_cb.set_yticks([0, 1])
    ax_cb.set_aspect('equal')

    plt.show()


def spatial_contours(
    adata: AnnData,
    genes: Union[str, Sequence[str]],
    spatial_key: str = 'spatial',
    summary_func: Callable[[np.ndarray], np.ndarray] = np.mean,
    spot_size: float = 30,
    alpha: float = 0.8,
    log_transform: bool = True,
    clip_percentiles: tuple = (1, 99),
    cmap: str = 'viridis',
    contour_kwargs: dict = None,
    scatter_kwargs: dict = None
) -> None:
    """Scatter spatial expression of one or more genes with smooth contour overlay.

    If multiple genes are provided, each is preprocessed (log1p → clip
    → normalize), then combined per cell via `summary_func` (e.g. mean, sum,
    max) on the normalized values. A smooth contour of the summarized signal
    is overlaid onto the spatial scatter.

    Args:
        adata: AnnData with spatial coordinates in `adata.obsm[spatial_key]`.
        genes: Single gene name or list of gene names to plot (must be in `adata.var_names`).
        spatial_key: Key in `.obsm` for an (n_obs, 2) coords array.
        summary_func: Function to combine multiple normalized gene arrays
            (takes an (n_obs, n_genes) array, returns length-n_obs array).
            Defaults to `np.mean`.
        spot_size: Scatter marker size.
        alpha: Scatter alpha transparency.
        log_transform: If True, apply `np.log1p` to raw expression before clipping.
        clip_percentiles: Tuple `(low_pct, high_pct)` percentiles to clip each gene.
        cmap: Colormap name for the scatter (e.g. 'viridis').
        contour_kwargs: Dict of parameters for smoothing & contouring:
            - levels: int or list of levels (default 6)
            - grid_res: int grid resolution (default 200)
            - smooth_sigma: float Gaussian blur sigma (default 2)
            - contour_kwargs: dict of line style kwargs (default {'colors':'k','linewidths':1})
        scatter_kwargs: Extra kwargs passed to `ax.scatter`.

    Raises:
        ValueError: If any gene is missing or spatial coords are malformed.
    """
    # ensure genes is list
    gene_list = [genes] if isinstance(genes, str) else list(genes)
    for g in gene_list:
        if g not in adata.var_names:
            raise ValueError(f"Gene '{g}' not found in adata.var_names.")

    # helper to extract numpy
    def _get_array(x):
        return x.toarray().flatten() if hasattr(x, 'toarray') else x.flatten()

    # preprocess each gene: extract, log1p, clip, normalize to [0,1]
    normed = []
    for g in gene_list:
        vals = _get_array(adata[:, g].X)
        if log_transform:
            vals = np.log1p(vals)
        lo, hi = np.percentile(vals, clip_percentiles)
        vals = np.clip(vals, lo, hi)
        normed.append((vals - lo) / (hi - lo) if hi > lo else np.zeros_like(vals))
    # stack into (n_obs, n_genes)
    M = np.column_stack(normed)
    # summarize across genes
    summary = summary_func(M, axis=1)

    # fetch spatial coords
    coords = adata.obsm.get(spatial_key)
    if coords is None or coords.shape[1] < 2:
        raise ValueError(f"adata.obsm['{spatial_key}'] must be an (n_obs, 2) array.")
    x, y = coords[:, 0], coords[:, 1]

    # scatter
    fig, ax = plt.subplots(figsize=(6, 6))
    sc_kw = {"c": summary, "cmap": cmap, "s": spot_size, "alpha": alpha}
    if scatter_kwargs:
        sc_kw.update(scatter_kwargs)
    sc = ax.scatter(x, y, **sc_kw)
    ax.set_aspect('equal')
    title = (
        gene_list[0] if len(gene_list) == 1
        else f"{len(gene_list)} genes ({summary_func.__name__})"
    )
    ax.set_title(f"Spatial expression: {title}")
    ax.set_xlabel('x'); ax.set_ylabel('y')
    fig.colorbar(sc, ax=ax, label="summarized (normalized)")

    # smooth + contour
    # default contour params
    ck = {
        "levels": 6,
        "grid_res": 200,
        "smooth_sigma": 2,
        "contour_kwargs": {"colors": "k", "linewidths": 1}
    }
    if contour_kwargs:
        ck.update(contour_kwargs)
    _smooth_contour(
        x, y, summary,
        levels=ck["levels"],
        grid_res=ck["grid_res"],
        smooth_sigma=ck["smooth_sigma"],
        contour_kwargs=ck["contour_kwargs"]
    )
    plt.tight_layout()
    plt.show()
