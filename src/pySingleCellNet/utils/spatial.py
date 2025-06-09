import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy import sparse
# import string
import igraph as ig
from typing import Dict, List, Literal, Optional, Sequence, Callable

from typing import Union
from anndata import AnnData
from scipy.stats import pearsonr, spearmanr



def rotate_spatial_coordinates(adata, key, degrees):
    """
    Rotate spatial coordinates stored in adata.obsm[key] by the specified number of degrees clockwise.
    
    Parameters:
    -----------
    adata : AnnData
        Annotated data object containing spatial coordinates.
    key : str
        Key in adata.obsm where the spatial coordinates are stored (e.g., 'spatial').
        The coordinates should be an array of shape (n_cells, 2).
    degrees : int or float
        The angle in degrees to rotate the coordinates clockwise.
    
    This function modifies adata in place and does not return anything.
    """
    # Retrieve the original coordinates
    coords = adata.obsm[key]
    
    # Convert clockwise degrees to radians (positive rotation is counterclockwise, so negate)
    angle_rad = np.deg2rad(-degrees)
    
    # Build rotation matrix
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad),  np.cos(angle_rad)]
    ])
    
    # Apply rotation to each coordinate (assuming coords shape is (n_cells, 2))
    rotated = coords.dot(rotation_matrix.T)
    
    # Update the AnnData object in place
    adata.obsm[key] = rotated



def flip_spatial_coordinates(adata, key, axis='x'):
    """
    Flip spatial coordinates stored in adata.obsm[key] along the specified axis.

    Parameters:
    -----------
    adata : AnnData
        Annotated data object containing spatial coordinates.
    key : str
        Key in adata.obsm where the spatial coordinates are stored (e.g., 'spatial').
        The coordinates should be an array of shape (n_cells, 2) with columns [x, y].
    axis : str
        Axis along which to flip:
        - 'x' (default): Flip vertically (cells at the bottom move to the top). This negates the y-coordinates.
        - 'y': Flip horizontally (cells on the left move to the right). This negates the x-coordinates.

    This function modifies adata in place and does not return anything.
    """
    # Retrieve the original coordinates
    coords = adata.obsm[key]
    
    # Ensure there are two columns
    if coords.shape[1] != 2:
        raise ValueError(f"Expected coordinates with shape (n_cells, 2), got {coords.shape}")

    # Flip based on axis
    if axis == 'x':
        # Flip y-coordinate
        coords[:, 1] = -coords[:, 1]
    elif axis == 'y':
        # Flip x-coordinate
        coords[:, 0] = -coords[:, 0]
    else:
        raise ValueError("Axis must be either 'x' or 'y'")

    # Update the AnnData object in place
    adata.obsm[key] = coords


def annotate_spatially_variable_genes(
    adata: AnnData,
    uns_key: str = 'moranI',
    pval_column: str = 'pval_norm',
    var_pval_name: str = 'moranI_pval_norm',
    var_flag_name: str = 'spatially_variable',
    pval_cutoff: float = 0.01
) -> None:
    """ 
    Annotate `adata.var` with Moran’s I p-values and a Boolean “spatially_variable” flag.

    This function looks in `adata.uns[uns_key]` for a DataFrame whose index is
    gene names (matching `adata.var_names`). It then:
    
      1. Re-indexes that DataFrame to `adata.var_names`.
      2. Copies the specified p-value column into `adata.var[var_pval_name]`.
      3. Creates a boolean column `adata.var[var_flag_name]`, set to True
         wherever `pval < pval_cutoff`, False otherwise (and False if pval is NaN).

    Args:
        adata: Annotated data matrix with `.var_names` matching Moran’s I index.
        uns_key: Key in `adata.uns` where the Moran’s I DataFrame lives.
        pval_column: Name of the column in `adata.uns[uns_key]` holding normalized p-values.
        var_pval_name: Column name to use in `adata.var` for storing p-values.
        var_flag_name: Column name to use in `adata.var` for storing the Boolean flag.
        pval_cutoff: Genes with `pval < pval_cutoff` will be flagged True. Defaults to 0.05.

    Raises:
        KeyError: if `adata.uns[uns_key]` is missing or not a DataFrame, or if `pval_column` is not found.
        ValueError: if `adata.var_names` cannot be aligned with the Moran’s I table index.
    """
    # 1) Validate uns_key and pval_column
    if uns_key not in adata.uns:
        raise KeyError(f"adata.uns['{uns_key}'] not found. Expected a DataFrame with index=genes.")
    moran_df = adata.uns[uns_key]
    if not hasattr(moran_df, 'loc') or pval_column not in moran_df.columns:
        raise KeyError(
            f"Expected `adata.uns['{uns_key}']` to be a DataFrame with a column '{pval_column}'."
        )

    # 2) Re‐index p-values to match adata.var_names
    #    (this will insert NaN for any gene not present in uns index)
    moran_pvals = moran_df[pval_column].reindex(adata.var_names)

    # 3) Store the p-values in adata.var
    adata.var[var_pval_name] = moran_pvals.values

    # 4) Create the Boolean flag
    spatial_flag = (moran_pvals < pval_cutoff).astype(bool).values
    adata.var[var_flag_name] = spatial_flag


def compute_contour_profiles(
    adata: AnnData,
    mask_vars: Optional[Sequence[str]] = None,
    module_key: Optional[str] = None,
    contour_levels: Union[int, Sequence[float]] = 6,
    spatial_key: str = 'spatial',
    log_transform: bool = True,
    clip_percentiles: tuple = (1, 99),
    grid_res: int = 200,
    smooth_sigma: float = 2.0,
    profile_key: str = 'contour_profiles'
) -> None:
    """Compute contour profiles for either individual genes or gene modules.

    If `module_key` is provided, `adata.uns[module_key]` must be a dictionary
    mapping module names to sequences of gene names. In this case, for each
    module, we summarize expression across all genes in that module, then
    compute and assign contour levels per cell. The result is an (n_cells ×
    n_modules) DataFrame with each column named by module, stored in
    `adata.obsm[profile_key]`.

    If `module_key` is None, `mask_vars` must be provided as a sequence of
    gene names. Then the function computes contour levels for each gene in
    `mask_vars` individually. The result is an (n_cells × n_genes) DataFrame
    with columns named by gene, stored in `adata.obsm[profile_key]`.

    Args:
        adata: AnnData with spatial coordinates in `adata.obsm[spatial_key]`.
        mask_vars: List of gene names to process if not using modules.
            Must be a subset of `adata.var_names`. Mutually exclusive with `module_key`.
        module_key: Key in `adata.uns` pointing to a dict mapping module names
            to lists of gene names. If provided, `mask_vars` is ignored.
        contour_levels: If int N, generate N equally spaced thresholds in the
            positive range of the smoothed grid. If sequence of floats in [0,1],
            treat as fractions of [vmin, vmax]. If sequence of absolute numbers,
            use them directly.
        spatial_key: Key in `adata.obsm` for an (n_obs, 2) array of x/y coords.
        log_transform: If True, apply np.log1p to raw expression before clipping.
        clip_percentiles: Tuple `(low_pct, high_pct)` for percentile clipping each gene.
        grid_res: Resolution of the regular grid used for interpolation.
        smooth_sigma: Sigma for Gaussian filter on the gridded field.
        profile_key: Key under which the resulting DataFrame of profiles will be stored
            in `adata.obsm`.

    Raises:
        ValueError: If neither `mask_vars` nor `module_key` is provided, if any
            specified gene is missing, if spatial coords are missing, or if
            `adata.uns[module_key]` is not a dict of sequences.
    """
    # Determine modules or single genes
    if module_key is not None:
        if mask_vars is not None:
            print(f"Warning: Ignoring mask_vars because module_key='{module_key}' was provided.")
        if module_key not in adata.uns or not isinstance(adata.uns[module_key], dict):
            raise ValueError(f"adata.uns['{module_key}'] must be a dict mapping module names to gene lists.")
        modules: Dict[str, Sequence[str]] = adata.uns[module_key]
        # Validate that modules contain genes
        for mod_name, genes in modules.items():
            if not isinstance(genes, (list, tuple)) or any(g not in adata.var_names for g in genes):
                raise ValueError(f"Module '{mod_name}' in adata.uns['{module_key}'] "
                                 f"must be a list of valid gene names.")
        items = list(modules.items())  # (module_name, gene_list)
        col_names = [name for name, _ in items]
    else:
        if mask_vars is None:
            raise ValueError("Either `mask_vars` or `module_key` must be provided.")
        # Validate genes
        for g in mask_vars:
            if g not in adata.var_names:
                raise ValueError(f"Gene '{g}' not found in adata.var_names.")
        # Treat each gene as its own "module"
        items = [(g, [g]) for g in mask_vars]
        col_names = mask_vars

    # Fetch spatial coordinates
    coords = adata.obsm.get(spatial_key)
    if coords is None or coords.shape[1] < 2:
        raise ValueError(f"adata.obsm['{spatial_key}'] must be an (n_obs, 2) array.")
    x, y = coords[:, 0], coords[:, 1]
    n_cells = x.shape[0]

    # Prepare array to hold contour-level values
    n_items = len(items)
    profiles = np.zeros((n_cells, n_items), dtype=float)

    # Helper to extract a numpy array from AnnData X
    def _get_array(x):
        return x.toarray().flatten() if hasattr(x, 'toarray') else x.flatten()

    # Loop over each gene or module
    for idx, (name, gene_list) in enumerate(items):
        # 1) Extract and combine raw expression for gene_list
        #    If single gene, gene_list=[g], else list of genes
        M_list = []
        for g in gene_list:
            raw = adata[:, g].X
            vals = _get_array(raw)
            if log_transform:
                vals = np.log1p(vals)
            lo, hi = np.percentile(vals, clip_percentiles)
            clipped = np.clip(vals, lo, hi)
            normed = (clipped - lo) / (hi - lo) if hi > lo else np.zeros_like(clipped)
            M_list.append(normed)
        M = np.column_stack(M_list)  # shape: (n_cells, len(gene_list))

        # 2) Summarize across genes in this module
        summary = np.mean(M, axis=1)  # always use mean for module summarization

        # 3) Build grid and interpolate with fill_value=0
        xi = np.linspace(x.min(), x.max(), grid_res)
        yi = np.linspace(y.min(), y.max(), grid_res)
        Xi, Yi = np.meshgrid(xi, yi)
        Zi = griddata((x, y), summary, (Xi, Yi), method='cubic', fill_value=0.0)

        # 4) Smooth
        Zi_s = gaussian_filter(Zi, sigma=smooth_sigma, mode='nearest')
        vmin_raw, vmax_raw = np.nanmin(Zi_s), np.nanmax(Zi_s)
        vmin = max(0.0, vmin_raw)
        vmax = vmax_raw

        # 5) Determine thresholds
        if vmax <= vmin:
            thresholds = np.array([0.0])
        else:
            if isinstance(contour_levels, int):
                N = contour_levels
                thresholds = np.linspace(vmin, vmax, N + 2)[1:-1]
            else:
                lv = np.array(contour_levels, dtype=float)
                if np.all((0 <= lv) & (lv <= 1)):
                    thresholds = vmin + (vmax - vmin) * lv
                else:
                    thresholds = lv.copy()

        # 6) Sample smoothed grid at each cell (nearest)
        pts_grid = np.vstack((Xi.flatten(), Yi.flatten())).T
        vals_grid = Zi_s.flatten()
        z_cells = griddata(pts_grid, vals_grid, (x, y), method='nearest')

        # 7) Assign highest threshold ≤ z_cells
        annotation = np.zeros(n_cells, dtype=float)
        for t in sorted(thresholds):
            mask = z_cells >= t
            annotation[mask] = t

        profiles[:, idx] = annotation

    # Construct DataFrame and store in adata.obsm
    df = pd.DataFrame(profiles, index=adata.obs_names, columns=col_names)
    adata.obsm[profile_key] = df




def annotate_axis_association(
    adata: AnnData,
    spatial_key: str,
    axis: Literal['x', 'y'],
    mask_vars: Optional[str] = None,
    partition_axis: Optional[Literal['x', 'y']] = None,
    partition_bins: Union[int, Sequence[float]] = 2,
    method: Literal['pearson', 'spearman'] = 'pearson'
) -> None:
    """Identify genes whose expression is associated with a spatial axis,
    optionally limited to a Boolean mask in `.var` and/or partitioned on the other axis.

    If `mask_vars` is provided, it must be the name of a Boolean column in `adata.var`.
    Only genes for which `adata.var[mask_vars] == True` are tested. Otherwise, all genes
    in `adata.var_names` are tested.

    Optionally, if `partition_axis` is set to the opposite axis, cells are binned along
    that axis (using `partition_bins`), and correlation is computed separately in each bin.
    The best (lowest p-value) bin’s result is recorded as the gene’s statistic.

    Results are stored in `adata.var` (one row per gene):
      - `{axis}_corr`: correlation coefficient (overall or best partition)
      - `{axis}_pval`: raw p-value (overall or best partition)
      - `{axis}_adjp`: FDR-adjusted p-value
      - If partitioning:
        - `{axis}_corr_part`: correlation in best bin
        - `{axis}_pval_part`: p-value in best bin
        - `{axis}_part`: index of best bin (0-based)

    Args:
        adata: AnnData with expression data and spatial coords in `.obsm[spatial_key]`.
        spatial_key: Key in `adata.obsm` for an (n_obs, 2) array of coordinates.
        axis: Which axis to test against: 'x'  (column 0) or 'y' (column 1).
        mask_vars: Optional name of a Boolean column in `adata.var`; tests only genes
            where `adata.var[mask_vars] == True`. If None, all genes are tested.
        partition_axis: If provided, must be 'x' or 'y' and different from `axis`. Cells
            will be binned along this axis, and correlation computed within each bin.
        partition_bins: If int, number of equal-width bins; if sequence of floats, those bin edges.
        method: Correlation method: 'pearson' or 'spearman'.

    Raises:
        ValueError: If `spatial_key` is missing or malformed, if `axis` is not 'x'/'y',
                    if `mask_vars` is not a Boolean column in `.var`, if `partition_axis`
                    is invalid, or if `partition_bins` is invalid.
    """
    # 1) Validate spatial_key
    if spatial_key not in adata.obsm:
        raise ValueError(f"adata.obsm['{spatial_key}'] not found.")
    coords = adata.obsm[spatial_key]
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"adata.obsm['{spatial_key}'] must be shape (n_obs, 2).")

    # 2) Choose test axis values
    if axis == 'x':
        test_vals = coords[:, 0]
    elif axis == 'y':
        test_vals = coords[:, 1]
    else:
        raise ValueError("`axis` must be 'x' or 'y'.")

    # 3) Determine genes to test via mask_vars
    if mask_vars is None:
        genes = np.array(adata.var_names)
    else:
        if mask_vars not in adata.var.columns:
            raise ValueError(f"'{mask_vars}' not found in adata.var.")
        col = adata.var[mask_vars]
        if col.dtype != bool and not np.issubdtype(col.dtype, np.bool_):
            raise ValueError(f"adata.var['{mask_vars}'] must be Boolean.")
        genes = np.array(adata.var_names)[col.values]  # only True rows

    if genes.size == 0:
        raise ValueError("No genes selected by `mask_vars`.")

    # 4) Prepare partitioning if requested
    do_partition = partition_axis is not None
    if do_partition:
        if partition_axis not in ('x', 'y'):
            raise ValueError("`partition_axis` must be 'x' or 'y'.")
        if partition_axis == axis:
            raise ValueError("`partition_axis` must differ from `axis`.")
        # Extract partition-axis values
        if partition_axis == 'x':
            part_vals = coords[:, 0]
        else:
            part_vals = coords[:, 1]

        # Determine bin edges
        if isinstance(partition_bins, int):
            if partition_bins < 1:
                raise ValueError("`partition_bins` as int must be >=1.")
            edges = np.linspace(part_vals.min(), part_vals.max(), partition_bins + 1)
        else:
            edges = np.array(partition_bins, dtype=float)
            if edges.ndim != 1 or edges.size < 2:
                raise ValueError("`partition_bins` must be int or 1D sequence length>=2.")
        # Assign each cell to a bin index [0 .. n_bins-1]
        bin_idx = np.digitize(part_vals, edges) - 1
        bin_idx = np.clip(bin_idx, 0, edges.size - 2)
        n_bins = edges.size - 1

    # 5) Extract expression matrix (convert sparse if needed)
    expr_mat = adata.X.toarray() if hasattr(adata.X, 'toarray') else np.asarray(adata.X)

    n_total = adata.n_vars
    n_test = genes.size

    # Preallocate full-length arrays for var columns
    full_corr = np.zeros(n_total, dtype=float)
    full_pval = np.ones(n_total, dtype=float)
    full_adjp = np.ones(n_total, dtype=float)

    if do_partition:
        full_corr_part = np.zeros(n_total, dtype=float)
        full_pval_part = np.ones(n_total, dtype=float)
        full_part = np.zeros(n_total, dtype=int)

    # Temporary arrays for masked genes
    corrs = np.zeros(n_test, dtype=float)
    pvals = np.ones(n_test, dtype=float)
    if do_partition:
        best_corr = np.zeros(n_test, dtype=float)
        best_pval = np.ones(n_test, dtype=float)
        best_bin = np.zeros(n_test, dtype=int)

    # 6) Loop through selected genes
    for i, g in enumerate(genes):
        vidx = adata.var_names.get_loc(g)
        gene_expr = expr_mat[:, vidx]

        if do_partition:
            part_corrs = np.zeros(n_bins, dtype=float)
            part_pvals = np.ones(n_bins, dtype=float)
            # Compute per-bin correlations
            for b in range(n_bins):
                idxs = np.where(bin_idx == b)[0]
                if idxs.size < 3:
                    part_corrs[b] = 0.0
                    part_pvals[b] = 1.0
                    continue
                se = gene_expr[idxs]
                tv = test_vals[idxs]
                if np.all(se == se[0]):
                    part_corrs[b] = 0.0
                    part_pvals[b] = 1.0
                else:
                    if method == 'pearson':
                        c, p = pearsonr(se, tv)
                    else:
                        c, p = spearmanr(se, tv)
                    part_corrs[b] = c
                    part_pvals[b] = p
            # Choose best bin by minimal p-value
            bidx = np.nanargmin(part_pvals)
            best_corr[i] = part_corrs[bidx]
            best_pval[i] = part_pvals[bidx]
            best_bin[i] = bidx
            # Record for full-length
            corrs[i] = part_corrs[bidx]
            pvals[i] = part_pvals[bidx]
        else:
            # No partition: test on all cells
            if np.all(gene_expr == gene_expr[0]):
                corrs[i] = 0.0
                pvals[i] = 1.0
            else:
                if method == 'pearson':
                    c, p = pearsonr(gene_expr, test_vals)
                else:
                    c, p = spearmanr(gene_expr, test_vals)
                corrs[i] = c
                pvals[i] = p

    # 7) FDR correction (Benjamini–Hochberg) on masked p-values
    order = np.argsort(pvals)
    ranked_p = pvals[order]
    m = float(n_test)
    bh_raw = ranked_p * m / (np.arange(n_test) + 1)
    bh_adj = np.minimum.accumulate(bh_raw[::-1])[::-1]
    bh_adj = np.clip(bh_adj, 0, 1.0)
    adjp_masked = np.empty(n_test, dtype=float)
    adjp_masked[order] = bh_adj

    # 8) Populate full-length arrays in adata.var
    for i, g in enumerate(genes):
        vidx = adata.var_names.get_loc(g)
        full_corr[vidx] = corrs[i]
        full_pval[vidx] = pvals[i]
        full_adjp[vidx] = adjp_masked[i]
        if do_partition:
            full_corr_part[vidx] = best_corr[i]
            full_pval_part[vidx] = best_pval[i]
            full_part[vidx] = best_bin[i]

    # 9) Store in adata.var
    adata.var[f'{axis}_corr'] = full_corr
    adata.var[f'{axis}_pval'] = full_pval
    adata.var[f'{axis}_adjp'] = full_adjp
    if do_partition:
        adata.var[f'{axis}_corr_part'] = full_corr_part
        adata.var[f'{axis}_pval_part'] = full_pval_part
        adata.var[f'{axis}_part'] = full_part














