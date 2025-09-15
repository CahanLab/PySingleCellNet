
from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt

def plot_module_heatmap(
    adata,
    *,
    uns_key: str = "knn_modules",
    modules: Optional[Dict[str, List[str]]] = None,   # overrides uns_key if provided
    cell_type_col: str = "cell_type",
    layer: Optional[str] = None,                      # None => adata.X
    top_genes: Optional[Dict[str, List[str]]] = None, # output of subset_modules_top_genes
    top_n_per_module: Optional[int] = 20,             # used only if top_genes is None
    zscore_genes: bool = True,
    max_cells_per_type: Optional[int] = 250,
    order_modules_by_similarity: bool = True,
    module_similarity_on: str = "celltype_means",     # 'celltype_means' or 'cells'
    cmap: str = "vlag",
    use_marsilea: str = "auto",                       # 'auto' | 'yes' | 'no'
    width_per_gene: float = 0.12,                     # Marsilea width in inches per gene
    height_per_cell: float = 0.012,                   # Marsilea height in inches per cell
    show_gene_labels: bool = False,                   # use Marsilea Labels (heavy for many genes)
    show: bool = True
):
    # --- fetch modules ---
    if modules is None:
        if uns_key not in adata.uns or not isinstance(adata.uns[uns_key], dict):
            raise ValueError(f"Expected modules dict in adata.uns['{uns_key}'].")
        modules = adata.uns[uns_key]

    if top_genes is None and top_n_per_module is not None:
        top_genes = {m: genes[:top_n_per_module] for m, genes in modules.items()}
    elif top_genes is None:
        top_genes = modules
    
    var_index = pd.Index(adata.var_names)
    top_genes = {m: [g for g in genes if g in var_index] for m, genes in top_genes.items()}
    top_genes = {m: v for m, v in top_genes.items() if len(v) > 0}
    if not top_genes:
        raise ValueError("No module genes found in adata.var_names after filtering.")
    
    # --- group rows by cell type (with optional downsampling) ---
    if cell_type_col not in adata.obs.columns:
        raise ValueError(f"'{cell_type_col}' not found in adata.obs")
    ct = adata.obs[cell_type_col].astype("category")
    cell_types = list(ct.cat.categories)
    
    rng = np.random.default_rng(0)
    per_ct_indices = []
    for c in cell_types:
        idx = np.where(ct.to_numpy() == c)[0]
        if (max_cells_per_type is not None) and (len(idx) > max_cells_per_type):
            idx = rng.choice(idx, size=max_cells_per_type, replace=False)
        per_ct_indices.append(idx)
    row_index = np.concatenate(per_ct_indices) if per_ct_indices else np.arange(adata.n_obs)
    
    # --- assemble expression matrix ---
    X = adata.layers[layer] if layer is not None else adata.X
    module_names = list(top_genes.keys())
    gene_blocks = [top_genes[m] for m in module_names]
    all_genes = [g for block in gene_blocks for g in block]
    col_index = var_index.get_indexer(all_genes)
    
    if sparse.issparse(X):
        M = X[row_index, :][:, col_index].toarray()
    else:
        M = np.asarray(X)[np.ix_(row_index, col_index)]
    
    if zscore_genes:
        mu = M.mean(axis=0, keepdims=True)
        sd = M.std(axis=0, ddof=0, keepdims=True) + 1e-8
        M = (M - mu) / sd
    
    # --- robust module ordering by centroid similarity ---
    def _robust_module_order(M: np.ndarray, gene_blocks: List[List[str]], ct_codes: np.ndarray):
        if len(gene_blocks) <= 1:
            return np.arange(len(gene_blocks), dtype=int)

        sizes = [len(b) for b in gene_blocks]
        starts = np.cumsum([0] + sizes[:-1])
        ends = np.cumsum(sizes)

        # build module centroids
        if module_similarity_on == "celltype_means":
            ct_cats = np.unique(ct_codes)
            C = []
            col = 0
            for s, e in zip(starts, ends):
                mod_profile = M[:, s:e].mean(axis=1)
                prof_by_ct = [mod_profile[ct_codes == cat].mean() if np.any(ct_codes == cat) else 0.0
                              for cat in ct_cats]
                C.append(prof_by_ct)
            C = np.array(C, dtype=float)
        else:
            C = np.stack([M[:, s:e].mean(axis=1) for s, e in zip(starts, ends)], axis=0)

        # center rows; try cosine (corr) then fallback to euclidean on z-scored rows
        C = C - C.mean(axis=1, keepdims=True)
        try:
            D = pdist(C, metric="cosine")
            if not np.isfinite(D).all():
                raise FloatingPointError
        except Exception:
            sd = C.std(axis=1, keepdims=True) + 1e-12
            Cz = C / sd
            D = pdist(Cz, metric="euclidean")
            if not np.isfinite(D).all():
                return np.arange(len(gene_blocks), dtype=int)
        return leaves_list(linkage(D, method="average"))
    
    if order_modules_by_similarity:
        ct_codes = ct.to_numpy()[row_index]
        order_idx = _robust_module_order(M, gene_blocks, ct_codes)
        module_names = [module_names[i] for i in order_idx]
        gene_blocks = [gene_blocks[i] for i in order_idx]
        new_cols = [g for block in gene_blocks for g in block]
        pos = {g: i for i, g in enumerate(all_genes)}
        M = M[:, [pos[g] for g in new_cols]]
        all_genes = new_cols
    
    # --- color palettes for sidebars ---
    import matplotlib.colors as mcolors
    tab20 = list(mcolors.TABLEAU_COLORS.values())
    mod_palette = {m: tab20[i % len(tab20)] for i, m in enumerate(module_names)}
    ct_palette = {c: tab20[i % len(tab20)] for i, c in enumerate(cell_types)}
    
    # prepare category arrays (1xN for columns; Nx1 for rows)
    gene_to_module = []
    for m, block in zip(module_names, gene_blocks):
        gene_to_module.extend([m] * len(block))
    col_cats = np.array(gene_to_module)[None, :]        # shape (1, n_genes)
    row_cats = ct.to_numpy()[row_index][:, None]        # shape (n_rows, 1)
    
    # --- Marsilea branch (fixed) ---
    use_flag = use_marsilea
    if use_marsilea == "auto":
        try:
            import marsilea as ma
            import marsilea.plotter as mp
            use_flag = "yes"
        except Exception:
            use_flag = "no"
    
    df = pd.DataFrame(M, index=adata.obs_names[row_index], columns=all_genes)
    
    if use_flag == "yes":
        import marsilea as ma
        import marsilea.plotter as mp

        # 1) Create the heatmap canvas with explicit size
        w = max(6.0, width_per_gene * df.shape[1])
        h = max(4.0, height_per_cell * df.shape[0])
        hmap = ma.Heatmap(df, cmap=cmap, width=w, height=h)  # Heatmap canvas; call .render() later. :contentReference[oaicite:1]{index=1}

        # ---- BUILD 1-D CATEGORY ARRAYS (IMPORTANT) ----
        # columns: length == n_cols
        col_labels = np.array([m for m, block in zip(module_names, gene_blocks) for _ in block], dtype=object)
        # rows: length == n_rows
        row_labels = adata.obs[cell_type_col].to_numpy()[row_index]

        # sanity checks (clearer error than an AssertionError deep in Marsilea)
        n_rows, n_cols = df.shape
        if row_labels.ndim != 1 or row_labels.shape[0] != n_rows:
            raise ValueError(f"Row labels must be 1-D with length {n_rows}; got shape {row_labels.shape}")
        if col_labels.ndim != 1 or col_labels.shape[0] != n_cols:
            raise ValueError(f"Column labels must be 1-D with length {n_cols}; got shape {col_labels.shape}")

        # 2) Add top module color bar and left cell-type color bar (1-D arrays)
        top_colors  = mp.Colors(col_labels, palette=mod_palette, label="Module")         # 1-D for columns :contentReference[oaicite:2]{index=2}
        left_colors = mp.Colors(row_labels, palette=ct_palette, label=cell_type_col)     # 1-D for rows   :contentReference[oaicite:3]{index=3}
        hmap.add_top(top_colors, size=0.15, pad=0.02)
        hmap.add_left(left_colors, size=0.20, pad=0.02)

        # 3) (Optional) gene labels
        if show_gene_labels:
            hmap.add_bottom(mp.Labels(df.columns.to_list(), rotation=90), size=0.40, pad=0.02)

        # 4) Legends & render
        # You can add global legends after adding plotters; Marsilea supports legend stacking. :contentReference[oaicite:4]{index=4}
        # hmap.add_legends(side="right", pad=0.1)  # optional
        if show:
            hmap.render()
        return hmap
    
    # --- seaborn fallback (kept unchanged) ---
    import seaborn as sns
    module_colors = pd.Series([mod_palette[m] for m in gene_to_module], index=all_genes, name="Module")
    row_colors = pd.Series([ct_palette[c] for c in ct.to_numpy()[row_index]],
                           index=adata.obs_names[row_index], name=cell_type_col)
    
    g = sns.clustermap(
        df, row_cluster=False, col_cluster=False, cmap=cmap,
        xticklabels=True, yticklabels=False,
        figsize=(max(6, 0.09 * df.shape[1]), max(4, 0.012 * df.shape[0])),
        col_colors=[module_colors], row_colors=[row_colors],
        dendrogram_ratio=(0.05, 0.05), cbar_pos=(0.02, 0.8, 0.02, 0.15)
    )
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=90, fontsize=8)
    # module separators
    x = 0
    for block in gene_blocks:
        x += len(block)
        g.ax_heatmap.axvline(x, color="black", linewidth=0.5)
    if show:
        plt.show()
    return g

