import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import anndata as ad
from palettable.colorbrewer.qualitative import Set2_6
from palettable.tableau import GreenOrange_6
from palettable.cartocolors.qualitative import Safe_6
from palettable.cartocolors.qualitative import Vivid_4
from palettable.cartocolors.qualitative import Vivid_6
from palettable.cartocolors.qualitative import Vivid_10
from palettable.scientific.diverging import Roma_20
from palettable.scientific.sequential import LaJolla_20
from palettable.scientific.sequential import Batlow_20
from anndata import AnnData
from scipy.sparse import csr_matrix
from ..utils import *
import marsilea as ml
import marsilea.plotter as mp
from sklearn.metrics import classification_report
from matplotlib.colors import LinearSegmentedColormap
import re

def heatmap_clustering_eval(
    df: pd.DataFrame,
    index_col: str = "label_col",
    metrics=("n_clusters", "unique_strict_genes", "unique_naive_genes", "frac_pairs_with_at_least_n_strict"),
    bar_sum_cols=("unique_strict_genes", "unique_naive_genes"),
    cmap_eval: str = "viridis",
    scale_eval: str = "zscore",        # 'zscore' | 'minmax' | 'none' (per column)
    linewidth: float = 0.5,
    value_fmt: dict | None = None,     # e.g., {"frac_pairs_with_at_least_n_strict": "{:.2f}"}
    title: str = "Clustering parameter sweep (select best rows)",
    render: bool = True,
    set_default_font: bool = True,     # avoids 'pc/k/res' being misread as font family
):
    """
    Marsilea heatmap to guide clustering parameter selection.

    Left:   textual columns for parsed parameters (pc, k, res)
    Center: eval heatmap with raw numbers printed in cells (includes n_clusters as first column)
    Right:  bar = unique_strict_genes + unique_naive_genes; rows sorted descending by this score

    Row names (index strings) are NOT shown.
    """

    # Optional: force a sane default font to avoid font-family warnings
    if set_default_font:
        try:
            import matplotlib as mpl
            mpl.rcParams["font.family"] = "DejaVu Sans"
        except Exception:
            pass

    # --- Normalize selections and validate columns
    metrics      = list(metrics)
    bar_sum_cols = list(bar_sum_cols)

    need = {index_col, *metrics, *bar_sum_cols}
    missing = need - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns: {sorted(missing)}; available={sorted(df.columns)}")

    # --- Unique rows by index_col & keep needed cols
    base = (
        df[[index_col, *metrics]]
        .drop_duplicates(subset=[index_col])
        .set_index(index_col)
        .copy()
    )

    # --- selection score = sum of chosen gene-count columns
    score = (
        df[[index_col, *bar_sum_cols]]
        .drop_duplicates(subset=[index_col])
        .set_index(index_col)
        .sum(axis=1)
    )

    # --- order rows by descending score
    order = score.sort_values(ascending=False).index
    base  = base.loc[order]
    score = score.loc[order]

    # --- raw values to print in cells
    X_raw = base.loc[:, metrics].astype(float)

    if value_fmt is None:
        value_fmt = {
            col: "{:.3f}" if "frac" in col or "ratio" in col else "{:.0f}"
            for col in metrics
        }
        if "n_clusters" in X_raw.columns:
            value_fmt["n_clusters"] = "{:.0f}"

    text_matrix = np.array(
        [[value_fmt.get(c, "{:.3f}").format(v) for c, v in zip(X_raw.columns, row)]
         for row in X_raw.values]
    )

    # --- color matrix for evals heatmap (column-wise scaling)
    X_color = X_raw.copy()
    if scale_eval == "zscore":
        X_color = (X_color - X_color.mean(axis=0)) / X_color.std(axis=0).replace(0, np.nan)
        X_color = X_color.fillna(0.0)
    elif scale_eval == "minmax":
        rng = (X_color.max(axis=0) - X_color.min(axis=0)).replace(0, np.nan)
        X_color = (X_color - X_color.min(axis=0)) / rng
        X_color = X_color.fillna(0.0)
    elif scale_eval != "none":
        raise ValueError("scale_eval must be one of {'zscore','minmax','none'}")

    # --- parse pc / k / res from index_col strings like: "autoc_pc20_pct1.00_s01_k10_res0.05"
    def _extract_num(pat, s, cast=float):
        m = re.search(pat, s)
        return cast(m.group(1)) if m else np.nan

    labels_series = order.to_series()
    params_df = pd.DataFrame({
        "pc":  labels_series.map(lambda s: _extract_num(r"pc(\d+)", s, int)),
        "k":   labels_series.map(lambda s: _extract_num(r"k(\d+)", s, int)),
        "res": labels_series.map(lambda s: _extract_num(r"res([0-9]*\.?[0-9]+)", s, float)),
    }, index=order)

    # --- Marsilea plotting
    import marsilea as ma
    import marsilea.plotter as mp

    # Evals heatmap (no row name labels)
    h_eval = ma.Heatmap(
        X_color.values,
        linewidth=linewidth,
        label="Evals",
        cmap=cmap_eval,
    )
    h_eval.add_top(mp.Labels(list(X_color.columns)))      # show metric names, not row labels
    h_eval.add_layer(mp.TextMesh(text_matrix))            # overlay raw numbers

    # Right-side bar (with clear label & padding)
    h_eval.add_right(
        mp.Numbers(score.values, label="unique_strict + unique_naive"),
        size=0.9,
        pad=0.15,   # generous so the label is clear
    )

    # Title & legends
    h_eval.add_legends()
    h_eval.add_title(title)

    # --- LEFT textual parameter columns: pc | k | res (aligned with rows)
    # Use label + label_props so 'pc/k/res' are treated as titles, not font families.
    pc_col = mp.Labels(
        params_df["pc"].astype("Int64").astype(str).replace("<NA>", "NA"),
        label="pc",
        label_loc="top",
        label_props={"family": "DejaVu Sans", "weight": "bold"},
    )
    k_col = mp.Labels(
        params_df["k"].astype("Int64").astype(str).replace("<NA>", "NA"),
        label="k",
        label_loc="top",
        label_props={"family": "DejaVu Sans", "weight": "bold"},
    )
    res_col = mp.Labels(
        params_df["res"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "NA"),
        label="res",
        label_loc="top",
        label_props={"family": "DejaVu Sans", "weight": "bold"},
    )

    # Attach the three text columns on the left (sizes tuned for readability)
    h_eval.add_left(pc_col,  size=0.7, pad=0.05)
    h_eval.add_left(k_col,   size=0.7, pad=0.05)
    h_eval.add_left(res_col, size=0.9, pad=0.10)

    if render:
        h_eval.render()

    return {
        "canvas": h_eval,
        "row_order": order.to_list(),
        "score": score,
        "params": params_df,
    }





def heatmap_classifier_report(df: pd.DataFrame,
    width=2.5,
    height=7):

    # Separate the special averages from the rest of the labels
    special_averages = df[df['Label'].isin(['micro avg', 'macro avg', 'weighted avg'])]
    non_avg_df = df[~df['Label'].isin(['micro avg', 'macro avg', 'weighted avg'])]
    row_names = mp.Labels(df['Label'], align="right")
    row_types = ["class"] * non_avg_df.shape[0] + ["avg"] * special_averages.shape[0]
    col_names = mp.Labels(["Precision", "Recall", "F1-Score"], align="center")
    
     # Create custom colormap
    colors = [(0, "#add8e6"), (0.8, "#ffffff"), (0.81, "#ffcccc"), (1, "#ff0000")]
    cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)
    # Create Marsilea plot
    support_count = mp.Numbers(df["Support"], color="#fac858", label="support")
    
    # Combine non-average rows and special averages
    combined_df = pd.concat([non_avg_df, special_averages])
    metrics = ["Precision", "Recall", "F1-Score"]
    heatmap_data = combined_df.set_index("Label")[metrics]
    # h = ml.Heatmap(heatmap_data, linewidth=1, cmap = cmap, annot=True, width=width, height=height)
    h = ml.Heatmap(heatmap_data, linewidth=1, annot=True, width=width, height=height)
    
    # Group the rows to separate special averages
    h.group_rows(row_types, order=["class","avg"])
    # h.group_rows(list(special_averages.index), label="Averages", color="#add8e6", pad=0.1)
    
    # Add bar chart for support values
    h.add_right(support_count, pad=0.1, size=0.5)
    
    # Add labels
    h.add_left(row_names, pad=0.02)
    h.add_top(col_names, pad=0.03)
    h.add_legends("right", align_stacks="center", align_legends="top", pad=0.2)
    # Set margins and render the plot
    h.set_margin(0.5)
    h.render()


def heatmap_scores(
    adata: AnnData, 
    groupby: str, 
    vmin: float = 0, 
    vmax: float = 1, 
    obsm_name='SCN_score', 
    order_by: str = None,
    figure_subplot_bottom: float = 0.4
):
    """
    Plots a heatmap of single cell scores, grouping cells according to a specified .obs column and optionally ordering within each group.

    Args:
        adata (AnnData): An AnnData object containing the single cell data.
        groupby (str): The name of the column in .obs used for grouping cells in the heatmap.
        vmin (float, optional): Minimum value for color scaling. Defaults to 0.
        vmax (float, optional): Maximum value for color scaling. Defaults to 1.
        obsm_name (str, optional): The key in .obsm to retrieve the matrix for plotting. Defaults to 'SCN_score'.
        order_by (str, optional): The name of the column in .obs used for ordering cells within each group. Defaults to None.

    Returns:
        None: The function plots a heatmap and does not return any value.
    """
    # Create a temporary AnnData object with the scores matrix and all original observations
    adTemp = AnnData(adata.obsm[obsm_name], obs=adata.obs)
    
    # Determine sorting criteria
    if order_by is not None:
        sort_criteria = [groupby, order_by]
    else:
        sort_criteria = [groupby]
    
    # Determine the order of cells by sorting based on the criteria
    sorted_order = adTemp.obs.sort_values(by=sort_criteria).index
    
    # Reorder adTemp according to the sorted order
    adTemp = adTemp[sorted_order, :]
    
    # Set figure dimensions and subplot adjustments
    # fsize = [5, 6]
    # plt.rcParams['figure.subplot.bottom'] = figure_subplot_bottom
    
    # Plot the heatmap with the sorted and grouped data
    with plt.rc_context({'figure.subplot.bottom': figure_subplot_bottom}):
        sc.pl.heatmap(adTemp, adTemp.var_names.values, groupby=groupby, cmap=Batlow_20.mpl_colormap,dendrogram=False, swap_axes=True, vmin=vmin, vmax=vmax)


def heatmap_gsea(
    gmat,
    clean_signatures=False,
    clean_cells=False,
    column_colors=None,
    figsize=(8, 6),
    label_font_size=7,
    cbar_pos=[0.2, 0.92, 0.6, 0.02],  # Positioned at the top
    dendro_ratio=(0.3, 0.1),
    cbar_title='NES',
    col_cluster=False,
    row_cluster=False,
):
    """
    Generates a heatmap with hierarchical clustering for gene set enrichment analysis (GSEA) results.
    
    Args:
        gmat (pd.DataFrame):
            A matrix of GSEA scores with gene sets as rows and samples as columns.
        clean_signatures (bool, optional):
            If True, removes gene sets with zero enrichment scores across all samples. Defaults to False.
        clean_cells (bool, optional):
            If True, removes samples with zero enrichment scores across all gene sets. Defaults to False.
        column_colors (pd.Series or pd.DataFrame, optional):
            Colors to annotate columns, typically representing sample groups. Defaults to None.
        figsize (tuple, optional):
            Figure size in inches (width, height). Defaults to (8, 6).
        label_font_size (int, optional):
            Font size for axis and colorbar labels. Defaults to 7.
        cbar_pos (list, optional):
            Position of the colorbar [left, bottom, width, height]. Defaults to [0.2, 0.92, 0.6, 0.02] for a horizontal top placement.
        dendro_ratio (tuple, optional):
            Proportion of the figure allocated to the row and column dendrograms. Defaults to (0.3, 0.1).
        cbar_title (str, optional):
            Title of the colorbar. Defaults to 'NES'.
        col_cluster (bool, optional):
            If True, performs hierarchical clustering on columns. Defaults to False.
        row_cluster (bool, optional):
            If True, performs hierarchical clustering on rows. Defaults to False.
    
    Returns:
        None
    
    Displays:
        A heatmap with optional hierarchical clustering and a horizontal colorbar at the top.
    """
    gsea_matrix = gmat.copy()
    if clean_cells:
        gsea_matrix = gsea_matrix.loc[:, gsea_matrix.sum(0) != 0]
    if clean_signatures:
        gsea_matrix = gsea_matrix.loc[gsea_matrix.sum(1) != 0, :]
    
    # plt.figure(constrained_layout=True)
    ax = sns.clustermap(
        data=gsea_matrix,
        cmap=Roma_20.mpl_colormap.reversed(),
        center=0,
        yticklabels=1,
        xticklabels=1,
        linewidth=.05,
        linecolor='white',
        method='average',
        metric='euclidean',
        dendrogram_ratio=dendro_ratio,
        col_colors=column_colors,
        figsize=figsize,
        row_cluster=row_cluster,
        col_cluster=col_cluster,
        cbar_pos=cbar_pos,
        cbar_kws={'orientation': 'horizontal'}
    )
    
    ax.ax_cbar.set_title(cbar_title, fontsize=label_font_size, pad=10)
    ax.ax_cbar.tick_params(labelsize=label_font_size, direction='in')
    
    # Adjust tick labels and heatmap appearance
    ax.ax_row_dendrogram.set_visible(False)
    ax.ax_col_dendrogram.set_visible(False)
    ax.ax_heatmap.set_yticklabels(ax.ax_heatmap.get_ymajorticklabels(), fontsize=label_font_size)
    ax.ax_heatmap.set_xticklabels(ax.ax_heatmap.get_xmajorticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=label_font_size)
    
    # plt.subplots_adjust(top=0.85)
    plt.show()


def heatmap_gsea_middle(
    gmat,
    clean_signatures = False,
    clean_cells = False,
    column_colors = None,
    figsize=(8,6),
    label_font_size = 7,
    cbar_pos = [0.07, .3, .02, .4],
    dendro_ratio = (0.3, 0.1),
    cbar_title='NES' ,
    col_cluster=False,
    row_cluster=False,
):
    
    gsea_matrix = gmat.copy()
    if clean_cells:
        gsea_matrix = gsea_matrix.loc[:,gsea_matrix.sum(0) != 0]

    if clean_signatures:
        gsea_matrix = gsea_matrix.loc[gsea_matrix.sum(1) != 0,:]

    # should add a check on matrix dims, and print message a dim is 0
    ax = sns.clustermap(data=gsea_matrix, cmap=Roma_20.mpl_colormap.reversed(), center=0,
        yticklabels=1, xticklabels=1, linewidth=.05, linecolor='white',
        method='average', metric='euclidean', dendrogram_ratio=dendro_ratio,
        col_colors= column_colors, figsize=figsize, row_cluster=row_cluster, col_cluster=col_cluster)
    # put gene set labels on left side, and right justify them <- does not work well    
    # ax.ax_heatmap.yaxis.tick_left()
    # ax.ax_heatmap.yaxis.set_label_position("left")
    # for label in ax.ax_heatmap.get_yticklabels():
    #     label.set_horizontalalignment('right')

    # uncomment the following 2 lines to put x-axis labels at top of heatmap
    # Let the horizontal axes labeling appear on top.
    # ax.tick_params(top=True, bottom=False,
    #               labeltop=True, labelbottom=False)

    ax.ax_cbar.set_axis_off()
    ax.figure.tight_layout(h_pad=.05, w_pad=0.02, rect=[0, 0, 1, 1])
    ax.ax_cbar.set_axis_on()
    # ax.ax_cbar.set_position((0.02, 0.8, 0.05, 0.18))

    ax.ax_cbar.set_position(cbar_pos)
    ax.ax_cbar.yaxis.tick_left()
    ax.ax_cbar.set_title(cbar_title, fontsize=label_font_size)
    ax.ax_cbar.tick_params(labelsize=label_font_size) # set_ylabel(size = label_font_size)
    
    ax.ax_row_dendrogram.set_visible(False)
    ax.ax_heatmap.set_yticklabels(ax.ax_heatmap.get_ymajorticklabels(), fontsize = label_font_size)
    ax.ax_heatmap.set_xticklabels(ax.ax_heatmap.get_xmajorticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize = label_font_size)
    plt.show()

def heatmap_gsea_old(
    gmat,
    clean_signatures = False,
    clean_cells = False,
    column_colors = None,
    figsize=(8,6),
    label_font_size = 7,
    cbar_pos = [0.07, .3, .02, .4],
    dendro_ratio = (0.3, 0.1),
    cbar_title='NES' 
):
    
    gsea_matrix = gmat.copy()
    if clean_cells:
        gsea_matrix = gsea_matrix.loc[:,gsea_matrix.sum(0) != 0]

    if clean_signatures:
        gsea_matrix = gsea_matrix.loc[gsea_matrix.sum(1) != 0,:]

    # should add a check on matrix dims, and print message a dim is 0
    ax = sns.clustermap(data=gsea_matrix, cmap=Roma_20.mpl_colormap.reversed(), center=0,
        yticklabels=1, xticklabels=1, linewidth=.05, linecolor='white',
        method='average', metric='euclidean', dendrogram_ratio=dendro_ratio,
        col_cluster = False, col_colors= column_colors, figsize=figsize)
    # put gene set labels on left side, and right justify them <- does not work well    
    # ax.ax_heatmap.yaxis.tick_left()
    # ax.ax_heatmap.yaxis.set_label_position("left")
    # for label in ax.ax_heatmap.get_yticklabels():
    #     label.set_horizontalalignment('right')

    # uncomment the following 2 lines to put x-axis labels at top of heatmap
    # Let the horizontal axes labeling appear on top.
    # ax.tick_params(top=True, bottom=False,
    #               labeltop=True, labelbottom=False)

    ax.ax_cbar.set_axis_off()
    ax.figure.tight_layout(h_pad=.05, w_pad=0.02, rect=[0, 0, 1, 1])
    ax.ax_cbar.set_axis_on()
    # ax.ax_cbar.set_position((0.02, 0.8, 0.05, 0.18))

    ax.ax_cbar.set_position(cbar_pos)
    ax.ax_cbar.yaxis.tick_left()
    ax.ax_cbar.set_title(cbar_title, fontsize=label_font_size)
    ax.ax_cbar.tick_params(labelsize=label_font_size) # set_ylabel(size = label_font_size)
    
    ax.ax_row_dendrogram.set_visible(False)
    ax.ax_heatmap.set_yticklabels(ax.ax_heatmap.get_ymajorticklabels(), fontsize = label_font_size)
    ax.ax_heatmap.set_xticklabels(ax.ax_heatmap.get_xmajorticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize = label_font_size)
    plt.show()




def heatmap_genes(adQuery, adTrain=None, cgenes_list={}, list_of_types_toshow=[], list_of_training_to_show=[], number_of_genes_toshow=3, query_annotation_togroup='SCN_class', training_annotation_togroup='SCN_class', split_show=False, save = False):
    if list_of_types_toshow.__len__() == list_of_training_to_show.__len__():
        if list_of_types_toshow.__len__() == 0:
            list_of_types_toshow = np.unique(adQuery.obs['SCN_class'])
            num = list_of_types_toshow.__len__()
            list_of_training_to_show = [False for _ in range(num)]
            # Default: all shown and False
        else:
            if adTrain is None:
                print("No adTrain given")
                return
            else:
                pass
    elif list_of_training_to_show.__len__() == 0:
        num = list_of_types_toshow.__len__()
        list_of_training_to_show = [False for _ in range(num)]
        # Default: all False
    else:
        print("Boolean list not correspondent to list of types")
        return


    list_1 = []
    list_2 = []
    SCN_annot = []
    annot = []

    MQ = adQuery.X.todense()
    if adTrain is not None:
        MT = adTrain.X.todense()
    else:
        pass
    

    for i in range(list_of_types_toshow.__len__()):
        type = list_of_types_toshow[i]
        for j in range(np.size(MQ, 0)):
            if adQuery.obs['SCN_class'][j] == type:
                list_1.append(j)
                SCN_annot.append(type)
                if split_show:
                    annot.append(adQuery.obs[query_annotation_togroup][j]+'_Query')
                else:
                    annot.append(adQuery.obs[query_annotation_togroup][j])

    for i in range(list_of_training_to_show.__len__()):
        type = list_of_types_toshow[i]            
        if list_of_training_to_show[i]:
            for j in range(np.size(MT, 0)):
                if adTrain.obs['SCN_class'][j] == type:
                    list_2.append(j)
                    SCN_annot.append(type)
                    if split_show:
                        annot.append(adTrain.obs[training_annotation_togroup][j]+'_Train')
                    else:
                        annot.append(adTrain.obs[training_annotation_togroup][j])
        else:
            pass
    
    SCN_annot = pd.DataFrame(SCN_annot)
    annot = pd.DataFrame(annot)

    M_1 = MQ[list_1,:]
    if adTrain is not None:
        M_2 = MT[list_2,:]
    else:
        pass

    if adTrain is None:
        Mdense = M_1
    else:
        Mdense = np.concatenate((M_1,M_2))
    New_Mat = csr_matrix(Mdense)
    adTrans = sc.AnnData(New_Mat)

    adTrans.obs['SCN_class'] = SCN_annot.values
    adTrans.obs['Cell_Type'] = annot.values
    adTrans.var_names = adQuery.var_names

    sc.pp.normalize_per_cell(adTrans, counts_per_cell_after=1e4)
    sc.pp.log1p(adTrans)

    RankList = []

    for type in list_of_types_toshow:
        for i in range(number_of_genes_toshow):
            RankList.append(cgenes_list[type][i])

    fig = sc.pl.heatmap(adTrans, RankList, groupby='Cell_Type', cmap='viridis', dendrogram=False, swap_axes=True, save=save)
    return fig