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
    h = ml.Heatmap(heatmap_data, cmap = cmap, annot=True, width=width, height=height)
    
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


def heatmap_scores(adata: AnnData, groupby: str, vmin: float = 0, vmax: float = 1, obsm_name='SCN_score', order_by: str = None):
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
    fsize = [5, 6]
    plt.rcParams['figure.subplot.bottom'] = 0.25
    
    # Plot the heatmap with the sorted and grouped data
    sc.pl.heatmap(adTemp, adTemp.var_names.values, groupby=groupby, 
                  cmap=Batlow_20.mpl_colormap,
                  dendrogram=False, swap_axes=True, vmin=vmin, vmax=vmax, 
                  figsize=fsize)


def heatmap_gsea(
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