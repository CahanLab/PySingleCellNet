import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import umap
import anndata as ad
from palettable.cartocolors.qualitative import Vivid_4
from palettable.cartocolors.qualitative import Vivid_3
from palettable.cartocolors.qualitative import Vivid_10
from palettable.scientific.diverging import Roma_20
from palettable.scientific.sequential import LaJolla_20
from palettable.scientific.sequential import Batlow_20
from anndata import AnnData
from scipy.sparse import csr_matrix
from sklearn.metrics import f1_score
# import altair as alt
from .utils import *

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


def plot_cell_type_proportions(adata_list, obs_column='SCN_class', labels=None):
    """
    Plots a stacked bar chart of category proportions for a list of AnnData objects.

    This function takes a list of AnnData objects, and for a specified column in the `.obs` attribute,
    it plots a stacked bar chart. Each bar represents an AnnData object with segments showing the proportion
    of each category within that object.

    Args:
        adata_list (List[anndata.AnnData]): A list of AnnData objects.
        obs_column (str, optional): The name of the `.obs` column to use for categories. Defaults to 'SCN_class'.
        labels (List[str], optional): Custom labels for each AnnData object to be displayed on the x-axis.
            If not provided, defaults to 'AnnData {i+1}' for each object. The length of `labels` must match
            the number of AnnData objects provided.

    Raises:
        ValueError: If the length of `labels` does not match the number of AnnData objects.

    Examples:
        >>> plot_cell_type_proportions([adata1, adata2], obs_column='your_column_name', labels=['Sample 1', 'Sample 2'])
    """
    
    # Ensure labels are provided, or create default ones
    if labels is None:
        labels = [f'AnnData {i+1}' for i in range(len(adata_list))]
    elif len(labels) != len(adata_list):
        raise ValueError("Length of 'labels' must match the number of AnnData objects provided.")
    
    # Extracting category proportions
    category_counts = []
    categories = set()
    for adata in adata_list:
        counts = adata.obs[obs_column].value_counts(normalize=True)
        category_counts.append(counts)
        categories.update(counts.index)
    
    categories = sorted(categories)
    
    # Preparing the data for plotting
    proportions = np.zeros((len(categories), len(adata_list)))
    for i, counts in enumerate(category_counts):
        for category in counts.index:
            j = categories.index(category)
            proportions[j, i] = counts[category]
    
    # Plotting
    fig, ax = plt.subplots()
    bottom = np.zeros(len(adata_list))
    for i, category in enumerate(categories):
        ax.bar(range(len(adata_list)), proportions[i], bottom=bottom, label=category)
        bottom += proportions[i]
    
    ax.set_xticks(range(len(adata_list)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel('Percent')
    ax.set_title(f'{obs_column} proportions')
    ax.legend(title='Categories', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()



def barplot_class_proportions_list(
    ads, 
    titles=None,
    scn_classes_to_display=None,
    bar_height=0.8,
    bar_groups_obsname = 'SCN_class',
    bar_subgroups_obsname = 'SCN_class_type',
    ncell_min = None,
):
    dfs = [adata.obs for adata in ads]
    num_dfs = len(dfs)
    # Determine the titles for each subplot
    if titles is None:
        titles = ['SCN Class Proportions'] * num_dfs
    elif len(titles) != num_dfs:
        raise ValueError("The length of 'titles' must match the number of annDatas.")
    # Determine the SCN classes to display
    all_classes_union = set().union(*(df[bar_groups_obsname].unique() for df in dfs))
    if scn_classes_to_display is not None:
        if not all(cls in all_classes_union for cls in scn_classes_to_display):
            raise ValueError("Some values in 'scn_classes_to_display' do not match available 'SCN_class' values in the provided DataFrames.")
        all_classes = scn_classes_to_display
    else:
        all_classes = all_classes_union
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(1, num_dfs, figsize=(6 * num_dfs, 8), sharey=True)
    for ax, df, title in zip(axes, dfs, titles):
        # Reindex and filter each DataFrame
        counts = df.groupby(bar_groups_obsname)[bar_subgroups_obsname].value_counts().unstack().reindex(all_classes).fillna(0)
        total_counts = counts.sum(axis=1)
        if ncell_min is not None:
            total_counts = total_counts.where(total_counts >= ncell_min, 0)

        proportions = counts.divide(total_counts, axis=0).fillna(0).replace([np.inf, -np.inf], 0, inplace=False)
        total_percent = (total_counts / total_counts.sum() * 100).round(1)  # Converts to percentage and round

        # Plotting
        proportions.plot(kind='barh', stacked=True, colormap=Vivid_3.mpl_colormap, width=bar_height, ax=ax, legend=False)
        # Modify colors from colormap to include alpha transparency
        # colors = [Vivid_3.mpl_colormap(i) for i in range(Vivid_3.mpl_colormap.N)]
        # new_colors = [(r,g,b,alpha_values.iloc[i]) for i, (r,g,b,a) in enumerate(colors)]  # Modify alpha for each color
        # new_colormap = mcolors.LinearSegmentedColormap.from_list("CustomMap", new_colors)
        
        # Plotting with the modified colormap
        # proportions.plot(kind='barh', stacked=True, colormap=new_colormap, width=bar_height, ax=ax, legend=False)        
        
        # Adding adjusted internal total counts within each bar
        text_size = max(min(12 - len(all_classes) // 2, 10), 7)  # Adjust text size
        for i, (count, percent) in enumerate(zip(total_counts, total_percent)):
            text = f'{int(count)} ({percent}%)'  # Text to display
            # text = f'({percent}%)'  # Text to display
            ax.text(0.95, i, text, ha='right', va='center', color='white', fontsize=text_size)

        ax.set_xlabel('Proportion')
        ax.set_title(title)

    # Setting the y-label for the first subplot only
    axes[0].set_ylabel('SCN Class')
    # Adding the legend after the last subplot
    axes[-1].legend(title='SCN Class Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    return plt



def barplot_scn_categories(
    adata: AnnData,
    bar_height = 0.8
):
    df = adata.obs.copy()
    # Group by 'SCN_class' and get dummies for 'SCN_class_type'
    counts = df.groupby('SCN_class')['SCN_class_type'].value_counts().unstack().fillna(0)
    proportions = counts.divide(counts.sum(axis=1), axis=0)
    total_counts = counts.sum(axis=1)
    # Determine the number of unique SCN_classes to adjust text size
    num_classes = len(df['SCN_class'].unique())
    text_size = max(min(12 - num_classes // 2, 10), 5)  # Adjust text size based on number of classes
    # Plotting
    ax = proportions.plot(kind='barh', stacked=True, colormap=Vivid_3.mpl_colormap, width=bar_height)
    ax.set_xlabel('Proportion')
    ax.set_ylabel('SCN Class')
    ax.set_title('Proportions of SCN Class Types with Total Counts')
    # Adding total counts to the right of each bar
    for i, value in enumerate(total_counts):
        ax.text(0.95, i, int(value), ha='right', va='center', color='white',fontsize=text_size)
    plt.legend(title='SCN Class Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    return plt


def dotplot_deg(
    adata: AnnData,
    diff_gene_dict: dict,
    #samples_obsvals: list = [],
    groupby_obsname: str = "comb_sampname",
    cellgrp_obsname: str = "comb_cellgrp",    
    cellgrp_obsvals: list = [],
    num_genes: int = 10, 
    order_by = 'scores',
    new_obsname = 'grp_by_samp',
    use_raw=False
):

    # remove celltypes unspecified in diff_gene_dict
    dd_dict = diff_gene_dict['geneTab_dict']
    tokeep = list(dd_dict.keys())

    # also remove cell_types not listed in celltype_names
    # default for celltype_names is all celltypes included in dd_dict
    if len(cellgrp_obsvals) > 0:
        cellgrp_obsvals = list(set(cellgrp_obsvals).intersection(set(tokeep)))
    else:
        cellgrp_obsvals = tokeep

    adNew = adata.copy()
    adNew = adNew[adNew.obs[cellgrp_obsname].isin(cellgrp_obsvals)].copy()
    
    # remove categories unspecified in diff_gene_dict
    dictkey = list(diff_gene_dict.keys())[0]
    sample_names = diff_gene_dict[dictkey]
    adNew = adNew[adNew.obs[groupby_obsname].isin(sample_names)].copy()

    # add column 'grp_by_samp' to obs that indicates cellgrp by sample
    adNew.obs[new_obsname] = adNew.obs[cellgrp_obsname].astype(str) + "_X_" + adNew.obs[groupby_obsname].astype(str)
    
    # define dict of marker genes based on threshold
    genes_to_plot = dict()
    for cellgrp in cellgrp_obsvals:
        print(f"{cellgrp}")
        for sname in sample_names:
            print(f"{sname}")
            genes_to_plot[cellgrp + "_X_" + sname] = pull_out_genes_v2(diff_gene_dict, cell_type = cellgrp, category = sname, num_genes = num_genes, order_by=order_by) 

    # return adNew, genes_to_plot
    plt.rcParams['figure.constrained_layout.use'] = True
    #xplot = sc.pl.DotPlot(adNew, genes_to_plot, 'ct_by_cat', cmap='RdPu', var_group_rotation = 0) #, dendrogram=True,ax=ax2, show=False)
    xplot = sc.pl.DotPlot(adNew, genes_to_plot, new_obsname, cmap=LaJolla_20.mpl_colormap, var_group_rotation = 0, use_raw=False) #, dendrogram=True,ax=ax2, show=False)
    xplot.swap_axes(True) # see sc.pl.DotPlot docs for useful info
    return xplot


def dotplot_diff_gene(
    adata: AnnData,
    diff_gene_dict: dict,
    num_genes: int = 10,
    celltype_groupby: str = "SCN_class",
    category_groupby: str = "SCN_class_type",
    category_names: str = ["None", "Singular"],
    celltype_names: list = [],
    order_by = 'scores'
):

    # remove celltypes unspecified in diff_gene_dict
    dd_dict = diff_gene_dict['geneTab_dict']
    tokeep = list(dd_dict.keys())

    # also remove cell_types not listed in celltype_names
    # default for celltype_names is all celltypes included in dd_dict
    if len(celltype_names) > 0:
        celltype_names = list(set(celltype_names).intersection(set(tokeep)))
    else:
        celltype_names = tokeep

    adNew = adata.copy()
    # adNew = adNew[adNew.obs[celltype_groupby].isin(tokeep)].copy()
    adNew = adNew[adNew.obs[celltype_groupby].isin(celltype_names)].copy()
    
    # remove categories unspecified in diff_gene_dict
    category_names = diff_gene_dict['category_names']
    adNew = adNew[adNew.obs[category_groupby].isin(category_names)].copy()

    

    # add column 'ct_by_cat' to obs that indicates celltype X category
    adNew.obs['ct_by_cat'] = adNew.obs[celltype_groupby].astype(str) + "_X_" + adNew.obs[category_groupby].astype(str)
    
    # define dict of marker genes based on threshold
    genes_to_plot = dict()
    for celltype in celltype_names:
        print(f"{celltype}")
        for scn_category in category_names:
            print(f"{scn_category}")
            genes_to_plot[celltype + "_X_" + scn_category] = pull_out_genes(diff_gene_dict, cell_type = celltype, category = scn_category, num_genes = num_genes, order_by=order_by) 

    # return adNew, genes_to_plot
    plt.rcParams['figure.constrained_layout.use'] = True
    #xplot = sc.pl.DotPlot(adNew, genes_to_plot, 'ct_by_cat', cmap='RdPu', var_group_rotation = 0) #, dendrogram=True,ax=ax2, show=False)
    xplot = sc.pl.DotPlot(adNew, genes_to_plot, 'ct_by_cat', cmap=LaJolla_20.mpl_colormap, var_group_rotation = 0) #, dendrogram=True,ax=ax2, show=False)
    xplot.swap_axes(True) # see sc.pl.DotPlot docs for useful info
    return xplot
    


def barplot_classifier_f1(adata: AnnData, ground_truth: str = "celltype", class_prediction: str = "SCN_class"):
    fscore = f1_score(adata.obs[ground_truth], adata.obs[class_prediction], average=None, labels = adata.obs[ground_truth].cat.categories)
    cates = list(adata.obs[ground_truth].cat.categories)
    f1_score_dict = {class_label: f1_score_x for class_label, f1_score_x in zip(cates, fscore)}
    
    # Calculate the number of observations per class
    class_counts = adata.obs[ground_truth].value_counts().to_dict()

    plt.rcParams['figure.constrained_layout.use'] = True
    sns.set_theme(style="whitegrid")
    ax = sns.barplot(x=list(f1_score_dict.values()), y=list(f1_score_dict.keys()))
    plt.xlabel('F1-Score')
    plt.title('F1-Scores per Class')
    plt.xlim(0, 1.1) # Set x-axis limits to ensure visibility of all bars

    # Add the number of observations per class as text within the barplot
    for i, bar in enumerate(ax.containers[0]):
        width = bar.get_width()
        print(f"{width}")
        label = f"n = {class_counts[cates[i]]}"
        fcolor = "white"
        if width < 0.20:
            fcolor = "black"

        ax.text( 0.03, bar.get_y() + bar.get_height() / 2, label, ha='left', va='center', color = fcolor)

    plt.show()

def dotplot_scn_scores(
    adata: AnnData,
    groupby: str,
    expression_cutoff = 0.1,
    obsm_name = 'SCN_score'
):    
    adTemp = AnnData(adata.obsm[obsm_name], obs=adata.obs)
    adTemp.obs[groupby] = adata.obs[groupby]
    sc.pl.dotplot(adTemp, adTemp.var_names.values, groupby=groupby, expression_cutoff=expression_cutoff, cmap=Batlow_20.mpl_colormap, colorbar_title="SCN score")

def umap_scores(
    adata: AnnData,
    scn_classes: list,
    obsm_name = 'SCN_score'
):    
    adTemp = AnnData(adata.obsm[obsm_name], obs=adata.obs)
    adTemp.obsm['X_umap'] = adata.obsm['X_umap'].copy()
    sc.pl.umap(adTemp,color=scn_classes, alpha=.75, s=10, vmin=0, vmax=1)



def heatmap_gsea(
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



def hm_genes(adQuery, adTrain=None, cgenes_list={}, list_of_types_toshow=[], list_of_training_to_show=[], number_of_genes_toshow=3, query_annotation_togroup='SCN_class', training_annotation_togroup='SCN_class', split_show=False, save = False):
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