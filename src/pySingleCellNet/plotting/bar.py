import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import umap
import anndata as ad
import igraph as ig
# from igraph import Graph
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
from sklearn.metrics import f1_score
from ..utils import *
from pySingleCellNet.config import SCN_CATEGORY_COLOR_DICT

def stackedbar_composition(
    adata: AnnData, 
    groupby: str, 
    obs_column = 'SCN_class', 
    labels = None, 
    bar_width: float = 0.75, 
    color_dict = None
):
    """
    Plots a stacked bar chart of cell type proportions for a single AnnData object grouped by a specified column.

    This function takes an AnnData object, and for a specified column in the `.obs` attribute, it groups the data
    and plots a stacked bar chart. Each bar represents a group with segments showing the proportion of each category
    within that group.

    Args:
        adata (anndata.AnnData): An AnnData object.
        groupby (str): The column in `.obs` to group by.
        obs_column (str, optional): The name of the `.obs` column to use for categories. Defaults to 'SCN_class'.
        labels (List[str], optional): Custom labels for each group to be displayed on the x-axis.
            If not provided, the unique values of the groupby column will be used. The length of `labels` must match
            the number of unique groups.
        bar_width (float, optional): The width of the bars in the plot. Defaults to 0.75.
        color_dict (Dict[str, str], optional): A dictionary mapping categories to specific colors. If not provided,
            default colors will be used.

    Raises:
        ValueError: If the length of `labels` does not match the number of unique groups.

    Examples:
        >>> plot_grouped_cell_type_proportions(adata, groupby='sample', obs_column='your_column_name')
    """
    
    # Ensure the groupby column exists in .obs
    if groupby not in adata.obs.columns:
        raise ValueError(f"The groupby column '{groupby}' does not exist in the .obs attribute.")
    
    # Extract unique groups and ensure labels are provided or create default ones
    # unique_groups = adata.obs[groupby].unique()
    unique_groups = adata.obs[groupby].cat.categories.to_list()
    if labels is None:
        labels = unique_groups
    elif len(labels) != len(unique_groups):
        raise ValueError("Length of 'labels' must match the number of unique groups.")
    
    if color_dict is None:
        color_dict = adata.uns['SCN_class_colors'] 

    # Extracting category proportions per group
    category_counts = []
    categories = set()
    for group in unique_groups:
        subset = adata[adata.obs[groupby] == group]
        counts = subset.obs[obs_column].value_counts(normalize=True)
        category_counts.append(counts)
        categories.update(counts.index)
    
    categories = sorted(categories)
    
    # Preparing the data for plotting
    proportions = np.zeros((len(categories), len(unique_groups)))
    for i, counts in enumerate(category_counts):
        for category in counts.index:
            j = categories.index(category)
            proportions[j, i] = counts[category]
    
    # Plotting
    fig, ax = plt.subplots()
    bottom = np.zeros(len(unique_groups))
    for i, category in enumerate(categories):
        color = color_dict[category] if color_dict and category in color_dict else None
        ax.bar(
            range(len(unique_groups)), 
            proportions[i], 
            bottom=bottom, 
            label=category, 
            width=bar_width, 
            edgecolor='white', 
            linewidth=.5,
            color=color
        )
        bottom += proportions[i]
    
    ax.set_xticks(range(len(unique_groups)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel('Percent')
    ax.set_title(f'{obs_column} proportions by {groupby}')
    ax.legend(title='Categories', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()


def stackedbar_composition_list(
    adata_list, 
    obs_column = 'SCN_class', 
    labels = None, 
    bar_width: float = 0.75, 
    color_dict = None
):
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
        bar_width (float, optional): The width of the bars in the plot. Defaults to 0.75.
        color_dict (Dict[str, str], optional): A dictionary mapping categories to specific colors. If not provided,
            default colors will be used.

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
    
    if color_dict is None:
        color_dict = adata_list[0].uns['SCN_class_colors'] # should parameterize this

    # Plotting
    fig, ax = plt.subplots()
    bottom = np.zeros(len(adata_list))
    for i, category in enumerate(categories):
        color = color_dict[category] if color_dict and category in color_dict else None
        ax.bar(
            range(len(adata_list)), 
            proportions[i], 
            bottom=bottom, 
            label=category, 
            width=bar_width, 
            edgecolor='white', 
            linewidth=.5,
            color=color
        )
        bottom += proportions[i]
    
    ax.set_xticks(range(len(adata_list)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel('Percent')
    ax.set_title(f'{obs_column} proportions')
    ax.legend(title='Categories', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()

def stackedbar_categories_list(
    ads, 
    titles=None,
    scn_classes_to_display=None,
    bar_height=0.8,
    bar_groups_obsname = 'SCN_class',
    bar_subgroups_obsname = 'SCN_class_type',
    ncell_min = None,
    color_dict = None
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

    # setup colors
    if color_dict is None:
        color_dict = SCN_CATEGORY_COLOR_DICT 

    all_categories = set().union(*(df[bar_subgroups_obsname].unique() for df in dfs))
    

    # Create a figure with multiple subplots
    fig, axes = plt.subplots(1, num_dfs, figsize=(6 * num_dfs, 8), sharey=True)
    for ax, df, title in zip(axes, dfs, titles):


        # Ensure SCN_class categories are consistent
        # I think if adatas were classified with same clf, then they should have the same SCN_class categories even if not all cell types are predicted in a given adata
        df['SCN_class'] = df['SCN_class'].astype('category')
        df['SCN_class_type'] = df['SCN_class_type'].astype('category')

        df['SCN_class'] = df['SCN_class'].cat.set_categories(df['SCN_class'].cat.categories)
        df['SCN_class_type'] = df['SCN_class_type'].cat.set_categories(df['SCN_class_type'].cat.categories)

        # Reindex and filter each DataFrame
        counts = df.groupby(bar_groups_obsname)[bar_subgroups_obsname].value_counts().unstack().reindex(all_classes).fillna(0)
        total_counts = counts.sum(axis=1)
        if ncell_min is not None:
            total_counts = total_counts.where(total_counts >= ncell_min, 0)

        proportions = counts.divide(total_counts, axis=0).fillna(0).replace([np.inf, -np.inf], 0, inplace=False)
        total_percent = (total_counts / total_counts.sum() * 100).round(1)  # Converts to percentage and round

        # Plotting
        ### proportions.plot(kind='barh', stacked=True, colormap=Vivid_3.mpl_colormap, width=bar_height, ax=ax, legend=False)

        proportions.plot(kind='barh', stacked=True, color=color_dict, width=bar_height, ax=ax, legend=False)
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

def stackedbar_categories(
    adata: AnnData,
    scn_classes_to_display = None, 
    # color_list = GreenOrange_6.mpl_colors,
    bar_height=0.8,
    color_dict = None
):
    # Copy the obs DataFrame to avoid modifying the original data
    df = adata.obs.copy()
    
    all_classes = df['SCN_class_type'].unique()
    if scn_classes_to_display is not None:
        if not all(cls in all_classes for cls in scn_classes_to_display):
            raise ValueError("Some values in 'scn_classes_to_display' do not match available 'SCN_class' values in the provided DataFrames.")
        all_classes = scn_classes_to_display
    
    # Ensure the columns 'SCN_class' and 'SCN_class_type' exist
    if 'SCN_class' not in df.columns or 'SCN_class_type' not in df.columns:
        raise KeyError("Columns 'SCN_class' and 'SCN_class_type' must be present in adata.obs")
    
    # Ensure SCN_class categories are consistent
    df['SCN_class'] = df['SCN_class'].astype('category')
    df['SCN_class_type'] = df['SCN_class_type'].astype('category')

    df['SCN_class'] = df['SCN_class'].cat.set_categories(df['SCN_class'].cat.categories)
    df['SCN_class_type'] = df['SCN_class_type'].cat.set_categories(df['SCN_class_type'].cat.categories)

    # Group by 'SCN_class' and get value counts for 'SCN_class_type'
    try:
        counts = df.groupby('SCN_class')['SCN_class_type'].value_counts().unstack().fillna(0)
    except Exception as e:
        print("Error during groupby and value_counts operations:", e)
        return
    
    # Calculate proportions
    proportions = counts.divide(counts.sum(axis=1), axis=0)
    
    # Calculate total counts
    total_counts = counts.sum(axis=1)
    total_percent = (total_counts / total_counts.sum() * 100).round(1)  # Converts to percentage and round

    # setup colors
    if color_dict is None:
        color_dict = SCN_CATEGORY_COLOR_DICT 
    
    # Determine the number of unique SCN_classes to adjust text size
    num_classes = len(df['SCN_class'].unique())
    # text_size = max(min(12 - num_classes // 2, 10), 5)  # Adjust text size based on number of classes
    
    # Plotting
    # ax = proportions.plot(kind='barh', stacked=True, colormap=Vivid_3, width=bar_height)
    #### ax = proportions.plot(kind='barh', stacked=True, colormap=Vivid_3.mpl_colormap, width=bar_height)
    #### ax = proportions.plot(kind='barh', stacked=True, colormap=Vivid_6.mpl_colormap, width=bar_height)
    
    #### SCN_class_category_names = ["Singular", "None", "Parent.Child", "Sibling", "Hybrid", "Gp.Gc"]
    #### SCN_class_category_color_dict = dict(zip(SCN_class_category_names, color_list))
    #### ax = proportions.plot(kind='barh', stacked=True, color=SCN_class_category_color_dict, width=bar_height)
    ax = proportions.plot(kind='barh', stacked=True, color=color_dict, width=bar_height)    

    # Set axis labels and title
    ax.set_xlabel('Proportion')
    ax.set_ylabel('SCN Class')
    ax.set_title('Cell typing categorization')
    
    # Adding adjusted internal total counts within each bar
    text_size = max(min(12 - num_classes // 2, 10), 7) # Adjust text size
    for i, (count, percent) in enumerate(zip(total_counts, total_percent)):
        text = f'{int(count)} ({percent}%)'  # Text to display
        # text = f'({percent}%)'  # Text to display
        ax.text(0.95, i, text, ha='right', va='center', color='white', fontsize=text_size)

    # for i, value in enumerate(total_counts):
    #    ax.text(0.95, i, int(value), ha='right', va='center', color='white',fontsize=text_size)

    # Add legend
    plt.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    return plt


def bar_classifier_f1(adata: AnnData, ground_truth: str = "celltype", class_prediction: str = "SCN_class"):
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
