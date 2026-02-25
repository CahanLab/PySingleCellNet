import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import anndata as ad
from anndata import AnnData
from scipy.sparse import csr_matrix
from sklearn.metrics import f1_score
from pySingleCellNet.config import SCN_CATEGORY_COLOR_DICT
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, leaves_list

def bar_compare_celltype_composition(adata1, adata2, celltype_col, min_delta, colors=None, metric="log_ratio"):
    """Compare cell type proportions between two AnnData objects and plot either log-ratio or differences for significant changes.

    Args:
        adata1 (AnnData): First AnnData object.
        adata2 (AnnData): Second AnnData object.
        celltype_col (str): Column name in `.obs` indicating cell types.
        min_delta (float): Minimum absolute difference in percentages to include in the plot.
        colors (dict, optional): Dictionary with cell types as keys and colors as values for the bars.
        metric (str, optional): "log_ratio" or "difference" to specify which metric to plot. Defaults to "log_ratio".

    Returns:
        None: Displays the bar plot.
    """
    # Compute cell type percentages for both AnnData objects
    def compute_percentages(adata, celltype_col):
        cell_counts = adata.obs[celltype_col].value_counts(normalize=True) * 100
        return cell_counts
    
    percentages_adata1 = compute_percentages(adata1, celltype_col)
    percentages_adata2 = compute_percentages(adata2, celltype_col)
    
    # Align indices to ensure comparison
    all_celltypes = percentages_adata1.index.union(percentages_adata2.index)
    percentages_adata1 = percentages_adata1.reindex(all_celltypes, fill_value=0)
    percentages_adata2 = percentages_adata2.reindex(all_celltypes, fill_value=0)
    
    # Compute the differences and log-ratio
    differences = percentages_adata1 - percentages_adata2
    log_ratios = np.log2((percentages_adata1 + 1e-6) / (percentages_adata2 + 1e-6))  # Avoid division by zero
    
    # Choose the metric to plot
    if metric == "log_ratio":
        plot_values = log_ratios
        xlabel = "Log2(Percent in A / Percent in B)"
        title = "Log2 Ratio of Cell Type Percentages"
    elif metric == "difference":
        plot_values = differences
        xlabel = "Difference in Percentages (A - B)"
        title = "Difference in Cell Type Percentages"
    else:
        raise ValueError("Invalid metric. Choose either 'log_ratio' or 'difference'.")
    
    # Filter cell types by the threshold
    significant_celltypes = plot_values[abs(differences) > min_delta].index
    
    # Prepare data for plotting
    plot_data = plot_values[significant_celltypes].sort_values()
    
    # Determine colors for the bars (align with sorted data)
    if colors:
        bar_colors = [tuple(map(float, colors[cell_type])) if cell_type in colors else 'skyblue' for cell_type in plot_data.index]
    else:
        bar_colors = 'skyblue'
    
    # Debugging: Log the bar colors and sorted cell types
    # print("Sorted Cell Types:", plot_data.index.tolist())
    # print("Bar Colors:", bar_colors)
    
    # Create the horizontal bar plot
    plt.figure(figsize=(10, 6))
    plot_data.plot(kind='barh', color=bar_colors, edgecolor='black')
    # plt.axvline(0, color='gray', linestyle='--', linewidth=1)
    plt.axvline(0, color='black', linewidth=1)
    # plt.title(title)
    plt.xlabel(xlabel)
    # plt.ylabel("Cell Types")
    plt.tight_layout()
    plt.show()



def stackedbar_composition(
    adata: AnnData, 
    groupby: str, 
    obs_column='SCN_class', 
    labels=None, 
    bar_width: float = 0.75, 
    color_dict=None, 
    ax=None,
    order_by_similarity: bool = False,
    similarity_metric: str = 'correlation',
    include_legend: bool = True,
    legend_rows: int = 10
):
    """
    Plots a stacked bar chart of cell type proportions for a single AnnData object grouped by a specified column.
    
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
        ax (matplotlib.axes.Axes, optional): The axis to plot on. If not provided, a new figure and axis will be created.
        order_by_similarity (bool, optional): Whether to order the bars by similarity in composition. Defaults to False.
        similarity_metric (str, optional): The metric to use for similarity ordering. Defaults to 'correlation'.
        include_legend (bool, optional): Whether to include a legend in the plot. Defaults to True.
        legend_rows (int, optional): The number of rows in the legend. Defaults to 10.
    
    Returns:
        matplotlib.axes.Axes or None: The axes object if `ax` was provided, otherwise None (displays the plot).

    Raises:
        ValueError: If the length of `labels` does not match the number of unique groups.

    Examples:
        >>> stackedbar_composition(adata, groupby='sample', obs_column='your_column_name')
        >>> fig, ax = plt.subplots()
        >>> stackedbar_composition(adata, groupby='sample', obs_column='your_column_name', ax=ax, include_legend=False, legend_rows=5)
    """
    # Ensure the groupby column exists in .obs
    if groupby not in adata.obs.columns:
        raise ValueError(f"The groupby column '{groupby}' does not exist in the .obs attribute.")

    # Check if groupby column is categorical or not
    if pd.api.types.is_categorical_dtype(adata.obs[groupby]):
        unique_groups = adata.obs[groupby].cat.categories.to_list()
    else:
        unique_groups = adata.obs[groupby].unique().tolist()

    # Extract unique groups and ensure labels are provided or create default ones
    if labels is None:
        labels = unique_groups
    elif len(labels) != len(unique_groups):
        raise ValueError("Length of 'labels' must match the number of unique groups.")

    if color_dict is None:
        color_dict = adata.uns.get('SCN_class_colors', {})
    
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
    
    # Ordering groups by similarity if requested
    if order_by_similarity:
        dist_matrix = pdist(proportions.T, metric=similarity_metric)
        linkage_matrix = linkage(dist_matrix, method='average')
        order = leaves_list(linkage_matrix)
        proportions = proportions[:, order]
        unique_groups = [unique_groups[i] for i in order]
        labels = [labels[i] for i in order]
    
    # Plotting
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    
    bottom = np.zeros(len(unique_groups))
    for i, category in enumerate(categories):
        color = color_dict.get(category, None)
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
    ax.set_ylabel('Proportion')
    ax.set_title(f'{obs_column} proportions by {groupby}')
    
    if include_legend:
        num_columns = int(np.ceil(len(categories) / legend_rows))
        ax.legend(title='Classes', bbox_to_anchor=(1.05, 1), loc='upper left', ncol=num_columns)
    
    if ax is None:
        plt.tight_layout()
        plt.show()
    else:
        return ax




def stackedbar_composition3(
    adata: AnnData, 
    groupby: str, 
    obs_column='SCN_class', 
    labels=None, 
    bar_width: float = 0.75, 
    color_dict=None, 
    ax=None,
    order_by_similarity: bool = False,
    similarity_metric: str = 'correlation',
    include_legend: bool = True
):
    """
    Plots a stacked bar chart of cell type proportions for a single AnnData object grouped by a specified column.
    
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
        ax (matplotlib.axes.Axes, optional): The axis to plot on. If not provided, a new figure and axis will be created.
        order_by_similarity (bool, optional): Whether to order the bars by similarity in composition. Defaults to False.
        similarity_metric (str, optional): The metric to use for similarity ordering. Defaults to 'correlation'.
        include_legend (bool, optional): Whether to include a legend in the plot. Defaults to True.
    
    Returns:
        matplotlib.axes.Axes or None: The axes object if `ax` was provided, otherwise None (displays the plot).

    Raises:
        ValueError: If the length of `labels` does not match the number of unique groups.

    Examples:
        >>> stackedbar_composition(adata, groupby='sample', obs_column='your_column_name')
        >>> fig, ax = plt.subplots()
        >>> stackedbar_composition(adata, groupby='sample', obs_column='your_column_name', ax=ax, include_legend=False)
    """
    # Ensure the groupby column exists in .obs
    if groupby not in adata.obs.columns:
        raise ValueError(f"The groupby column '{groupby}' does not exist in the .obs attribute.")

    # Check if groupby column is categorical or not
    if pd.api.types.is_categorical_dtype(adata.obs[groupby]):
        unique_groups = adata.obs[groupby].cat.categories.to_list()
    else:
        unique_groups = adata.obs[groupby].unique().tolist()

    # Extract unique groups and ensure labels are provided or create default ones
    if labels is None:
        labels = unique_groups
    elif len(labels) != len(unique_groups):
        raise ValueError("Length of 'labels' must match the number of unique groups.")

    if color_dict is None:
        color_dict = adata.uns.get('SCN_class_colors', {})
    
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
    
    # Ordering groups by similarity if requested
    if order_by_similarity:
        dist_matrix = pdist(proportions.T, metric=similarity_metric)
        linkage_matrix = linkage(dist_matrix, method='average')
        order = leaves_list(linkage_matrix)
        proportions = proportions[:, order]
        unique_groups = [unique_groups[i] for i in order]
        labels = [labels[i] for i in order]
    
    # Plotting
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    
    bottom = np.zeros(len(unique_groups))
    for i, category in enumerate(categories):
        color = color_dict.get(category, None)
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
    ax.set_ylabel('Proportion')
    ax.set_title(f'{obs_column} proportions by {groupby}')
    
    if include_legend:
        ax.legend(title='Classes', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    if ax is None:
        plt.tight_layout()
        plt.show()
    else:
        return ax




def stackedbar_composition_list(
    adata_list, 
    obs_column = 'SCN_class', 
    labels = None, 
    bar_width: float = 0.75, 
    color_dict = None,
    legend_loc = "outside center right"
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

    Returns:
        matplotlib.figure.Figure: The generated figure object.

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
    #### fig, ax = plt.subplots()
    fig, ax = plt.subplots(constrained_layout=True)
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
    ax.set_ylabel('Proportion')
    ax.set_title(f'{obs_column} proportions')
    # ax.legend(title='Classes', bbox_to_anchor=(1.05, 1), loc='upper left')
    ## legend = fig.legend(title='Classes', loc="outside right upper", frameon=False)#, bbox_to_anchor=(1.05, 1), loc='upper left')
    ## legend_height = legend.get_window_extent().height / fig.dpi  # in inches

    # Add legend
    legend_handles = [mpatches.Patch(color=color, label=label) for label, color in color_dict.items()]
    # legend = ax.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
    ##### legend = fig.legend(title='Classes', loc="outside right upper", frameon=False)
    fig.legend(handles=legend_handles, loc=legend_loc, frameon=False)
    ##### legend_height = legend.get_window_extent().height / fig.dpi  # in inches

    # fig_height = fig.get_size_inches()[1]  # current height in inches
    #### fig.set_size_inches(fig.get_size_inches()[0], legend_height )
    # plt.tight_layout()
    # plt.show()
    return fig



def stackedbar_categories(
    adata: AnnData,
    scn_classes_to_display = None, 
    bar_height=0.8,
    color_dict = None,
    class_col_name = 'SCN_class_argmax',
    category_col_name = 'SCN_class_type',
    title = None,
    show_pct_total = False,
    legend_loc = "best"
):
    """Plot horizontal stacked bar chart of SCN classification categories per cell type.

    Args:
        adata (AnnData): An AnnData object containing SCN classification results.
        scn_classes_to_display (list, optional): Subset of SCN classes to include. Defaults to None (all classes).
        bar_height (float, optional): Height of the horizontal bars. Defaults to 0.8.
        color_dict (dict, optional): Dictionary mapping category names to colors. Defaults to None (uses SCN_CATEGORY_COLOR_DICT).
        class_col_name (str, optional): Column name in `.obs` for the cell type labels. Defaults to 'SCN_class_argmax'.
        category_col_name (str, optional): Column name in `.obs` for the SCN category labels. Defaults to 'SCN_class_type'.
        title (str, optional): Title for the plot. Defaults to None ('Cell typing categorization').
        show_pct_total (bool, optional): Whether to display count and percentage text inside bars. Defaults to False.
        legend_loc (str, optional): Location of the legend. Defaults to "best".

    Returns:
        matplotlib.figure.Figure: The generated figure object.
    """
    # Copy the obs DataFrame to avoid modifying the original data
    df = adata.obs.copy()
    
    # Ensure the necessary columns exist
    if class_col_name not in df.columns or category_col_name not in df.columns:
        raise KeyError(f"Columns '{class_col_name}' and '{category_col_name}' must be present in adata.obs")
    
    # Convert specified columns to categorical
    df[class_col_name] = df[class_col_name].astype('category')
    df[category_col_name] = df[category_col_name].astype('category')

    #df[class_col_name] = df[class_col_name].cat.set_categories(df[class_col_name].cat.categories)
    #df[category_col_name] = df[category_col_name].cat.set_categories(df[category_col_name].cat.categories)

    # Group by cell class and count occurrences of each category
    try:
        counts = df.groupby(class_col_name)[category_col_name].value_counts().unstack().fillna(0)
    except Exception as e:
        print("Error during groupby and value_counts operations:", e)
        return
    
    # Calculate proportions and total counts
    proportions = counts.divide(counts.sum(axis=1), axis=0)
    total_counts = counts.sum(axis=1)
    total_percent = (total_counts / total_counts.sum() * 100).round(1)  # Converts to percentage and round

    all_classes = df[class_col_name].unique()
    if scn_classes_to_display is not None:
        if not all(cls in all_classes for cls in scn_classes_to_display):
            raise ValueError("Some values in 'scn_classes_to_display' do not match available 'SCN_class' values in the provided DataFrames.")
        all_classes = scn_classes_to_display
        proportions = proportions.loc[all_classes]
        total_counts = total_counts[all_classes]
        total_percent = total_percent[all_classes]


    # setup colors
    if color_dict is None:
        color_dict = SCN_CATEGORY_COLOR_DICT

    # specify category order per bar and filter
    cat_order = list(color_dict.keys())
    existing_order = [cat for cat in cat_order if cat in proportions.columns]
    proportions = proportions[existing_order]

    fig, ax = plt.subplots()
    proportions.plot(kind='barh', stacked=True, color=color_dict, width=bar_height, ax=ax)    

    # Set axis labels and title
    if title is None:
        title = 'Cell typing categorization'

    ax.set_title(title)
    ax.set_xlabel('Proportion')
    ax.set_ylabel('Cell Group')

    if show_pct_total:    
        # Adding adjusted internal total counts within each bar
        num_classes = len(all_classes)
        text_size = max(min(12 - num_classes // 2, 10), 7) # Adjust text size
        for i, (count, percent) in enumerate(zip(total_counts, total_percent)):
            text = f'{int(count)} ({percent}%)'  # Text to display
            ax.text(0.95, i, text, ha='right', va='center', color='white', fontsize=text_size)

    # Add legend
    legend_handles = [mpatches.Patch(color=color, label=label) for label, color in color_dict.items()]
    ax.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc=legend_loc)

    fig.tight_layout()
    return fig


def stackedbar_categories_list(
    ads, 
    titles=None,
    scn_classes_to_display=None,
    bar_height=0.8,
    bar_groups_obsname='SCN_class_argmax',
    bar_subgroups_obsname='SCN_class_type',
    ncell_min=None,
    color_dict=None,
    show_pct_total = False,
    legend_loc = "outside center right"
):
    """Plot side-by-side horizontal stacked bar charts of SCN categories for multiple AnnData objects.

    Args:
        ads (list[AnnData]): List of AnnData objects to plot.
        titles (list[str], optional): Titles for each subplot. Defaults to None ('SCN Class Proportions' for each).
        scn_classes_to_display (list, optional): Subset of SCN classes to include. Defaults to None (all classes).
        bar_height (float, optional): Height of the horizontal bars. Defaults to 0.8.
        bar_groups_obsname (str, optional): Column name in `.obs` for the cell type groups. Defaults to 'SCN_class_argmax'.
        bar_subgroups_obsname (str, optional): Column name in `.obs` for the SCN category subgroups. Defaults to 'SCN_class_type'.
        ncell_min (int, optional): Minimum number of cells required to display a class. Defaults to None.
        color_dict (dict, optional): Dictionary mapping category names to colors. Defaults to None (uses SCN_CATEGORY_COLOR_DICT).
        show_pct_total (bool, optional): Whether to display count and percentage text inside bars. Defaults to False.
        legend_loc (str, optional): Location of the legend. Defaults to "outside center right".

    Returns:
        matplotlib.figure.Figure: The generated figure object.
    """
    dfs = [adata.obs for adata in ads]
    num_dfs = len(dfs)
    
    # Determine the titles for each subplot
    if titles is None:
        titles = ['SCN Class Proportions'] * num_dfs
    elif len(titles) != num_dfs:
        raise ValueError("The length of 'titles' must match the number of annDatas.")
    
    # Determine the SCN classes to display
    all_classes = set().union(*(df[bar_groups_obsname].unique() for df in dfs))
    if scn_classes_to_display is not None:
        if not all(cls in all_classes for cls in scn_classes_to_display):
            raise ValueError("Some values in 'scn_classes_to_display' do not match available 'SCN_class' values in the provided DataFrames.")
        else:
            classes_to_display = scn_classes_to_display
    else:
        classes_to_display = all_classes
    
    # Set up colors
    if color_dict is None:
        color_dict = SCN_CATEGORY_COLOR_DICT

    # specify category order per bar
    cat_order = list(color_dict.keys())

    all_categories = set().union(*(df[bar_subgroups_obsname].unique() for df in dfs))
    
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(1, num_dfs, figsize=(6 * num_dfs, 8), sharey=True)
    for ax, df, title in zip(axes, dfs, titles):

        tmp_classes_to_display = list(classes_to_display)  # Copy to prevent mutation
        
        # Ensure SCN_class and SCN_class_type categories are consistent
        df[bar_groups_obsname] = df[bar_groups_obsname].astype('category')
        df[bar_subgroups_obsname] = df[bar_subgroups_obsname].astype('category')

        # Reindex and filter each DataFrame to ensure all SCN_class values are present
        counts = df.groupby(bar_groups_obsname)[bar_subgroups_obsname].value_counts().unstack().fillna(0)
        counts = counts.reindex(classes_to_display, fill_value=0)  # Handle missing SCN_class categories
        
        # Calculate total counts and proportions
        total_counts = counts.sum(axis=1)
        total_percent = (total_counts / total_counts.sum() * 100).round(1)  # Converts to percentage and round
        proportions = counts.divide(total_counts, axis=0).fillna(0).replace([np.inf, -np.inf], 0)
        
        # Filter by ncell_min if provided
        if ncell_min is not None:
            passing_classes = total_counts[total_counts >= ncell_min].index.to_list()
            tmp_classes_to_display = list(set(passing_classes) & set(tmp_classes_to_display))    

        # Filter proportions and counts to display the relevant SCN classes
        proportions = proportions.loc[tmp_classes_to_display]
        total_counts = total_counts[tmp_classes_to_display]
        total_percent = total_percent[tmp_classes_to_display]

        # specify category order per bar
        # proportions = proportions[cat_order]
        existing_order = [cat for cat in cat_order if cat in proportions.columns]
        proportions = proportions[existing_order]

        # Plot the proportions as a stacked bar chart
        proportions.plot(kind='barh', stacked=True, color=color_dict, width=bar_height, ax=ax, legend=False)
        
        if show_pct_total:
            # Add text with counts and percentages inside the bars
            text_size = max(min(12 - len(classes_to_display) // 2, 10), 7)  # Adjust text size
            for i, (count, percent) in enumerate(zip(total_counts, total_percent)):
                text = f'{int(count)} ({percent}%)'  # Text to display
                ax.text(0.95, i, text, ha='right', va='center', color='white', fontsize=text_size)

        ax.set_title(title)

    # Setting the y-label for the first subplot only
    axes[0].set_ylabel('Cell group')
    
    # Adding the legend after the last subplot
    legend_handles = [mpatches.Patch(color=color, label=label) for label, color in color_dict.items()]
    fig.legend(handles=legend_handles, loc=legend_loc, frameon=False)
    # fig.legend(handles=legend_handles, frameon=False)

    return fig



def bar_classifier_f1(adata: AnnData, ground_truth: str = "celltype", class_prediction: str = "SCN_class", bar_height=0.8):
    """
    Plots a bar graph of F1 scores per class based on ground truth and predicted classifications.

    Args:
        adata (AnnData): Annotated data matrix.
        ground_truth (str, optional): The column name in `adata.obs` containing the true class labels. Defaults to "celltype".
        class_prediction (str, optional): The column name in `adata.obs` containing the predicted class labels. Defaults to "SCN_class".

    Returns:
        None
    """
    # Calculate F1 scores
    fscore = f1_score(
        adata.obs[ground_truth], 
        adata.obs[class_prediction], 
        average=None, 
        labels=adata.obs[ground_truth].cat.categories
    )
    
    # Get category names
    cates = list(adata.obs[ground_truth].cat.categories)
    
    # Create a DataFrame for F1 scores
    f1_scores_df = pd.DataFrame({
        'Class': cates,
        'F1-Score': fscore,
        'Count': adata.obs[ground_truth].value_counts().reindex(cates).values
    })
    
    # Get colors from the .uns dictionary
    color_key = 'SCN_class_colors' if 'SCN_class_colors' in adata.uns else 'SCN_class_argmax_colors'
    f1_scores_df['Color'] = f1_scores_df['Class'].map(adata.uns[color_key])

    plt.rcParams['figure.constrained_layout.use'] = True
    # sns.set_theme(style="whitegrid")
    
    # fig, ax = plt.subplots(layout="constrained")
    fig, ax = plt.subplots()

    text_size = max(min(12 - len(cates) // 2, 10), 7) # Adjust text size
    # Plot the F1 scores with colors
    ax = f1_scores_df.plot.barh(
        x='Class', 
        y='F1-Score', 
        color=f1_scores_df['Color'], 
        legend=False,
        width=bar_height
    )
    
    ax.set_xlabel('F1-Score')
    ax.set_title('F1-Scores per Class')
    ax.set_xlim(0, 1.1)  # Set x-axis limits to ensure visibility of all bars

    # Add the number of observations per class as text within the barplot
    for i, (count, fscore) in enumerate(zip(f1_scores_df['Count'], f1_scores_df['F1-Score'])):
        ax.text(0.03, i, f"n = {count}", ha='left', va='center', color='white' if fscore >= 0.20 else 'black', fontsize=text_size)

    # plt.show()
    return fig


