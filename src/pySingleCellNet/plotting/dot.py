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
from palettable.scientific.sequential import LaJolla_20, Batlow_20
from anndata import AnnData
from scipy.sparse import csr_matrix
from scipy import sparse
from typing import Optional, Callable
from ..utils import *

def umi_counts_ranked(adata, total_counts_column="total_counts"):
    """Identify and plot the knee point of the UMI count distribution in an AnnData object.

    Args:
        adata (AnnData): The input AnnData object.
        total_counts_column (str): Column in `adata.obs` containing total UMI counts. Defaults to "total_counts".

    Returns:
        None: Displays a log-log plot with the knee point.
    """
    # Extract total UMI counts
    umi_counts = adata.obs[total_counts_column]
    
    # Sort UMI counts in descending order
    sorted_umi_counts = np.sort(umi_counts)[::-1]
    
    # Compute cumulative UMI counts (normalized to a fraction)
    cumulative_counts = np.cumsum(sorted_umi_counts)
    cumulative_fraction = cumulative_counts / cumulative_counts[-1]
    
    # Compute derivatives to identify the knee point
    first_derivative = np.gradient(cumulative_fraction)
    second_derivative = np.gradient(first_derivative)
    
    # Find the index of the maximum curvature (knee point)
    knee_idx = np.argmax(second_derivative)
    knee_point_value = sorted_umi_counts[knee_idx]
    
    # Generate log-log plot
    cell_ranks = np.arange(1, len(sorted_umi_counts) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(cell_ranks, sorted_umi_counts, marker='o', markersize=2, linestyle='-', linewidth=0.5, label="UMI Counts")
    plt.axvline(cell_ranks[knee_idx], color="red", linestyle="--", label=f"Knee Point: {knee_point_value}")
    plt.title('UMI Counts Per Cell (Log-Log Scale)', fontsize=14)
    plt.xlabel('Cell Rank (Descending)', fontsize=12)
    plt.ylabel('Total UMI Counts', fontsize=12)
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()
    

def rela_graph(gra, color_dict):
    """Plot a PAGA-derived relationship graph with colored vertices.

    Renders an igraph graph using the Fruchterman-Reingold layout with vertex colors
    from the provided color dictionary and vertex sizes scaled by cell count.

    Args:
        gra (igraph.Graph): An igraph Graph object with vertex attributes 'name' and 'ncells'.
        color_dict (dict): Dictionary mapping vertex names to RGB color tuples.

    Returns:
        None: Displays the graph plot.
    """
    ig.config['plotting.backend'] = 'matplotlib'
    v_style = {}
    v_style["layout"] = "fruchterman_reingold"
    v_style["vertex_label_dist"] = 2.5
    v_style["vertex_label_angle"] = 3 # in radians
    v_style["bbox"] = (600,600)
    v_style["margin"] = (50)

    for vertex in gra.vs:
        # vertex["color"] = convert_color(color_dict.get(vertex["name"], np.array([0.5, 0.5, 0.5])))
        vertex["color"] = tuple(color_dict.get(vertex["name"], np.array([0.5, 0.5, 0.5])))  

    # Normalize node sizes for better visualization
    max_size = 50  # Maximum size for visualization
    min_size = 10  # Minimum size for visualization
    ncells = gra.vs["ncells"]
    node_sizes = [min_size + (size / max(ncells)) * (max_size - min_size) for size in ncells]
    fig, ax = plt.subplots()
    ig.plot(gra, **v_style, vertex_size=node_sizes)
    plt.show()



def ontogeny_graph(gra, color_dict):
    """Plot an ontogeny relationship graph with colored vertices.

    Renders an igraph graph using the Fruchterman-Reingold layout with vertex colors
    from the provided color dictionary and vertex sizes scaled by cell count.

    Args:
        gra (igraph.Graph): An igraph Graph object with vertex attributes 'name' and 'ncells'.
        color_dict (dict): Dictionary mapping vertex names to RGB color tuples.

    Returns:
        None: Displays the graph plot.
    """
    ig.config['plotting.backend'] = 'matplotlib'
    v_style = {}
    v_style["layout"] = "fruchterman_reingold"
    v_style["vertex_label_dist"] = 2.5
    v_style["vertex_label_angle"] = 3 # in radians
    v_style["bbox"] = (600,600)
    v_style["margin"] = (50)

    for vertex in gra.vs:
        # vertex["color"] = convert_color(color_dict.get(vertex["name"], np.array([0.5, 0.5, 0.5])))
        vertex["color"] = tuple(color_dict.get(vertex["name"], np.array([0.5, 0.5, 0.5])))  

    # Normalize node sizes for better visualization
    max_size = 50  # Maximum size for visualization
    min_size = 10  # Minimum size for visualization
    ncells = gra.vs["ncells"]
    node_sizes = [min_size + (size / max(ncells)) * (max_size - min_size) for size in ncells]
    fig, ax = plt.subplots()
    ig.plot(gra, **v_style, vertex_size=node_sizes)
    plt.show()




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
    """Create a DotPlot of differentially expressed genes across cell type/category combinations.

    Builds a combined 'ct_by_cat' grouping from cell type and category columns, selects
    top differentially expressed genes per group from the provided dictionary, and renders
    a scanpy DotPlot with swapped axes.

    Args:
        adata (AnnData): An AnnData object with SCN classification results.
        diff_gene_dict (dict): Dictionary containing 'geneTab_dict' (mapping cell types to
            gene tables) and 'category_names' (list of category names).
        num_genes (int, optional): Number of top genes to display per group. Defaults to 10.
        celltype_groupby (str, optional): Column in `.obs` for cell type grouping. Defaults to "SCN_class".
        category_groupby (str, optional): Column in `.obs` for category grouping. Defaults to "SCN_class_type".
        category_names (list[str], optional): Category names to include. Defaults to ["None", "Singular"].
        celltype_names (list, optional): Subset of cell types to include. Defaults to [] (all in diff_gene_dict).
        order_by (str, optional): Column name to order genes by in the gene tables. Defaults to 'scores'.

    Returns:
        sc.pl.DotPlot: A scanpy DotPlot object (call .show() or .render() to display).
    """
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
    

def dotplot_scn_scores(
    adata: AnnData,
    groupby: str,
    expression_cutoff = 0.1,
    obsm_name = 'SCN_score'
):
    """Create a dot plot of SCN classification scores grouped by a specified column.

    Constructs a temporary AnnData from the SCN score matrix and renders a scanpy
    dot plot colored by score intensity.

    Args:
        adata (AnnData): An AnnData object with SCN scores stored in `.obsm`.
        groupby (str): Column name in `.obs` to group cells by.
        expression_cutoff (float, optional): Minimum score threshold for dot display. Defaults to 0.1.
        obsm_name (str, optional): Key in `.obsm` containing the SCN score matrix. Defaults to 'SCN_score'.

    Returns:
        None: Displays the dot plot.
    """
    adTemp = AnnData(adata.obsm[obsm_name], obs=adata.obs)
    adTemp.obs[groupby] = adata.obs[groupby]
    sc.pl.dotplot(adTemp, adTemp.var_names.values, groupby=groupby, expression_cutoff=expression_cutoff, cmap=Batlow_20.mpl_colormap, colorbar_title="SCN score")


def umap_scores(
    adata: AnnData,
    scn_classes: list,
    obsm_name='SCN_score',
    alpha=0.75,
    s=10,
    display=True
):
    """
    Plots UMAP projections of scRNA-seq data with specified scores.

    Args:
        adata (AnnData): 
            The AnnData object containing the scRNA-seq data.
        scn_classes (list): 
            A list of SCN classes to visualize on the UMAP.
        obsm_name (str, optional): 
            The name of the obsm key containing the SCN scores. Defaults to 'SCN_score'.
        alpha (float, optional): 
            The transparency level of the points on the UMAP plot. Defaults to 0.75.
        s (int, optional): 
            The size of the points on the UMAP plot. Defaults to 10.
        display (bool, optional): 
            If True, the plot is displayed immediately. If False, the axis object is returned. Defaults to True.

    Returns:
        matplotlib.axes.Axes or None: 
            If `display` is False, returns the matplotlib axes object. Otherwise, returns None.
    """
    # Create a temporary AnnData object with the desired obsm
    adTemp = AnnData(adata.obsm[obsm_name], obs=adata.obs)
    adTemp.obsm['X_umap'] = adata.obsm['X_umap'].copy()
    
    # Create the UMAP plot
    ax = sc.pl.umap(adTemp, color=scn_classes, alpha=alpha, s=s, vmin=0, vmax=1, show=False)
    
    # Display or return the axis
    if display:
        plt.show()
    else:
        return ax

