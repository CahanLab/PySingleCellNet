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
# from palettable.colorbrewer.qualitative import Set2_6
# from palettable.tableau import GreenOrange_6
# from palettable.cartocolors.qualitative import Safe_6
# from palettable.cartocolors.qualitative import Vivid_4
# from palettable.cartocolors.qualitative import Vivid_6
# from palettable.cartocolors.qualitative import Vivid_10
# from palettable.scientific.diverging import Roma_20
from palettable.scientific.sequential import LaJolla_20
from palettable.scientific.sequential import Batlow_20
from anndata import AnnData
from scipy.sparse import csr_matrix
from scipy import sparse
from typing import Optional, Callable
from ..utils import *



def umi_counts_ranked(adata, total_counts_column="total_counts"):
    """
    Identifies and plors the knee point of the UMI count distribution in an AnnData object.

    Parameters:
        adata (AnnData): The input AnnData object.
        total_counts_column (str): Column in `adata.obs` containing total UMI counts. Default is "total_counts".
        show (bool): If True, displays a log-log plot with the knee point. Default is True.

    Returns:
        float: The UMI count value at the knee point.
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

def dotplot_deg(
    adata: AnnData,
    diff_gene_dict: dict,
    #samples_obsvals: list = [],
    groupby_obsname: str = "comb_sampname", # I think that this is the column that splits cells for DEG
    cellgrp_obsname: str = "comb_cellgrp",    # I think that this is the column that splits into discint clusters or cell types
    cellgrp_obsvals: list = [], # this is the subset of clusters or cell types to limit this visualization to
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
    

def dotplot_scn_scores(
    adata: AnnData,
    groupby: str,
    expression_cutoff = 0.1,
    obsm_name = 'SCN_score'
):    
    adTemp = AnnData(adata.obsm[obsm_name], obs=adata.obs)
    adTemp.obs[groupby] = adata.obs[groupby]
    sc.pl.dotplot(adTemp, adTemp.var_names.values, groupby=groupby, expression_cutoff=expression_cutoff, cmap=Batlow_20.mpl_colormap, colorbar_title="SCN score")

def umap_scores_old(
    adata: AnnData,
    scn_classes: list,
    obsm_name = 'SCN_score'
):    
    adTemp = AnnData(adata.obsm[obsm_name], obs=adata.obs)
    adTemp.obsm['X_umap'] = adata.obsm['X_umap'].copy()
    sc.pl.umap(adTemp,color=scn_classes, alpha=.75, s=10, vmin=0, vmax=1)


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

