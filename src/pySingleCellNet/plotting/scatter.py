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

def scatter_qc_adata(adata, title_suffix=""):
    """
    Creates a figure with two scatter plot panels for visualizing data from an AnnData object.

    The first panel shows 'total_counts' vs 'n_genes_by_counts', colored by 'pct_counts_mt'.
    The second panel shows 'n_genes_by_counts' vs 'pct_counts_mt'. An optional title suffix
    can be added to customize the axis titles.

    Args:
        adata (AnnData): The AnnData object containing the dataset.
                         Must contain 'total_counts', 'n_genes_by_counts', and 'pct_counts_mt' in `adata.obs`.
        title_suffix (str, optional): A string to append to the axis titles, useful for specifying
                                      experimental conditions (e.g., "C11 day 2"). Defaults to an empty string.

    Returns:
        None: The function displays a matplotlib figure with two scatter plots.

    Example:
        >>> plot_scatter_with_contours(adata, title_suffix="C11 day 2")
    """
    
    # Extract necessary columns from the adata object
    total_counts = adata.obs['total_counts']
    n_genes_by_counts = adata.obs['n_genes_by_counts']
    pct_counts_mt = adata.obs['pct_counts_mt']

    # Create a figure with two subplots (1 row, 2 columns)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # First subplot: total_counts vs n_genes_by_counts, colored by pct_counts_mt
    scatter1 = axes[0].scatter(total_counts, n_genes_by_counts, c=pct_counts_mt, cmap='viridis', alpha=0.5, s=1)
    axes[0].set_xlabel(f'Total Counts ({title_suffix})')
    axes[0].set_ylabel(f'Number of Genes by Counts ({title_suffix})')
    axes[0].set_title(f'Total Counts vs Genes ({title_suffix})')
    # Add a colorbar
    fig.colorbar(scatter1, ax=axes[0], label='% Mito')

    # Second subplot: n_genes_by_counts vs pct_counts_mt
    scatter2 = axes[1].scatter(n_genes_by_counts, pct_counts_mt, alpha=0.5, s=1)
    axes[1].set_xlabel(f'Number of Genes by Counts ({title_suffix})')
    axes[1].set_ylabel(f'% Mito ({title_suffix})')
    axes[1].set_title(f'Genes vs % Mito ({title_suffix})')

    # Adjust layout to avoid overlap
    plt.tight_layout()

    # Show the plot
    plt.show()










def scatter_qc_adata(adata, title_suffix=""):
    # Extract necessary columns from the adata object
    total_counts = adata.obs['total_counts']
    n_genes_by_counts = adata.obs['n_genes_by_counts']
    pct_counts_mt = adata.obs['pct_counts_mt']

    # Create a figure with two subplots (1 row, 2 columns)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # First subplot: total_counts vs n_genes_by_counts, colored by pct_counts_mt
    scatter1 = axes[0].scatter(total_counts, n_genes_by_counts, c=pct_counts_mt, cmap='viridis', alpha=0.5, s=1)
    axes[0].set_xlabel(f'Total Counts ({title_suffix})')
    axes[0].set_ylabel(f'Number of Genes by Counts ({title_suffix})')
    axes[0].set_title(f'Total Counts vs Genes ({title_suffix})')
    # Add a colorbar
    fig.colorbar(scatter1, ax=axes[0], label='% Mito')

    # Second subplot: n_genes_by_counts vs pct_counts_mt
    scatter2 = axes[1].scatter(n_genes_by_counts, pct_counts_mt, alpha=0.5, s=1)
    axes[1].set_xlabel(f'Number of Genes by Counts ({title_suffix})')
    axes[1].set_ylabel(f'% Mito ({title_suffix})')
    axes[1].set_title(f'Genes vs % Mito ({title_suffix})')

    # Adjust layout to avoid overlap
    plt.tight_layout()

    # Show the plot
    plt.show()


