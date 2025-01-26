import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from anndata import AnnData
# from sklearn.ensemble import RandomForestClassifier
# from scipy.sparse import csr_matrix
import warnings
# from .utils import *
# from .tsp_rf import *
import gseapy as gp
import os
# import anndata
# import pySingleCellNet as pySCN
import copy
import igraph as ig
from collections import defaultdict


def convert_diffExp_to_dict(
    adata,
    uns_name: str = 'rank_genes_groups'
):
    """Convert differential expression results from AnnData into a dictionary of DataFrames.

    This function extracts differential expression results stored in `adata.uns[uns_name]` 
    using Scanpy's `get.rank_genes_groups_df`, cleans the data, and organizes it into 
    a dictionary where each key corresponds to a group and each value is a DataFrame 
    of differential expression results for that group.

    Args:
        adata (AnnData): Annotated data matrix containing differential expression results 
            in `adata.uns`.
        uns_name (str, optional): Key in `adata.uns` where rank_genes_groups results 
            are stored. Defaults to 'rank_genes_groups'.

    Returns:
        dict: Dictionary mapping each group to a DataFrame of its differential 
        expression results, with rows corresponding to genes and relevant statistics 
        for each gene.
    """
    import scanpy as sc  # Ensure Scanpy is imported
    tempTab = sc.get.rank_genes_groups_df(adata, group=None, key=uns_name)
    tempTab = tempTab.dropna()
    groups = tempTab['group'].cat.categories.to_list()

    ans = {}
    for g in groups:
        ans[g] = tempTab[tempTab['group'] == g].copy()
    return ans


def gsea_on_deg(
    deg_res: dict,
    genesets_name: str,
    genesets: dict,
    permutation_num: int = 100,
    threads: int = 4,
    seed: int = 3,
    min_size: int = 10,
    max_size: int = 500
) -> dict:
    """Perform Gene Set Enrichment Analysis (GSEA) on differential expression results.

    Applies GSEA using `gseapy.prerank` for each group in the differential 
    expression results dictionary against provided gene sets.

    Args:
        deg_res (dict): Dictionary mapping cell group names to DataFrames 
            of differential expression results. Each DataFrame must contain 
            columns 'names' (gene names) and 'scores' (ranking scores).
        genesets_name (str): Name of the gene set collection (not actively used).
        genesets (dict): Dictionary of gene sets where keys are gene set 
            names and values are lists of genes.
        permutation_num (int, optional): Number of permutations for GSEA. 
            Defaults to 100.
        threads (int, optional): Number of parallel threads to use. Defaults to 4.
        seed (int, optional): Random seed for reproducibility. Defaults to 3.
        min_size (int, optional): Minimum gene set size to consider. Defaults to 10.
        max_size (int, optional): Maximum gene set size to consider. Defaults to 500.

    Returns:
        dict: Dictionary where keys are cell group names and values are 
            GSEA result objects returned by `gseapy.prerank`.

    Example:
        >>> deg_results = {
        ...     'Cluster1': pd.DataFrame({'names': ['GeneA', 'GeneB'], 'scores': [2.5, -1.3]}),
        ...     'Cluster2': pd.DataFrame({'names': ['GeneC', 'GeneD'], 'scores': [1.2, -2.1]})
        ... }
        >>> gene_sets = {'Pathway1': ['GeneA', 'GeneC'], 'Pathway2': ['GeneB', 'GeneD']}
        >>> results = gsea_on_deg(deg_results, 'ExampleGeneSets', gene_sets)
    """
    ans = dict()
    diff_gene_tables = deg_res
    cellgrp_vals = list(diff_gene_tables.keys())
    for cellgrp in cellgrp_vals:
        atab = diff_gene_tables[cellgrp]
        atab = atab[['names', 'scores']]
        atab.columns = ['0', '1']
        pre_res = gp.prerank(
            rnk=atab,
            gene_sets=genesets,
            permutation_num=permutation_num,
            ascending=False,
            threads=threads,
            no_plot=True,
            seed=seed,
            min_size=min_size,
            max_size=max_size
        )
        ans[cellgrp] = pre_res
    return ans


def collect_gsea_results_from_dict(
    gsea_dict2: dict,
    fdr_thr=0.25
):
    """Collect and filter GSEA results from a dictionary of GSEA objects.

    Aggregates normalized enrichment scores (NES) for each cell type 
    across all gene sets, applying an FDR threshold to filter out 
    insignificant results.

    Args:
        gsea_dict2 (dict): Dictionary mapping cell types to GSEA result objects.
        fdr_thr (float, optional): FDR threshold above which NES values are set to 0. 
            Defaults to 0.25.

    Returns:
        pd.DataFrame: DataFrame with gene sets as rows and cell types as columns 
            containing filtered NES values.
    """
    import copy  # Ensure copy is imported if not already
    gsea_dict = copy.deepcopy(gsea_dict2)
    pathways = pd.Index([])
    cell_types = list(gsea_dict.keys())

    for cell_type in cell_types:
        tmpRes = gsea_dict[cell_type].res2d
        gene_set_names = list(tmpRes['Term'])
        pathways = pathways.union(gene_set_names)
        
    nes_df = pd.DataFrame(0, columns=cell_types, index=pathways)

    for cell_type in cell_types:
        ct_df = gsea_dict[cell_type].res2d
        ct_df.index = ct_df['Term']
        ct_df.loc[lambda df: df['FDR q-val'] > fdr_thr, "NES"] = 0
        nes_df[cell_type] = ct_df["NES"]

    nes_df = nes_df.apply(pd.to_numeric, errors='coerce')
    return nes_df

    
