import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from anndata import AnnData
import warnings
import gseapy as gp
import os
import copy
import igraph as ig
from collections import defaultdict

def deg(
    adata: AnnData,
    sample_obsvals: list = [],  # Impacts the sign of the test statistic
    limitto_obsvals: list = [],  # Specifies which cell groups to test; if empty, tests all
    cellgrp_obsname: str = 'comb_cellgrp',  # .obs column name holding the cell sub-groups to iterate over
    groupby_obsname: str = 'comb_sampname',  # .obs column name to group by for differential expression
    ncells_per_sample: int = 30,  # Minimum number of cells per sample required to perform the test
    test_name: str = 't-test',  # Name of the statistical test to use
    mask_var: str = 'highly_variable'
) -> dict:
    """
    Perform differential expression analysis on an AnnData object across specified cell groups and samples.

    This function iterates over specified or all cell groups within the `adata` object and performs
    differential expression analysis using the specified statistical test (e.g., t-test). It filters
    groups based on the minimum number of cells per sample and returns the results in a structured dictionary.

    Args:
        adata (AnnData): The annotated data matrix containing observations and variables.
        sample_obsvals (list, optional): List of sample observation values to include. Defaults to an empty list.
            Impacts the sign of the test statistic.
        limitto_obsvals (list, optional): List of cell group observation values to limit the analysis to.
            If empty, all cell groups in `adata` are tested. Defaults to an empty list.
        cellgrp_obsname (str, optional): The `.obs` column name in `adata` that holds the cell sub-groups.
            Defaults to 'comb_cellgrp'.
        groupby_obsname (str, optional): The `.obs` column name in `adata` used to group observations for differential expression.
            Defaults to 'comb_sampname'.
        ncells_per_sample (int, optional): The minimum number of cells per sample required to perform the test.
            Groups with fewer cells are skipped. Defaults to 30.
        test_name (str, optional): The name of the statistical test to use for differential expression.
            Defaults to 't-test'.
        mask_var (str, optional): The name of the .var column indicating highly variable genes
            Defaults to 'highly_variable'.

    Returns:
        dict: A dictionary containing:
            - 'sample_names': List of sample names used in the analysis.
            - 'geneTab_dict': A dictionary where each key is a cell group name and each value is a DataFrame
              of differential expression results for that group.
    """
    ans = dict()
    
    # Keys for the rank_genes_groups object
    subset_keys = ['names', 'scores', 'pvals', 'pvals_adj', 'logfoldchanges']
    
    # If no specific sample observation values are provided, use all unique values from adata
    if len(sample_obsvals) == 0:
        sample_obsvals = adata.obs[groupby_obsname].unique().tolist()
    
    # Store the sample names in the result dictionary for later ordering of differential expression DataFrame
    ans['sample_names'] = sample_obsvals
    
    # Retrieve unique cell group names from the AnnData object
    cellgroup_names_in_anndata = adata.obs[cellgrp_obsname].unique()
    
    # If limitto_obsvals is provided, validate and set the cell groups to test
    if len(limitto_obsvals) > 0:
        # Identify any provided cell groups that are not present in adata
        unique_to_input = [x for x in limitto_obsvals if x not in cellgroup_names_in_anndata]
        if len(unique_to_input) > 0:
            print(f"The argument cellgrp_obsname has values that are not present in adata: {unique_to_input}")
        else:
            cellgroup_names = limitto_obsvals
    else:
        # If no limit is set, use all available cell groups
        cellgroup_names = cellgroup_names_in_anndata
    
    # Initialize a temporary dictionary to store differential expression results
    tmp_dict = dict()
    
    # Create a mask to filter adata for the specified sample observation values
    mask = adata.obs[groupby_obsname].isin(sample_obsvals)
    adata = adata[mask].copy()
    
    def convert_rankGeneGroup_to_df(rgg: dict, list_of_keys: list) -> pd.DataFrame:
        """
        Convert the rank_genes_groups result from AnnData to a pandas DataFrame.

        Args:
            rgg (dict): The rank_genes_groups result from AnnData.
            list_of_keys (list): List of keys to extract from the rank_genes_groups result.

        Returns:
            pd.DataFrame: A DataFrame containing the extracted rank genes information.
        """
        # Initialize a dictionary to hold arrays for each key
        arrays_dict = {}
        for key in list_of_keys:
            recarray = rgg[key]
            field_name = recarray.dtype.names[0]  # Get the first field name from the structured array
            arrays_dict[key] = recarray[field_name]
        
        # Convert the dictionary of arrays to a DataFrame
        return pd.DataFrame(arrays_dict)
    
    # Iterate over each cell group to perform differential expression analysis
    for cell_group in cellgroup_names:
        print(f"cell group: {cell_group}")
        
        # Subset the AnnData object for the current cell group
        adTmp = adata[adata.obs[cellgrp_obsname] == cell_group].copy()
        
        # Count the number of cells per sample within the cell group
        vcounts = adTmp.obs[groupby_obsname].value_counts()
        
        # Check if there are exactly two samples and each has at least ncells_per_sample cells
        if (len(vcounts) == 2) and (vcounts >= ncells_per_sample).all():
            # Perform differential expression analysis using the specified test
            sc.tl.rank_genes_groups(
                adTmp,
                use_raw=False,
                groupby=groupby_obsname,
                groups=[sample_obsvals[0]],
                reference=sample_obsvals[1],
                method=test_name,
                mask_var=mask_var
            )
            
            # Convert the rank_genes_groups result to a DataFrame and store it in tmp_dict
            tmp_dict[cell_group] = convert_rankGeneGroup_to_df(adTmp.uns['rank_genes_groups'].copy(), subset_keys)
            # Alternative method to get the DataFrame (commented out)
            # tmp_dict[cell_group] = sc.get.rank_genes_groups_df(adTmp, cell_group)
    
    # Store the differential expression results in the result dictionary
    ans['geneTab_dict'] = tmp_dict
    
    return ans





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
    fdr_thr: float = 0.25,
    top_n: int = 3
):
    """
    Collect and filter GSEA results from a dictionary of GSEA objects.
    
    For each cell type:
      1. Sets NES=0 for any gene set with FDR > fdr_thr.
      2. Selects up to top_n sets with the largest positive NES and 
         top_n with the most negative NES.
    
    The final output is limited to the union of all such selected sets
    across all cell types, with zeroes preserved for cell types in which
    the pathway is not among the top_n or fails the FDR threshold.
    
    Args:
        gsea_dict2 (dict): Dictionary mapping cell types to GSEA result objects.
            Each object has a .res2d DataFrame with columns ["Term", "NES", "FDR q-val"].
        fdr_thr (float, optional): FDR threshold above which NES values are set to 0. 
            Defaults to 0.25.
        top_n (int, optional): Maximum number of positive and negative results 
            (by NES) to keep per cell type. Defaults to 10.
    
    Returns:
        pd.DataFrame: A DataFrame whose rows are the union of selected gene sets 
            across all cell types, and whose columns are cell types. Entries 
            are filtered NES values (0 where FDR fails, or if not in the top_n).
    """
    import copy
    
    # Make a copy of the input to avoid in-place modifications
    gsea_dict = copy.deepcopy(gsea_dict2)
    
    # Collect all possible gene set names and cell types
    pathways = pd.Index([])
    cell_types = list(gsea_dict.keys())
    
    for cell_type in cell_types:
        tmpRes = gsea_dict[cell_type].res2d
        gene_set_names = list(tmpRes['Term'])
        pathways = pathways.union(gene_set_names)
    
    # Initialize NES DataFrame
    nes_df = pd.DataFrame(0, columns=cell_types, index=pathways)
    
    # Apply FDR threshold and fill NES
    for cell_type in cell_types:
        ct_df = gsea_dict[cell_type].res2d.copy()
        ct_df.index = ct_df['Term']
        # Zero out NES where FDR is too high
        ct_df.loc[ct_df['FDR q-val'] > fdr_thr, "NES"] = 0
        nes_df[cell_type] = ct_df["NES"]
    
    # Convert NES to numeric just in case
    nes_df = nes_df.apply(pd.to_numeric, errors='coerce')
    
    # Determine top_n positive and top_n negative for each cell type
    selected_sets = set()
    for cell_type in cell_types:
        ct_values = nes_df[cell_type]
        # Filter non-zero for positives and negatives
        pos_mask = ct_values > 0
        neg_mask = ct_values < 0

        # Select top_n largest positive NES
        top_pos_index = ct_values[pos_mask].sort_values(ascending=False).head(top_n).index
        # Select top_n most negative NES (smallest ascending)
        top_neg_index = ct_values[neg_mask].sort_values(ascending=True).head(top_n).index

        selected_sets.update(top_pos_index)
        selected_sets.update(top_neg_index)
    
    # Restrict DataFrame to the union of selected sets, converting the set to a list
    selected_sets_list = list(selected_sets)
    nes_df = nes_df.loc[selected_sets_list]
    
    return nes_df




