import numpy as np
import pandas as pd
# from anndata import AnnData
import scanpy as sc
import anndata as ad
# import string
import igraph as ig
from typing import Dict, List
from .adataTools import find_elbow

def score_gene_modules(
    adata,
    gene_dict: dict,
    key_added: str = "module_scores"
):

    # Number of cells and clusters
    n_cells = adata.shape[0]
    # Initialize an empty matrix for scores
    # scores_matrix = np.zeros((n_cells, n_clusters))
    scores_df = pd.DataFrame(index=adata.obs_names)
    # For each cluster, calculate the gene scores and store in the DataFrame
    for cluster, genes in gene_dict.items():
        score_name = cluster
        sc.tl.score_genes(adata, gene_list=genes, score_name=score_name, use_raw=False)        
        # Store the scores in the DataFrame
        scores_df[score_name] = adata.obs[score_name].values
        del(adata.obs[score_name])
    # Assign the scores DataFrame to adata.obsm
    obsm_name = key_added
    adata.obsm[obsm_name] = scores_df

def find_knn_modules(
    adata,
    mean_cluster: bool = True,
    groupby: str = 'leiden',
    use_hvg: bool = True,
    knn: int = 5,
    leiden_resolution: float = 0.5,
    prefix: str = 'gmod_',
    npcs_adjust: int = 1
):
    """
    Finds gene modules using k-nearest neighbors on PCA-reduced data and stores the modules in adata.uns['knn_modules'].

    Parameters:
    - adata: AnnData object to process.
    - mean_cluster: If True, compute the mean expression per cluster defined in `groupby`.
    - groupby: Column in adata.obs to use for clustering when mean_cluster is True.
    - use_hvg: If True, subset the data to highly variable genes (adata.var['highly_variable'] must exist).
    - knn: Number of neighbors to use in the kNN graph.
    - leiden_resolution: Resolution parameter for the Leiden clustering.
    - prefix: Prefix to add to each module name.
    - npcs_adjust: Additional PCs to add to the elbow estimate.
    
    Returns:
    - Updated adata object with 'knn_modules' stored in adata.uns.
    """
    # Work on a copy to avoid modifying the original object
    adata_subset = adata.copy()

    # If using highly variable genes, verify the flag exists and subset accordingly.
    if use_hvg:
        if 'highly_variable' not in adata_subset.var.columns:
            raise ValueError("'highly_variable' column not found in adata.var.")
        hvg_names = adata_subset.var.index[adata_subset.var['highly_variable']].tolist()
        if not hvg_names:
            raise ValueError("No highly variable genes found in adata.var.")
        adata_subset = adata_subset[:, hvg_names].copy()

    # If computing mean expression per cluster, check and compute it.
    if mean_cluster:
        if groupby not in adata_subset.obs.columns:
            raise ValueError(f"Column '{groupby}' not found in adata.obs.")
        # It is assumed that this function computes and stores the result in adata_subset.uns['mean_expression']
        
        # compute_mean_expression_per_cluster(adata_subset, groupby)
        adata_subset = sc.get.aggregate(adata_subset, by=groupby, func='mean')
        adata_subset.X  = adata_subset.layers['mean']

        # Replace the data with the mean expression matrix
        # if 'mean_expression' not in adata_subset.uns:
        #    raise ValueError("Mean expression not found in .uns after computing clusters.")
        # adata_subset = adata_subset.uns['mean_expression'].copy()

    # Transpose so that genes (or clusters) become observations
    adata_transposed = adata_subset.T.copy()

    # Run PCA and determine number of PCs from the elbow method
    sc.tl.pca(adata_transposed)
    elbow = find_elbow(adata_transposed)
    n_pcs = elbow + npcs_adjust

    # Compute the kNN graph using correlation as the metric
    # sc.pp.neighbors(adata_transposed, n_neighbors=knn, n_pcs=n_pcs, metric='correlation')
    sc.pp.neighbors(adata_transposed, n_neighbors=knn, n_pcs=n_pcs, metric='euclidean')
    
    # Perform Leiden clustering with an explicit resolution keyword
    sc.tl.leiden(adata_transposed, resolution=leiden_resolution)
    
    # Group indices by Leiden clusters and add prefix to module names
    clusters = (
        adata_transposed.obs.groupby('leiden', observed=True)
        .apply(lambda df: df.index.tolist())
        .to_dict()
    )
    modules = {f"{prefix}{cluster}": indices for cluster, indices in clusters.items()}

    # Save the modules in the original adata uns field
    adata.uns['knn_modules'] = modules


def what_module_has_gene(
    adata,
    target_gene,
    mod_slot='knn_modules'
) -> list: 
    if mod_slot not in adata.uns.keys():
        raise ValueError(mod_slot + " have not been identified.")
    genemodules = adata.uns[mod_slot]
    return [key for key, genes in genemodules.items() if target_gene in genes]







def extract_top_bottom_genes(
    deg_res: dict,
    ngenes: int,
    sort_by: str = 'scores',
    extraction_map: Dict[str, str] = None
) -> Dict[str, List[str]]:
    """
    Extracts top and bottom ngenes from each gene table in deg_res and organizes them
    into a dictionary with combined keys of group and sample names.

    Parameters:
    -----------
    deg_res : dict
        A dictionary containing differential expression results with keys:
            - 'sample_names': List of sample names (e.g., ['Singular', 'None'])
            - 'geneTab_dict': Dictionary where each key is a group name and each value is
                              a Pandas DataFrame with gene information.
    ngenes : int
        The number of top or bottom genes to extract from each gene table.
    sort_by : str, optional (default='scores')
        The column name in the gene tables to sort by.
    extraction_map : dict, optional
        A dictionary mapping sample names to extraction behavior ('top' or 'bottom').
        If not provided, defaults to:
            - First sample name: 'top'
            - Second sample name: 'bottom'
            - Additional sample names: 'top'

    Returns:
    --------
    result_dict : dict
        A dictionary where each key is a combination of group and sample name
        (e.g., 'Meso.Nascent_Singular') and each value is a list of gene names.
        - For 'top', the list contains the top ngenes based on sort_by.
        - For 'bottom', the list contains the bottom ngenes based on sort_by.

    Raises:
    -------
    KeyError:
        If 'sample_names' or 'geneTab_dict' keys are missing in deg_res,
        or if 'sort_by' is not a column in the gene tables.
    ValueError:
        If ngenes is not a positive integer or if 'sample_names' does not contain at least one entry.
    TypeError:
        If the input types are incorrect.
    """
    # Input Validation
    required_keys = ['sample_names', 'geneTab_dict']
    for key in required_keys:
        if key not in deg_res:
            raise KeyError(f"Key '{key}' not found in deg_res.")

    sample_names = deg_res['sample_names']
    geneTab_dict = deg_res['geneTab_dict']

    if not isinstance(sample_names, list):
        raise TypeError(f"'sample_names' should be a list, got {type(sample_names)}.")

    if not isinstance(geneTab_dict, dict):
        raise TypeError(f"'geneTab_dict' should be a dict, got {type(geneTab_dict)}.")

    if not isinstance(ngenes, int) or ngenes <= 0:
        raise ValueError(f"'ngenes' must be a positive integer, got {ngenes}.")

    if len(sample_names) < 1:
        raise ValueError(f"'sample_names' should contain at least one entry, got {len(sample_names)}.")

    result_dict = {}

    # Define default extraction behavior if extraction_map is not provided
    if extraction_map is None:
        extraction_map = {}
        for idx, sample in enumerate(sample_names):
            if idx == 0:
                extraction_map[sample] = 'top'
            elif idx == 1:
                extraction_map[sample] = 'bottom'
            else:
                extraction_map[sample] = 'top'  # Default behavior for additional samples

    else:
        # Validate extraction_map
        if not isinstance(extraction_map, dict):
            raise TypeError(f"'extraction_map' should be a dict, got {type(extraction_map)}.")
        for sample, behavior in extraction_map.items():
            if behavior not in ['top', 'bottom']:
                raise ValueError(f"Invalid extraction behavior '{behavior}' for sample '{sample}'. Must be 'top' or 'bottom'.")

    # Iterate over each group in geneTab_dict
    for group, df in geneTab_dict.items():
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected DataFrame for group '{group}', got {type(df)}.")

        # Ensure required columns exist
        required_columns = ['names', sort_by]
        for col in required_columns:
            if col not in df.columns:
                raise KeyError(f"Column '{col}' not found in gene table for group '{group}'.")

        # Sort the DataFrame by 'sort_by' column
        sorted_df = df.sort_values(by=sort_by, ascending=False).reset_index(drop=True)

        # Iterate over sample names to determine extraction behavior
        for sample in sample_names:
            combined_key = f"{group}_{sample}"
            behavior = extraction_map.get(sample, 'top')  # Default to 'top' if not specified

            if behavior == 'top':
                # Extract top ngenes
                top_genes = sorted_df['names'].head(ngenes).tolist()
                result_dict[combined_key] = top_genes
            elif behavior == 'bottom':
                # Extract bottom ngenes
                bottom_genes = sorted_df['names'].tail(ngenes).tolist()
                result_dict[combined_key] = bottom_genes

    return result_dict


def rank_genes_subsets(
    adata,
    groupby,
    grpA,
    grpB,
    pval = 0.01,
    layer=None
):
    """
    Subset an AnnData object to specified groups, create a new .obs column labeling cells
    as group A or B, and run rank_genes_groups for differential expression analysis. Necessary because the scanpy reference does not seem to work
    
    Parameters:
        adata (AnnData): The AnnData object.
        groupby (str): The .obs column to group cells by.
        grpA (list): Values used to subset cells into group A.
        grpB (list): Values used to subset cells into group B.
        layer (str, optional): Layer to use for expression values.
        
    Returns:
        AnnData: Subsetted and labeled AnnData object after running rank_genes_groups.
    """
    # Subset the data to cells in either grpA or grpB
    subset = adata[adata.obs[groupby].isin(grpA + grpB)].copy()
    # Create a new .obs column labeling cells as 'grpA' or 'grpB'
    subset.obs["comparison_group"] = subset.obs[groupby].apply(
        lambda x: "grpA" if x in grpA else "grpB"
    )
    # Run rank_genes_groups
    sc.tl.rank_genes_groups(
        subset,
        groupby="comparison_group",
        layer=layer,
        pts = True,
        use_raw=False
    )
    # return subset
    ans = sc.get.rank_genes_groups_df(subset, group='grpA', pval_cutoff=pval)
    return ans



def pull_out_genes(
    diff_genes_dict: dict, 
    cell_type: str,
    category: str, 
    num_genes: int = 0,
    order_by = "logfoldchanges", 
    threshold = 2) -> list:

    ans = []
    #### xdat = diff_genes_dict[cell_type]
    xdat = diff_genes_dict['geneTab_dict'][cell_type]
    xdat = xdat[xdat['pvals_adj'] < threshold].copy()

    category_names = diff_genes_dict['category_names']
    category_index = category_names.index(category)
    
    # any genes left?
    if xdat.shape[0] > 0:

        if num_genes == 0:
            num_genes = xdat.shape[0]

        if category_index == 0:
            xdat.sort_values(by=[order_by], inplace=True, ascending=False)
        else:
            xdat.sort_values(by=[order_by], inplace=True, ascending=True)

        ans = list(xdat.iloc[0:num_genes]["names"])

    return ans


def pull_out_genes_v2(
    diff_genes_dict: dict, 
    cell_type: str,
    category: str, 
    num_genes: int = 0,
    order_by = "logfoldchanges", 
    threshold = 2) -> list:

    ans = []
    #### xdat = diff_genes_dict[cell_type]
    xdat = diff_genes_dict['geneTab_dict'][cell_type]
    xdat = xdat[xdat['pvals_adj'] < threshold].copy()

    dictkey = list(diff_genes_dict.keys())[0]
    category_names = diff_genes_dict[dictkey]
    category_index = category_names.index(category)
    
    # any genes left?
    if xdat.shape[0] > 0:

        if num_genes == 0:
            num_genes = xdat.shape[0]

        if category_index == 0:
            xdat.sort_values(by=[order_by], inplace=True, ascending=False)
        else:
            xdat.sort_values(by=[order_by], inplace=True, ascending=True)

        ans = list(xdat.iloc[0:num_genes]["names"])

    return ans



def remove_genes(adata, genes_to_exclude=None):
    adnew = adata[:,~adata.var_names.isin(genes_to_exclude)].copy()
    return adnew



def limit_anndata_to_common_genes(anndata_list):
    # Find the set of common genes across all anndata objects
    common_genes = set(anndata_list[0].var_names)
    for adata in anndata_list[1:]:
        common_genes.intersection_update(set(adata.var_names))
    
    # Limit the anndata objects to the common genes
    # latest anndata update broke this:
    if common_genes:
         for adata in anndata_list:
            adata._inplace_subset_var(list(common_genes))

    #return anndata_list
    #return common_genes


def add_ambient_rna(
    adata,
    obs_key: str,
    obs_val: str,
    n_cells_to_sample: int = 10,
    weight_of_ambient: float = 0.05
) -> ad.AnnData:

    # What cells will be updated?
    non_cluster_cells = adata[adata.obs[obs_key] != obs_val, :].copy()
    n_mean_cells = non_cluster_cells.n_obs

    # Generate the sample means
    sample_means_adata = create_mean_cells(adata, obs_key, obs_val, n_mean_cells, n_cells_to_sample)
    
    # If there are more non-cluster cells than samples, raise an error
    # if non_cluster_cells.shape[0] > nSamples:
    #    raise ValueError("The number of samples is less than the number of non-cluster cells.")
    
    # Update the non-cluster cells' expression states to be the weighted mean of their current states and the sample means
    non_cluster_cells.X = (1 - weight_of_ambient) * non_cluster_cells.X + weight_of_ambient * sample_means_adata.X
    
    # Update the original adata object with the new expression states
    adata[adata.obs[obs_key] != obs_val, :] = non_cluster_cells
    
    return adata


# to do: params for other species; error handling
def get_Y_chr_genes(adata):
    gene_chrs = sc.queries.biomart_annotations("mmusculus",["mgi_symbol", "ensembl_gene_id", "chromosome_name"],).set_index("mgi_symbol")
    ygenes = gene_chrs[gene_chrs["chromosome_name"]=='Y']
    ygenes = ygenes[ygenes.index.isin(adata.var.index)]
    return ygenes.index.tolist()


def score_sex(
    adata, 
    y_genes=['Eif2s3y', 'Ddx3y', 'Uty'], 
    x_inactivation_genes=['Xist', 'Tsix']
):
    """
    Adds sex chromosome expression scores to an AnnData object.
    
    This function calculates two scores for each cell in a scRNA-seq AnnData object:
      - Y_score: the sum of expression values for a set of Y-chromosome specific genes.
      - X_inact_score: the sum of expression values for genes involved in X-chromosome inactivation.
      
    The scores are added to the AnnData object's `.obs` DataFrame with the keys 'Y_score' and 'X_inact_score'.
    
    Parameters
    ----------
    adata : AnnData
        An AnnData object containing scRNA-seq data, with gene names in `adata.var_names`.
    y_genes : list of str, optional
        List of Y-chromosome specific marker genes (default is ['Eif2s3y', 'Ddx3y', 'Uty']).
    x_inactivation_genes : list of str, optional
        List of genes involved in X-chromosome inactivation (default is ['Xist', 'Tsix']).
        
    Raises
    ------
    ValueError
        If none of the Y-specific or X inactivation genes are found in `adata.var_names`.
    
    Returns
    -------
    None
        The function modifies the AnnData object in place by adding the score columns to `adata.obs`.
    """
    # Filter for genes that are available in the dataset.
    available_y_genes = [gene for gene in y_genes if gene in adata.var_names]
    available_x_genes = [gene for gene in x_inactivation_genes if gene in adata.var_names]
    
    if not available_y_genes:
        raise ValueError("None of the Y-specific genes were found in the dataset.")
    if not available_x_genes:
        raise ValueError("None of the X inactivation genes were found in the dataset.")
    
    # Compute the sum of expression for the Y-specific genes.
    y_expression = adata[:, available_y_genes].X
    if hasattr(y_expression, "toarray"):
        y_expression = y_expression.toarray()
    adata.obs['Y_score'] = np.sum(y_expression, axis=1)
    
    # Compute the sum of expression for the X inactivation genes.
    x_expression = adata[:, available_x_genes].X
    if hasattr(x_expression, "toarray"):
        x_expression = x_expression.toarray()
    adata.obs['X_inact_score'] = np.sum(x_expression, axis=1)
    
    # Optionally, you could log some output:
    print("Added 'Y_score' and 'X_inact_score' to adata.obs for {} cells.".format(adata.n_obs))
    
# Example usage:
# Assuming 'adata' is your AnnData object:
# add_sex_scores(adata)
# print(adata.obs[['Y_score', 'X_inact_score']].head())




