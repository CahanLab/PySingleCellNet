import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scipy.stats import median_abs_deviation

def call_outlier_cells(adata, metric = ["total_counts"], nmads = 5):
    """Determines whether obs[metric] exceeds nmads.

    Args:
        adata: The input AnnData object containing single-cell data.
        metric: The column name in `adata.obs` holding cell metric.
        nmads: The number of median abs deviations to define a cell as an outlier. Defaults to 5.

    Returns:
        None. The function adds a new column to `adata.obs` named "outlier_" + metric, but does not return anything.
    """
    M = adata.obs[metric]
    outlier = (M < np.median(M) - nmads * median_abs_deviation(M)) | (
        np.median(M) + nmads * median_abs_deviation(M) < M
    )

    new_col = "outlier_" + nmads + "_" + metric
    adata.obs[new_col] = outlier
    


# also see pl.umi_counts_ranked
def find_knee_point(adata, total_counts_column="total_counts"):
    """Identifies the knee point of the UMI count distribution in an AnnData object.

    Args:
        adata: The input AnnData object.
        total_counts_column: Column in `adata.obs` containing total UMI counts. Defaults to "total_counts".

    Returns:
        The UMI count value at the knee point.
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
    
    return knee_point_value


def mito_rib(adQ: AnnData, species: str = "MM", log1p = True, clean: bool = True) -> AnnData:
    """Calculate mitochondrial and ribosomal QC metrics and add them to the `.var` attribute of the AnnData object.

    Args:
        adQ: Annotated data matrix with observations (cells) and variables (features).
        species: The species of the input data. Can be "MM" (Mus musculus) or "HS" (Homo sapiens). Defaults to "MM".
        log1p: Whether to log1p-transform QC metrics. Defaults to True.
        clean: Whether to remove mitochondrial and ribosomal genes from the data. Defaults to True.

    Returns:
        Annotated data matrix with QC metrics added to the `.var` attribute.
    """
    # Create a copy of the input data
    adata = adQ.copy()

    # Define mitochondrial and ribosomal gene prefixes based on the species
    if species == 'MM':
        mt_prefix = "mt-"
        ribo_prefix = ("Rps","Rpl")
    else:
        mt_prefix = "MT-"
        ribo_prefix = ("RPS","RPL")

    # Add mitochondrial and ribosomal gene flags to the `.var` attribute
    adata.var['mt'] = adata.var_names.str.startswith((mt_prefix))
    adata.var['ribo'] = adata.var_names.str.startswith(ribo_prefix)

    # Calculate QC metrics using Scanpy's `calculate_qc_metrics` function
    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=['ribo', 'mt'],
        percent_top=None,
        log1p=log1p,
        inplace=True
    )

    # Optionally remove mitochondrial and ribosomal genes from the data
    if clean:
        mito_genes = adata.var_names.str.startswith(mt_prefix)
        ribo_genes = adata.var_names.str.startswith(ribo_prefix)
        remove = np.add(mito_genes, ribo_genes)
        keep = np.invert(remove)
        adata = adata[:,keep].copy()
        # sc.pp.calculate_qc_metrics(adata,percent_top=None,log1p=False,inplace=True)
    
    return adata


def score_sex(
    adata, 
    y_genes=['Eif2s3y', 'Ddx3y', 'Uty'], 
    x_inactivation_genes=['Xist', 'Tsix']
):
    """Adds sex chromosome expression scores to an AnnData object.

    This function calculates two scores for each cell in a scRNA-seq AnnData object:
      - Y_score: the sum of expression values for a set of Y-chromosome specific genes.
      - X_inact_score: the sum of expression values for genes involved in X-chromosome inactivation.

    The scores are added to the AnnData object's `.obs` DataFrame with the keys 'Y_score' and 'X_inact_score'.

    Args:
        adata: An AnnData object containing scRNA-seq data, with gene names in `adata.var_names`.
        y_genes: List of Y-chromosome specific marker genes. Defaults to ['Eif2s3y', 'Ddx3y', 'Uty'].
        x_inactivation_genes: List of genes involved in X-chromosome inactivation. Defaults to ['Xist', 'Tsix'].

    Returns:
        None. The function modifies the AnnData object in place by adding the score columns to `adata.obs`.

    Raises:
        ValueError: If none of the Y-specific or X inactivation genes are found in `adata.var_names`.
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






