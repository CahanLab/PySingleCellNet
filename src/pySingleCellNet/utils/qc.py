import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
# import anndata as ad

# also see pl.umi_counts_ranked
def find_knee_point(adata, total_counts_column="total_counts"):
    """
    Identifies the knee point of the UMI count distribution in an AnnData object.

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
    
    return knee_point_value



def mito_rib_heme(adQ: AnnData,
                  species: str = "MM",
                  clean: dict = None) -> AnnData:
    """
    Calculate mitochondrial, ribosomal, and hemoglobin QC metrics 
    and add them to the `.var` attribute of the AnnData object.
    
    Parameters
    ----------
    adQ : AnnData
        Annotated data matrix with observations (cells) and variables (features).
    species : str, optional (default: "MM")
        The species of the input data. Can be "MM" (Mus musculus) or "HS" (Homo sapiens).
    clean : dict, optional (default: {'ribo': True, 'mt': True, 'heme': True})
        Dictionary controlling whether to remove:
          - 'ribo': ribosomal genes
          - 'mt': mitochondrial genes
          - 'heme': hemoglobin genes
    
    Returns
    -------
    AnnData
        Annotated data matrix with QC metrics added to the `.var` attribute,
        and optionally with certain gene classes removed.
    """
    # -------------------------
    # 1. Set default if clean is None
    # -------------------------
    if clean is None:
        clean = {'ribo': True, 'mt': True, 'heme': True}
    else:
        # Ensure all three keys exist; if not, set them to default True
        for k in ['ribo', 'mt', 'heme']:
            if k not in clean:
                clean[k] = True
    
    # -------------------------
    # 2. Copy the input data
    # -------------------------
    adata = adQ.copy()
    
    # -------------------------
    # 3. Define gene prefixes based on species
    # -------------------------
    if species == 'MM':
        # MOUSE
        mt_prefix = "mt-"
        ribo_prefix = ("Rps", "Rpl")
        # Common mouse hemoglobin genes often start with 'Hba-' or 'Hbb-'
        heme_prefix = ("Hba-", "Hbb-")
    
    else:
        # HUMAN
        mt_prefix = "MT-"
        ribo_prefix = ("RPS", "RPL")
        # Human hemoglobin genes typically start with 'HB...' 
        # (HBA, HBB, HBD, HBE, HBG, HBZ, HBM, HBQ, etc.)
        # Using just "HB" can be too broad in some annotations, 
        # so here's a more explicit tuple:
        heme_prefix = ("HBA", "HBB", "HBD", "HBE", "HBG", "HBZ", "HBM", "HBQ")
    
    # -------------------------
    # 4. Flag MT, Ribo, and Heme genes in .var
    # -------------------------
    adata.var['mt'] = adata.var_names.str.startswith(mt_prefix)
    adata.var['ribo'] = adata.var_names.str.startswith(ribo_prefix)
    adata.var['heme'] = adata.var_names.str.startswith(heme_prefix)
    
    # -------------------------
    # 5. Calculate QC metrics 
    #    (Scanpy automatically calculates .var['total_counts'] etc.)
    # -------------------------
    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=['ribo', 'mt', 'heme'],
        percent_top=None,
        log1p=True,
        inplace=True
    )
    
    # -------------------------
    # 6. Optionally remove genes
    # -------------------------
    remove_mask = np.zeros(adata.shape[1], dtype=bool)
    
    if clean['mt']:
        remove_mask |= adata.var['mt'].values
    if clean['ribo']:
        remove_mask |= adata.var['ribo'].values
    if clean['heme']:
        remove_mask |= adata.var['heme'].values
    
    keep_mask = ~remove_mask
    
    adata = adata[:, keep_mask].copy()
    
    # -------------------------
    # 7. Return the modified AnnData
    # -------------------------
    return adata



def mito_rib(adQ: AnnData, species: str = "MM", log1p = True, clean: bool = True) -> AnnData:
    """
    Calculate mitochondrial and ribosomal QC metrics and add them to the `.var` attribute of the AnnData object.

    Parameters
    ----------
    adQ : AnnData
        Annotated data matrix with observations (cells) and variables (features).
    species : str, optional (default: "MM")
        The species of the input data. Can be "MM" (Mus musculus) or "HS" (Homo sapiens).
    clean : bool, optional (default: True)
        Whether to remove mitochondrial and ribosomal genes from the data.

    Returns
    -------
    AnnData
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










