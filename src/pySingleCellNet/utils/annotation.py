import warnings
import numpy as np
import pandas as pd
import scanpy as sc
import re
import csv
from collections import defaultdict

def create_gene_structure_dict_by_stage(file_path, stage):
    """Create a dictionary mapping structures to lists of genes expressed at a specific stage.

    Designed for parsing output from Jax Labs MGI data.

    Args:
        file_path: Path to the gene expression file.
        stage: The Theiler Stage to filter the data.

    Returns:
        A dictionary where keys are structures and values are lists of genes expressed in those structures.
    """
    structure_dict = defaultdict(set)  # Using a set to avoid duplicate gene symbols
    
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file, delimiter='\t')  # Use tab-delimiter based on previous example
        for row in reader:
            if row['Theiler Stage'] == str(stage):  # Subset by stage
                structure = row['Structure']
                gene_symbol = row['Gene Symbol']
                if structure and gene_symbol:  # Ensure both fields are not empty
                    structure_dict[structure].add(gene_symbol)
    
    # Convert sets to lists for final output
    structure_dict = {structure: list(genes) for structure, genes in structure_dict.items()}
    return structure_dict

def filter_genes_dict(gene_dict, x):
    """Filter gene lists to remove genes appearing in more than x lists.

    Useful to trim results of create_gene_structure_dict_by_stage.

    Args:
        gene_dict: Dictionary where keys are identifiers and values are lists of genes.
        x: Maximum number of lists a gene can appear in to be retained.

    Returns:
        A new dictionary with the same keys but with genes filtered out that appear in more than x lists.
    """
    gene_occurrences = {}
    
    # Count occurrences of each gene across all lists
    for genes in gene_dict.values():
        for gene in genes:
            if gene in gene_occurrences:
                gene_occurrences[gene] += 1
            else:
                gene_occurrences[gene] = 1
    
    # Create a new dictionary with filtered gene lists
    filtered_gene_dict = {}
    for key, genes in gene_dict.items():
        filtered_genes = [gene for gene in genes if gene_occurrences[gene] <= x]
        filtered_gene_dict[key] = filtered_genes
    
    return filtered_gene_dict


def write_gmt(gene_list, filename, collection_name, prefix=""):
    """Write a .gmt file from a gene list.

    Args:
        gene_list: Dictionary of gene sets (keys are gene set names, values are lists of genes).
        filename: The name of the file to write to.
        collection_name: The name of the gene set collection.
        prefix: A prefix to add to each gene set name. Defaults to "".

    Returns:
        None. Writes the GMT file to disk.
    """
    with open(filename, mode='w') as fo:
        for akey in gene_list:
            # replace whitespace with a "_"
            gl_name = re.sub(r'\s+', '_', akey)
            if prefix:
                pfix = prefix + "_" + gl_name
            else:
                pfix = gl_name
            preface = pfix + "\t" + collection_name + "\t"
            output = preface + "\t".join(gene_list[akey])
            print(output, file=fo)


def read_gmt(file_path: str) -> dict:
    """
    Read a Gene Matrix Transposed (GMT) file and return a dictionary of gene sets.

    Args:
        file_path (str): Path to the GMT file.

    Returns:
        dict: A dictionary where keys are gene set names and values are lists of associated genes.
    """
    gene_sets = {}
    
    with open(file_path, 'r') as gmt_file:
        for line in gmt_file:
            columns = line.strip().split('\t')
            gene_set_name = columns[0]
            description = columns[1]  # This can be ignored if not needed
            genes = columns[2:]
            
            gene_sets[gene_set_name] = genes
            
    return gene_sets


def filter_gene_list(genelist, min_genes, max_genes=1e6):
    """Filter the gene lists in the provided dictionary based on their lengths.

    Args:
        genelist: Dictionary with keys as identifiers and values as lists of genes.
        min_genes: Minimum number of genes a list should have.
        max_genes: Maximum number of genes a list should have. Defaults to 1e6.

    Returns:
        Filtered dictionary with lists that have a length between min_genes and max_genes (inclusive).
    """
    filtered_dict = {key: value for key, value in genelist.items() if min_genes <= len(value) <= max_genes}
    return filtered_dict

def ann_set_up(species: str = "mmusculus") -> pd.DataFrame:
    """Fetch gene annotations from BioMart.

    Retrieves gene names and GO IDs from Ensembl BioMart for the specified species.

    Args:
        species: Species name for BioMart query. Defaults to "mmusculus".

    Returns:
        DataFrame with columns 'external_gene_name' and 'go_id'.

    Example:
        >>> annot = ann_set_up(species="hsapiens")
    """
    annot = sc.queries.biomart_annotations(species, ["external_gene_name", "go_id"])
    return annot


def annSetUp(species: str = "mmusculus") -> pd.DataFrame:
    """Deprecated: Use :func:`ann_set_up` instead.

    Args:
        species: Species name for BioMart query. Defaults to "mmusculus".

    Returns:
        DataFrame with columns 'external_gene_name' and 'go_id'.
    """
    warnings.warn(
        "annSetUp is deprecated, use ann_set_up instead",
        DeprecationWarning,
        stacklevel=2
    )
    return ann_set_up(species)


def get_genes_from_go(go_id, ann_list: pd.DataFrame) -> np.ndarray:
    """Get genes associated with a GO term.

    Args:
        go_id: GO term ID (string) or list of GO term IDs.
        ann_list: DataFrame from ann_set_up with 'external_gene_name' and 'go_id' columns.

    Returns:
        Array of gene names associated with the GO term(s).

    Example:
        >>> genes = get_genes_from_go("GO:0006915", annot)
    """
    if isinstance(go_id, str):
        return ann_list.loc[ann_list.go_id == go_id, :].external_gene_name.sort_values().to_numpy()
    else:
        return ann_list.loc[ann_list.go_id.isin(go_id), :].external_gene_name.sort_values().to_numpy()


def getGenesFromGO(GOID, annList: pd.DataFrame) -> np.ndarray:
    """Deprecated: Use :func:`get_genes_from_go` instead.

    Args:
        GOID: GO term ID (string) or list of GO term IDs.
        annList: DataFrame from ann_set_up with 'external_gene_name' and 'go_id' columns.

    Returns:
        Array of gene names associated with the GO term(s).
    """
    warnings.warn(
        "getGenesFromGO is deprecated, use get_genes_from_go instead",
        DeprecationWarning,
        stacklevel=2
    )
    return get_genes_from_go(GOID, annList)
