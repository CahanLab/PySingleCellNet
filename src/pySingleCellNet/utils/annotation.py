import datetime
import numpy as np
import pandas as pd
from anndata import AnnData
import matplotlib.pyplot as plt
import scanpy as sc
#import mygene
import anndata as ad
from scipy.sparse import issparse
import re
from alive_progress import alive_bar
import csv
from collections import defaultdict

def create_gene_structure_dict_by_stage(file_path, stage):
    """
    Create a dictionary mapping structures to lists of genes expressed at a specific stage. Designed for parsing output from Jax Labs MGI data
    
    Parameters:
        file_path (str): Path to the gene expression file.
        stage (str or int): The Theiler Stage to filter the data.
    
    Returns:
        dict: A dictionary where keys are structures and values are lists of genes expressed in those structures.
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
    # Create a dictionary to keep track of gene occurrences
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


# to do: params for other species; error handling
def get_Y_chr_genes(adata):
    gene_chrs = sc.queries.biomart_annotations("mmusculus",["mgi_symbol", "ensembl_gene_id", "chromosome_name"],).set_index("mgi_symbol")
    ygenes = gene_chrs[gene_chrs["chromosome_name"]=='Y']
    ygenes = ygenes[ygenes.index.isin(adata.var.index)]
    return ygenes.index.tolist()


def write_gmt(gene_list, filename, collection_name, prefix=""):
    """
    Write a .gmt file from a gene list.

    Parameters:
    gene_list: dict
        Dictionary of gene sets (keys are gene set names, values are lists of genes).
    filename: str
        The name of the file to write to.
    collection_name: str
        The name of the gene set collection.
    prefix: str, optional
        A prefix to add to each gene set name.
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

def convert_ensembl_to_symbol(adata, species = 'mouse', batch_size=1000):
    # mg = mygene.MyGeneInfo()

    ensembl_ids = adata.var_names.tolist()
    total_ids = len(ensembl_ids)
    chunks = [ensembl_ids[i:i + batch_size] for i in range(0, total_ids, batch_size)]

    id_symbol_dict = {}

    # Querying in chunks
    for chunk in chunks:
        result = mg.querymany(chunk, scopes='ensembl.gene', fields='symbol', species=species)
        chunk_dict = {item['query']: item['symbol'] for item in result if 'symbol' in item}
        id_symbol_dict.update(chunk_dict)

    # Find IDs without a symbol
    unmapped_ids = set(ensembl_ids) - set(id_symbol_dict.keys())
    print(f"Found {len(unmapped_ids)} Ensembl IDs without a gene symbol")

    # Remove unmapped IDs
    adata = adata[:, ~adata.var_names.isin(list(unmapped_ids))]

    # Convert the remaining IDs to symbols and update 'var' DataFrame index
    adata.var.index = [id_symbol_dict[id] for id in adata.var_names]

    # Ensure index has no name to avoid '_index' conflict
    adata.var.index.name = None
    
    return adata


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
    """
    Filter the gene lists in the provided dictionary based on their lengths.

    Parameters:
    - genelist : dict
        Dictionary with keys as identifiers and values as lists of genes.
    - min_genes : int
        Minimum number of genes a list should have.
    - max_genes : int
        Maximum number of genes a list should have.

    Returns:
    - dict
        Filtered dictionary with lists that have a length between min_genes and max_genes (inclusive of min_genes and max_genes).
    """
    filtered_dict = {key: value for key, value in genelist.items() if min_genes <= len(value) <= max_genes}
    return filtered_dict



def annSetUp(species="mmusculus"):
    annot = sc.queries.biomart_annotations(species,["external_gene_name", "go_id"],)
    return annot

def getGenesFromGO(GOID, annList):
    if (str(type(GOID)) != "<class 'str'>"):
        return annList.loc[annList.go_id.isin(GOID),:].external_gene_name.sort_values().to_numpy()
    else:
        return annList.loc[annList.go_id==GOID,:].external_gene_name.sort_values().to_numpy()





