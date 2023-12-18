import datetime
import numpy as np
import pandas as pd
from anndata import AnnData
import scanpy as sc
import mygene
import anndata as ad
import pySingleCellNet as pySCN
from scipy.sparse import issparse
import re

# convert adata.uns[x] into a dict of data.frames
def convert_diffExp_to_dict(
    adata,
    uns_name: str = 'rank_genes_groups'
):
    
    # sc.tl.rank_genes_groups(adTemp, dLevel, use_raw=False, method=test_name)
    # tempTab = pd.DataFrame(adata.uns[uns_name]['names']).head(topX)
    tempTab = sc.get.rank_genes_groups_df(adata, group = None, key=uns_name)
    tempTab = tempTab.dropna()
    groups = tempTab['group'].cat.categories.to_list()
    #groups = np.unique(grps)

    ans = {}
    for g in groups:
        ans[g] = tempTab[tempTab['group']==g].copy()
    return ans

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
    mg = mygene.MyGeneInfo()

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


def old_convert_ensembl_to_symbol(adata, batch_size=1000):
    mg = mygene.MyGeneInfo()

    ensembl_ids = adata.var_names.tolist()
    total_ids = len(ensembl_ids)
    chunks = [ensembl_ids[i:i + batch_size] for i in range(0, total_ids, batch_size)]

    id_symbol_dict = {}

    # Querying in chunks
    for chunk in chunks:
        result = mg.querymany(chunk, scopes='ensembl.gene', fields='symbol', species='mouse')
        chunk_dict = {item['query']: item['symbol'] for item in result if 'symbol' in item}
        id_symbol_dict.update(chunk_dict)

    # Find IDs without a symbol
    unmapped_ids = set(ensembl_ids) - set(id_symbol_dict.keys())
    print(f"Found {len(unmapped_ids)} Ensembl IDs without a gene symbol")

    # Remove unmapped IDs
    adata = adata[:, ~adata.var_names.isin(list(unmapped_ids))]

    # Convert the remaining IDs to symbols
    adata.var_names = [id_symbol_dict[id] for id in adata.var_names]
    
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

def filter_gene_list(genelist, min_genes, max_genes):
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


def convert_rankGeneGroup_to_df( rgg: dict, list_of_keys: list) -> pd.DataFrame:
# Annoying but necessary function to deal with recarray format of .uns['rank_genes_groups'] to make sorting/extracting easier

    arrays_dict = {}
    for key in list_of_keys:
        recarray = rgg[key]
        field_name = recarray.dtype.names[0]  # Get the first field name
        arrays_dict[key] = recarray[field_name]

    return pd.DataFrame(arrays_dict)


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


def read_broken_geo_mtx(path: str, prefix: str) -> AnnData:
    # assumes that obs and var in .mtx _could_ be switched
    # determines which is correct by size of genes.tsv and barcodes.tsv

    adata = sc.read_mtx(path + prefix + "matrix.mtx")
    cell_anno = pd.read_csv(path + prefix + "barcodes.tsv", delimiter='\t', header=None)
    n_cells = cell_anno.shape[0]
    cell_anno.rename(columns={0:'cell_id'},inplace=True)

    gene_anno = pd.read_csv(path + prefix + "genes.tsv", header=None, delimiter='\t')
    n_genes = gene_anno.shape[0]
    gene_anno.rename(columns={0:'gene'},inplace=True)

    if adata.shape[0] == n_genes:
        adata = adata.T

    adata.obs = cell_anno.copy()
    adata.obs_names = adata.obs['cell_id']
    adata.var = gene_anno.copy()
    adata.var_names = adata.var['gene']
    return adata

def mito_rib(adQ: AnnData, species: str = "MM", clean: bool = True) -> AnnData:
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
        log1p=False,
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


def norm_hvg_scale_pca(
    adQ: AnnData,
    tsum: float = 1e4,
    min_mean: float = 0.0125,
    max_mean: float = 6,
    min_disp: float = 0.25,
    scale_max: float = 10,
    n_comps: int = 100,
    gene_scale: bool = False,
    use_hvg: bool = True
) -> AnnData:
    """
    Normalize, detect highly variable genes, optionally scale, and perform PCA on an AnnData object.

    Parameters
    ----------
    adQ : AnnData
        Annotated data matrix with observations (cells) and variables (features).
    tsum : float, optional (default: 1e4)
        The total count to which data is normalized.
    min_mean : float, optional (default: 0.0125)
        The minimum mean expression value of genes to be considered as highly variable.
    max_mean : float, optional (default: 6)
        The maximum mean expression value of genes to be considered as highly variable.
    min_disp : float, optional (default: 0.25)
        The minimum dispersion value of genes to be considered as highly variable.
    scale_max : float, optional (default: 10)
        The maximum value of scaled expression data.
    n_comps : int, optional (default: 100)
        The number of principal components to compute.
    gene_scale : bool, optional (default: False)
        Whether to scale the expression values of highly variable genes.

    Returns
    -------
    AnnData
        Annotated data matrix with normalized, highly variable, optionally scaled, and PCA-transformed data.
    """
    # Create a copy of the input data
    adata = adQ.copy()

    # Normalize the data to a target sum
    sc.pp.normalize_total(adata, target_sum=tsum)

    # Log-transform the data
    sc.pp.log1p(adata)

    # Detect highly variable genes
    sc.pp.highly_variable_genes(
        adata,
        min_mean=min_mean,
        max_mean=max_mean,
        min_disp=min_disp
    )

    # Optionally scale the expression values of highly variable genes
    if gene_scale:
        sc.pp.scale(adata, max_value=scale_max)

    # Perform PCA on the data
    sc.tl.pca(adata, n_comps=n_comps, use_highly_variable=use_hvg)

    return adata


def reduce_cells(
    adata: AnnData,
    n_cells: int = 5,
    cluster_key: str = "cluster",
    use_raw: bool = True
) -> AnnData:
    """
    Reduce the number of cells in an AnnData object by combining transcript counts across clusters.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with observations (cells) and variables (features).
    n_cells : int, optional (default: 5)
        The number of cells to combine into a meta-cell.
    cluster_key : str, optional (default: "cluster")
        The key in `adata.obs` that specifies the cluster identity of each cell.
    use_raw : bool, optional (default: True)
        Whether to use the raw count matrix in `adata.raw` instead of `adata.X`.

    Returns
    -------
    AnnData
        Annotated data matrix with reduced number of cells.
    """
    # Create a copy of the input data
    adata = adata.copy()

    # Use the raw count matrix if specified
    if use_raw:
        adata.X = adata.raw.X

    # Get the k-nearest neighbor graph
    knn_graph = adata.uns['neighbors']['connectivities']

    # Get the indices of the cells in each cluster
    clusters = np.unique(adata.obs[cluster_key])
    cluster_indices = {c: np.where(adata.obs[cluster_key] == c)[0] for c in clusters}

    # Calculate the number of meta-cells to make
    n_metacells = int(sum(np.ceil(adata.obs[cluster_key].value_counts() / n_cells)))

    # Create a list of new AnnData objects to store the combined transcript counts
    ad_list = []

    # Loop over each cluster
    for cluster in clusters:
        # Get the indices of the cells in this cluster
        indices = cluster_indices[cluster]

        # If there are fewer than n_cells cells in the cluster, skip it
        if len(indices) < n_cells:
            continue

        # Compute the total transcript count across n_cells cells in the cluster
        num_summaries = int(np.ceil(len(indices) / n_cells))
        combined = []
        used_indices = set()
        for i in range(num_summaries):
            # Select n_cells cells at random from the remaining unused cells
            unused_indices = list(set(indices) - used_indices)
            np.random.shuffle(unused_indices)
            selected_indices = unused_indices[:n_cells]

            # Add the transcript counts for the selected cells to the running total
            combined.append(np.sum(adata.X[selected_indices,:], axis=0))

            # Add the selected indices to the set of used indices
            used_indices.update(selected_indices)

        # Create a new AnnData object to store the combined transcript counts for this cluster
        tmp_adata = AnnData(X=np.array(combined), var=adata.var)

        # Add the cluster identity to the `.obs` attribute of the new AnnData object
        tmp_adata.obs[cluster_key] = cluster

        # Append the new AnnData object to the list
        ad_list.append(tmp_adata)

    # Concatenate the new AnnData objects into a single AnnData object
    adata2 = anndata.concat(ad_list, join='inner')

    return adata2

def create_mean_cells(adata, obs_key, obs_value, n_mean_cells, n_cells):
    """
    Average n_cells randomly sampled cells from obs_key == obs_value obs, n_mean_cells times

    Parameters:
    adata: anndata.AnnData
        The annotated data matrix of shape n_obs x n_vars. Rows correspond to cells and columns to genes.
    cluster_key: str
        The key in the obs field that identifies the clusters.
    n_mean_cells: int
        The number of meta-cells to create.
    n_cells: int
        The number of cells to randomly select from each cluster.

    Returns:
    anndata.AnnData
        Annotated data matrix with meta-cells.
    """
    if obs_key not in adata.obs:
        raise ValueError(f"Key '{obs_key}' not found in adata.obs")

    # Create a new anndata object to store the mean-cells
    mean_cells = None
    
    # Get cells in this cluster
    cluster_cells = adata[adata.obs[obs_key] == obs_value].copy()
    seeds = np.random.choice(n_mean_cells, n_mean_cells, replace=False)
    for i in range(n_mean_cells):
        # Randomly select cells
        selected_cells = sc.pp.subsample(cluster_cells, n_obs = n_cells, random_state = seeds[i], copy=True)

        # Average their gene expression
        avg_expression = selected_cells.X.mean(axis=0)

        # Create a new cell in the mean_cells anndata object
        new_cell = sc.AnnData(avg_expression.reshape(1, -1),
            var=adata.var, 
            obs=pd.DataFrame(index=[f'{obs_value}_mean_{i}']))

        # Append new cell to the meta_cells anndata object
        if mean_cells is None:
            mean_cells = new_cell
        else:
            mean_cells = mean_cells.concatenate(new_cell)

    return mean_cells


def add_ambient_rna(
    adata,
    obs_key: str,
    obs_val: str,
    n_cells_to_sample: int = 10,
    weight_of_ambient: float = 0.05
) -> AnnData:

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



def create_hybrid_cells(
    adata: AnnData,
    celltype_counts: dict,
    groupby: str,
    n_hybrid_cells: int
) -> AnnData:
    """
    Generate hybrid cells by taking the mean of transcript counts for randomly selected cells
    from the specified groups in proportions as indicated by celltypes

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with observations (cells) and variables (genes).
    celltype_counts : dict
        keys indicate the subset of cells and the values are the number of cells to sample from each cell type.
    groupby : str
        The name of the column in `adata.obs` to group cells by before selecting cells at random.
    n_hybrid_cells : int
        The number of hybrid cells to generate.

    Returns
    -------
    AnnData
        Annotated data matrix containing only the hybrid cells.
    """
    hybrid_list = []
    for i in range(n_hybrid_cells):
        # Randomly select cells from the specified clusters
        r_cells = pySCN.sample_cells(adata, celltype_counts, groupby)

        # Calculate the average transcript counts for the selected cells
        ### x_counts = np.average(r_cells.X, axis=0)
        x_counts = np.mean(r_cells.X, axis=0)

        # Create an AnnData object for the hybrid cell
        hybrid = AnnData(
            x_counts.reshape(1, -1),
            obs={'hybrid': [f'Hybrid Cell {i+1}']},
            var=adata.var
        )

        # Append the hybrid cell to the hybrid list
        hybrid_list.append(hybrid)

    # Concatenate the hybrid cells into a single AnnData object
    hybrid = ad.concat(hybrid_list, join='outer')

    return hybrid

def sample_cells(
    adata: AnnData,
    celltype_counts: dict,
    groupby: str
)-> AnnData:
    """
    Sample cells as specified by celltype_counts and groub_by

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with observations (cells) and variables (genes).
    celltype_counts: dict
        keys indicate the subset of cells and the values are the number of cells to sample from each cell type.
    groupby : str
        The name of the column in `adata.obs` to group cells by before selecting cells at random.

    Returns
    -------
    AnnData
        Annotated data matrix containing only the selected cells.
    """

    sampled_cells = []

    for celltype, count in celltype_counts.items():
        # subset the AnnData object by .obs[groupby], most often will be something like cluster, leiden, SCN_class, celltype
        subset = adata[adata.obs[groupby] == celltype]
        
        # sample cells from the subset
        # sampled_subset = subset.sample(n=count, random_state=1)
        cell_ids = np.random.choice(subset.obs.index, count, replace = False)
        adTmp = subset[np.isin(subset.obs.index, cell_ids, assume_unique = True),:].copy()
        
        # append the sampled cells to the list
        sampled_cells.append(adTmp)

    # concatenate the sampled cells into a single AnnData object
    sampled_adata = sampled_cells[0].concatenate(sampled_cells[1:])

    return sampled_adata

from scipy.sparse import issparse

def compute_mean_expression_per_cluster(
    adata,
    cluster_key
):
    """
    Compute mean gene expression for each gene in each cluster, create a new anndata object, and store it in adata.uns.

    Parameters:
    - adata : anndata.AnnData
        The input AnnData object with labeled cell clusters.
    - cluster_key : str
        The key in adata.obs where the cluster labels are stored.

    Returns:
    - anndata.AnnData
        The modified AnnData object with the mean expression anndata stored in uns['mean_expression'].
    """
    if cluster_key not in adata.obs.columns:
        raise ValueError(f"{cluster_key} not found in adata.obs")

    # Extract unique cluster labels
    clusters = adata.obs[cluster_key].unique().tolist()

    # Compute mean expression for each cluster
    mean_expressions = []
    for cluster in clusters:
        cluster_cells = adata[adata.obs[cluster_key] == cluster, :]
        mean_expression = np.mean(cluster_cells.X, axis=0).A1 if issparse(cluster_cells.X) else np.mean(cluster_cells.X, axis=0)
        mean_expressions.append(mean_expression)

    # Convert to matrix
    mean_expression_matrix = np.vstack(mean_expressions)
    
    # Create a new anndata object
    mean_expression_adata = sc.AnnData(X=mean_expression_matrix, 
                                       var=pd.DataFrame(index=adata.var_names), 
                                       obs=pd.DataFrame(index=clusters))
    
    # Store this new anndata object in adata.uns
    adata.uns['mean_expression'] = mean_expression_adata
    #return adata


def find_elbow(
    adata
):
    """
    Find the "elbow" index in the variance explained by principal components.

    Parameters:
    - variance_explained : list or array
        Variance explained by each principal component, typically in decreasing order.

    Returns:
    - int
        The index corresponding to the "elbow" in the variance explained plot.
    """
    variance_explained = adata.uns['pca']['variance_ratio']
    # Coordinates of all points
    n_points = len(variance_explained)
    all_coords = np.vstack((range(n_points), variance_explained)).T
    # Line vector from first to last point
    line_vec = all_coords[-1] - all_coords[0]
    line_vec_norm = line_vec / np.sqrt(np.sum(line_vec**2))
    # Vector being orthogonal to the line
    vec_from_first = all_coords - all_coords[0]
    scalar_prod = np.sum(vec_from_first * np.tile(line_vec_norm, (n_points, 1)), axis=1)
    vec_from_first_parallel = np.outer(scalar_prod, line_vec_norm)
    vec_to_line = vec_from_first - vec_from_first_parallel
    # Distance to the line
    dist_to_line = np.sqrt(np.sum(vec_to_line ** 2, axis=1))
    # Index of the point with max distance to the line
    elbow_idx = np.argmax(dist_to_line)
    return elbow_idx




def ctMerge(sampTab, annCol, ctVect, newName):
    oldRows=np.isin(sampTab[annCol], ctVect)
    newSampTab= sampTab.copy()
    newSampTab.loc[oldRows,annCol]= newName
    return newSampTab

def ctRename(sampTab, annCol, oldName, newName):
    oldRows=sampTab[annCol]== oldName
    newSampTab= sampTab.copy()
    newSampTab.loc[oldRows,annCol]= newName
    return newSampTab

#adapted from Sam's code
def splitCommonAnnData(adata, ncells, dLevel="cell_ontology_class", cellid = None, cells_reserved = 3):
    if cellid == None: 
         adata.obs["cellid"] = adata.obs.index
         cellid = "cellid"
    cts = set(adata.obs[dLevel])

    trainingids = np.empty(0)
    for ct in cts:
        print(ct, ": ")
        aX = adata[adata.obs[dLevel] == ct, :]
        ccount = aX.n_obs - cells_reserved
        ccount = min([ccount, ncells])
        print(aX.n_obs)
        trainingids = np.append(trainingids, np.random.choice(aX.obs[cellid].values, ccount, replace = False))

    val_ids = np.setdiff1d(adata.obs[cellid].values, trainingids, assume_unique = True)
    aTrain = adata[np.isin(adata.obs[cellid], trainingids, assume_unique = True),:]
    aTest = adata[np.isin(adata.obs[cellid], val_ids, assume_unique = True),:]
    return([aTrain, aTest])

def splitCommon(expData, ncells,sampTab, dLevel="cell_ontology_class", cells_reserved = 3):
    cts = set(sampTab[dLevel])
    trainingids = np.empty(0)
    for ct in cts:
        aX = expData.loc[sampTab[dLevel] == ct, :]
        print(ct, ": ")
        ccount = len(aX.index) - cells_reserved
        ccount = min([ccount, ncells])
        print(ccount)
        trainingids = np.append(trainingids, np.random.choice(aX.index.values, ccount, replace = False))
    val_ids = np.setdiff1d(sampTab.index, trainingids, assume_unique = True)
    aTrain = expData.loc[np.isin(sampTab.index.values, trainingids, assume_unique = True),:]
    aTest = expData.loc[np.isin(sampTab.index.values, val_ids, assume_unique = True),:]
    return([aTrain, aTest])

def annSetUp(species="mmusculus"):
    annot = sc.queries.biomart_annotations(species,["external_gene_name", "go_id"],)
    return annot

def getGenesFromGO(GOID, annList):
    if (str(type(GOID)) != "<class 'str'>"):
        return annList.loc[annList.go_id.isin(GOID),:].external_gene_name.sort_values().to_numpy()
    else:
        return annList.loc[annList.go_id==GOID,:].external_gene_name.sort_values().to_numpy()

def dumbfunc(aNamedList):
    return aNamedList.index.values

def GEP_makeMean(expDat,groupings,type='mean'):
    if (type=="mean"):
        return expDat.groupby(groupings).mean()
    if (type=="median"):
        return expDat.groupby(groupings).median()

def utils_myDist(expData):
    numSamps=len(expData.index)
    result=np.subtract(np.ones([numSamps, numSamps]), expData.T.corr())
    del result.index.name
    del result.columns.name
    return result

def utils_stripwhite(string):
    return string.strip()

def utils_myDate():
    d = datetime.datetime.today()
    return d.strftime("%b_%d_%Y")

def utils_strip_fname(string):
    sp=string.split("/")
    return sp[len(sp)-1]

def utils_stderr(x):
    return (stats.sem(x))

def zscore(x,meanVal,sdVal):
    return np.subtract(x,meanVal)/sdVal

def zscoreVect(genes, expDat, tVals,ctt, cttVec):
    res={}
    x=expDat.loc[cttVec == ctt,:]
    for gene in genes:
        xvals=x[gene]
        res[gene]= pd.series(data=zscore(xvals, tVals[ctt]['mean'][gene], tVals[ctt]['sd'][gene]), index=xvals.index.values)
    return res

def downSampleW(vector,total=1e5, dThresh=0):
    vSum=np.sum(vector)
    dVector=total/vSum
    res=dVector*vector
    res[res<dThresh]=0
    return res

def weighted_down(expDat, total, dThresh=0):
    rSums=expDat.sum(axis=1)
    dVector=np.divide(total, rSums)
    res=expDat.mul(dVector, axis=0)
    res[res<dThresh]=0
    return res

def trans_prop(expDat, total, dThresh=0):
    rSums=expDat.sum(axis=1)
    dVector=np.divide(total, rSums)
    res=expDat.mul(dVector, axis=0)
    res[res<dThresh]=0
    return np.log(res + 1)

def trans_zscore_col(expDat):
    return expDat.apply(stats.zscore, axis=0)

def trans_zscore_row(expDat):
    return expDat.T.apply(stats.zscore, axis=0).T

def trans_binarize(expData,threshold=1):
    expData[expData<threshold]=0
    expData[expData>0]=1
    return expData

def getUniqueGenes(genes, transID='id', geneID='symbol'):
    genes2=genes.copy()
    genes2.index=genes2[transID]
    genes2.drop_duplicates(subset = geneID, inplace= True, keep="first")
    del genes2.index.name
    return genes2

def removeRed(expData,genes, transID="id", geneID="symbol"):
    genes2=getUniqueGenes(genes, transID, geneID)
    return expData.loc[:, genes2.index.values]

def cn_correctZmat_col(zmat):
    def myfuncInf(vector):
        mx=np.max(vector[vector<np.inf])
        mn=np.min(vector[vector>(np.inf * -1)])
        res=vector.copy()
        res[res>mx]=mx
        res[res<mn]=mn
        return res
    return zmat.apply(myfuncInf, axis=0)

def cn_correctZmat_row(zmat):
    def myfuncInf(vector):
        mx=np.max(vector[vector<np.inf])
        mn=np.min(vector[vector>(np.inf * -1)])
        res=vector.copy()
        res[res>mx]=mx
        res[res<mn]=mn
        return res
    return zmat.apply(myfuncInf, axis=1)

def makeExpMat(adata):
    expMat = pd.DataFrame(adata.X, index = adata.obs_names, columns = adata.var_names)
    return expMat

def makeSampTab(adata):
    sampTab = adata.obs
    return sampTab
