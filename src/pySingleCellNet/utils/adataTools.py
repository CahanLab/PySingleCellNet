import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from typing import Dict, List, Optional
from scipy.sparse import issparse
import igraph as ig

def rename_cluster_labels(
    adata: ad,AnnData,
    old_col: str = "cluster",
    new_col: str = "short_cluster"
) -> None:
    """
    Renames cluster labels in the specified .obs column with multi-letter codes.
    
    - All unique labels (including NaN) are mapped in order of appearance to 
      a base-26 style ID: 'A', 'B', ..., 'Z', 'AA', 'AB', etc.
    - The new labels are stored as a categorical column in `adata.obs[new_col]`.
    
    Args:
        adata (AnnData):
            The AnnData object containing the cluster labels.
        old_col (str, optional):
            The name of the .obs column that has the original cluster labels.
            Defaults to "cluster".
        new_col (str, optional):
            The name of the new .obs column that will store the shortened labels.
            Defaults to "short_cluster".
    
    Returns:
        None: The function adds a new column to `adata.obs` in place.
    """
    
    # 1. Extract unique labels (including NaN), in the order they appear
    unique_labels = adata.obs[old_col].unique()
    
    # 2. Helper function for base-26 labeling
    def index_to_label(idx: int) -> str:
        """
        Convert a zero-based index to a base-26 letter code:
        0 -> A
        1 -> B
        ...
        25 -> Z
        26 -> AA
        27 -> AB
        ...
        """
        letters = []
        while True:
            remainder = idx % 26
            letter = chr(ord('A') + remainder)
            letters.append(letter)
            idx = idx // 26 - 1
            if idx < 0:
                break
        return ''.join(letters[::-1])
    
    # 3. Build the mapping (including NaN -> next code)
    label_map = {}
    for i, lbl in enumerate(unique_labels):
        label_map[lbl] = index_to_label(i)
    
    # 4. Apply the mapping to create the new column
    adata.obs[new_col] = adata.obs[old_col].map(label_map)
    adata.obs[new_col] = adata.obs[new_col].astype("category")


def generate_joint_graph(adata, connectivity_keys, weights, output_key='jointNeighbors'):
    """Create a joint graph by combining multiple connectivity graphs with specified weights.
    
    This function computes the weighted sum of selected connectivity and distance matrices 
    in an AnnData object and stores the result in `.obsp`.
    
    Args:
        adata (AnnData): 
            The AnnData object containing connectivity matrices in `.obsp`.
        connectivity_keys (list of str): 
            A list of keys in `adata.obsp` corresponding to connectivity matrices to combine.
        weights (list of float): 
            A list of weights for each connectivity matrix. Must match the length of `connectivity_keys`.
        output_key (str, optional): 
            The base key under which to store the combined graph in `.obsp`. 
            The default is `'jointNeighbors'`.
    
    Raises:
        ValueError: If the number of `connectivity_keys` does not match the number of `weights`.
        KeyError: If any key in `connectivity_keys` or its corresponding distances key is not found in `adata.obsp`.
    
    Returns:
        None: 
            Updates the AnnData object in place by adding the combined connectivity and distance matrices
            to `.obsp` and metadata to `.uns`.
    
    Example:
        >>> generate_joint_graph(adata, ['neighbors_connectivities', 'umap_connectivities'], [0.7, 0.3])
        >>> adata.obsp['jointNeighbors_connectivities']
        >>> adata.uns['jointNeighbors']
    """

    if len(connectivity_keys) != len(weights):
        raise ValueError("The number of connectivity keys must match the number of weights.")
    
    # Initialize the joint graph and distances matrix with zeros
    joint_graph = None
    joint_distances = None
    # Loop through each connectivity key and weight
    for key, weight in zip(connectivity_keys, weights):
        if key not in adata.obsp:
            raise KeyError(f"'{key}' not found in adata.obsp.")
        
        # Retrieve the connectivity matrix
        connectivity_matrix = adata.obsp[key]
        # Assume corresponding distances key exists
        distances_key = key.replace('connectivities', 'distances')
        if distances_key not in adata.obsp:
            raise KeyError(f"'{distances_key}' not found in adata.obsp.")
        distances_matrix = adata.obsp[distances_key]
        # Initialize or accumulate the weighted connectivity and distances matrices
        if joint_graph is None:
            joint_graph = weight * connectivity_matrix
            joint_distances = weight * distances_matrix
        else:
            joint_graph += weight * connectivity_matrix
            joint_distances += weight * distances_matrix
        
    # Save the resulting joint graph and distances matrix in the specified keys of .obsp
    
    adata.obsp[output_key + '_connectivities'] = joint_graph
    adata.obsp[output_key + '_distances'] = joint_distances
    
    # Save metadata about the joint graph in .uns
    adata.uns[output_key] = {
        'connectivities_key': output_key + '_connectivities',
        'distances_key': output_key + '_distances',
        'params': {
            'connectivity_keys': connectivity_keys,
            'weights': weights,
            'method': "umap"
        }
    }


def combine_pca_scores(adata, n_pcs=50, score_key='SCN_score'):
    """Combine principal components and gene set scores into a single matrix.

    This function merges the top principal components (PCs) and gene set scores 
    into a combined matrix stored in `.obsm`.

    Args:
        adata (AnnData): 
            AnnData object containing PCA results and gene set scores in `.obsm`.
        n_pcs (int, optional): 
            Number of top PCs to include. Default is 50.
        score_key (str, optional): 
            Key in `.obsm` where gene set scores are stored. Default is `'SCN_score'`.

    Raises:
        ValueError: If `'X_pca'` is not found in `.obsm`.  
        ValueError: If `score_key` is missing in `.obsm`.

    Returns:
        None: Updates `adata` by adding the combined matrix to `.obsm['X_pca_scores_combined']`.

    Example:
        >>> combine_pca_scores(adata, n_pcs=30, score_key='GeneSet_Score')
    """

    # Ensure that the required data exists in .obsm
    if 'X_pca' not in adata.obsm:
        raise ValueError("X_pca not found in .obsm. Perform PCA before combining.")
    
    if score_key not in adata.obsm:
        raise ValueError(f"{score_key} not found in .obsm. Please provide valid gene set scores.")
    
    # Extract the top n_pcs from .obsm['X_pca']
    pca_matrix = adata.obsm['X_pca'][:, :n_pcs]
    
    # Extract the gene set scores from .obsm
    score_matrix = adata.obsm[score_key]
    
    # Combine PCA matrix and score matrix horizontally (along columns)
    combined_matrix = np.hstack([pca_matrix, score_matrix])
    
    # Add the combined matrix back into .obsm with a new key
    adata.obsm['X_pca_scores_combined'] = combined_matrix

    print(f"Combined matrix with {n_pcs} PCs and {score_matrix.shape[1]} gene set scores added to .obsm['X_pca_scores_combined'].")



def build_knn_graph(correlation_matrix, labels, k=5):
    """
    Build a k-nearest neighbors (kNN) graph from a correlation matrix.
    
    Parameters:
        correlation_matrix (ndarray): Square correlation matrix.
        labels (list): Node labels corresponding to the rows/columns of the correlation matrix.
        k (int): Number of nearest neighbors to connect each node to.
    
    Returns:
        igraph.Graph: kNN graph.
    """

    # import igraph as ig

    # Ensure the correlation matrix is square
    assert correlation_matrix.shape[0] == correlation_matrix.shape[1], "Matrix must be square."
    
    # Initialize the graph
    n = len(labels)
    g = ig.Graph()
    g.add_vertices(n)
    g.vs["name"] = labels  # Add node labels
    
    # Build kNN edges
    for i in range(n):
        # Get k largest correlations (excluding self-correlation)
        neighbors = np.argsort(correlation_matrix[i, :])[-(k + 1):-1]  # Exclude the node itself
        for j in neighbors:
            g.add_edge(i, j, weight=correlation_matrix[i, j])
    
    return g


def filter_anndata_slots(adata, slots_to_keep):
    """
    Creates a copy of an AnnData object and filters it to retain only the specified 
    slots and elements within those slots. Unspecified slots or elements are removed from the copy.
    
    The function operates on a copy of the provided AnnData object, ensuring that the original
    data remains unchanged. This approach allows users to maintain data integrity while
    exploring different subsets or representations of their dataset.
    
    Args:
        adata (AnnData): The AnnData object to be copied and filtered. This object
            represents a single-cell dataset with various annotations and embeddings.
        slots_to_keep (dict): A dictionary specifying which slots and elements within 
            those slots to keep. The keys should be the slot names ('obs', 'var', 'obsm',
            'obsp', 'varm', 'varp'), and the values should be lists of the names within 
            those slots to preserve. If a slot is not mentioned or its value is None, 
            all its contents are removed in the copy. Example format:
            {'obs': ['cluster'], 'var': ['gene_id'], 'obsm': ['X_pca']}
            
    Returns:
        AnnData: A copy of the original AnnData object filtered according to the specified
        slots to keep. This copy contains only the data and annotations specified by the
        `slots_to_keep` dictionary, with all other data and annotations removed.

    Example:
        adata = sc.datasets.pbmc68k_reduced()
        slots_to_keep = {
            'obs': ['n_genes', 'percent_mito'],
            'var': ['n_cells'],
            # Assuming we want to clear these unless specified to keep
            'obsm': None,
            'obsp': None,
            'varm': None,
            'varp': None,
        }
        filtered_adata = filter_anndata_slots(adata, slots_to_keep)
        # `filtered_adata` is the modified copy, `adata` remains unchanged.
    """
    
    # Create a copy of the AnnData object to work on
    adata_copy = adata.copy()
    
    # Define all possible slots
    all_slots = ['obs', 'var', 'obsm', 'obsp', 'varm', 'varp']
    
    for slot in all_slots:
        if slot not in slots_to_keep or slots_to_keep[slot] is None:
            # If slot is not mentioned or is None, remove all its contents
            if slot in ['obs', 'var']:
                setattr(adata_copy, slot, pd.DataFrame(index=getattr(adata_copy, slot).index))
            else:
                setattr(adata_copy, slot, {})
        else:
            # Specific elements within the slot are specified to be kept
            elements_to_keep = slots_to_keep[slot]
            
            if slot in ['obs', 'var']:
                # Filter columns for 'obs' and 'var'
                df = getattr(adata_copy, slot)
                columns_to_drop = [col for col in df.columns if col not in elements_to_keep]
                df.drop(columns=columns_to_drop, inplace=True)
                
            elif slot in ['obsm', 'obsp', 'varm', 'varp']:
                # Filter keys for 'obsm', 'obsp', 'varm', 'varp'
                mapping = getattr(adata_copy, slot)
                keys_to_drop = [key for key in mapping.keys() if key not in elements_to_keep]
                for key in keys_to_drop:
                    del mapping[key]
    
    return adata_copy


def read_broken_geo_mtx(path: str, prefix: str) -> ad.AnnData:
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


# outdated
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

