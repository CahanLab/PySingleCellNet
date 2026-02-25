from __future__ import annotations
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import igraph as ig

def build_knn_graph(correlation_matrix, labels, k=5):
    """Build a k-nearest neighbors (kNN) graph from a correlation matrix.

    Args:
        correlation_matrix: Square correlation matrix.
        labels: Node labels corresponding to the rows/columns of the correlation matrix.
        k: Number of nearest neighbors to connect each node to. Defaults to 5.

    Returns:
        kNN graph as an igraph.Graph.
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

