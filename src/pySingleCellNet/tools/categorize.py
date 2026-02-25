import numpy as np
import pandas as pd
import igraph as ig
from anndata import AnnData
import scanpy as sc

def _add_scn_class_cat(adata: AnnData):
    """Add a new '.obs' column 'SCN_class_cat' to an AnnData object.

    The value of 'SCN_class_cat' is taken from `.obs['SCN_class_emp']` when 
    `.obs['SCN_class_type']` is "Singular". Otherwise, it is taken from 
    `.obs['SCN_class_type']`.

    Args:
        adata (AnnData): The AnnData object to modify.

    Raises:
        ValueError: If 'SCN_class_emp' or 'SCN_class_type' is not present in `adata.obs`.

    Returns:
        None: Modifies the AnnData object in place.
    """
    if 'SCN_class_emp' not in adata.obs or 'SCN_class_type' not in adata.obs:
        raise ValueError("The input AnnData object must have both 'SCN_class_emp' and 'SCN_class_type' in .obs.")
    
    adata.obs['SCN_class_cat'] = adata.obs.apply(
        lambda row: row['SCN_class_emp'] if row['SCN_class_type'] == 'Singular' else row['SCN_class_type'], axis=1
    )

def categorize_classification(
    adata_c: AnnData,
    thresholds: pd.DataFrame,
    graph: ig.Graph = None,
    k: int = 3,
    columns_to_ignore: list = ["rand"],
    inplace: bool = True,
    class_obs_name: str = 'SCN_class_argmax'
):
    """Classify cells based on SCN scores and thresholds, then categorize 
    multi-class cells as either 'Intermediate' or 'Hybrid'.

    Classification rules:
      - If exactly one cell type exceeds threshold: "Singular"
      - If zero cell types exceed threshold: "None"
      - If more than one cell type exceeds threshold:
          * If all pairs of high-scoring cell types are within `k` edges 
            in the provided graph: "Intermediate"
          * Otherwise: "Hybrid"
      - If predicted cell type is 'rand': Set classification to "Rand"

    Args:
        adata_c (AnnData): Annotated data matrix containing:
            - `.obsm["SCN_score"]`: DataFrame of SCN scores for each cell type.
            - `.obs[class_obs_name]`: Predicted cell type (argmax classification).
        thresholds (pd.DataFrame): Thresholds for each cell type. Expected to 
            match the columns in `SCN_score`.
        graph (ig.Graph): An iGraph describing relationships between cell types. 
            Must have vertex names matching the cell-type columns in SCN_score.
        k (int, optional): Maximum graph distance to consider cell types "Intermediate". Defaults to 3.
        columns_to_ignore (list, optional): List of SCN score columns to ignore. Defaults to ["rand"].
        inplace (bool, optional): If True, modify `adata_c` in place. Otherwise, return a new AnnData object. Defaults to True.
        class_obs_name (str, optional): The name of the `.obs` column with argmax classification. Defaults to 'SCN_class_argmax'.

    Raises:
        ValueError: If `graph` is None.
        ValueError: If "SCN_score" is missing in `adata_c.obsm`.
        ValueError: If `class_obs_name` is not found in `adata_c.obs`.
        ValueError: If the provided graph does not have vertex "name" attributes.

    Returns:
        AnnData or None: Returns modified AnnData if `inplace` is False, otherwise None.
    """
    if graph is None:
        raise ValueError("A valid iGraph 'graph' must be provided. None was given.")
    
    if "SCN_score" not in adata_c.obsm:
        raise ValueError("No 'SCN_score' in adata_c.obsm. Please provide SCN scores.")
    
    SCN_scores = adata_c.obsm["SCN_score"].copy()
    SCN_scores.drop(columns=columns_to_ignore, inplace=True, errors='ignore')
    
    exceeded = SCN_scores.sub(thresholds.squeeze(), axis=1) > 0
    true_counts = exceeded.sum(axis=1)
    
    result_list = [
        [col for col in exceeded.columns[exceeded.iloc[row].values]]
        for row in range(exceeded.shape[0])
    ]
    
    class_type = pd.Series(["None"] * len(true_counts), index=true_counts.index, name="SCN_class_type")
    
    singular_mask = (true_counts == 1)
    class_type.loc[singular_mask] = "Singular"
    
    if "name" in graph.vs.attributes():
        type2index = {graph.vs[i]["name"]: i for i in range(graph.vcount())}
    else:
        raise ValueError("graph does not have a 'name' attribute for vertices.")
    
    def is_all_within_k_edges(cell_types):
        """Check if all pairs of cell types are within k edges in the graph.

        Args:
            cell_types (list): List of cell type names.

        Returns:
            bool: True if all pairs are within k edges, False otherwise.
        """
        if len(cell_types) <= 1:
            return True
        for i in range(len(cell_types)):
            for j in range(i + 1, len(cell_types)):
                ct1, ct2 = cell_types[i], cell_types[j]
                if ct1 not in type2index or ct2 not in type2index:
                    return False
                idx1 = type2index[ct1]
                idx2 = type2index[ct2]
                dist = graph.shortest_paths(idx1, idx2)[0][0]
                if dist >= k:
                    return False
        return True
    
    multi_mask = (true_counts > 1)
    multi_indices = np.where(multi_mask)[0]
    
    for i in multi_indices:
        c_types = result_list[i]
        if is_all_within_k_edges(c_types):
            class_type.iloc[i] = "Intermediate"
        else:
            class_type.iloc[i] = "Hybrid"
    
    ans = ['_'.join(lst) if lst else 'None' for lst in result_list]
    
    adata_c.obs['SCN_class_emp'] = ans
    adata_c.obs['SCN_class_type'] = class_type
    
    if class_obs_name not in adata_c.obs:
        raise ValueError(f"{class_obs_name} not found in adata_c.obs.")
    
    adata_c.obs['SCN_class_emp'] = adata_c.obs.apply(
        lambda row: 'Rand' if row[class_obs_name] == 'rand' else row['SCN_class_emp'],
        axis=1
    )
    adata_c.obs['SCN_class_type'] = adata_c.obs.apply(
        lambda row: 'Rand' if row[class_obs_name] == 'rand' else row['SCN_class_type'],
        axis=1
    )
    
    _add_scn_class_cat(adata_c)

    if inplace:
        return None
    else:
        return adata_c


def comp_ct_thresh(adata_c: AnnData, qTile: int = 0.05, obs_name='SCN_class_argmax') -> pd.DataFrame:
    """Compute quantile thresholds for each cell type based on SCN scores.

    For each cell type (excluding "rand"), this function calculates the qTile 
    quantile of the SCN scores for cells predicted to belong to that type.

    Args:
        adata_c (AnnData): Annotated data matrix with:
            - `.obsm["SCN_score"]`: DataFrame of SCN scores.
            - `.obs`: Observation metadata containing predictions.
        qTile (int, optional): The quantile to compute (e.g., 0.05 for 5th percentile). Defaults to 0.05.
        obs_name (str, optional): The column in `.obs` containing cell type predictions. Defaults to 'SCN_class_argmax'.

    Returns:
        pd.DataFrame: A DataFrame where each row corresponds to a cell type 
        (excluding 'rand') and contains the computed quantile threshold.
        Returns None if 'SCN_score' is not present in `adata_c.obsm`.
    """
    if "SCN_score" not in adata_c.obsm_keys():
        print("No .obsm['SCN_score'] was found in the AnnData provided. You may need to run PySingleCellNet.scn_classify()")
        return
    else:
        sampTab = adata_c.obs.copy()
        scnScores = adata_c.obsm["SCN_score"].copy()

        cts = scnScores.columns.drop('rand')
        thrs = pd.DataFrame(np.zeros(len(cts)), index=cts)

        for ct in cts:
            templocs = sampTab[sampTab[obs_name] == ct].index
            tempscores = scnScores.loc[templocs, ct]
            if len(tempscores) == 0:
                thrs.loc[ct, 0] = 0.0
            else:
                thrs.loc[ct, 0] = np.quantile(tempscores, q=qTile)
    
        return thrs


def paga_connectivities_to_igraph(
    adInput,
    n_neighbors=10,
    use_rep='X_pca',
    n_comps=30,
    threshold=0.05, 
    paga_key="paga", 
    connectivities_key="connectivities", 
    group_key="auto_cluster"
):
    """Convert a PAGA adjacency matrix to an undirected iGraph object and add 'ncells' 
    attribute for each vertex based on the number of cells in each cluster.
    
    This function extracts the PAGA connectivity matrix from `adata.uns`, thresholds 
    the edges, constructs an undirected iGraph graph, and assigns vertex names and 
    the number of cells in each cluster.
    
    Args:
        adInput (AnnData): The AnnData object containing:
            - `adata.uns[paga_key][connectivities_key]`: The PAGA adjacency matrix (CSR format).
            - `adata.obs[group_key].cat.categories`: The node labels.
        n_neighbors (int, optional): Number of neighbors for computing nearest neighbors. Defaults to 10.
        use_rep (str, optional): The representation to use. Defaults to 'X_pca'.
        n_comps (int, optional): Number of principal components. Defaults to 30.
        threshold (float, optional): Minimum edge weight to include. Defaults to 0.05.
        paga_key (str, optional): Key in `adata.uns` for PAGA results. Defaults to "paga".
        connectivities_key (str, optional): Key for connectivity matrix in `adata.uns[paga_key]`. Defaults to "connectivities".
        group_key (str, optional): The `.obs` column name with cluster labels. Defaults to "auto_cluster".
    
    Returns:
        ig.Graph: An undirected graph with edges meeting the threshold, edge weights assigned, 
        vertex names set to cluster categories when possible, and each vertex has an 'ncells' attribute.
    """
    # Copy so as to avoid altering the original AnnData object
    adata = adInput.copy()
    
    # Compute PCA, knn, and PAGA
    sc.tl.pca(adata, n_comps, mask_var='highly_variable')
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep=use_rep, n_pcs=n_comps)
    sc.tl.paga(adata, groups=group_key)
    
    # Extract the PAGA connectivity matrix
    adjacency_csr = adata.uns[paga_key][connectivities_key]
    adjacency_coo = adjacency_csr.tocoo()
    
    # Build edge list based on threshold
    edges = []
    weights = []
    for i, j, val in zip(adjacency_coo.row, adjacency_coo.col, adjacency_coo.data):
        if i < j and val >= threshold:
            edges.append((i, j))
            weights.append(val)
    
    # Create the graph
    g = ig.Graph(n=adjacency_csr.shape[0], edges=edges, directed=False)
    g.es["weight"] = weights
    
    # Assign vertex names and 'ncells' attribute if group_key exists in adata.obs
    if group_key in adata.obs:
        # Get cluster categories
        categories = adata.obs[group_key].cat.categories
        
        # Calculate the number of cells per category
        cell_counts_series = adata.obs[group_key].value_counts().reindex(categories, fill_value=0)
        cell_counts = list(cell_counts_series)
        
        if len(categories) == adjacency_csr.shape[0]:
            # Assign vertex names and 'ncells' attribute
            g.vs["name"] = list(categories)
            g.vs["label"] = list(categories)
            g.vs["ncells"] = cell_counts
        else:
            print(
                f"Warning: adjacency matrix size ({adjacency_csr.shape[0]}) "
                f"differs from number of categories ({len(categories)}). "
                "Vertex names and 'ncells' will not be fully assigned."
            )
            # Even if the sizes don't match, still assign available 'ncells' for existing categories
            g.vs["ncells"] = cell_counts
    else:
        print(
            f"Warning: {group_key} not found in adata.obs; "
            "vertex names and 'ncells' will not be assigned."
        )
    
    return g



def graph_from_nodes_and_edges(edge_dataframe, node_dataframe, attribution_column_names, directed=True):
    """Create an iGraph graph from provided node and edge dataframes.

    This function constructs an iGraph graph using nodes defined in 
    `node_dataframe` and edges defined in `edge_dataframe`. Each vertex 
    is assigned attributes based on specified columns, and edges are 
    created according to 'from' and 'to' columns in the edge dataframe.

    Args:
        edge_dataframe (pd.DataFrame): A DataFrame containing edge 
            information with at least 'from' and 'to' columns indicating 
            source and target node identifiers.
        node_dataframe (pd.DataFrame): A DataFrame containing node 
            information. Must include an 'id' column for vertex identifiers 
            and any other columns specified in `attribution_column_names`.
        attribution_column_names (list of str): List of column names from 
            `node_dataframe` whose values will be assigned as attributes 
            to the corresponding vertices in the graph.
        directed (bool, optional): Whether the graph should be directed. 
            Defaults to True.

    Returns:
        ig.Graph: An iGraph graph constructed from the given nodes and edges, 
        with vertex attributes and labels set according to the provided data.
    """
    gra = ig.Graph(directed=directed)
    attr = {}
    for attr_names in attribution_column_names:
        attr[attr_names] = node_dataframe[attr_names].to_numpy()
    
    gra.add_vertices(n=node_dataframe.id.to_numpy(), attributes=attr)
    for ind in edge_dataframe.index:
        tempsource = edge_dataframe.loc[ind].loc['from']
        temptarget = edge_dataframe.loc[ind].loc['to']
        gra.add_edges([(tempsource, temptarget)])
    
    gra.vs["label"] = gra.vs["id"]
    return gra





