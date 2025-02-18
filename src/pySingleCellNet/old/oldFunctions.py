import numpy as np
import pandas as pd
from anndata import AnnData
import matplotlib.pyplot as plt
import scanpy as sc
import anndata as ad

import mygene

from collections import defaultdict


# Obsolete
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



def old_split_adata_indices(
    adata: AnnData,
    n_cells: int = 100,
    groupby: str = "cell_ontology_class",
    cellid: str = None,
    strata_col: str  = None
) -> tuple:
    """
    Splits an AnnData object into training and validation indices based on stratification by cell type
    and optionally by another categorical variable.

    Args:
        adata (AnnData): The annotated data matrix to split.
        n_cells (int): The number of cells to sample per cell type.
        groupby(str, optional): The column name in adata.obs that specifies the cell type. Defaults to "cell_ontology_class".
        cellid (str, optional): The column in adata.obs to use as a unique identifier for cells. If None, it defaults to using the index.
        strata_col (str, optional): The column name in adata.obs used for secondary stratification, such as developmental stage, gender, or disease status.

    Returns:
        tuple: A tuple containing two lists:
            - training_indices (list): List of indices for the training set.
            - validation_indices (list): List of indices for the validation set.

    Raises:
        ValueError: If any specified column names do not exist in the DataFrame.
    """
    if cellid is None:
        adata.obs["cellid"] = adata.obs.index
        cellid = "cellid"
    if groupby not in adata.obs.columns or (strata_col and strata_col not in adata.obs.columns):
        raise ValueError("Specified column names do not exist in the DataFrame.")

    cts = set(adata.obs[groupby])
    trainingids = []

    for ct in cts:
        # print(ct, ": ")
        subset = adata[adata.obs[groupby] == ct]

        if strata_col:
            stratified_ids = []
            strata_groups = subset.obs[strata_col].unique()
            
            for group in strata_groups:
                group_subset = subset[subset.obs[strata_col] == group]
                ccount = min(group_subset.n_obs, n_cells // len(strata_groups))
                if ccount > 0:
                    sampled_ids = np.random.choice(group_subset.obs[cellid].values, ccount, replace=False)
                    stratified_ids.extend(sampled_ids)

            trainingids.extend(stratified_ids)
        else:
            ccount = min(subset.n_obs, n_cells)
            sampled_ids = np.random.choice(subset.obs[cellid].values, ccount, replace=False)
            trainingids.extend(sampled_ids)

        # print(subset.n_obs)

    # Get all unique IDs
    all_ids = adata.obs[cellid].values
    # Determine validation IDs
    assume_unique = adata.obs_names.is_unique
    val_ids = np.setdiff1d(all_ids, trainingids, assume_unique=assume_unique)

    # Return indices instead of actual subsets
    return trainingids, val_ids



def old_paga_connectivities_to_igraph(
    adInput,
    n_neighbors = 10,
    use_rep = 'X_pca',
    n_comps = 30,
    threshold=0.05, 
    paga_key="paga", 
    connectivities_key="connectivities", 
    group_key="auto_cluster"
):
    """Convert a PAGA adjacency matrix to an undirected iGraph object.

    This function extracts the PAGA connectivity matrix from `adata.uns`, 
    thresholds the edges, and constructs an undirected iGraph graph. 
    Vertex names are assigned based on cluster categories if available.

    Args:
        adata (AnnData): The AnnData object containing:
            - `adata.uns[paga_key][connectivities_key]`: The PAGA adjacency matrix (CSR format).
            - `adata.obs[group_key].cat.categories`: The node labels.
        threshold (float, optional): Minimum edge weight to include. Defaults to 0.05.
        paga_key (str, optional): Key in `adata.uns` for PAGA results. Defaults to "paga".
        connectivities_key (str, optional): Key for connectivity matrix in `adata.uns[paga_key]`. Defaults to "connectivities".
        group_key (str, optional): The `.obs` column name with cluster labels. Defaults to "auto_cluster".

    Returns:
        ig.Graph: An undirected graph with edges meeting the threshold, edge weights assigned, 
        and vertex names set to cluster categories when possible.
    """

    # copy so as to avoid altering adata
    adata = adInput.copy()
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep=use_rep, n_pcs = n_comps)
    sc.tl.paga(adata, groups = groupby)
   
    adjacency_csr = adata.uns[paga_key][connectivities_key]
    adjacency_coo = adjacency_csr.tocoo()
    
    edges = []
    weights = []
    for i, j, val in zip(adjacency_coo.row, adjacency_coo.col, adjacency_coo.data):
        if i < j and val >= threshold:
            edges.append((i, j))
            weights.append(val)
    
    g = ig.Graph(n=adjacency_csr.shape[0], edges=edges, directed=False)
    g.es["weight"] = weights
    
    if group_key in adata.obs:
        categories = adata.obs[group_key].cat.categories
        if len(categories) == adjacency_csr.shape[0]:
            g.vs["name"] = list(categories)
        else:
            print(
                f"Warning: adjacency matrix size ({adjacency_csr.shape[0]}) "
                f"differs from number of categories ({len(categories)}). "
                "Vertex names will not be assigned."
            )
    else:
        print(
            f"Warning: {group_key} not found in adata.obs; "
            "vertex names will not be assigned."
        )
    
    return g



def split_common_anndata(adata, ncells, dLevel="cell_ontology_class", cellid = None, cells_reserved = 3):
    if cellid == None: 
         adata.obs["cellid"] = adata.obs.index
         cellid = "cellid"
    cts = set(adata.obs[dLevel])

    trainingids = np.empty(0)
    
    n_cts = len(cts)
    with alive_bar(n_cts, title="Splitting data") as bar:
        for ct in cts:
            # print(ct, ": ")
            aX = adata[adata.obs[dLevel] == ct, :]
            ccount = aX.n_obs - cells_reserved
            ccount = min([ccount, ncells])
            # print(aX.n_obs)
            trainingids = np.append(trainingids, np.random.choice(aX.obs[cellid].values, ccount, replace = False))
            bar()

    val_ids = np.setdiff1d(adata.obs[cellid].values, trainingids, assume_unique = True)
    aTrain = adata[np.isin(adata.obs[cellid], trainingids, assume_unique = True),:]
    aTest = adata[np.isin(adata.obs[cellid], val_ids, assume_unique = True),:]
    return([aTrain, aTest])

def splitCommon(expData, ncells,sampTab, dLevel="cell_ontology_class", cells_reserved = 3):
    cts = set(sampTab[dLevel])
    trainingids = np.empty(0)
    for ct in cts:
        aX = expData.loc[sampTab[dLevel] == ct, :]
        # print(ct, ": ")
        ccount = len(aX.index) - cells_reserved
        ccount = min([ccount, ncells])
        # print(ccount)
        trainingids = np.append(trainingids, np.random.choice(aX.index.values, ccount, replace = False))
    val_ids = np.setdiff1d(sampTab.index, trainingids, assume_unique = True)
    aTrain = expData.loc[np.isin(sampTab.index.values, trainingids, assume_unique = True),:]
    aTest = expData.loc[np.isin(sampTab.index.values, val_ids, assume_unique = True),:]
    return([aTrain, aTest])



def norm_hvg_scale_pca_oct(
    adQ: AnnData,
    tsum: float = 1e4,
    n_top_genes = 3000,
    batch_key = None,
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
        n_top_genes = n_top_genes,
        batch_key = batch_key
    )
    
    # Optionally scale the expression values of highly variable genes
    if gene_scale:
        sc.pp.scale(adata, max_value=scale_max)
    
    # Perform PCA on the data
    sc.tl.pca(adata, n_comps=n_comps, use_highly_variable=use_hvg)
    
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

def convert_rankGeneGroup_to_df( rgg: dict, list_of_keys: list) -> pd.DataFrame:
# Annoying but necessary function to deal with recarray format of .uns['rank_genes_groups'] to make sorting/extracting easier

    arrays_dict = {}
    for key in list_of_keys:
        recarray = rgg[key]
        field_name = recarray.dtype.names[0]  # Get the first field name
        arrays_dict[key] = recarray[field_name]

    return pd.DataFrame(arrays_dict)






def stackedbar_categories_list_old(
    ads, 
    titles=None,
    scn_classes_to_display=None,
    bar_height=0.8,
    bar_groups_obsname = 'SCN_class',
    bar_subgroups_obsname = 'SCN_class_type',
    ncell_min = None,
    color_dict = None,
    show_pct_total = False
):
    dfs = [adata.obs for adata in ads]
    num_dfs = len(dfs)
    # Determine the titles for each subplot
    if titles is None:
        titles = ['SCN Class Proportions'] * num_dfs
    elif len(titles) != num_dfs:
        raise ValueError("The length of 'titles' must match the number of annDatas.")
    # Determine the SCN classes to display
    ### all_classes_union = set().union(*(df[bar_groups_obsname].unique() for df in dfs))
    all_classes = set().union(*(df[bar_groups_obsname].unique() for df in dfs))
    if scn_classes_to_display is not None:
        if not all(cls in all_classes for cls in scn_classes_to_display):
            raise ValueError("Some values in 'scn_classes_to_display' do not match available 'SCN_class' values in the provided DataFrames.")
        else:
            classes_to_display = scn_classes_to_display
    else:
        classes_to_display = all_classes

    # setup colors
    if color_dict is None:
        color_dict = SCN_CATEGORY_COLOR_DICT 

    all_categories = set().union(*(df[bar_subgroups_obsname].unique() for df in dfs))
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(1, num_dfs, figsize=(6 * num_dfs, 8), sharey=True)
    for ax, df, title in zip(axes, dfs, titles):

        tmp_classes_to_display = classes_to_display.copy()
        # Ensure SCN_class categories are consistent
        # When adatas are classified with same clf, then they will have the same SCN_class categories even if not all cell types are predicted in a given adata
        df['SCN_class'] = df['SCN_class'].astype('category')
        df['SCN_class_type'] = df['SCN_class_type'].astype('category')
        df['SCN_class'] = df['SCN_class'].cat.set_categories(df['SCN_class'].cat.categories)
        df['SCN_class_type'] = df['SCN_class_type'].cat.set_categories(df['SCN_class_type'].cat.categories)

        # Reindex and filter each DataFrame
        ### counts = df.groupby(bar_groups_obsname)[bar_subgroups_obsname].value_counts().unstack().reindex(all_classes).fillna(0)
        counts = df.groupby('SCN_class')['SCN_class_type'].value_counts().unstack().fillna(0)
        total_counts = counts.sum(axis=1)
        total_percent = (total_counts / total_counts.sum() * 100).round(1)  # Converts to percentage and round
        proportions = counts.divide(total_counts, axis=0).fillna(0).replace([np.inf, -np.inf], 0, inplace=False)

        if ncell_min is not None:
            passing_classes = total_counts[total_counts >= ncell_min].index.to_list()
            tmp_classes_to_display = list(set(passing_classes) & set(tmp_classes_to_display))    

        proportions = proportions.loc[tmp_classes_to_display]
        total_counts = total_counts[tmp_classes_to_display]
        total_percent = total_percent[tmp_classes_to_display]

        # Plotting
        proportions.plot(kind='barh', stacked=True, color=color_dict, width=bar_height, ax=ax, legend=False)
        
        # Adding adjusted internal total counts within each bar
        text_size = max(min(12 - len(all_classes) // 2, 10), 7)  # Adjust text size
        for i, (count, percent) in enumerate(zip(total_counts, total_percent)):
            text = f'{int(count)} ({percent}%)'  # Text to display
            # text = f'({percent}%)'  # Text to display
            ax.text(0.95, i, text, ha='right', va='center', color='white', fontsize=text_size)

        # ax.set_xlabel('Proportion')
        ax.set_title(title)

    # Setting the y-label for the first subplot only
    axes[0].set_ylabel('SCN Class')
    # Adding the legend after the last subplot
    # axes[-1].legend(title='SCN Class Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add legend
    legend_handles = [mpatches.Patch(color=color, label=label) for label, color in color_dict.items()]
    ## legend = ax.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left')
    ## fig.legend(title='Categories', loc="outside right upper", frameon=False)
    legend = fig.legend(handles=legend_handles, loc="outside right upper", frameon=False)
    ### plt.tight_layout()
    ### plt.show()
    return fig
    




