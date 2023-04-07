import datetime
import numpy as np
import pandas as pd
from anndata import AnnData
import scanpy as sc




def limit_anndata_to_common_genes(anndata_list):
    # Find the set of common genes across all anndata objects
    common_genes = set(anndata_list[0].var_names)
    for adata in anndata_list[1:]:
        common_genes.intersection_update(set(adata.var_names))
    
    # Limit the anndata objects to the common genes
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
        sc.pp.calculate_qc_metrics(adata,percent_top=None,log1p=False,inplace=True)
    
    return adata


def norm_hvg_scale_pca(
    adQ: AnnData,
    tsum: float = 1e4,
    min_mean: float = 0.0125,
    max_mean: float = 6,
    min_disp: float = 0.25,
    scale_max: float = 10,
    n_comps: int = 100,
    gene_scale: bool = False
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
    sc.tl.pca(adata, n_comps=n_comps)

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

def create_hybrid_cells(
    adata: AnnData,
    clusters: list,
    groupby: str,
    n_hybrid_cells: int,
    weights: list
) -> AnnData:
    """
    Generate hybrid cells by calculating the weighted average of transcript counts for randomly selected cells
    from the specified clusters.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with observations (cells) and variables (genes).
    clusters : list
        A list of integers representing the clusters to use for hybrid cell creation.
    groupby : str
        The name of the column in `adata.obs` to group cells by before selecting cells at random.
    n_hybrid_cells : int
        The number of hybrid cells to generate.
    weights : list
        A list of floats representing the weights to use for the weighted average.

    Returns
    -------
    AnnData
        Annotated data matrix containing only the hybrid cells.
    """
    hybrid_list = []
    for i in range(n_hybrid_cells):
        # Randomly select cells from the specified clusters
        cells = adata.obs_names[adata.obs[groupby].isin(clusters)]
        cells = np.random.choice(cells, size=10, replace=False)

        # Calculate the weighted average transcript counts for the selected cells
        weighted_counts = np.average(adata[cells].X, axis=0, weights=weights)

        # Create an AnnData object for the hybrid cell
        hybrid = AnnData(
            weighted_counts.reshape(1, -1),
            obs={'hybrid': [f'Hybrid Cell {i+1}']},
            var=adata.var
        )

        # Append the hybrid cell to the hybrid list
        hybrid_list.append(hybrid)

    # Concatenate the hybrid cells into a single AnnData object
    hybrid = ad.concat(hybrid_list, join='outer')

    return hybrid



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
