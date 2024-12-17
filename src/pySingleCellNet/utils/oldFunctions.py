import numpy as np
import pandas as pd
from anndata import AnnData
import matplotlib.pyplot as plt
import scanpy as sc
import anndata as ad

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









