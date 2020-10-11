import datetime
import numpy as np
import pandas as pd
from anndata import AnnData
import scanpy as sc

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
def splitCommonAnnData(adata, ncells, dLevel="cell_ontology_class", cellid = "cell", cells_reserved = 3):
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
