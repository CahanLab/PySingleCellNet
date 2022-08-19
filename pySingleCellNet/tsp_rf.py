import pandas as pd
import numpy as np
import scanpy as sc
from sklearn import linear_model
from itertools import combinations
from .stats import * 

### from stats import * 


def csRenameOrth(adQuery,adTrain,orthTable,speciesQuery='human',speciesTrain='mouse'):
    _,_,cgenes=np.intersect1d(adQuery.var_names.values, orthTable[speciesQuery], return_indices=True)
    _,_,ccgenes=np.intersect1d(adTrain.var_names.values, orthTable[speciesTrain], return_indices=True)
    temp1=np.zeros(len(orthTable.index.values), dtype=bool)
    temp2=np.zeros(len(orthTable.index.values), dtype=bool)
    temp1[cgenes]=True
    temp2[ccgenes]=True
    common=np.logical_and(temp1, temp2)
    oTab=orthTable.loc[common.T,:]
    adT=adTrain[:, oTab[speciesTrain]]
    adQ=adQuery[:, oTab[speciesQuery]]
    adQ.var_names = adT.var_names
    return [adQ, adT]


def csRenameOrth2(expQuery,expTrain,orthTable,speciesQuery='human',speciesTrain='mouse'):
    _,_,cgenes=np.intersect1d(expQuery.columns.values, orthTable[speciesQuery], return_indices=True)
    _,_,ccgenes=np.intersect1d(expTrain.columns.values, orthTable[speciesTrain], return_indices=True)
    temp1=np.zeros(len(orthTable.index.values), dtype=bool)
    temp2=np.zeros(len(orthTable.index.values), dtype=bool)
    temp1[cgenes]=True
    temp2[ccgenes]=True
    common=np.logical_and(temp1, temp2)
    oTab=orthTable.loc[common.T,:]
    expT=expTrain.loc[:, oTab[speciesTrain]]
    expQ=expQuery.loc[:, oTab[speciesQuery]]
    expQ.columns= expT.columns
    return [expQ, expT]


def makePairTab(genes):
    pairs = list(combinations(genes,2))
    labels = ['genes1', 'genes2']
    pTab = pd.DataFrame(data = pairs, columns = labels)
    pTab['gene_pairs'] = pTab['genes1'] + '_' + pTab['genes2']
    return(pTab)

def gnrAll(expDat,cellLabels):
    myPatternG=sc_sampR_to_pattern(cellLabels)
    res={}
    groups=np.unique(cellLabels)
    for i in range(0, len(groups)):
        res[groups[i]]=sc_testPattern(myPatternG[groups[i]], expDat)
    return res

def getClassGenes(diffRes, topX=25, bottom=True):
    xi = ~pd.isna(diffRes["cval"])
    diffRes = diffRes.loc[xi,:]
    sortRes= diffRes.sort_values(by="cval", ascending=False)
    ans=sortRes.index.values[0:topX]
    if bottom:
        l= len(sortRes)-topX
        ans= np.append(ans, sortRes.index.values[l:] ).flatten()
    return ans

def addRandToSampTab(classRes, sampTab, desc, id="cell_name"):
    cNames= classRes.index.values
    snames= sampTab.index.values
    rnames= np.setdiff1d(cNames, snames)
    stNew= pd.DataFrame()
    stNew["rid"]=rnames
    stNew["rdesc"]="rand"
    stTop=sampTab[[id, desc]]
    stNew.columns= [id, desc]
    ans = stTop.append(stNew)
    return ans

def ptSmall(expMat, pTab):
    npairs = len(pTab.index)
    genes1 = pTab['genes1'].values
    genes2 = pTab['genes2'].values
    expTemp=expMat.loc[:,np.unique(np.concatenate([genes1,genes2]))]
    ans = pd.DataFrame(0, index = expTemp.index, columns = np.arange(npairs))
    ans = ans.astype(pd.SparseDtype("int", 0))
    temp1= expTemp.loc[:,genes1]
    temp2= expTemp.loc[:,genes2]
    temp1.columns=np.arange(npairs)
    temp2.columns=np.arange(npairs)
    boolArray = temp1 > temp2
    ans = boolArray.astype(int)
    ans.columns = list(pTab[['gene_pairs']].values.T)
    return(ans)

def findBestPairs(xdiff, n=50, maxPer=3):
    xdiff = xdiff.sort_values(by = ['cval'], ascending = False)
    genes=[]
    genesTemp = list(xdiff.index.values)
    for g in genesTemp:
        genes.append(g[0].split("_"))
    genes = np.unique(np.array(genes).flatten())
    countList = dict(zip(genes, np.zeros(genes.shape)))
    i = 1
    ans = np.empty(0)
    xdiff_index = 0
    pair_names = xdiff.index.values
    while i<n:
        tmpAns = pair_names[xdiff_index]
        tgp = tmpAns[0].split('_')
        if countList[tgp[0]] < maxPer and countList[tgp[1]] < maxPer:
            ans = np.append(ans, tmpAns)
            countList[tgp[0]] = countList[tgp[0]] + 1
            countList[tgp[1]] = countList[tgp[1]] + 1
            i = i + 1
        xdiff_index = xdiff_index + 1
    return(np.array(ans))

def query_transform(expMat, genePairs):
    npairs = len(genePairs)
    ans = pd.DataFrame(0, index = expMat.index, columns = np.arange(npairs))
    genes1=[]
    genes2=[]
    for g in genePairs:
        sp=g.split("_")
        genes1.append(sp[0])
        genes2.append(sp[1])
    expTemp=expMat.loc[:,np.unique(np.concatenate([genes1,genes2]))]
    ans = pd.DataFrame(0, index = expTemp.index, columns = np.arange(npairs))
    ans = ans.astype(pd.SparseDtype("int", 0))
    temp1= expTemp.loc[:,genes1]
    temp2= expTemp.loc[:,genes2]
    temp1.columns=np.arange(npairs)
    temp2.columns=np.arange(npairs)
    boolArray = temp1 > temp2
    ans = boolArray.astype(int)
    ans.columns = genePairs
    return(ans)

def pair_transform(expMat):
    pTab=makePairTab(expMat)
    npairs = len(pTab.index)
    ans = pd.DataFrame(0, index = expMat.index, columns = np.arange(npairs))
    genes1 = pTab['genes1'].values
    genes2 = pTab['genes2'].values
    expTemp=expMat.loc[:,np.unique(np.concatenate([genes1,genes2]))]
    ans = pd.DataFrame(0, index = expTemp.index, columns = np.arange(npairs))
    ans = ans.astype(pd.SparseDtype("int", 0))
    temp1= expTemp.loc[:,genes1]
    temp2= expTemp.loc[:,genes2]
    temp1.columns=np.arange(npairs)
    temp2.columns=np.arange(npairs)
    boolArray = temp1 > temp2
    ans = boolArray.astype(int)
    ans.columns = genePairs
    return(ans)

def gnrBP(expDat,cellLabels,topX=50):
    myPatternG=sc_sampR_to_pattern(cellLabels)
    levels=list(myPatternG.keys())
    ans={}
    for i in range(0, len(levels)):
        xres=sc_testPattern(myPatternG[levels[i]],expDat)
        tmpAns=findBestPairs(xres, topX)
        ans[levels[i]]=tmpAns
    return ans

def ptGetTop (expDat, cell_labels, cgenes_list=None, topX=50, sliceSize = 5000, quickPairs = True):
    if not quickPairs:
        genes=expDat.columns.values
        grps=np.unique(cell_labels)
        myPatternG=sc_sampR_to_pattern(cell_labels)
        pairTab=makePairTab(genes)
        nPairs = len(pairTab)
        start = 0
        stp = np.min([sliceSize, nPairs])
        tmpTab = pairTab.iloc[start:stp,:]
        tmpPdat = ptSmall(expDat, tmpTab)
        statList= dict((k, sc_testPattern(v, tmpPdat)) for k, v in myPatternG.items())
        start= stp
        stp= start + sliceSize
        while start < nPairs:
            print(start)
            if stp > nPairs:
                stp =  nPairs
            tmpTab = pairTab.iloc[start:stp,:]
            tmpPdat = ptSmall(expDat, tmpTab)
            tmpAns=dict((k, sc_testPattern(v, tmpPdat)) for k, v in myPatternG.items())
            for g in grps:
                statList[g]=pd.concat([statList[g], tmpAns[g]])
            start= stp
            stp= start + sliceSize
        res=[]
        for g in grps:
            tmpAns=findBestPairs(statList[g], topX)
            res.append(tmpAns)
        return np.unique(np.array(res).flatten())


    else:
        myPatternG= sc_sampR_to_pattern(cell_labels)
        res=[]
        grps=np.unique(cell_labels)
        for g in grps:
            print(g)
            genes=cgenes_list[g]
            pairTab=makePairTab(genes)
            nPairs=len(pairTab)
            tmpPdat=ptSmall(expDat, pairTab)
            tmpAns=findBestPairs(sc_testPattern(myPatternG[g],tmpPdat), topX)
            res.append(tmpAns)
        return np.unique(np.array(res).flatten())

def findClassyGenes(expDat, sampTab,dLevel, topX=25, dThresh=0, alpha1=0.05,alpha2=.001, mu=2):
    gsTrain=sc_statTab(expDat, dThresh=dThresh)
    ggenes=sc_filterGenes(gsTrain, alpha1=alpha1, alpha2=alpha2, mu=mu)
    grps= sampTab[dLevel]
    xdiff=gnrAll(expDat.loc[:,ggenes], grps)
    groups=np.unique(grps)
    res=[]
    cgenes={}
    for g in groups:
        temp=getClassGenes(xdiff[g], topX)
        cgenes[g]=temp
        res.append(temp)
    cgenes2=np.unique(np.array(res).flatten())
    return [cgenes2, grps, cgenes]

def findClassyGenes_edit(adDat, dLevel, topX=25):
    adTemp = adDat.copy()
    grps = adDat.obs[dLevel]
    groups = np.unique(grps)

    sc.tl.rank_genes_groups(adTemp, dLevel, method='wilcoxon')
    tempTab = pd.DataFrame(adTemp.uns['rank_genes_groups']['names']).head(topX)

    res = []
    cgenes = {}

    for g in groups:
        temp = tempTab[g]
        res.append(temp)
        cgenes[g] = temp.to_numpy()
    cgenes2 = np.unique(np.array(res).flatten())

    print('new functionality run')

    return [cgenes2, grps, cgenes]