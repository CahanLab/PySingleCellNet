import pandas as pd
import statsmodels.api as sm
import statsmodels.stats.multitest as smt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
from scipy import stats

# takes expression df, gene names(optional if overide expDats) and dThresh (default 0) and gives out
# df (rows=genes, cols= differen stats)
def sc_statTab(expDat, dThresh=0):
    geneNames=expDat.columns.values
    muAll=sc_compMu(expDat, threshold = dThresh);
    alphaAll=sc_compAlpha(expDat,threshold = dThresh);
    meanAll=expDat.apply(np.mean, axis = 0);
    covAll=expDat.apply(sc_cov, axis = 0);
    fanoAll=expDat.apply(sc_fano, axis = 0 );
    maxAll=expDat.apply(np.max, axis = 0 );
    sdAll=expDat.apply(np.std, axis = 0 );
    statTabAll=pd.concat([muAll, alphaAll, meanAll, covAll, fanoAll, maxAll, sdAll], axis=1);
    statTabAll.columns=["mu", "alpha","overall_mean", "cov", "fano", "max_val", "sd"]
    return statTabAll;


def sc_compAlpha(expDat, threshold=0, pseudo=False):
    def singleGene(col, thresh, pseu):
        if pseudo:
            return (np.sum(col>thresh)+1)/float(len(col)+1)
        else:
            return np.sum(col>thresh)/float(len(col))
    return expDat.apply(singleGene, axis=0, args=(threshold,pseudo,))

def sc_compMu(expDat, threshold=0, ):
    def singleGene(col, thresh):
           return np.sum(col[col>thresh])/float(len(col[col>thresh]))
    return expDat.apply(singleGene, axis=0, args=(threshold,)).fillna(0)

def repNA(df):
    return df.fillna(0)

def sc_fano(vector):
    return np.true_divide(np.var(vector),np.mean(vector))

def sc_cov(vector):
    return np.true_divide(np.std(vector),np.mean(vector))

def sc_filterGenes(geneStats,alpha1=0.1,alpha2=0.01,mu=2):
    return geneStats[np.logical_or(geneStats.alpha>alpha1, np.logical_and(geneStats.alpha>alpha2,geneStats.mu>mu))].index.values

def sc_filterCells(sampTab, minVal=1e3, maxValQuant=0.95):
    q=np.quantile(sampTab.umis, maxValQuant)
    return sampTab[np.logical_and(sampTab.umis>minVal, sampTab.umis<q)].index.values

def sc_findEnr(expDat, sampTab, dLevel="group"):
    summ=expDat.groupby(sampTab[dLevel]).median()
    dict={}
    for n in range(0, summ.index.size):
        temp= np.subtract(summ.iloc[n,:],summ.drop(index=summ.index.values[n]).apply(np.median, axis=0))
        dict[summ.index.values[n]]=summ.columns.values[np.argsort(-1*temp)].tolist()
    return dict

def enrDiff(expDat, sampTab, dLevel="group"):
    groups=np.unique(sampTab[dLevel])
    summ=expDat.groupby(sampTab[dLevel]).median()
    ref=summ.copy()
    for n in range(0, ref.index.size):
        summ.iloc[n,:]= np.subtract(summ.iloc[n,:],ref.drop(index=ref.index.values[n]).apply(np.median, axis=0))
    return summ

def binGenesAlpha(geneStats, nbins=20):
    max=np.max(geneStats['alpha'])
    min=np.min(geneStats['alpha'])
    rrange=max-min;
    inc= rrange/nbins
    threshs=np.arange(max, min, -1*inc)
    res=pd.DataFrame(index=geneStats.index.values, data=np.arange(0,geneStats.index.size,1), columns=["bin"] )
    for i in range(0, len(threshs)):
        res.loc[geneStats["alpha"]<=threshs[i],0]=len(threshs)-i
    return res

def binGenes(geneStats, nbins=20, meanType="overall_mean"):
    max=np.max(geneStats[meanType])
    min=np.min(geneStats[meanType])
    rrange=max-min;
    inc= rrange/nbins
    threshs=np.arange(max, min, -1*inc)
    res=pd.DataFrame(index=geneStats.index.values, data=np.arange(0,geneStats.index.size,1), columns=["bin"] )
    for i in range(0, len(threshs)):
        res.loc[geneStats[meanType]<=threshs[i],"bin"]=len(threshs)-i
    return res

def findVarGenes(geneStats,zThresh=2,meanType="overall_mean"):
    zscs=pd.DataFrame(index=geneStats.index.values, data=np.zeros([geneStats.index.size, 3]), columns=["alpha", meanType, "mu"])
    mTypes=["alpha", meanType, "mu"]
    scaleVar=["fano","fano","cov"]
    for i in range(0,3):
        sg=binGenes(geneStats, meanType=mTypes[i])
        bbins=np.unique(sg["bin"])
        for b in bbins:
            if(np.unique(geneStats.loc[sg.bin==b, scaleVar[i]]).size>1):
                tmpZ=stats.zscore(geneStats.loc[sg.bin==b, scaleVar[i]])
            else:
                tmpZ=np.zeros(geneStats.loc[sg.bin==b, scaleVar[i]].index.size).T
            zscs.loc[sg.bin==b, mTypes[i]]=tmpZ
    return(zscs.loc[np.logical_and(zscs.iloc[:,0]>zThresh, np.logical_and(zscs.iloc[:,1]>zThresh,zscs.iloc[:,2]>zThresh))].index.values)


def sc_sampR_to_pattern(sampR):
    d_ids = np.unique(sampR)
    nnnc = len(sampR)
    dict = {}
    for d_id in d_ids:
        x = np.zeros(nnnc)
        x[np.where(np.isin(sampR, d_id))] = 1
        dict[d_id] = x
    return dict

def minTab(sampTab, dLevel):
    myMin=np.min(sampTab[dLevel].value_counts())
    grouped = sampTab.groupby(dLevel,as_index=False)
    res=grouped.apply(lambda x: x.sample(n=myMin, replace=False)).reset_index(level=0, drop=True)
    return res

def sc_testPattern(pattern,expDat):
    def oneTest(y,X):
        X = sm.add_constant(X)
        model = sm.OLS(y,X)
        results = model.fit()
        ccor = results.rsquared**0.5*np.sign(results.tvalues[1])
        pval = results.pvalues[1]
        return np.array([pval, ccor])
    tempDf = expDat.apply(oneTest, axis=0, args=(pattern,))
    tempDf = np.transpose(tempDf)
    res = pd.DataFrame(tempDf)
    res.columns = ["pval","cval"]
    _,res["holm"],_,_ = smt.multipletests(res["pval"].values, method="holm")
    return res

def par_findSpecGenes(expDat, sampTab, dLevel="group", minSet=True):
    if minSet:
        samps=minTab(sampTab, dLevel)
    else:
        samps=sampTab.copy()
    pats=sc_sampR_to_pattern(samps[dLevel])
    exps= expDat.loc[samps.index,:]
    res={}
    levels=list(pats.keys())
    for i in range(0, len(levels)):
        res[levels[i]]=sc_testPattern(pats[levels[i]],exps)
    return res

def getTopGenes(xDat, topN=3):
    return xDat.sort_values(by='cval', ascending=False).index.values[0:topN]

def getSpecGenes(xDatList, topN=50):
    groups=list(xDatList.keys())
    allG=[]
    for i in range(0, len(groups)):
        topNs=getTopGenes(xDatList[groups[i]], topN)
        allG.append(topNs)
    allG=np.array(allG).reshape(-1,1)
    u, c = np.unique(allG, return_counts=True)
    u[c>1]=np.nan
    allG[~np.isin(allG, u)]=np.nan
    specGenes=allG.reshape(len(groups), topN)
    res={}
    for i in range(0, len(specGenes)):
        res[groups[i]]=specGenes[i, ~pd.isnull(specGenes[i])].tolist()
    return res

def getTopGenesList(xDatList, topN = 50):
    groups=list(xDatList.keys())
    temp=[]
    for i in range(0, len(groups)):
        topNs=getTopGenes(xDatList[groups[i]], topN)
        res=", ".join(topNs)
        temp.append(res)
    res={};
    for i in range(0, len(groups)):
        res[groups[i]]=temp[i]
    return res
    
