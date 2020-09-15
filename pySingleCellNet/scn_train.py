import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import warnings
from .utils import *
from .tsp_rf import *


def randomize(expDat, num=50):
    temp=expDat.to_numpy()
    temp=np.array([np.random.choice(x, len(x), replace=False) for x in temp])
    temp=temp.T
    temp=np.array([np.random.choice(x, len(x), replace=False) for x in temp]).T
    return pd.DataFrame(data=temp, columns=expDat.columns).iloc[0:num,:]

def sc_trans_rnaseq(aDat,total = 10000 ):
    sc.pp.normalize_per_cell(aDat, counts_per_cell_after=total)
    sc.pp.log1p(aDat)
    sc.pp.scale(aDat, max_value=10)
    #return np.log(1+expCountDnW)
    return aDat

def sc_makeClassifier(expTrain, genes, groups, nRand=70, ntrees=2000, stratify=False):
    randDat = randomize(expTrain, num=nRand)
    expT = pd.concat([expTrain, randDat])
    allgenes = expT.columns.values
    missingGenes = np.setdiff1d(np.unique(genes), allgenes)
    ggenes= np.intersect1d(np.unique(genes), allgenes)
    if not stratify:
        clf = RandomForestClassifier(n_estimators=ntrees, random_state=100)
    else:
        clf = RandomForestClassifier(n_estimators=ntrees,class_weight="balanced", random_state=100)
    ggroups=np.append(np.array(groups), np.repeat("rand", nRand)).flatten()
    clf.fit(expT.loc[:,ggenes].to_numpy(), ggroups)
    return clf

def scn_train(aTrain,dLevel,nTopGenes = 10,nTopGenePairs = 25,nRand = 70,nTrees = 1000,stratify=False, weightedDown_total = 10000, weightedDown_dThresh = 0.25):
    warnings.filterwarnings('ignore')
    stTrain= aTrain.obs
    expTnorm= pd.DataFrame(data=aTrain.X,  index= aTrain.obs.index.values, columns= aTrain.var.index.values)
    expTnorm=expTnorm.loc[stTrain.index.values]
    print("Matrix normalized")
    cgenesA, grps, cgenes_list =findClassyGenes(expTnorm,stTrain, dLevel = dLevel, topX = nTopGenes)
    print("There are ", len(cgenesA), " classification genes\n")
    xpairs= ptGetTop(expTnorm.loc[:,cgenesA], grps, cgenes_list, topX=nTopGenePairs, sliceSize=5000)
    print("There are", len(xpairs), "top gene pairs\n")
    pdTrain= query_transform(expTnorm.loc[:,cgenesA], xpairs)
    print("Finished pair transforming the data\n")
    tspRF=sc_makeClassifier(pdTrain.loc[:, xpairs], genes=xpairs, groups=grps, nRand = nRand, ntrees = nTrees, stratify=stratify)
    return [cgenesA, xpairs, tspRF]

def scn_predict(cgenes, xpairs, rf_tsp, aDat, nrand = 2, weightedDown_total = 10000, weightedDown_dThresh = 0.25):
    expDat= pd.DataFrame(data=aDat.X, index= aDat.obs.index.values, columns= aDat.var.index.values)
    expValTrans=query_transform(expDat.reindex(labels=cgenes, axis='columns', fill_value=0), xpairs)
    classRes_val=rf_classPredict(rf_tsp, expValTrans, numRand=nrand)
    return classRes_val

def rf_classPredict(rfObj,expQuery,numRand=50):
    if numRand > 0 :
        randDat=randomize(expQuery, num=numRand)
        expQuery=pd.concat([expQuery, randDat])
    xpreds= pd.DataFrame(rfObj.predict_proba(expQuery), columns= rfObj.classes_, index=expQuery.index)
    return xpreds
