from sklearn import metrics
import numpy as np
import pandas as pd
import seaborn as sns
from .stats import *
from .scn_train import *
import matplotlib
import matplotlib.pyplot as plt

    cts = set(sampTab[dLevel])
    trainingids = np.empty(0)
    for ct in cts:
        aX = sampTab.loc[sampTab[dLevel] == ct, :]
        ccount = len(aX.index)
        trainingids = np.append(trainingids, np.random.choice(aX.index.values, int(ccount*prop), replace = False))
    val_ids = np.setdiff1d(sampTab.index, trainingids, assume_unique = True)
    sampTrain = sampTab.loc[trainingids,:]
    sampVal = sampTab.loc[val_ids,:]
    return([sampTrain, sampVal])

def sc_classAssess(stDat,washedDat, dLevel = "description1", dLevelSID="sample_name", minCells = 40, dThresh = 0, propTrain=0.25, nRand = 50, nTrees=2000):
    goodGrps = np.unique(stTrain.newAnn)[stTrain.newAnn.value_counts()>minCells]
    stTmp=stDat.loc[np.isin(stDat[dlevel], goodGrps) , :]
    expDat_good = washedDat["expDat"].loc[stTmp.index, :]
    stTrain, stVal = divide_sampTab(stTmp, propTrain, dLevel = dLevel)
    expTrain=expDat_good.loc[stTrain.index,:]
    expVal=expDat_good.loc[stVal.index,:]
    varGenes = findVarGenes(expDat_good, washedDat["geneStats"])
    cellgrps=stTrain[dLevel]
    testRFs=sc_makeClassifier(expTrain, genes=varGenes, groups=cellgrps, nRand=nRand, ntrees=nTrees)
    ct_scores=rf_classPredict(testRFs, expVal)
    assessed= [ct_scores, stVal, stTrain]
    return assessed

def sc_classThreshold(vect, classification, thresh):
    TP=0;
    FN=0;
    FP=0;
    TN=0;
    calledPos = vect.loc[vect>thresh].index.values
    calledNeg = vect.loc[vect<=thresh].index.values

    if (np.isin(classification, calledPos)):
        TP = 1
        FN = 0
        FP = len(calledPos) - 1
        TN = len(calledNeg)
    else:
        TP = 0
        FN = 1
        FP = len(calledPos)
        TN = len(calledNeg) -1
    Accu = (TP + TN)/(TP + TN + FP + FN)
    return Accu

def cn_clPerf(vect, sampTab, dLevel, classification, thresh, dLevelSID="sample_id"):
    TP=0;
    FN=0;
    FP=0;
    TN=0;
    sampIDs = vect.index.values;
    classes = sampTab.loc[sampIDs,dLevel];
    actualPos = sampTab.loc[sampTab[dLevel]==classification,dLevelSID]
    actualNeg = sampTab.loc[sampTab[dLevel]!=classification,dLevelSID]
    calledPos = vect.loc[vect>thresh].index.values
    calledNeg = vect.loc[vect<=thresh].index.values
    TP = len(np.intersect1d(actualPos, calledPos));
    FP = len(np.intersect1d(actualNeg, calledPos));
    FN = len(actualPos)-TP;
    TN = len(actualNeg)-FP;
    return([TP, FN, FP, TN]);


def cn_eval(vect, sampTab, dLevel, classification, threshs=np.arange(0,1,0.05),dLevelSID="sample_id"):
    ans=np.zeros([len(threshs), 7])
    for i in range(0, len(threshs)):
        thresh = threshs[i];
        ans[i,0:4] = cn_clPerf(vect, sampTab, dLevel, classification, thresh, dLevelSID=dLevelSID);
    ans[:,4] = threshs;
    ans=pd.DataFrame(data=ans, columns=["TP", "FN", "FP", "TN", "thresh","FPR", "TPR"]);
    TPR=ans['TP']/(ans['TP']+ans['FN']);
    FPR=ans['FP']/(ans['TN']+ans['FP']);
    ans['TPR']=TPR;
    ans['FPR']=FPR;
    return ans

def cn_classAssess(ct_scores, stVal, classLevels="description2", dLevelSID="sample_id", resolution=0.005):
    allROCs = {}
    evalAll=np.zeros([len(ct_scores.columns),2])
    classifications= ct_scores.columns.values;
    i=0
    for xname in classifications:
        classification=classifications[i];
        tmpROC= cn_eval(ct_scores[xname],stVal,classLevels,xname,threshs=np.arange(0,1,resolution), dLevelSID=dLevelSID);
        allROCs[xname] = tmpROC;
        i = i + 1;
    return allROCs;

def assess_comm(ct_scores, aTrain, aQuery, resolution = 0.005, nRand = 50, dLevelSID = "sample_name", classTrain = "cell_ontology_class", classQuery = "description2"):
    stTrain= aTrain.obs
    stQuery= aQuery.obs
    shared_cell_type = np.intersect1d(np.unique(stTrain[classTrain]), np.unique(stQuery[classQuery]))
    stVal_com = stQuery.loc[np.isin(stQuery[classQuery], shared_cell_type),:]
    if(nRand > 0):
        tmp = np.empty([nRand, len(stVal_com.columns)], dtype=np.object)
        tmp[:]="rand"
        tmp=pd.DataFrame(data=tmp, columns=stVal_com.columns.values )
        tmp[dLevelSID] = ct_scores.index.values[(len(ct_scores.index) - nRand):len(ct_scores.index)]
        tmp.index= tmp[dLevelSID]
        stVal_com= pd.concat([stVal_com, tmp])
    cells_sub = stVal_com[dLevelSID]
    ct_score_com = ct_scores.loc[cells_sub,:]
    report= {}
    ct_scores_t = ct_score_com.T
    true_label = stVal_com[classQuery]
    y_true=true_label.str.get_dummies()
    eps = 1e-15
    y_pred = np.maximum(np.minimum(ct_scores, 1 - eps), eps)
    multiLogLoss = (-1 / len(ct_scores_t.index)) * np.sum(np.matmul(y_true.T.values, np.log(y_pred.values)))
    pred_label = ct_scores.idxmax(axis=1)
    cm=pd.crosstab(true_label, pred_label)
    #in case of misclassfication where there are classifiers that are not used
    if (len(np.setdiff1d(np.unique(true_label), np.unique(pred_label))) != 0):
        misCol = np.setdiff1d(np.unique(true_label), np.unique(pred_label))
        for i in range(0, len(misCol)):
            added = pd.DataFrame(np.zeros([len(cm.index), 1]), index=cm.index)
            cm = pd.concat([cm, added], axis=1)
        cm.columns.values[(len(cm.columns) - len(misCol)) : len(cm.columns)] = misCol
    if (len(np.setdiff1d(np.unique(pred_label), np.unique(true_label))) != 0):
        misRow = np.setdiff1d(np.unique(pred_label), np.unique(true_label))
        for i in range(0, len(misRow)):
            added = pd.DataFrame(np.zeros([1, len(cm.columns)]), columns= cm.columns)
            cm = pd.concat([cm, added], axis=0)
        cm.index.values[(len(cm.index) - len(misRow)) : len(cm.index)] = misRow
    cm= cm.loc[cm.index.values,:]
    n = np.sum(np.sum(cm))
    nc = len(cm.index)
    diag = np.diag(cm)
    rowsums = np.sum(cm, axis=1)
    colsums = np.sum(cm, axis=0)
    p = rowsums / n
    q = colsums / n
    expAccuracy = np.dot(p,q)
    accuracy = np.sum(diag) / n
    confusionMatrix = cn_classAssess(ct_score_com, stVal_com, classLevels= classQuery, dLevelSID=dLevelSID, resolution=resolution)
    PR_ROC = cal_class_PRs(confusionMatrix)
    nonNA_PR = PR_ROC.dropna(subset=['recall'], axis="rows").copy()
    nonNA_PR.loc[np.logical_and(nonNA_PR["TP"] == 0 , nonNA_PR["FP"] ==0), "precision"] = 1
    w = []
    areas = []
    for i in range(0,  len(np.unique(nonNA_PR["ctype"]))):
        tmp = nonNA_PR.loc[np.isin(nonNA_PR["ctype"], np.unique(nonNA_PR["ctype"])[i]),:]
        area = metrics.auc(tmp["recall"], tmp["precision"])
        areas.append(area)
        w.append(np.sum(np.isin(stVal_com[classQuery], np.unique(nonNA_PR["ctype"])[i]))/len(stVal_com.index))
    report['accuracy'] = accuracy
    report['kappa'] = (accuracy - expAccuracy) / (1 - expAccuracy)
    report['AUPRC_w'] = np.mean(areas)
    report['AUPRC_wc'] = np.average(a=areas, weights=w)
    report['multiLogLoss'] = multiLogLoss
    report['cm'] = cm
    report['confusionMatrix'] = confusionMatrix
    report['nonNA_PR'] = nonNA_PR
    report['PR_ROC'] = PR_ROC
    return(report)

def cal_class_PRs(assessed):
    ctts = list(assessed.keys())
    dat = list(assessed.values())
    for i in range(0, len(dat)):
        dat[i]["ctype"]=ctts[i]
        dat[i]["precision"]=dat[i]["TP"]/(dat[i]["TP"]+ dat[i]["FP"])
        dat[i]["recall"]=dat[i]["TP"]/(dat[i]["TP"]+ dat[i]["FN"])
    return (pd.concat(dat, axis=0))


def plot_metrics(assessed):
    plt.bar([0,1], [assessed["kappa"],assessed["AUPRC_w"]])
    plt.xticks([0,1], ("cohen's kappa", "mean_AUPRC"))
    plt.ylim([0,1])
    plt.xlabel("metric")
    plt.ylabel("value")

def plot_PRs(assessed):
    att = assessed["nonNA_PR"]
    g = sns.FacetGrid(att, col="ctype", col_wrap=5, sharex=False, sharey=False)
    g = g.map(plt.plot, "recall", "precision", marker=".")
