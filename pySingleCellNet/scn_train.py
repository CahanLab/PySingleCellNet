import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from sklearn.ensemble import RandomForestClassifier
from scipy.sparse import csr_matrix
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

def scn_train(aTrain,dLevel,nTopGenes = 100,nTopGenePairs = 100,nRand = 100, nTrees = 1000,stratify=False,counts_per_cell_after=1e4, scaleMax=10, limitToHVG=False, normalization = True, include_all_genes = False):
    warnings.filterwarnings('ignore')
    stTrain= aTrain.obs
    
    expRaw = aTrain.to_df()
    expRaw = expRaw.loc[stTrain.index.values]

    adNorm = aTrain.copy()
    if normalization == True:
        sc.pp.normalize_per_cell(adNorm, counts_per_cell_after=counts_per_cell_after)
        sc.pp.log1p(adNorm)

        print("HVG")
        if limitToHVG:
            sc.pp.highly_variable_genes(adNorm, min_mean=0.0125, max_mean=4, min_disp=0.5)
            adNorm = adNorm[:, adNorm.var.highly_variable]

        sc.pp.scale(adNorm, max_value=scaleMax)

    expTnorm = adNorm.to_df()
    expTnorm=expTnorm.loc[stTrain.index.values]

    ### expTnorm= pd.DataFrame(data=aTrain.X,  index= aTrain.obs.index.values, columns= aTrain.var.index.values)
    ### expTnorm=expTnorm.loc[stTrain.index.values]
    print("Matrix normalized")
    ### cgenesA, grps, cgenes_list =findClassyGenes(expTnorm,stTrain, dLevel = dLevel, topX = nTopGenes)
    if include_all_genes == False:
        cgenesA, grps, cgenes_list =findClassyGenes(expTnorm,stTrain, dLevel = dLevel, topX = nTopGenes)
    else: 
        cgenesA = np.array(aTrain.var.index)
        grps = aTrain.obs[dLevel]
        cgenes_list = dict()
        for g in np.unique(grps):
            cgenes_list[g] = cgenesA

    print("There are ", len(cgenesA), " classification genes\n")
    ### xpairs= ptGetTop(expTnorm.loc[:,cgenesA], grps, cgenes_list, topX=nTopGenePairs, sliceSize=5000)
    xpairs= ptGetTop(expTnorm.loc[:,cgenesA], grps, cgenes_list, topX=nTopGenePairs, sliceSize=5000)

    print("There are", len(xpairs), "top gene pairs\n")
    pdTrain= query_transform(expRaw.loc[:,cgenesA], xpairs)
    print("Finished pair transforming the data\n")
    tspRF=sc_makeClassifier(pdTrain.loc[:, xpairs], genes=xpairs, groups=grps, nRand = nRand, ntrees = nTrees, stratify=stratify)
    return [cgenesA, xpairs, tspRF]

def scn_classify(adata, cgenes, xpairs, rf_tsp, nrand = 0 ):
    classRes = scn_predict(cgenes, xpairs, rf_tsp, adata, nrand = nrand)
    categories = classRes.columns.values
    adNew = ad.AnnData(classRes, obs=adata.obs, var=pd.DataFrame(index=categories))
    # adNew.obs['category'] =  classRes.idxmax(axis=1)
    adNew.obs['SCN_class'] =  classRes.idxmax(axis=1)
    return adNew

def add_classRes(adata: AnnData, adClassRes, copy=False) -> AnnData:
    cNames = adClassRes.var_names
    for cname in cNames:
        adata.obs[cname] = adClassRes[:,cname].X
    # adata.obs['category'] = adClassRes.obs['category']
    adata.obs['SCN_class'] = adClassRes.obs['SCN_class']
    return adata if copy else None

def check_adX(adata: AnnData) -> AnnData:
    from scipy import sparse
    if( isinstance(adata.X, np.ndarray)):
        adata.X = sparse.csr_matrix(adata.X)


def scn_predict(cgenes, xpairs, rf_tsp, aDat, nrand = 2):
    if isinstance(aDat.X,np.ndarray):
        # in the case of aDat.X is a numpy array 
        aDat.X = ad._core.views.ArrayView(aDat.X)
###    expDat= pd.DataFrame(data=aDat.X, index= aDat.obs.index.values, columns= aDat.var.index.values)
    expDat= pd.DataFrame(data=aDat.X.toarray(), index= aDat.obs.index.values, columns= aDat.var.index.values)
    expValTrans=query_transform(expDat.reindex(labels=cgenes, axis='columns', fill_value=0), xpairs)
    classRes_val=rf_classPredict(rf_tsp, expValTrans, numRand=nrand)
    return classRes_val

def rf_classPredict(rfObj,expQuery,numRand=50):
    if numRand > 0 :
        randDat=randomize(expQuery, num=numRand)
        expQuery=pd.concat([expQuery, randDat])
    xpreds= pd.DataFrame(rfObj.predict_proba(expQuery), columns= rfObj.classes_, index=expQuery.index)
    return xpreds

def scn_train_edit(aTrain,dLevel,nTopGenes = 100,nTopGenePairs = 100,nRand = 100, nTrees = 1000,stratify=False,counts_per_cell_after=1e4, scaleMax=10, limitToHVG=True, normalization = True, include_all_genes = False):
    warnings.filterwarnings('ignore')
    stTrain= aTrain.obs
    
    expRaw = aTrain.to_df()
    expRaw = expRaw.loc[stTrain.index.values]

    adNorm = aTrain.copy()
    if normalization == True:
        sc.pp.normalize_per_cell(adNorm, counts_per_cell_after=counts_per_cell_after)
        sc.pp.log1p(adNorm)

        print("HVG")
        if limitToHVG:
            sc.pp.highly_variable_genes(adNorm, min_mean=0.0125, max_mean=4, min_disp=0.5)
            adNorm = adNorm[:, adNorm.var.highly_variable]

        sc.pp.scale(adNorm, max_value=scaleMax)

    expTnorm = adNorm.to_df()
    expTnorm=expTnorm.loc[stTrain.index.values]

    ### expTnorm= pd.DataFrame(data=aTrain.X,  index= aTrain.obs.index.values, columns= aTrain.var.index.values)
    ### expTnorm=expTnorm.loc[stTrain.index.values]
    print("Matrix normalized")
    ### cgenesA, grps, cgenes_list =findClassyGenes(expTnorm,stTrain, dLevel = dLevel, topX = nTopGenes)
    if include_all_genes == False:
        cgenesA, grps, cgenes_list =findClassyGenes_edit(adNorm, dLevel = dLevel, topX = nTopGenes)
    else: 
        cgenesA = np.array(aTrain.var.index)
        grps = aTrain.obs[dLevel]
        cgenes_list = dict()
        for g in np.unique(grps):
            cgenes_list[g] = cgenesA

    print("There are ", len(cgenesA), " classification genes\n")
    ### xpairs= ptGetTop(expTnorm.loc[:,cgenesA], grps, cgenes_list, topX=nTopGenePairs, sliceSize=5000)
    xpairs= ptGetTop(expTnorm.loc[:,cgenesA], grps, cgenes_list, topX=nTopGenePairs, sliceSize=5000)

    print("There are", len(xpairs), "top gene pairs\n")
    pdTrain= query_transform(expRaw.loc[:,cgenesA], xpairs)
    print("Finished pair transforming the data\n")
    tspRF=sc_makeClassifier(pdTrain.loc[:, xpairs], genes=xpairs, groups=grps, nRand = nRand, ntrees = nTrees, stratify=stratify)
    return [cgenesA, xpairs, tspRF, cgenes_list]

def add_classRes_result(adata, adClassRes, copy=False):
    # cNames = adClassRes.var_names
    # for cname in cNames:
    #     adata.obs[cname] = adClassRes[:,cname].X
    # adata.obs['category'] = adClassRes.obs['category']
    adata.obs['SCN_result'] = adClassRes.obs['SCN_class']
    return adata if copy else None

def add_training_dlevel(adata, dlevel):
    adata.obs['SCN_result'] = adata.obs[dlevel]
    return adata

def select_type_pairs(adTrain, adQuery, Cell_Type, threshold, dlevel = 'cell_ontology_class'):
    ind_train = []
    ind_query = []

    for i in range(adTrain.obs.shape[0]):
        if adTrain.obs[dlevel][i] == Cell_Type:
            ind_train.append(i)

    for i in range(adQuery.obs.shape[0]):
        if adQuery.obs['SCN_class'][i] == Cell_Type and adQuery.obs[Cell_Type][i] >= threshold:
            ind_query.append(i)

    
    if type(adTrain.X) is csr_matrix:
        Mat_Train = adTrain.X.todense()
    elif type(adTrain.X) is np.matrix:
        Mat_Train = adTrain.X
    else:
        Mat_Train = np.array(adTrain.X)

    Mat_Train = Mat_Train[ind_train,:]
    Obs_Train = ['Train' for _ in range(ind_train.__len__())]


    if type(adQuery.X) is csr_matrix:
        Mat_Query = adQuery.X.todense()
    elif type(adQuery.X) is np.matrix:
        Mat_Query = adQuery.X
    else:
        Mat_Query = np.array(adQuery.X)

    Mat_Query = Mat_Query[ind_query,:]
    Obs_Query = ['Query' for _ in range(ind_query.__len__())]


    Mat_ranking = np.concatenate((Mat_Train, Mat_Query))
    Obs_ranking = Obs_Train+Obs_Query

    adRanking = sc.AnnData(Mat_ranking, obs=pd.DataFrame(Obs_ranking, columns=['source']))
    adRanking.var_names = adTrain.var_names

    sc.tl.rank_genes_groups(adRanking, groupby='source', groups=['Query'], reference='Train', method='t-test')


    Pvalues = []
    pvals = adRanking.uns['rank_genes_groups']['pvals']
    for pval in pvals:
        Pvalue = pval[0]
        Pvalues.append(Pvalue)

    logFCs = []
    logfoldchanges = adRanking.uns['rank_genes_groups']['logfoldchanges']
    for logfoldchange in logfoldchanges:
        logFC = logfoldchange[0]
        logFCs.append(logFC)

    gene_names = []
    names = adRanking.uns['rank_genes_groups']['names']
    for name in names:
        gene_name = name[0]
        gene_names.append(gene_name)

    voc_tab = pd.DataFrame(data = [Pvalues,logFCs])
    voc_tab = voc_tab.transpose()
    voc_tab.columns = ['Pvalue', 'logFC']
    voc_tab.index = gene_names

    voc_tab['st'] = voc_tab.index
    voc_tab['st'] = voc_tab['st'].astype('category')
    voc_tab['st'].cat.reorder_categories(adRanking.var_names, inplace=True)
    voc_tab.sort_values('st', inplace=True)
    voc_tab.set_index(['st'])
    voc_tab = voc_tab.drop(columns=['st'])

    adRanking.varm['voc_tab'] = voc_tab

    return adRanking