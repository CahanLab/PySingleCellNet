import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from sklearn.ensemble import RandomForestClassifier
import warnings
from .utils import *
from .tsp_rf import *

def randomize(expDat, num=50):

    """Create random cell profiles from existing data
    Parameteres
    -----------
    expDat:
        expDat (pandas dataframe): Gene Expression matrix
    num: int
        number of profiles randomly generated

    Returns:
    --------
        a randomized gene expression matrix (pandas dataframe) with added random cell profiles

    """

    temp=expDat.to_numpy()
    temp=np.array([np.random.choice(x, len(x), replace=False) for x in temp])
    temp=temp.T
    temp=np.array([np.random.choice(x, len(x), replace=False) for x in temp]).T
    return pd.DataFrame(data=temp, columns=expDat.columns).iloc[0:num,:]

def sc_trans_rnaseq(aDat,total = 10000 ):

    """Nomarlize and log-transform the raw expression count
    Parameteres
    -----------
    aDat:
        expDat (pandas dataframe): Gene Expression matrix (raw counts)
    total: int    
         total count after normalization
    
    Returns:
    --------
        normalized and log-transformed gene expression matrix
    """

    sc.pp.normalize_per_cell(aDat, counts_per_cell_after=total)
    sc.pp.log1p(aDat)
    sc.pp.scale(aDat, max_value=10)
    #return np.log(1+expCountDnW)
    return aDat

def sc_makeClassifier(expTrain, genes, groups, nRand=70, ntrees=2000, stratify=False):
    
    """Build random forest classifier
    Parameteres
    -----------
    expTrain:
        expTrain (pandas dataframe): Gene Expression matrix (raw counts)
    genes: str
        gene used for training
    groups: str
        annotation used for training
    nRand: int
        number of random cell profiles generated
    ntrees: int
        number of decision trees used in RandomForestClassifier
    stratify: bool
        whether to stratify samples for each class
    
    Returns
    -----------
        RandomForestClassifier
    """ 

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

def scn_train(aTrain,dLevel,nTopGenes = 100,nTopGenePairs = 100,nRand = 100, nTrees = 1000,stratify=False,counts_per_cell_after=1e4, scaleMax=10, limitToHVG=False):

    """Train for pySCN classifier
    Parameteres
    -----------
    aTrain:
        adata that contains training data for pySCN
    dLevel: str
        train labels
    nTopGenes: int
        numbers of top genes used to compute for topPairs
    nTopGenePairs: int
        number of topPairs used to construct pySCN classifier
    ntrees: int
        number of decision trees used in RandomForestClassifier
    stratify: bool
        whether to stratify samples for each class
    counts_per_cell_after: int
        total count after normalization
    scaleMax: int
        scaling factor 
    limitToHVG: bool
        whether to limit the gene selection to highly_variable_genes
    
    Returns
    -----------
        pySCN classifier
    """ 

    warnings.filterwarnings('ignore')
    stTrain= aTrain.obs
    
    expRaw = pd.DataFrame(data=aTrain.X.toarray(),  index= aTrain.obs.index.values, columns= aTrain.var.index.values)
    expRaw = expRaw.loc[stTrain.index.values]

    adNorm = aTrain.copy()
    sc.pp.normalize_per_cell(adNorm, counts_per_cell_after=counts_per_cell_after)
    sc.pp.log1p(adNorm)

    print("HVG")
    if limitToHVG:
        sc.pp.highly_variable_genes(adNorm, min_mean=0.0125, max_mean=4, min_disp=0.5)
        adNorm = adNorm[:, adNorm.var.highly_variable]

    sc.pp.scale(adNorm, max_value=scaleMax)
    expTnorm= pd.DataFrame(data=adNorm.X,  index= adNorm.obs.index.values, columns= adNorm.var.index.values)
    expTnorm=expTnorm.loc[stTrain.index.values]

    ### expTnorm= pd.DataFrame(data=aTrain.X,  index= aTrain.obs.index.values, columns= aTrain.var.index.values)
    ### expTnorm=expTnorm.loc[stTrain.index.values]
    print("Matrix normalized")
    ### cgenesA, grps, cgenes_list =findClassyGenes(expTnorm,stTrain, dLevel = dLevel, topX = nTopGenes)
    cgenesA, grps, cgenes_list =findClassyGenes(expTnorm,stTrain, dLevel = dLevel, topX = nTopGenes)
    print("There are ", len(cgenesA), " classification genes\n")
    ### xpairs= ptGetTop(expTnorm.loc[:,cgenesA], grps, cgenes_list, topX=nTopGenePairs, sliceSize=5000)
    xpairs= ptGetTop(expTnorm.loc[:,cgenesA], grps, cgenes_list, topX=nTopGenePairs, sliceSize=5000)

    print("There are", len(xpairs), "top gene pairs\n")
    pdTrain= query_transform(expRaw.loc[:,cgenesA], xpairs)
    print("Finished pair transforming the data\n")
    tspRF=sc_makeClassifier(pdTrain.loc[:, xpairs], genes=xpairs, groups=grps, nRand = nRand, ntrees = nTrees, stratify=stratify)
    return [cgenesA, xpairs, tspRF]

def scn_classify(adata, cgenes, xpairs, rf_tsp, nrand = 0 ):

    """Transform & predict labels with pySCN classifier and add the result to AnnData
    Parameteres
    -----------
    adata: AnnData
        query dataset
    cgenes: str
        intersected genes between query dataset and training dataset
    xpairs: str
        topPairs
    rf_tsp:
        pySCN classifier
    nrand: int
        random number of profiles generated for query

    Returns:
    -------
    aNew:AnnData
        SCN classification matrix  

    """
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
    """Transform the query data and predict labels with pySCN classifier
    Parameteres
    -----------
    adata: AnnData
        query dataset
    cgenes: str
        intersected genes between query dataset and training dataset
    xpairs: str
        topPairs
    rf_tsp:
        pySCN classifier
    nrand: int
        random number of profiles generated for query

    Returns:
    -------
    classRes_val:
        SCN classification matrix  
    """
###    expDat= pd.DataFrame(data=aDat.X, index= aDat.obs.index.values, columns= aDat.var.index.values)
    expDat= pd.DataFrame(data=aDat.X.toarray(), index= aDat.obs.index.values, columns= aDat.var.index.values)
    expValTrans=query_transform(expDat.reindex(labels=cgenes, axis='columns', fill_value=0), xpairs)
    classRes_val=rf_classPredict(rf_tsp, expValTrans, numRand=nrand)
    return classRes_val

def rf_classPredict(rfObj,expQuery,numRand=50):
    """Predict labels with pySCN classifier
    Parameteres
    -----------
    expQuery: 
        transformed query dataset
    rfObj:
        pySCN classifier
    numRand: int
        random number of profiles generated for query

    Returns:
    -------
    xpreds:pd DataFrame
        SCN classification matrix  
    """
    if numRand > 0 :
        randDat=randomize(expQuery, num=numRand)
        expQuery=pd.concat([expQuery, randDat])
    xpreds= pd.DataFrame(rfObj.predict_proba(expQuery), columns= rfObj.classes_, index=expQuery.index)
    return xpreds
