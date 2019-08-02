
"""
Created on Wed Jul 24 16:41:03 2019

@author: SamCrowl
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier as randomforest

def makeRandDF(adata, nrand = 50):
    """This should work, but has not yet been tested. This function may eventually be moved to another file, as it may be useful for other functions.
        
    """
    
    expMat = makeExpMat(adata)
    pdRand = pd.DataFrame()
    for i in range(nrand):
        tmp = np.empty()
        for j in adata.var.values:
            tmp = np.append(tmp, expMat.loc[j].sample(n=1).values)
        pdRand = pdRand.append(tmp)
    
    return(pdRand)
    

def sc_makeClassifier(aTrans, genes, groups, nrand = 50, ntrees = 1000):
    """Incomplete function. Once completed, will create a random forest based on transformed matrix and cell type groups
        
    """
    grps = aTrans.obs[groups]
    randDat = makeRandDF(aTrans, nrand)
    ggenes = np.empty(nrand); ggenes.fill('rand')
    
    missingGenes = np.setdiff1d(np.unique(genes), allgenes)
    print('Number of missing genes ', missingGenes.size)
    #First make instance of random forest classifier
    rf_tspAll = randomforest(n_estimators = ntrees)
    #Then use attribute .fit(X, y) to apply build random forest, where X is experimental matrix, y are labels
    rf_tspAll.fit(aTrans.X, grps)
    return(rf_tspAll)
    
