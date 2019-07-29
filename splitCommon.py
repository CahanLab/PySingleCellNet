
"""
Created on Wed Jul 17 10:45:39 2019

@author: SamCrowl
"""

import numpy as np

def splitCommon(adata, ncells, dLevel="cell_ontology_class"):
    """Split a AnnData object into groups for training and assessment
    
    Parameters:
        aData: AnnData object that contains both expression matrix and cell/gene meta data
        ncells: The number of cells to be seperated for training
        dLevel: Descriptor that labels the location of cell type annotations
        
    Returns:
        A list with two AnnData objects, one for training and one for assessment
    
    """
    
    cts = set(adata.obs[dLevel])
    trainingids = np.empty(0)
    for ct in cts:
        print(ct, ": ")
        aX = adata[adata.obs[dLevel] == ct, :]
        ccount = aX.n_obs - 3
        ccount = min([ccount, ncells])
        print(aX.n_obs)
        trainingids = np.append(trainingids, np.random.choice(aX.obs["sample_name"].values, ccount, replace = False))
        
    val_ids = np.setdiff1d(adata.obs["sample_name"].values, trainingids)
    aTrain = adata[np.isin(adata.obs["sample_name"], trainingids),:]
    aTest = adata[np.isin(adata.obs["sample_name"], val_ids),:]
    return([aTrain, aTest])


