
"""
Created on Mon Jul 22 16:53:53 2019

@author: SamCrowl
"""
import numpy as np
import pandas as pd
from anndata import AnnData

def pairSplit(pairArray):
    ans = pd.DataFrame(columns = ['gene1', 'gene2'])
    for i in range(pairArray.size):
        ans.loc[i] = pairArray[i].split('_')
        
    return(ans)
    
    
def query_transform(adata, genePairs):
    """Makes a complete gene-to-gene comparison
    
    Parameters:
        adata: AnnData object with expression data
        genePairs: gene pairs
    Returns:
        AnnData object that contains matrix indicating which gene of a pair is greater
    """
    
    genes = pairSplit(genePairs)
    ans = pd.DataFrame(columns = adata.obs.index)
    
    expMat = makeExpMat(adata)
    for i in range(genePairs.size):
        ans.loc[:, i] = (expMat.loc[:, genes.loc[i,1]].value 
               > expMat.loc[: genes.loc[i,2]].value).astype(int)
        
    ans2 = AnnData(ans, obs = adata.obs, var = pTab[['gene_pairs']])
    return(ans2)
        
