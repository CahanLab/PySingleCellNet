
"""
Created on Thu Jul 18 17:07:18 2019

@author: SamCrowl
"""
import scanpy as sc
import numpy as np
import pandas as pd

def rankGenes(aTrain, dLevel, topX = 10, method = 'wilcoxon'):
    sc.tl.rank_genes_groups(aTrain, dLevel, n_genes = topX, method = method)
    pdClassyGenes = pd.DataFrame(aTrain.uns['rank_genes_groups']['names'])
    return(pdClassyGenes)
    
def compileClassyGenes(pdClassyGenes, groups):
    classygenes = np.empty(0)
    for i in groups:
        classygenes = np.append(classygenes, pdClassyGenes[[i]])
        
    return(set(classygenes))
    
def findClassyGenes(aTrain, dLevel, topX = 10, method = 'wilcoxon'):
    pdClassyGenes = rankGenes(aTrain, dLevel, topX, method)
    groups = set(aTrain.obs[dLevel])
    classygenes = compileClassyGenes(pdClassyGenes, groups)
    return(classygenes)
