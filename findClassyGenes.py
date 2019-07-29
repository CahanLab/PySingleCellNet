
"""
Created on Thu Jul 18 17:07:18 2019

@author: SamCrowl
"""

def rankGenes(aTrain, dLevel, topX = 10, method = 'wilcoxon'):
    sc.tl.rank_genes_groups(aTrain, dLevel, n_genes = topX, method = method)
    pdClassyGenes = pd.DataFrame(aTrain.uns['rank_genes_groups']['names'])
    return(pdClassyGenes)
    
def compileClassyGenes(pdClassyGenes, dLevel):
    classygenes = np.empty(0)
    for i in set(aTrain.obs[dLevel]):
        classygenes = np.append(classygenes, pdClassyGenes[[i]])
        
    return(set(classygenes))
    
def findClassyGenes(aTrain, dLevel, topX = 10, method = 'wilcoxon'):
    pdClassyGenes = rankGenes(aTrain, dLevel, topX, method)
    classygenes = compileClassyGenes(pdClassyGenes, dLevel)
    return(classygenes)
