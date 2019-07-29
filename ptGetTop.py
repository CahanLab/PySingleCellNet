
"""
Created on Fri Jul 19 09:54:05 2019

@author: SamCrowl
"""
from sklearn import linear_model
from itertools import combinations

def makeExpMat(adata):
    atmp = adata.X
    expMat = pd.DataFrame(atmp, index = aTrain2.obs['sample_name'], columns = aTrain2.var['gene_ids'])
    

def makePairTab(genes):
    """Make pandas data frame that contains all possible gene pair combinations (works)
    
    Parameters:
        genes: numpy array of genes to be used for gene pairs
        
    Returns:
        pandas data frame that contains all possible gene pair combinations
    """
    pairs = list(combinations(genes,2))
    labeles = ['genes1', 'genes2']
    pTab = pd.DataFrame(data = pairs, columns = labels)
    pTab['gene_pairs'] = pTab['genes1'] + '_' + pTab['genes2']
    return(pTab)
    
def ptSmall(expMat, pTab):
    """Completed function, but may be ways to optimize. transforms a matrix based on gene pairs, then uses the transformed matrix to identify top gene pairs.
        
    """
    
    npairs = len(pTab.index)
    #May be a less memory heavy variable type, come back later
    ans = pd.DataFrame(0, index = expMat.index, columns = np.arange(npairs))
    genes1 = pTab[['genes1']].values
    genes2 = pTab[['genes2']].values
    
    for i in range(npairs):
        boolArray = expMat.loc[:, genes1[i]].values > expMat.loc[:, genes2[i]].values
        ans.loc[:, i] = boolArray.astype(int)
    
    #Not sure if there is an easier way to do this, but did not work when it was a numpy array
    ans.columns = list(pTab[['gene_pairs']].values.T)
    return(ans)
    
def sc_sampR_to_pattern(sampR):
    d_ids = np.unique(sampR)
    nnnc = np.prod(sampR.shape)
    ans = {}
    for d_id in d_ids:
        x = np.zeros(nnnc)
        x[np.where(np.isin(sampR, d_id))] = 1
        ans[d_id] = x
        
    return(ans)
    

  

def sc_testPattern(pattern, expDat):
    """Incomplete function.
        
    """
    pval = np.empty(0)
    cval = np.empty(0)
    geneids = expDat.columns.values
    lsregr = linear_model.LinearRegression()
    lsregr.fit(expDat, pattern)
    #Get regression statistics here
    
    
    
    
def findBestPairs(xdiff, n=50, maxPer=3):
    """Code is complete, but it is untested. Finds top gene pairs based upon the stats returned by sc_testPattern
        
    """
    
    xdiff = xdiff.sort_values(by = ['cval'], ascending = False)
    genes = xdiff.index.split('_')
    genes = np.unique(np.append(genes[0], genes[1]))
    #unsure if this will work
    countList = dict(zip(genes, np.zeros(genes.shape)))
    
    i = 1
    ans = np.empty(0)
    xdiff_index = 0
    pair_names = xdiff.index.values
    while i<n:
        tmpAns = pair_names[xdiff_index]
        tgp = tmpAns.split('_')
        if countList[tgp[0]] < maxPer and countList[tgp[1]] < maxPer:
            ans = np.append(ans, tmpAns)
            countList[tgp[0]] = countList[tgp[0]] + 1
            countList[tgp[1]] = countList[tgp[1]] + 1
            i = i + 1
            
        xdiff_index = xdiff_index + 1
        
    return(ans)
    
    

def ptGetTop(aTrain, dLevel, topX = 50, sliceSize = 5000):
    """Makes vector of gene pairs, iterates over this and computes pairDat, sc_testPattern, then, at the end, findBestPairs. (incomplete)
    
    Parameters:
        aTrain: AnnData object with expression data
        dLevel: descriptor used for grouping of cells
        topX: number of gene pairs to extract from each slice
        sliceSize: how many gene pairs to work with at a time
    
    Returns:
        numpy array of top gene pairs
    
    """
    ans = np.empty()
    genes = aTrain.var.index.values
    grps = np.unique(aTrain.obs[dLevel])
    #Need to figure out the meaning of this
    mcCores = 1
        #if(ncores > 1):
        #mcCores = ncores -1
        
        #print(ncores, "-->", mcCores)
    
    #make data frame of pairs of genes that will be sliced later
    pTab = makePairTab(genes)
    myPatternG = sc_sampR_to_pattern(aTrain.obs[dLevel].values)
    
    print("setup ans and make pattern")
    statDict = {}
    for grp in grps:
        statList[grp] = pd.DataFrame()
        
    #Make the pairedDat, run sc_testPattern equivalent
    print("make pairDat on slice and test")
    nPairs = len(pTab.index)
    print("nPairs = ", nPairs)
    
    #Difficult to work with AnnData here, so create pandas dataframe with necessary info
    expMat = makeExpMat(aTrain)
    start = 1
    stp = min([sliceSize, nPairs])
    while(start <= nPairs):
        if(stp > nPairs):
            stp = nPairs
        
        print(start, "-", stp, sep = '')
        tmpTab = pTab.loc[start-1:stp-1, :]
        tmpPdat = ptSmall(expMat, tmpTab)
        
        #need to figure out exactly what is happening in sc_testPattern. Do not fully understand it yet.
        #For now, assume I wrote it already
        tmpAns = dict((k, sc_testPattern(v, tmpPdat)) for k, v in myPatternG.items())
        
        #May need to adjust this, but I don't currently see why this wouldn't work the same as a for loop
        statDict = statDict.update(tmpAns)
        
        start = stp + 1
        stp = start + sliceSize - 1
        
        
    print("compile results")
    for grp in grps:
        tmpAns = findBestPairs(statList[grp], topX)
        ans = np.append(ans, tmpAns)
        
    return(np.unique(ans))
        
