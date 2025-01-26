import pandas as pd
import numpy as np
import scanpy as sc
from sklearn import linear_model
from itertools import combinations
from .stats import * 
import random as rand 
from pySingleCellNet.config import SCN_DIFFEXP_KEY
from .utils import build_knn_graph, rank_genes_subsets

def csRenameOrth(adQuery,adTrain,orthTable,speciesQuery='human',speciesTrain='mouse'):
    _,_,cgenes=np.intersect1d(adQuery.var_names.values, orthTable[speciesQuery], return_indices=True)
    _,_,ccgenes=np.intersect1d(adTrain.var_names.values, orthTable[speciesTrain], return_indices=True)
    temp1=np.zeros(len(orthTable.index.values), dtype=bool)
    temp2=np.zeros(len(orthTable.index.values), dtype=bool)
    temp1[cgenes]=True
    temp2[ccgenes]=True
    common=np.logical_and(temp1, temp2)
    oTab=orthTable.loc[common.T,:]
    adT=adTrain[:, oTab[speciesTrain]]
    adQ=adQuery[:, oTab[speciesQuery]]
    adQ.var_names = adT.var_names
    return [adQ, adT]


def csRenameOrth2(expQuery,expTrain,orthTable,speciesQuery='human',speciesTrain='mouse'):
    _,_,cgenes=np.intersect1d(expQuery.columns.values, orthTable[speciesQuery], return_indices=True)
    _,_,ccgenes=np.intersect1d(expTrain.columns.values, orthTable[speciesTrain], return_indices=True)
    temp1=np.zeros(len(orthTable.index.values), dtype=bool)
    temp2=np.zeros(len(orthTable.index.values), dtype=bool)
    temp1[cgenes]=True
    temp2[ccgenes]=True
    common=np.logical_and(temp1, temp2)
    oTab=orthTable.loc[common.T,:]
    expT=expTrain.loc[:, oTab[speciesTrain]]
    expQ=expQuery.loc[:, oTab[speciesQuery]]
    expQ.columns= expT.columns
    return [expQ, expT]


def makePairTab(genes):
    pairs = list(combinations(genes,2))
    labels = ['genes1', 'genes2']
    pTab = pd.DataFrame(data = pairs, columns = labels)
    pTab['gene_pairs'] = pTab['genes1'] + '_' + pTab['genes2']
    return(pTab)

def gnrAll(expDat,cellLabels):
    myPatternG=sc_sampR_to_pattern(cellLabels)
    res={}
    groups=np.unique(cellLabels)
    for i in range(0, len(groups)):
        res[groups[i]]=sc_testPattern(myPatternG[groups[i]], expDat)
    return res

def getClassGenes(diffRes, topX=25, bottom=True):
    xi = ~pd.isna(diffRes["cval"])
    diffRes = diffRes.loc[xi,:]
    sortRes= diffRes.sort_values(by="cval", ascending=False)
    ans=sortRes.index.values[0:topX]
    if bottom:
        l= len(sortRes)-topX
        ans= np.append(ans, sortRes.index.values[l:] ).flatten()
    return ans

def addRandToSampTab(classRes, sampTab, desc, id="cell_name"):
    cNames= classRes.index.values
    snames= sampTab.index.values
    rnames= np.setdiff1d(cNames, snames)
    stNew= pd.DataFrame()
    stNew["rid"]=rnames
    stNew["rdesc"]="rand"
    stTop=sampTab[[id, desc]]
    stNew.columns= [id, desc]
    ans = stTop.append(stNew)
    return ans

def ptSmall(expMat, pTab):
    npairs = len(pTab.index)
    genes1 = pTab['genes1'].values
    genes2 = pTab['genes2'].values
    expTemp=expMat.loc[:,np.unique(np.concatenate([genes1,genes2]))]
    ans = pd.DataFrame(0, index = expTemp.index, columns = np.arange(npairs))
    ans = ans.astype(pd.SparseDtype("int", 0))
    temp1= expTemp.loc[:,genes1]
    temp2= expTemp.loc[:,genes2]
    temp1.columns=np.arange(npairs)
    temp2.columns=np.arange(npairs)
    boolArray = temp1 > temp2
    ans = boolArray.astype(int)
    ans.columns = list(pTab[['gene_pairs']].values.T)
    return(ans)

def findBestPairs(xdiff, n=50, maxPer=3):
    xdiff = xdiff.sort_values(by = ['cval'], ascending = False)
    genes=[]
    genesTemp = list(xdiff.index.values)
    for g in genesTemp:
        genes.append(g[0].split("_"))
    genes = np.unique(np.array(genes).flatten())
    countList = dict(zip(genes, np.zeros(genes.shape)))
    i = 1
    ans = np.empty(0)
    xdiff_index = 0
    pair_names = xdiff.index.values
    while i<n:
        tmpAns = pair_names[xdiff_index]
        tgp = tmpAns[0].split('_')
        if countList[tgp[0]] < maxPer and countList[tgp[1]] < maxPer:
            ans = np.append(ans, tmpAns)
            countList[tgp[0]] = countList[tgp[0]] + 1
            countList[tgp[1]] = countList[tgp[1]] + 1
            i = i + 1
        xdiff_index = xdiff_index + 1
    return(np.array(ans))

def query_transform(expMat, genePairs):
    npairs = len(genePairs)
    ans = pd.DataFrame(0, index = expMat.index, columns = np.arange(npairs))
    genes1=[]
    genes2=[]
    for g in genePairs:
        sp=g.split("_")
        genes1.append(sp[0])
        genes2.append(sp[1])
    expTemp=expMat.loc[:,np.unique(np.concatenate([genes1,genes2]))]
    ans = pd.DataFrame(0, index = expTemp.index, columns = np.arange(npairs))
    ans = ans.astype(pd.SparseDtype("int", 0))
    temp1= expTemp.loc[:,genes1]
    temp2= expTemp.loc[:,genes2]
    temp1.columns=np.arange(npairs)
    temp2.columns=np.arange(npairs)
    boolArray = temp1 > temp2
    ans = boolArray.astype(int)
    ans.columns = genePairs
    return(ans)

# I think this is not used
def pair_transform(expMat):
    pTab=makePairTab(expMat)
    npairs = len(pTab.index)
    ans = pd.DataFrame(0, index = expMat.index, columns = np.arange(npairs))
    genes1 = pTab['genes1'].values
    genes2 = pTab['genes2'].values
    expTemp=expMat.loc[:,np.unique(np.concatenate([genes1,genes2]))]
    ans = pd.DataFrame(0, index = expTemp.index, columns = np.arange(npairs))
    ans = ans.astype(pd.SparseDtype("int", 0))
    temp1= expTemp.loc[:,genes1]
    temp2= expTemp.loc[:,genes2]
    temp1.columns=np.arange(npairs)
    temp2.columns=np.arange(npairs)
    boolArray = temp1 > temp2
    ans = boolArray.astype(int)
    ans.columns = genePairs
    return(ans)

def gnrBP(expDat,cellLabels,topX=50):
    myPatternG=sc_sampR_to_pattern(cellLabels)
    levels=list(myPatternG.keys())
    ans={}
    for i in range(0, len(levels)):
        xres=sc_testPattern(myPatternG[levels[i]],expDat)
        tmpAns=findBestPairs(xres, topX)
        ans[levels[i]]=tmpAns
    return ans


def generate_gene_pairs(
    cgenes_list: dict,
    npairs: int = 50,
    genes_const: list = None,
    prop_other: float = 0.4,
    prop_same: float = 0.4
) -> np.ndarray:
    """
    Generate a random selection of gene pairs for each list in cgenes_list and 
    return them as a flattened NumPy array of unique pair strings.
    
    :param cgenes_list: A dict where each value is a list of genes. 
                       Example: {'A': ['geneA1', 'geneA2'], 'B': ['geneB1', 'geneB2', ...], ...}
    :param npairs: Total number of gene pairs to generate for each key in cgenes_list.
    :param genes_const: Optional list of genes to draw from for the 'const' category.
    :param prop_other: Proportion of pairs (out of npairs) whose second gene will be from OTHER lists.
    :param prop_same: Proportion of pairs whose second gene will be from the SAME list as the first gene.
    :return: A 1D NumPy array of unique gene pairs (e.g. ["gene1_gene2", "geneA_geneB", ...]).
    
    The function ensures that:
      1) prop_other + prop_same <= 1.0
      2) Each list in cgenes_list has length > 0
      3) If genes_const is provided (and non-empty), then prop_other + prop_same < 1.0
      
    If genes_const is used, then prop_const = 1 - (prop_other + prop_same). 
    This will be the proportion of pairs whose second gene is drawn from genes_const.
    
    Additionally, no gene pair will be made of the same two genes (i.e. g1 != g2).
    """

    def pick_different_gene(current_gene, gene_pool):
        """
        Return a random gene from gene_pool that is not equal to current_gene.
        If no such gene is found, return None.
        """
        valid_genes = [g for g in gene_pool if g != current_gene]
        if not valid_genes:
            return None
        return rand.choice(valid_genes)
    
    # 1) Basic checks
    if (prop_other + prop_same) > 1.0:
        raise ValueError("prop_other + prop_same cannot exceed 1.0.")
    
    # 2) Check that each list in cgenes_list has length > 0
    for key, gene_list in cgenes_list.items():
        if len(gene_list) == 0:
            raise ValueError(f"The gene list for key '{key}' is empty. Each list must have length > 0.")
    
    # 3) If genes_const is given, check prop_other + prop_same < 1 and compute prop_const
    prop_const = 0.0
    if genes_const is not None and len(genes_const) > 0:
        if (prop_other + prop_same) >= 1.0:
            raise ValueError("If genes_const is used, prop_other + prop_same must be < 1.0.")
        prop_const = 1.0 - (prop_other + prop_same)
    else:
        # If genes_const is None or empty, we don't use the const category at all
        genes_const = []
    
    # This will hold *all* pairs (flattened) across all keys
    all_pairs = []
    
    # For each group in cgenes_list, generate npairs of gene pairs
    for key, gene_list in cgenes_list.items():
        # Determine how many pairs come from 'same', 'other', and 'const'
        same_count = int(round(npairs * prop_same))
        other_count = int(round(npairs * prop_other))
        const_count = int(round(npairs * prop_const))
        
        # Fix any rounding discrepancy
        total_assigned = same_count + other_count + const_count
        leftover = npairs - total_assigned
        
        # Distribute leftover to one of the categories (simplest: add to 'same')
        same_count += leftover
        
        # --- SAME pairs (both genes from same list, g2 != g1) ---
        for _ in range(same_count):
            g1 = rand.choice(gene_list)
            g2 = pick_different_gene(g1, gene_list)
            if g2 is not None:
                all_pairs.append(f"{g1}_{g2}")
        
        # --- OTHER pairs (second gene from other lists, g2 != g1) ---
        other_genes = []
        for other_key, other_list in cgenes_list.items():
            if other_key != key:
                other_genes.extend(other_list)
        
        for _ in range(other_count):
            g1 = rand.choice(gene_list)
            g2 = pick_different_gene(g1, other_genes)
            if g2 is not None:
                all_pairs.append(f"{g1}_{g2}")
        
        # --- CONST pairs (second gene from genes_const, g2 != g1) ---
        for _ in range(const_count):
            g1 = rand.choice(gene_list)
            g2 = pick_different_gene(g1, genes_const)
            if g2 is not None:
                all_pairs.append(f"{g1}_{g2}")
    
    # Convert to a NumPy array of unique values
    return np.unique(all_pairs)



def ptGetTop (expDat, cell_labels, cgenes_list=None, topX=50, sliceSize = 5000, quickPairs = True, propOther: float = 0.5):
    if not quickPairs:
        genes=expDat.columns.values
        grps=np.unique(cell_labels)
        myPatternG=sc_sampR_to_pattern(cell_labels)
        pairTab=makePairTab(genes)
        nPairs = len(pairTab)
        start = 0
        stp = np.min([sliceSize, nPairs])
        tmpTab = pairTab.iloc[start:stp,:]
        tmpPdat = ptSmall(expDat, tmpTab)
        statList= dict((k, sc_testPattern(v, tmpPdat)) for k, v in myPatternG.items())
        start= stp
        stp= start + sliceSize
        while start < nPairs:
            # print(start)
            if stp > nPairs:
                stp =  nPairs
            tmpTab = pairTab.iloc[start:stp,:]
            tmpPdat = ptSmall(expDat, tmpTab)
            tmpAns=dict((k, sc_testPattern(v, tmpPdat)) for k, v in myPatternG.items())
            for g in grps:
                statList[g]=pd.concat([statList[g], tmpAns[g]])
            start= stp
            stp= start + sliceSize
        res=[]
        for g in grps:
            tmpAns=findBestPairs(statList[g], topX)
            res.append(tmpAns)
        return np.unique(np.array(res).flatten())


    else:
        myPatternG= sc_sampR_to_pattern(cell_labels)
        maxOther = int(np.ceil(propOther * topX))
        res=[]
        grps=np.unique(cell_labels)
        for g in grps:
            # print(g)
            genes=cgenes_list[g]
            tmpMaxPer = 3
            if len(genes) < topX:
                tmpMaxPer = 8
            
            pairTab=makePairTab(genes)
            nPairs=len(pairTab)
            tmpPdat=ptSmall(expDat, pairTab)
            tmpAns=findBestPairs(sc_testPattern(myPatternG[g],tmpPdat), topX, maxPer = tmpMaxPer)
            res.append(tmpAns)
            # to add here also select from cgenes_list[-g]
            notg = [item for item in grps if item != g]
            for ng in notg:
                ng_genes = cgenes_list[ng]
                g1 = list(set(genes).difference(set(ng_genes)))
                g2 = list(set(ng_genes).difference(set(genes)))
                lastIndex = min([len(g1), len(g2), maxOther])
                if lastIndex > 0:
                    g1 = rand.sample(g1, lastIndex)
                    g2 = rand.sample(g2, lastIndex)
                    labels = ['genes1', 'genes2']
                    pTab = pd.DataFrame(data = list(zip(g1[0:lastIndex], g2[0:lastIndex])), columns = labels)
                    pxxx = pTab['genes1'] + '_' + pTab['genes2']
                    res.append( pxxx.to_numpy(dtype="<U32") )

                
        return np.unique( np.concatenate(res) )




# this is the same thing as findSigGenes() in rank_class.py
def findSignatureGenes(
    adata,
    dLevel: str = "leiden",
    topX: int = 25,
    uns_name: str = 'rank_genes_groups'
):
    grps = adata.obs[dLevel]
    groups = np.unique(grps)
    # sc.tl.rank_genes_groups(adTemp, dLevel, use_raw=False, method=test_name)
    # tempTab = pd.DataFrame(adata.uns[uns_name]['names']).head(topX)
    tempTab = sc.get.rank_genes_groups_df(adata, group = None, key=uns_name)
    tempTab = tempTab.dropna()
    cgenes = {}
    for g in groups:
        temp = tempTab[tempTab['group']==g]
        glen = len(temp) 
        if glen > 0:
            if glen > topX:
                cgenes[g] = temp[:topX]['names'].to_numpy()
            else:
                cgenes[g] = temp['names'].to_numpy()

    return cgenes

def get_classy_genes(adDat, dLevel, key_name = SCN_DIFFEXP_KEY, topX=25):
    adTemp = adDat.copy()
    grps = adDat.obs[dLevel]
    groups = np.unique(grps)

    tempTab = pd.DataFrame(adTemp.uns[key_name]['names']).head(topX)

    res = []
    cgenes = {}

    for g in groups:
        temp = tempTab[g] 
        res.append(temp)
        cgenes[g] = temp.to_numpy()
    cgenes2 = np.unique(np.array(res).flatten())

    return [cgenes2, grps, cgenes]    

def get_classy_genes_2(
    adata,
    groupby,
    key_name="rank_genes_groups",  # Default differential expression key
    topX_per_diff_type=10,
    pval=0.01,
    bottom_min_in=0.15,
    bottom_max_out=0.1,
    top_min_in=0.4,
    top_max_out=0.25,
    proportion_top=0.70,
    k_of_knn=1,
    layer="lognorm"
):
    """
    Identifies cell-type-specific genes using differential expression analysis and kNN graph-based comparisons.

    Args:
        adata (AnnData): The AnnData object containing single-cell data.
        groupby (str): The .obs column to group cells by.
        key_name (str, optional): Key in `.uns` to use for rank_genes_groups results. Defaults to "rank_genes_groups".
        topX_per_diff_type (int, optional): Number of top genes to select per differential expression comparison type. Defaults to 10.
        pval (float, optional): P-value cutoff for differential expression. Defaults to 0.01.
        bottom_min_in (float, optional): Minimum proportion of cells expressing a gene in the target group for "bottom" gene selection. Defaults to 0.15.
        bottom_max_out (float, optional): Maximum proportion of cells expressing a gene in reference groups for "bottom" gene selection. Defaults to 0.1.
        top_min_in (float, optional): Minimum proportion of cells expressing a gene in the target group for "top" gene selection. Defaults to 0.4.
        top_max_out (float, optional): Maximum proportion of cells expressing a gene in reference groups for "top" gene selection. Defaults to 0.25.
        proportion_top (float, optional): Proportion of genes selected from the "top" of the ranking vs. the "bottom." Defaults to 0.70.
        k_of_knn (int, optional): Number of neighbors to consider for kNN graph-based comparisons. Defaults to 1.
        layer (str, optional): Data layer to use for differential expression analysis. Defaults to "lognorm".

    Returns:
        tuple: A tuple containing:
            - cgenes2 (list): Combined list of all unique genes identified.
            - grps (pandas.Series): Group labels from the specified `.obs` column.
            - gene_dict (dict): Dictionary of genes specific to each group.
    """
    
    # Copy the AnnData object to avoid modifying the original
    adTemp = adata.copy()
    grps = adata.obs[groupby]
    groups = np.unique(grps)
    
    # Retrieve the general differential expression table
    diff_tab_general = sc.get.rank_genes_groups_df(
        adTemp, None, pval_cutoff=pval, key=key_name
    )
    diff_tab_general = diff_tab_general.sort_values(
        by=["group", "pct_nz_group"], ascending=[True, False]
    )
    
    gene_list = []
    gene_dict = {}
    
    # Build kNN graph from dendrogram correlation matrix
    celltype_graph = build_knn_graph(
        adata.uns["dendrogram_" + groupby]["correlation_matrix"],
        list(adata.obs[groupby].cat.categories),
        k_of_knn,
    )
    
    for g in groups:
        # Filter the differential expression table for the current group
        tempTab = diff_tab_general[diff_tab_general["group"] == g]
        
        # Get top genes from the current group
        xlist = get_top_genes_from_df(
            tempTab, topX=topX_per_diff_type, proportion_top=proportion_top
        )
        xlist += get_top_genes_from_df(
            tempTab,
            by_score=False,
            min_in=bottom_min_in,
            max_out=bottom_max_out,
            proportion_top=proportion_top,
        )
        xlist += get_top_genes_from_df(
            tempTab,
            by_score=False,
            min_in=top_min_in,
            max_out=top_max_out,
            proportion_top=proportion_top,
        )
        
        # Subset analysis comparing group g to its kNN neighbors
        other_groups = celltype_graph.vs[
            celltype_graph.neighbors(celltype_graph.vs.find(name=g).index)
        ]["name"]
        
        xdata = adata.copy()
        subsetDF = rank_genes_subsets(
            xdata, groupby=groupby, grpA=[g], grpB=other_groups, layer=layer
        )
        
        # Add genes from the subset analysis
        xlist += get_top_genes_from_df(
            subsetDF, topX=topX_per_diff_type, proportion_top=proportion_top
        )
        xlist += get_top_genes_from_df(
            subsetDF,
            by_score=False,
            min_in=bottom_min_in,
            max_out=bottom_max_out,
            proportion_top=proportion_top,
        )
        xlist += get_top_genes_from_df(
            subsetDF,
            by_score=False,
            min_in=top_min_in,
            max_out=top_max_out,
            proportion_top=proportion_top,
        )
        
        # Compile results for this cell type
        gene_list += xlist
        gene_dict[g] = list(set(xlist))
    
    # Combine all unique genes
    cgenes2 = list(set(gene_list))
    
    return cgenes2, grps, gene_dict








def findClassyGenes(adDat, dLevel, topX=25, test_name='wilcoxon'):
    adTemp = adDat.copy()
    grps = adDat.obs[dLevel]
    groups = np.unique(grps)

    sc.tl.rank_genes_groups(adTemp, dLevel, use_raw=False, method=test_name)
    tempTab = pd.DataFrame(adTemp.uns['rank_genes_groups']['names']).head(topX)

    res = []
    cgenes = {}

    for g in groups:
        temp = tempTab[g] 
        res.append(temp)
        cgenes[g] = temp.to_numpy()
    cgenes2 = np.unique(np.array(res).flatten())

    return [cgenes2, grps, cgenes]    
