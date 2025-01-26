




def add_training_dlevel(adata, dlevel):
    adata.obs['SCN_class'] = adata.obs[dlevel]
    return adata

def check_adX(adata: AnnData) -> AnnData:
    from scipy import sparse
    if( isinstance(adata.X, np.ndarray)):
        adata.X = sparse.csr_matrix(adata.X)










# I don't think that this is called anywhere
def add_classRes(adata: AnnData, adClassRes, copy=False) -> AnnData:
    cNames = adClassRes.var_names
    for cname in cNames:
        adata.obs[cname] = adClassRes[:,cname].X.toarray()
    # adata.obs['category'] = adClassRes.obs['category']
    adata.obs['SCN_class'] = adClassRes.obs['SCN_class']
    return adata if copy else None





def scn_train(aTrain,
    dLevel,
    nRand = None,
    cell_type_to_color = None,
    nTopGenes = 20,
    nTopGenePairs = 20,
    nTrees = 1000,
    propOther=0.5,
    counts_per_cell_after = 1e4,
    scaleMax = 10,
    limitToHVG = True,
    normalization = True,
    include_all_genes = False
):
    progress_total = 5
    with alive_bar(progress_total, title="Training classifier") as bar:
        warnings.filterwarnings('ignore')

        # auto determine nRand = mean number of cells per type
        if nRand is None:
            nRand = np.floor(np.mean(aTrain.obs[dLevel].value_counts()))

        stTrain= aTrain.obs
        expRaw = aTrain.to_df()
        expRaw = expRaw.loc[stTrain.index.values]
        adNorm = aTrain.copy()
        if normalization:
            sc.pp.normalize_per_cell(adNorm, counts_per_cell_after=counts_per_cell_after)
            sc.pp.log1p(adNorm)
            # print("HVG")
            if limitToHVG:
                try:
                    sc.pp.highly_variable_genes(adNorm, min_mean=0.0125, max_mean=4, min_disp=0.5)
                except Exception as e:
                    raise ValueError(f"PySCN encountered an error when selecting variable genes. This may be avoided if you do not call scale or regress_out on the training data. Original error text: {repr(e)}") 
                adNorm = adNorm[:, adNorm.var.highly_variable]

            sc.pp.scale(adNorm, max_value=scaleMax)
            #print("data normalized")

        expTnorm = adNorm.to_df()
        expTnorm = expTnorm.loc[stTrain.index.values]
        bar() # Bar 1

        ### expTnorm= pd.DataFrame(data=aTrain.X,  index= aTrain.obs.index.values, columns= aTrain.var.index.values)
        ### expTnorm=expTnorm.loc[stTrain.index.values]
        
        ### cgenesA, grps, cgenes_list =findClassyGenes(expTnorm,stTrain, dLevel = dLevel, topX = nTopGenes)
        if include_all_genes == False:
            cgenesA, grps, cgenes_list =findClassyGenes(adNorm, dLevel = dLevel, topX = nTopGenes)
        else: 
            cgenesA = np.array(aTrain.var.index)
            grps = aTrain.obs[dLevel]
            cgenes_list = dict()
            for g in np.unique(grps):
                cgenes_list[g] = cgenesA

        bar() # Bar 2
        # print("There are ", len(cgenesA), " classification genes\n")
        ### xpairs= ptGetTop(expTnorm.loc[:,cgenesA], grps, cgenes_list, topX=nTopGenePairs, sliceSize=5000)
        xpairs= ptGetTop(expTnorm.loc[:,cgenesA], grps, cgenes_list, topX=nTopGenePairs, sliceSize=5000, propOther=propOther)
        bar() # Bar 3
        # print("There are", len(xpairs), "top gene pairs\n")
        pdTrain= query_transform(expRaw.loc[:,cgenesA], xpairs)
        # print("Finished pair transforming the data\n")
       

        tspRF=sc_makeClassifier(pdTrain.loc[:, xpairs], genes=xpairs, groups=grps, nRand = nRand, ntrees = nTrees)
        bar() # Bar 4
    
    ## set celltype colors
        ## Do this here because we add a 'rand' celltype
        
        # Need to add checks that all classes have a color if ct_colors is provided
        if cell_type_to_color is None:
            ## assume this is a Series
            cell_types = stTrain[dLevel].cat.categories.to_list()
            cell_types.append('rand')
            unique_colors = get_unique_colors(len(cell_types))
            cell_type_to_color = {cell_type: color for cell_type, color in zip(cell_types, unique_colors)}
        bar() # Bar 5

    return {'tpGeneArray': cgenesA, 'topPairs':xpairs, 'classifier': tspRF, 'diffExpGenes':cgenes_list, 'ctColors':cell_type_to_color}




