
def dotplot_deg(
    adata: AnnData,
    diff_gene_dict: dict,
    #samples_obsvals: list = [],
    groupby_obsname: str = "comb_sampname", # I think that this is the column that splits cells for DEG
    cellgrp_obsname: str = "comb_cellgrp",    # I think that this is the column that splits into discint clusters or cell types
    cellgrp_obsvals: list = [], # this is the subset of clusters or cell types to limit this visualization to
    num_genes: int = 10, 
    order_by = 'scores',
    new_obsname = 'grp_by_samp',
    use_raw=False
):

    # remove celltypes unspecified in diff_gene_dict
    dd_dict = diff_gene_dict['geneTab_dict']
    tokeep = list(dd_dict.keys())

    # also remove cell_types not listed in celltype_names
    # default for celltype_names is all celltypes included in dd_dict
    if len(cellgrp_obsvals) > 0:
        cellgrp_obsvals = list(set(cellgrp_obsvals).intersection(set(tokeep)))
    else:
        cellgrp_obsvals = tokeep

    adNew = adata.copy()
    adNew = adNew[adNew.obs[cellgrp_obsname].isin(cellgrp_obsvals)].copy()
    
    # remove categories unspecified in diff_gene_dict
    dictkey = list(diff_gene_dict.keys())[0]
    sample_names = diff_gene_dict[dictkey]
    adNew = adNew[adNew.obs[groupby_obsname].isin(sample_names)].copy()

    # add column 'grp_by_samp' to obs that indicates cellgrp by sample
    adNew.obs[new_obsname] = adNew.obs[cellgrp_obsname].astype(str) + "_X_" + adNew.obs[groupby_obsname].astype(str)
    
    # define dict of marker genes based on threshold
    genes_to_plot = dict()
    for cellgrp in cellgrp_obsvals:
        print(f"{cellgrp}")
        for sname in sample_names:
            print(f"{sname}")
            genes_to_plot[cellgrp + "_X_" + sname] = pull_out_genes_v2(diff_gene_dict, cell_type = cellgrp, category = sname, num_genes = num_genes, order_by=order_by) 

    # return adNew, genes_to_plot
    plt.rcParams['figure.constrained_layout.use'] = True
    #xplot = sc.pl.DotPlot(adNew, genes_to_plot, 'ct_by_cat', cmap='RdPu', var_group_rotation = 0) #, dendrogram=True,ax=ax2, show=False)
    xplot = sc.pl.DotPlot(adNew, genes_to_plot, new_obsname, cmap=LaJolla_20.mpl_colormap, var_group_rotation = 0, use_raw=False) #, dendrogram=True,ax=ax2, show=False)
    xplot.swap_axes(True) # see sc.pl.DotPlot docs for useful info
    return xplot


def umap_scores_old(
    adata: AnnData,
    scn_classes: list,
    obsm_name = 'SCN_score'
):    
    adTemp = AnnData(adata.obsm[obsm_name], obs=adata.obs)
    adTemp.obsm['X_umap'] = adata.obsm['X_umap'].copy()
    sc.pl.umap(adTemp,color=scn_classes, alpha=.75, s=10, vmin=0, vmax=1)




