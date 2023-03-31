import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import umap
import anndata as ad
from anndata import AnnData
from scipy.sparse import csr_matrix



def heatmap_scn_scores(adata: AnnData, groupby: str, cbar_auto: str = False):    
    adTemp = AnnData(adata.obsm['SCN_score'], obs=adata.obs)
    adTemp.obs[groupby] = adata.obs[groupby]
    #if cbar_auto:
    #    sc.pl.heatmap(adTemp, adTemp.var_names.values, groupby=groupby, cmap='viridis', dendrogram=False, swap_axes=True)
    #else:
    #    sc.pl.heatmap(adTemp, adTemp.var_names.values, groupby=groupby, cmap='viridis', dendrogram=False, swap_axes=True, vim=0, vmax=1)
    sc.pl.heatmap(adTemp, adTemp.var_names.values, groupby=groupby, cmap='viridis', dendrogram=False, swap_axes=True)

def dotplot_scn_scores(adata: AnnData, groupby: str, expression_cutoff = 0.1):    
    adTemp = AnnData(adata.obsm['SCN_score'], obs=adata.obs)
    adTemp.obs[groupby] = adata.obs[groupby]
    sc.pl.dotplot(adTemp, adTemp.var_names.values, groupby=groupby, expression_cutoff=expression_cutoff, cmap='viridis', colorbar_title="SCN score")

def umap_scn_scores(adata: AnnData, groupby: str, expression_cutoff = 0.1):    
    adTemp = AnnData(adata.obsm['SCN_score'], obs=adata.obs)
    adTemp.obs[groupby] = adata.obs[groupby].copy()
    sc.pl.dotplot(adTemp, adTemp.var_names.values, groupby=groupby, expression_cutoff=expression_cutoff, cmap='viridis', colorbar_title="SCN score")


def hm_genes(adQuery, adTrain=None, cgenes_list={}, list_of_types_toshow=[], list_of_training_to_show=[], number_of_genes_toshow=3, query_annotation_togroup='SCN_class', training_annotation_togroup='SCN_class', split_show=False, save = False):
    if list_of_types_toshow.__len__() == list_of_training_to_show.__len__():
        if list_of_types_toshow.__len__() == 0:
            list_of_types_toshow = np.unique(adQuery.obs['SCN_class'])
            num = list_of_types_toshow.__len__()
            list_of_training_to_show = [False for _ in range(num)]
            # Default: all shown and False
        else:
            if adTrain is None:
                print("No adTrain given")
                return
            else:
                pass
    elif list_of_training_to_show.__len__() == 0:
        num = list_of_types_toshow.__len__()
        list_of_training_to_show = [False for _ in range(num)]
        # Default: all False
    else:
        print("Boolean list not correspondent to list of types")
        return


    list_1 = []
    list_2 = []
    SCN_annot = []
    annot = []

    MQ = adQuery.X.todense()
    if adTrain is not None:
        MT = adTrain.X.todense()
    else:
        pass
    

    for i in range(list_of_types_toshow.__len__()):
        type = list_of_types_toshow[i]
        for j in range(np.size(MQ, 0)):
            if adQuery.obs['SCN_class'][j] == type:
                list_1.append(j)
                SCN_annot.append(type)
                if split_show:
                    annot.append(adQuery.obs[query_annotation_togroup][j]+'_Query')
                else:
                    annot.append(adQuery.obs[query_annotation_togroup][j])

    for i in range(list_of_training_to_show.__len__()):
        type = list_of_types_toshow[i]            
        if list_of_training_to_show[i]:
            for j in range(np.size(MT, 0)):
                if adTrain.obs['SCN_class'][j] == type:
                    list_2.append(j)
                    SCN_annot.append(type)
                    if split_show:
                        annot.append(adTrain.obs[training_annotation_togroup][j]+'_Train')
                    else:
                        annot.append(adTrain.obs[training_annotation_togroup][j])
        else:
            pass
    
    SCN_annot = pd.DataFrame(SCN_annot)
    annot = pd.DataFrame(annot)

    M_1 = MQ[list_1,:]
    if adTrain is not None:
        M_2 = MT[list_2,:]
    else:
        pass

    if adTrain is None:
        Mdense = M_1
    else:
        Mdense = np.concatenate((M_1,M_2))
    New_Mat = csr_matrix(Mdense)
    adTrans = sc.AnnData(New_Mat)

    adTrans.obs['SCN_class'] = SCN_annot.values
    adTrans.obs['Cell_Type'] = annot.values
    adTrans.var_names = adQuery.var_names

    sc.pp.normalize_per_cell(adTrans, counts_per_cell_after=1e4)
    sc.pp.log1p(adTrans)

    RankList = []

    for type in list_of_types_toshow:
        for i in range(number_of_genes_toshow):
            RankList.append(cgenes_list[type][i])

    fig = sc.pl.heatmap(adTrans, RankList, groupby='Cell_Type', cmap='viridis', dendrogram=False, swap_axes=True, save=save)
    return fig