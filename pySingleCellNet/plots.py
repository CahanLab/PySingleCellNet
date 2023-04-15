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
from sklearn.metrics import f1_score
import altair as alt
from .utils import *

alt.data_transformers.disable_max_rows()


def dotplot_diff_gene(
    adata: AnnData,
    diff_gene_dict: dict,
    num_genes: int = 10,
    celltype_groupby: str = "SCN_class",
    category_groupby: str = "SCN_class_type",
    category_names: str = ["None", "Singular"],
    celltype_names: list = []
):

    # remove celltypes unspecified in diff_gene_dict
    dd_dict = diff_gene_dict['geneTab_dict']
    tokeep = list(dd_dict.keys())

    # also remove cell_types not listed in celltype_names
    # default for celltype_names is all celltypes included in dd_dict
    if len(celltype_names) > 0:
        celltype_names = list(set(celltype_names).intersection(set(tokeep)))
    else:
        celltype_names = tokeep

    adNew = adata.copy()
    # adNew = adNew[adNew.obs[celltype_groupby].isin(tokeep)].copy()
    adNew = adNew[adNew.obs[celltype_groupby].isin(celltype_names)].copy()
    
    # remove categories unspecified in diff_gene_dict
    category_names = diff_gene_dict['category_names']
    adNew = adNew[adNew.obs[category_groupby].isin(category_names)].copy()

    

    # add column 'ct_by_cat' to obs that indicates celltype X category
    adNew.obs['ct_by_cat'] = adNew.obs[celltype_groupby].astype(str) + "_X_" + adNew.obs[category_groupby].astype(str)
    
    # define dict of marker genes based on threshold
    genes_to_plot = dict()
    for celltype in celltype_names:
        print(f"{celltype}")
        for scn_category in category_names:
            print(f"{scn_category}")
            genes_to_plot[celltype + "_X_" + scn_category] = pull_out_genes(diff_gene_dict, cell_type = celltype, category = scn_category, num_genes = num_genes) 

    # return adNew, genes_to_plot
    plt.rcParams['figure.constrained_layout.use'] = True
    xplot = sc.pl.DotPlot(adNew, genes_to_plot, 'ct_by_cat', cmap='RdPu', var_group_rotation = 0) #, dendrogram=True,ax=ax2, show=False)
    xplot.swap_axes(True) # see sc.pl.DotPlot docs for useful info
    return xplot
    

def stackedbar_categories(
    ad_list: list, 
    adata_names: list, # a sample id or name to disinguish anndatas in plot
    celltype_groupby: str = "SCN_class", # what to group the cells on
    category_groupby: str = "SCN_class_type", # what to split celltypes on
    color_dict: dict = {'Hybrid': 'lightgreen','None': 'peachpuff', 'Singular': 'powderblue'}
):
    
    # make a pd of celltype, category, sample_id
    obs_all = pd.DataFrame()
    i = 0
    for adTemp in ad_list:
        obsTmp = adTemp.obs[ [celltype_groupby, category_groupby] ].copy()
        obsTmp["source"] = adata_names[i]
        obs_all = pd.concat([obs_all, obsTmp])
        i = i + 1

    xcar = alt.Chart(obs_all).mark_bar().encode(
        column=alt.Column('source',sort=["HeldOut", "Query 1"]),
        x=alt.X('count()', stack="normalize",axis=alt.Axis(format='%', title='Percent of cells')),
        y=celltype_groupby,
        color=alt.Color(
            category_groupby,
            # sort=["Singular", "Hybrid", "None"], 
            scale=alt.Scale(domain=list(color_dict.keys()),range=list(color_dict.values()))
        )
    )
    return xcar


def barplot_classifier_f1(adata: AnnData, ground_truth: str = "celltype", class_prediction: str = "SCN_class"):
    fscore = f1_score(adata.obs[ground_truth], adata.obs[class_prediction], average=None, labels = adata.obs[ground_truth].cat.categories)
    cates = list(adata.obs[ground_truth].cat.categories)
    f1_score_dict = {class_label: f1_score_x for class_label, f1_score_x in zip(cates, fscore)}
    
    # Calculate the number of observations per class
    class_counts = adata.obs[ground_truth].value_counts().to_dict()

    plt.rcParams['figure.constrained_layout.use'] = True
    sns.set_theme(style="whitegrid")
    ax = sns.barplot(x=list(f1_score_dict.values()), y=list(f1_score_dict.keys()))
    plt.xlabel('F1-Score')
    plt.title('F1-Scores per Class')
    plt.xlim(0, 1.1) # Set x-axis limits to ensure visibility of all bars

    # Add the number of observations per class as text within the barplot
    for i, bar in enumerate(ax.containers[0]):
        width = bar.get_width()
        print(f"{width}")
        label = f"n = {class_counts[cates[i]]}"
        fcolor = "white"
        if width < 0.20:
            fcolor = "black"

        ax.text( 0.03, bar.get_y() + bar.get_height() / 2, label, ha='left', va='center', color = fcolor)

    plt.show()


def heatmap_scn_scores(adata: AnnData, groupby: str, vmin: float = 0, vmax: float = 1):    
    adTemp = AnnData(adata.obsm['SCN_score'], obs=adata.obs)
    adTemp.obs[groupby] = adata.obs[groupby]
    # guess at appropriate dimensions
    fsize = [5, 6]
    plt.rcParams['figure.subplot.bottom'] = 0.25
    sc.pl.heatmap(adTemp, adTemp.var_names.values, groupby=groupby, cmap='viridis', dendrogram=False, swap_axes=True, vmin = vmin, vmax = vmax, figsize=fsize)

def dotplot_scn_scores(adata: AnnData, groupby: str, expression_cutoff = 0.1):    
    adTemp = AnnData(adata.obsm['SCN_score'], obs=adata.obs)
    adTemp.obs[groupby] = adata.obs[groupby]
    sc.pl.dotplot(adTemp, adTemp.var_names.values, groupby=groupby, expression_cutoff=expression_cutoff, cmap='viridis', colorbar_title="SCN score")

def umap_scn_scores(adata: AnnData, scn_classes: list):    
    adTemp = AnnData(adata.obsm['SCN_score'], obs=adata.obs)
    adTemp.obsm['X_umap'] = adata.obsm['X_umap'].copy()
    sc.pl.umap(adTemp,color=scn_classes, alpha=.75, s=10, vmin=0, vmax=1)

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