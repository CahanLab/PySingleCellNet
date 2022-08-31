import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import umap
import anndata as ad
from scipy.sparse import csr_matrix



def sc_hmScanpy(adata, categories, groupBy='leiden'):
    ctAll = adata.obs.columns.values[categories]
    var = pd.DataFrame(index = ctAll)

    #adNew = ad.AnnData(adM1Norm.obs[ctAll], obs=adM1Norm.obs, var=var)
    #ax = sc.pl.heatmap(adNew, ctAll, groupby='leiden', cmap='viridis', dendrogram=True, swap_axes=True)


def My_sc_hmClass(classMat, grps,cRow=False,cCol=False):
    warnings.filterwarnings('ignore')
    g_uni=np.unique(grps)
    my_palette = dict(zip(g_uni, sns.color_palette("hls",len(g_uni))))
    col_colors = grps.map(my_palette)
    if cRow:
        g=sns.clustermap(classMat.T, row_cluster=cRow, col_cluster=cCol, standard_scale=None, col_colors=col_colors, cmap="viridis",  cbar_kws={"orientation": "horizontal"})
    else:
        g=sns.clustermap(classMat.T, row_cluster=cRow, col_cluster=cCol, standard_scale=None, col_colors=col_colors, cmap="viridis")
    g.gs.update(left=0.00, right=0.6)
    gs2 = matplotlib.gridspec.GridSpec(1,1, left=0.925, right=1)
    ax2 = g.fig.add_subplot(gs2[0])
    ax2.grid(False)
    ax2.axis("off")
    ax = g.ax_heatmap
    ax.set_yticks(np.arange(len(classMat.columns.values)))
    ax.set_yticklabels(classMat.columns.values)
    ax.set_xticks([])
    for label in g_uni:
        g.ax_col_dendrogram.bar(0, 0, color=my_palette[label],
                                label=label, linewidth=0)
    handles, labels =  g.ax_col_dendrogram.get_legend_handles_labels()
    lgd=ax2.legend(handles, labels, loc="center left", ncol=2,bbox_to_anchor=(0, 0.4), frameon=False)
    if cCol and cRow:
        g.cax.set_position([0, 0.7, .015, .1])
    elif cRow:
        g.cax.set_position([0.125, 0.775, .475, .02])
    else:
        g.cax.set_position([0, 0.125, .03, .575])    
    plt.tight_layout()
    plt.show()



def sc_hmClass(classMat, grps,cRow=False,cCol=False):
    g_uni=np.unique(grps)
    my_palette = dict(zip(g_uni, sns.color_palette("hls",len(g_uni))))
    col_colors = grps.map(my_palette)
    if cRow:
        g=sns.clustermap(classMat.T, row_cluster=cRow, col_cluster=cCol, standard_scale=1, col_colors=col_colors, cmap="viridis",  cbar_kws={"orientation": "horizontal"})   
    else:
        g=sns.clustermap(classMat.T, row_cluster=cRow, col_cluster=cCol, standard_scale=1, col_colors=col_colors, cmap="viridis")
    g.gs.update(left=0.00, right=0.6)
    gs2 = matplotlib.gridspec.GridSpec(1,1, left=0.925, right=1)
    ax2 = g.fig.add_subplot(gs2[0])
    ax2.grid(False)
    ax2.axis("off")
    ax = g.ax_heatmap
    ax.set_yticks(np.arange(len(classMat.columns.values)))
    ax.set_yticklabels(classMat.columns.values)
    ax.set_xticks([])

    for label in g_uni:
        g.ax_col_dendrogram.bar(0, 0, color=my_palette[label],
                                label=label, linewidth=0)
    handles, labels =  g.ax_col_dendrogram.get_legend_handles_labels()
    lgd=ax2.legend(handles, labels, loc="center left", ncol=2,bbox_to_anchor=(0, 0.4), frameon=False)
    if cCol and cRow:
        g.cax.set_position([0, 0.7, .015, .1])
    elif cRow:
        g.cax.set_position([0.125, 0.775, .475, .02])
    else:
        g.cax.set_position([0, 0.125, .03, .575])
    
    plt.tight_layout()
    plt.show()

def sc_violinClass(aData, classRes, dLevel="cluster", threshold=0.20,  ncol =1, sub_cluster=[] ):
    sampTab= adata.obs.copy()
    maxes=classRes.apply(np.max, axis=0)
    temp=classRes.loc[:, maxes>=threshold]
    grps_uni=np.unique(sampTab[dLevel])
    temp=pd.concat([sampTab[dLevel], temp], axis=1).melt(id_vars=[dLevel])
    if sub_cluster != []:
        temp=temp.loc[temp.variable.isin(sub_cluster),:]
    temp["hue"]=temp.variable
    fig1, axes = plt.subplots(ncols=int(ncol), nrows=int(np.ceil(len(grps_uni)/ncol)), sharex=True,sharey=False, figsize=(ncol*4, np.ceil(len(grps_uni)/ncol)*4))
    if len(grps_uni)>1:
        for i in range(0,len(grps_uni)):
            heyo=sns.violinplot(x="variable", y="value",hue="hue", data=temp.loc[temp[dLevel]==grps_uni[i],:], ax=axes.flat[i], scale="width" , legend=False, dodge=False)
            axes.flat[i].set_xlabel("")
            axes.flat[i].get_legend().remove()
            axes.flat[i].set_ylim([0,1])
            axes.flat[i].set_ylabel("Class. Score")
            axes.flat[i].tick_params(labelrotation=90)
            axes.flat[i].set_title(grps_uni[i])
            sns.despine(ax=axes.flat[i])
    else:
        heyo=sns.violinplot(x="variable", y="value",hue="hue", data=temp, scale="width" , legend=False, dodge=False)
        plt.xlabel("")
        plt.ylabel("")
        plt.legend().remove()
        plt.ylim([0,1])
        plt.ylabel("Class. Score")
        plt.tick_params(labelrotation=90)
        plt.title(grps_uni)
        sns.despine()
    handles, labels = heyo.get_legend_handles_labels()
    space = fig1.add_axes([1,0, 0.01, 1])
    space.axis("off")
    space.legend(handles, labels, loc= "center left")

def plot_attr(classRes, aData, dLevel, sub_cluster = []):
    sampTab= adata.obs.copy()
    if sub_cluster !=[]:
        temp=pd.concat([sampTab.newAnn.loc[sampTab[dLevel].isin(sub_cluster)], classRes.loc[sampTab[dLevel].isin(sub_cluster),:].idxmax(axis=1)], axis=1)
    else:
        temp=pd.concat([sampTab[dLevel], classRes.idxmax(axis=1)], axis=1)
    temp.columns=[dLevel, "class"]
    temp=pd.DataFrame(temp.pivot_table(index=dLevel, columns='class', aggfunc='size', fill_value=0).to_records())
    temp.iloc[:, 1:]=temp.iloc[:, 1:].div(temp.iloc[:, 1:].sum(axis=1), axis=0)
    temp.set_index('newAnn').plot(kind='barh', stacked=True, figsize=(10,8))
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), ncol=2)
    plt.tight_layout()
    
def plot_attr2(adata, dLevel):
    temp= adata.obs.copy()
    temp.columns=[dLevel, "class"]
    temp=pd.DataFrame(sampTab.pivot_table(index=dLevel, columns='category', aggfunc='size', fill_value=0).to_records())
    temp.iloc[:, 1:]=temp.iloc[:, 1:].div(temp.iloc[:, 1:].sum(axis=1), axis=0)
    temp.set_index('newAnn').plot(kind='barh', stacked=True, figsize=(10,8))
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), ncol=2)
    plt.tight_layout()


def get_cate(classRes, aData, cThresh = 0):
    res=classRes.idxmax(axis=1)
    if cThresh>0:
        for i in range(0, len(res)):
            if classRes.loc[res.index[i], res[i]]<cThresh:
                res[i]="rand"
    st=adata.obs.copy()
    st["category"]=res
    return st
    
def plot_umap(aData, dLevel="category"):
    sampTab= adata.obs.copy()
    standard_embedding = umap.UMAP(random_state=42).fit_transform(aData.X)
    dat = pd.DataFrame(standard_embedding, index=aData.obs.index)
    dat = pd.concat([dat, sampTab[dLevel]], axis=1)
    dat.columns = ["component 1","component 2", "class"]
    grps=np.unique(dat["class"])
    for i in grps:
        plt.scatter(dat.loc[dat["class"]==i,"component 1"], dat.loc[dat["class"]==i, "component 2"], s=1, label=i)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(loc="center left", ncol=2, bbox_to_anchor=(1, 0.5))

def hm_mgenes(adQuery, adTrain=None, cgenes_list={}, list_of_types_toshow=[], list_of_training_to_show=[], number_of_genes_toshow=3, query_annotation_togroup='SCN_result', training_annotation_togroup='SCN_result', save = False):
    if list_of_types_toshow.__len__() == list_of_training_to_show.__len__():
        if list_of_types_toshow.__len__() == 0:
            list_of_types_toshow = np.unique(adQuery.obs['SCN_result'])
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
            if adQuery.obs['SCN_result'][j] == type:
                list_1.append(j)
                SCN_annot.append(type)
                annot.append(adQuery.obs[query_annotation_togroup][j])

    for i in range(list_of_training_to_show.__len__()):
        type = list_of_types_toshow[i]            
        if list_of_training_to_show[i]:
            for j in range(np.size(MT, 0)):
                if adTrain.obs['SCN_result'][j] == type:
                    list_2.append(j)
                    SCN_annot.append(type)
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

    adTrans.obs['SCN_result'] = SCN_annot.values
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