import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

def sc_hmClass(classMat, grps,cRow=False,cCol=False):
    warnings.filterwarnings('ignore')
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

def sc_violinClass(sampTab, classRes, dLevel="cluster", threshold=0.20,  ncol =1, sub_cluster=[] ):
    warnings.filterwarnings('ignore')
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

def plot_attr(classRes, sampTab, dLevel, sub_cluster = []):
    warnings.filterwarnings('ignore')
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

def plot_umap(expMat, sampTab, dLevel="category"):
    standard_embedding = umap.UMAP(random_state=42).fit_transform(expMat.values)
    dat = pd.DataFrame(standard_embedding, index=expMat.index)
    dat = pd.concat([dat, sampTab[dLevel]], axis=1)
    dat.columns = ["component 1","component 2", "class"]
    grps=np.unique(dat["class"])
    for i in grps:
        plt.scatter(dat.loc[dat["class"]==i,"component 1"], dat.loc[dat["class"]==i, "component 2"], s=1, label=i)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(loc="center left", ncol=2, bbox_to_anchor=(1, 0.5))

def get_cate(classRes, sampTab, cThresh = 0):
    res=classRes.idxmax(axis=1)
    if cThresh>0:
        for i in range(0, len(res)):
            if classRes.loc[res.index[i], res[i]]<cThresh:
                res[i]="rand"
    st=sampTab.copy()
    st["category"]=res
    return st
