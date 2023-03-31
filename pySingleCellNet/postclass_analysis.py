import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from sklearn.ensemble import RandomForestClassifier
from scipy.sparse import csr_matrix
import warnings
from .utils import *
from .tsp_rf import *


def select_type_pairs(adTrain, adQuery, Cell_Type, threshold, upper = True, dlevel = 'cell_ontology_class'):
    ind_train = []
    ind_query = []

    for i in range(adTrain.obs.shape[0]):
        if adTrain.obs[dlevel][i] == Cell_Type:
            ind_train.append(i)

    for i in range(adQuery.obs.shape[0]):
        if upper:
            if adQuery.obs['SCN_class'][i] == Cell_Type and adQuery.obs[Cell_Type][i] >= threshold:
                ind_query.append(i)
        else:
            if adQuery.obs['SCN_class'][i] == Cell_Type and adQuery.obs[Cell_Type][i] <= threshold:
                ind_query.append(i)

    
    if type(adTrain.X) is csr_matrix:
        Mat_Train = adTrain.X.todense()
    elif type(adTrain.X) is np.matrix:
        Mat_Train = adTrain.X
    else:
        Mat_Train = np.array(adTrain.X)

    Mat_Train = Mat_Train[ind_train,:]
    Obs_Train = ['Train' for _ in range(ind_train.__len__())]


    if type(adQuery.X) is csr_matrix:
        Mat_Query = adQuery.X.todense()
    elif type(adQuery.X) is np.matrix:
        Mat_Query = adQuery.X
    else:
        Mat_Query = np.array(adQuery.X)

    Mat_Query = Mat_Query[ind_query,:]
    Obs_Query = ['Query' for _ in range(ind_query.__len__())]


    Mat_ranking = np.concatenate((Mat_Train, Mat_Query))
    Obs_ranking = Obs_Train+Obs_Query

    adRanking = sc.AnnData(Mat_ranking, obs=pd.DataFrame(Obs_ranking, columns=['source']))
    adRanking.var_names = adTrain.var_names

    sc.tl.rank_genes_groups(adRanking, groupby='source', groups=['Query'], reference='Train', method='t-test')


    Pvalues = []
    pvals = adRanking.uns['rank_genes_groups']['pvals']
    for pval in pvals:
        Pvalue = pval[0]
        Pvalues.append(Pvalue)

    logFCs = []
    logfoldchanges = adRanking.uns['rank_genes_groups']['logfoldchanges']
    for logfoldchange in logfoldchanges:
        logFC = logfoldchange[0]
        logFCs.append(logFC)

    gene_names = []
    names = adRanking.uns['rank_genes_groups']['names']
    for name in names:
        gene_name = name[0]
        gene_names.append(gene_name)

    voc_tab = pd.DataFrame(data = [Pvalues,logFCs])
    voc_tab = voc_tab.transpose()
    voc_tab.columns = ['Pvalue', 'logFC']
    voc_tab.index = gene_names

    voc_tab['st'] = voc_tab.index
    voc_tab['st'] = voc_tab['st'].astype('category')
    voc_tab['st'].cat.reorder_categories(adRanking.var_names, inplace=True)
    voc_tab.sort_values('st', inplace=True)
    voc_tab.set_index(['st'])
    voc_tab = voc_tab.drop(columns=['st'])

    adRanking.varm['voc_tab'] = voc_tab

    return adRanking

    