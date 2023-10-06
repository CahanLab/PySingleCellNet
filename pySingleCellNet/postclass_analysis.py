import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from sklearn.ensemble import RandomForestClassifier
from scipy.sparse import csr_matrix
import warnings
from .utils import *
from .tsp_rf import *
import gseapy as gp
import os
import anndata
import pySingleCellNet as pySCN
import pacmap

# run pacmap dimendsion reduction an adata.X
# default parameters
def embed_pacmap(adata):
    embedding = pacmap.PaCMAP(n_components=2, n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0)
    xmat = adata.X
    xmat = xmat.toarray()
    X_transformed = embedding.fit_transform(xmat)
    adata.obsm['X_pacmap'] = X_transformed.copy()


def merge_for_diffExp(
    ad1: AnnData,
    ad2: AnnData,
    sample_name_1: str,
    sample_name_2: str,
    cell_groups = list,
    cell_group_name: str = "SCN_class",
    cell_subset_name: str = "SCN_class_type",
    cell_subset_1: str = "Singular",
    cell_subset_2: str = "Singular",
    new_obs_name = "cate_sample"
) -> AnnData:
    """
    Merges two AnnData objects for differential expression analysis.

    The function first filters the two AnnData objects based on the cell subset name and copies them. 
    Then, it adds a new observation to each AnnData object. After that, it filters the AnnData objects 
    based on the cell group name. It ensures that the two AnnData objects have the same genes before 
    concatenating them. The function returns the concatenated AnnData object.

    Args:
        ad1 (AnnData): First AnnData object.
        ad2 (AnnData): Second AnnData object.
        sample_name_1 (str): Name of the first sample.
        sample_name_2 (str): Name of the second sample.
        cell_groups (list, optional): List of cell groups. Defaults to list.
        cell_group_name (str, optional): Name of the cell group. Defaults to "SCN_class".
        cell_subset_name (str, optional): Name of the cell subset. Defaults to "SCN_class_type".
        cell_subset_1 (str, optional): First cell subset. Defaults to "Singular".
        cell_subset_2 (str, optional): Second cell subset. Defaults to "Singular".
        new_obs_name (str, optional): New observation name. Defaults to "cate_sample".

    Returns:
        AnnData: The concatenated AnnData object.
    """
    adTmp1 = ad1[ad1.obs[cell_subset_name] == cell_subset_1].copy()
    adTmp2 = ad2[ad2.obs[cell_subset_name] == cell_subset_2].copy()

    adTmp1.obs[new_obs_name] = sample_name_1 + "_" + adTmp1.obs[cell_subset_name]
    adTmp2.obs[new_obs_name] = sample_name_2 + "_" + adTmp2.obs[cell_subset_name]

    adTmp1 = adTmp1[adTmp1.obs[cell_group_name].isin(cell_groups)]
    adTmp2 = adTmp2[adTmp2.obs[cell_group_name].isin(cell_groups)]

    pySCN.limit_anndata_to_common_genes([adTmp1, adTmp2])
    adBoth = anndata.concat([adTmp1, adTmp2])
    return adBoth


# This should return the combined data frames AND an object that can be used to set colors of gsea heatmap
def combine_gsea_dfs(
    gsea_dfs: list,
    df_names: list
):
    """
    Combine multiple GSEA results (add as new renamed columns)

    Args:
        gsea_dfs: each produced by pySCN.collect_gsea_results_from_dict
        df_names: abbreviation or name to preprend to columns of corresponding df

    Returns:
        pd.DataFrame: in which gsea_dfs have been concatenated and column names updated
        pd.Series: index = column name of ^, value = df_name 
    """
    new_column_names = []
    column_annotations = []
    gsea_dfs_new = gsea_dfs.copy() 
    for i in range(len(gsea_dfs_new)):
        tmp = list(gsea_dfs_new[i].columns)
        new_cols = [df_names[i] + "_" + s for s in tmp]
        gsea_dfs_new[i].columns = new_cols
        tmp_col_anns = [df_names[i]] * gsea_dfs_new[i].shape[1]
        column_annotations.extend(tmp_col_anns)

    gsea_comb = pd.concat(gsea_dfs_new, axis=1)
    gsea_comb = gsea_comb.fillna(0)

    my_series = pd.Series(column_annotations, index=list(gsea_comb.columns))
    return gsea_comb, my_series


def gsea_on_diff_gene_dict(
    diff_gene_dict: dict,
    gene_set_name: str,
    gene_set_path: str,
    path_to_results: str,
    result_name: str, # this should indicate the data source(s), but omit cell types and categories
    permutation_num: int = 100,
    threads: int = 4,
    seed: int = 3,
    min_size: int = 10,
    max_size: int = 500
) -> dict:

    ans = dict()

    categories = diff_gene_dict['category_names']
    diff_gene_tables = diff_gene_dict['geneTab_dict']
    cell_types = list(diff_gene_tables.keys()) # this should be an optional parameter

    for cell_type in cell_types:
        # run_name = gene_set_name + "::" + cell_type + "_" + categories[0] + "_vs_" + categories[1]
        out_dir = path_to_results + "/" + result_name + "/" + gene_set_name + "/" + cell_type + "_" + categories[0] + "_vs_" + categories[1]
        os.makedirs(out_dir, exist_ok=True)
        
        # prep data
        atab = diff_gene_tables[cell_type]
        atab = atab[['names', 'scores']]
        atab.columns = ['0', '1']

        pre_res = gp.prerank(rnk=atab, gene_sets=gene_set_path, outdir=out_dir, 
            permutation_num = permutation_num, ascending = False, threads=threads,  no_plot = True, seed=seed, min_size = min_size, max_size=max_size)
        ans[cell_type] = pre_res

    return ans

def collect_gsea_results_from_dict(
    gsea_dict: dict,
    fdr_thr = 0.25
):
    
    # Initialize set of pathways. The order of these in prerank results and their composition will differ
    # so we need to get the union first

    pathways = pd.Index([])
    cell_types = list(gsea_dict.keys())

    for cell_type in cell_types:
        tmpRes = gsea_dict[cell_type].res2d
        gene_set_names = list(tmpRes['Term'])
        pathways = pathways.union(gene_set_names)
        
    # initialize an empty results data.frame
    nes_df = pd.DataFrame(0, columns = cell_types, index=pathways)

    # 
    for cell_type in cell_types:
        ct_df = gsea_dict[cell_type].res2d
        ct_df.index = ct_df['Term']
        ct_df.loc[lambda df: df['FDR q-val'] > fdr_thr, "NES"] = 0
        # nes_df.loc[ct_df.index,cell_type] = ct_df.loc[:,"NES"]
        nes_df[cell_type] = ct_df["NES"]

    nes_df = nes_df.apply(pd.to_numeric, errors='coerce')
    return nes_df


def make_diff_gene_dict(
    adata: AnnData,
    celltype_groupby: str = "SCN_class",
    category_groupby: str = "SCN_class_type",
    category_names: str = ["None", "Singular"],
    celltype_names: list = []
) -> dict:

    ans = dict()

    # these are the keys for the rank_genes_groups object
    subset_keys = ['names', 'scores', 'pvals', 'pvals_adj', 'logfoldchanges']

    # check for mis-matches
    celltype_names_in_anndata = adata.obs[celltype_groupby].unique()
    if len(celltype_names) > 0:
        unique_to_input = [x for x in celltype_names if x not in celltype_names_in_anndata]
        if len(unique_to_input) > 0:
            print(f"The argument celltype_names has values that are not present in adata: {unique_to_input}")
            # return dict()
        else:

            ans['category_names'] = category_names # this is used later to properly order diffExp DF
            tmp_dict = dict()

            if len(celltype_names)  == 0:
                celltype_names = celltype_names_in_anndata

            for ct in celltype_names:
                adTmp = adata[adata.obs[celltype_groupby] == ct].copy()
                # print(ct)
                sc.tl.rank_genes_groups(adTmp, use_raw=False, groupby=category_groupby, groups=[category_names[0]], reference=category_names[1], method="wilcoxon")
                #### ans[ct] = convert_rankGeneGroup_to_df(adTmp.uns['rank_genes_groups'].copy(), subset_keys)
                tmp_dict[ct] = convert_rankGeneGroup_to_df(adTmp.uns['rank_genes_groups'].copy(), subset_keys)
    
            ans['geneTab_dict'] = tmp_dict

    return ans
    
def class_by_threshold(adata_c: AnnData, thresholds: pd.DataFrame, columns_to_ignore: list = ["rand"], inplace = True ):
    SCN_scores = adata_c.obsm["SCN_score"].copy()
    SCN_scores.drop(columns = columns_to_ignore, inplace=True, errors = 'ignore')
    
    # create a boolean DataFrame indicating which values exceed the thresholds 
    exceeded = SCN_scores.sub(thresholds.squeeze(), axis=1) > 0 
    true_counts = exceeded.sum(axis=1)

    # categorize the classification results based on number of passing classes
    class_type = ["None"] * len(true_counts)
    class_type = pd.DataFrame( class_type, index=true_counts.index.copy() )
    singulars = true_counts == 1
    class_type.loc[ true_counts.index[singulars] ] = "Singular"
    hybrids = true_counts > 1
    class_type.loc[ true_counts.index[hybrids] ] = "Hybrid"

    # create a list of lists of column names where values exceed the thresholds 
    result_list = [[col for col in exceeded.columns[exceeded.iloc[row].values]] for row in range(exceeded.shape[0])]
    ans = ['_'.join(lst) for lst in result_list]
    ans = [i if len(i) != 0 else "None" for i in ans]

    if inplace == True:
        adata_c.obs['SCN_class_emp'] = ans.copy()
        adata_c.obs['SCN_class_type'] = class_type.copy()



def wrap_ct_categories(adata, adata_c, thrs, net, ct_name_column, distInc=0):
    cateres = cate_scn(adata_c, thrs)
    adata = wrap_max_min_path(adata, net, cateres, ct_name_column)
    adata = translate_dist_cate(adata)
    return adata

def translate_dist_cate(adata, distInc=1):
    adata.obs['ct_category'] = 'None'
    for ob in adata.obs_names:
        if adata.obs.loc[ob, 'nCates'] == 1:
            adata.obs.loc[ob, 'ct_category'] = 'Singular'
        elif adata.obs.loc[ob, 'nCates'] >1:
            if adata.obs.loc[ob, 'ct_dist'] <= adata.obs.loc[ob, 'nCates']-distInc:
                adata.obs.loc[ob, 'ct_category'] = 'Intermediate'
            else:
                adata.obs.loc[ob, 'ct_category'] = 'Hybrid'
    
    return adata


def wrap_max_min_path(adata, net, cateres, ct_name_column):
    adata.obs['nCates'] = 0
    for ob in adata.obs_names:
        adata.obs.loc[ob,'nCates'] = cateres[ob].__len__()
    
    adata.obs['ct_dist'] = -2
    for ob in adata.obs_names:
        if adata.obs.loc[ob, 'nCates'] == 0:
            adata.obs.loc[ob, 'ct_dist'] = -1
        elif adata.obs.loc[ob, 'nCates'] == 1:
            adata.obs.loc[ob, 'ct_dist'] = 0

    adata = comp_max_min_path(adata, net, cateres, 'ct')

    return adata

def comp_max_min_path(adata, net, cateres, ct_name_column):
    df = net.get_vertex_dataframe()
    for ob in adata.obs_names:
        node_ids = []
        if adata.obs.loc[ob, 'nCates'] >1:
            node_ids = cateres[ob]
            nodes_min_paths = []

            for i in range(node_ids.__len__()):
                for j in range(node_ids.__len__()):
                    if j>i:
                        start_type = node_ids[i]
                        target_type = node_ids[j]
                        start_id = df[df[ct_name_column] == start_type]['name'].to_list()[0]
                        target_id = df[df[ct_name_column] == target_type]['name'].to_list()[0]

                        path = net.get_shortest_paths(start_id, to=target_id)[0].__len__()-1
                        nodes_min_paths.append(path)
            
            adata.obs.loc[ob, 'ct_dist'] = np.max(nodes_min_paths)
    return adata


def cate_scn(adata_c, thrs):
    # check whether .obsm['SCN_score'] has been defined
    if "SCN_score" not in adata_c.obsm_keys():
        print("No .obsm['SCN_score'] was found in the annData provided. You may need to run PySingleCellNet.scn_classify()")
        return
    else:
        sampTab = adata_c.obs.copy()
        scnScores = adata_c.obsm["SCN_score"].copy()

        CateRes = {}
        for ob in scnScores.index:
            cts = []
            for ct in scnScores.columns:
                if ct == 'rand':
                    break
                else:
                    if scnScores.loc[ob].loc[ct]>thrs.loc[ct].loc[0]:
                        cts.append(ct)
            CateRes[ob] = cts

    return CateRes


def graph_from_data_frame(Net, edge_dataframe, node_dataframe, attribution_column_names, directed=True):
    attr = {}
    for attr_names in attribution_column_names:
        attr[attr_names] = node_dataframe[attr_names].to_numpy()
    
    Net.add_vertices(n=node_dataframe.id.to_numpy(), attributes=attr)

    for ind in edge_dataframe.index:
        tempsource = edge_dataframe.loc[ind].loc['from']
        temptarget = edge_dataframe.loc[ind].loc['to']
        Net.add_edges([(tempsource, temptarget)])
    
    return Net


def comp_ct_thresh(adata_c: AnnData, qTile: int = 0.05) -> pd.DataFrame:

    # check whether .obsm['SCN_score'] has been defined
    if "SCN_score" not in adata_c.obsm_keys():
        print("No .obsm['SCN_score'] was found in the annData provided. You may need to run PySingleCellNet.scn_classify()")
        return
    else:
        #scnScores = pd.DataFrame(adata_c.X)
        #scnScores.columns = adata_c.var_names
        #scnScores.index = adata_c.obs_names
        # sampTab = adata_c.obs
        sampTab = adata_c.obs.copy()
        scnScores = adata_c.obsm["SCN_score"].copy()

        cts = scnScores.columns
        cts = cts.drop('rand')
        thrs = pd.DataFrame(np.repeat(0, cts.__len__()))
        thrs.index = cts

        for ct in cts:
            print(ct)
            templocs = sampTab[sampTab.SCN_class == ct].index
            tempscores = scnScores.loc[templocs, ct]
            thrs.loc[[ct], [0]] = np.quantile(tempscores, q = qTile)
    
        return thrs





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

    