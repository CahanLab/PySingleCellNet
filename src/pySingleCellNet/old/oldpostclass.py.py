import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from anndata import AnnData
from sklearn.ensemble import RandomForestClassifier
from scipy.sparse import csr_matrix
import warnings
from .utils import *
from .tsp_rf import *
import gseapy as gp
import os
# import anndata
# import pySingleCellNet as pySCN
#import pacmap
import copy
import igraph as ig
from collections import defaultdict


def convert_rank_genes_groups_to_dict(
    adata: AnnData,
    groupby: str = 'leiden',
    key: str = 'rank_genes_groups',
    pvals_adj: int = 1
) -> dict:
    """Convert rank_genes_groups results from an AnnData object to a dictionary.

    Extracts differentially expressed genes per group from `adata` using 
    Scanpy's ranking functionality, optionally filtering by adjusted p-value.

    Args:
        adata (AnnData): Annotated data object containing gene expression data.
        groupby (str, optional): Column in `adata.obs` used for grouping. Defaults to 'leiden'.
        key (str, optional): Key in `adata.uns` where rank_genes_groups results are stored. 
            Defaults to 'rank_genes_groups'.
        pvals_adj (float, optional): Adjusted p-value threshold for filtering genes. 
            If not 1, filters genes with 'pvals_adj' less than this value. Defaults to 1.

    Returns:
        dict: Dictionary where each key is a group and each value is a list of gene symbols 
            differentially expressed in that group.
    """
    import scanpy as sc  # Ensure Scanpy is imported
    dedf_x = sc.get.rank_genes_groups_df(adata, group=None, key=key)
    dedf_x.dropna(subset=['names'], inplace=True)
    if pvals_adj != 1:
        dedf_x = dedf_x[dedf_x['pvals_adj'] < pvals_adj]
    groups = dedf_x['group'].cat.categories.to_list()
    tmp_dict = dict()
    if len(groups) > 0:
        for g in groups:
            tmpDF = dedf_x[dedf_x['group'] == g].copy()
            print(f"{g} .... {tmpDF.shape[0]}")
            tmp_dict[g] = tmpDF['names'].tolist()
    return tmp_dict




def determine_relationships(graph: ig.Graph, GpGc = False):
    relationships = defaultdict(dict)
    for node in graph.vs:
        node_label = node['name']
        # Parent-child relationships
        for neighbor in graph.neighbors(node, mode="OUT"):
            neighbor_label = graph.vs[neighbor]['name']
            relationships[node_label][neighbor_label] = "parent_child"
            relationships[neighbor_label][node_label] = "parent_child"  # Ensure symmetry
        # Sibling relationships
        parents = graph.neighbors(node, mode="IN")
        for parent in parents:
            parent_label = graph.vs[parent]['name']
            siblings = [graph.vs[sibling]['name'] for sibling in graph.neighbors(parent, mode="OUT") if sibling != node.index]
            for sibling in siblings:
                if sibling not in relationships[node_label]:
                    relationships[node_label][sibling] = "sibling"
                    relationships[sibling][node_label] = "sibling"  # Ensure symmetry
        # Grandparent-child relationships
        if GpGc:
            for parent in parents:
                grandparents = graph.neighbors(parent, mode="IN")
                for grandparent in grandparents:
                    grandparent_label = graph.vs[grandparent]['name']
                    if grandparent_label not in relationships[node_label]:
                        relationships[grandparent_label][node_label] = "grandparent_grandchild"
                        relationships[node_label][grandparent_label] = "grandparent_grandchild"  # Ensure symmetry

    return dict(relationships)


    def combine_adatas_for_deg(
    adatas: list,
    sample_obsvals: list,
    cellgrp_obsnames: list,
    new_cellgrp_obsname = 'comb_cellgrp',
    groupby_obsname = 'comb_sampname'
):
    # Check input
    if len(adatas) != 2 or len(sample_obsvals) != 2 or len(cellgrp_obsnames) != 2:
        raise ValueError("Input lists should have a length of 2.")
    # Create deep copies of the input AnnData objects
    adata1 = adatas[0].copy()
    adata2 = adatas[1].copy()
    # Assign sample names to the respective AnnData objects
    adata1.obs[groupby_obsname] = sample_obsvals[0]
    adata2.obs[groupby_obsname] = sample_obsvals[1]
    adata1.obs[new_cellgrp_obsname] = adata1.obs[cellgrp_obsnames[0]].copy()
    adata2.obs[new_cellgrp_obsname] = adata2.obs[cellgrp_obsnames[1]].copy()

    # pySCN.limit_anndata_to_common_genes([adata1, adata2])
    limit_anndata_to_common_genes([adata1, adata2])
    
    # Combine the AnnData objects
    # combined_adata = adata1.concatenate(adata2)
    combined_adata = anndata.concat([adata1, adata2])
    combined_adata.obs_names_make_unique()
    combined_adata.raw = combined_adata
    return combined_adata




    # run pacmap dimendsion reduction an adata.X
# default parameters
def embed_pacmap(adata, use_hvg = True):
    embedding = pacmap.PaCMAP(n_components=2, n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0)
    #adNorm = pySCN.norm_hvg_scale_pca(adNorm)
    adHVG = adata[:,adata.var['highly_variable']].copy()
    xmat = adHVG.X
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

    # pySCN.limit_anndata_to_common_genes([adTmp1, adTmp2])
    limit_anndata_to_common_genes([adTmp1, adTmp2])
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


# just run enrichR on result of convert_rank_genes_groups_to_dict() <- rank_genes_groups()
def enrichR_on_RGG(
    query_gene_lists: dict, # usually made with convert_rank_genes_groups_to_dict()
    ref_gene_lists: dict, # from source such as MSigDB
    background_genes: list, # list of possible genes
    seed: int = 3,
    min_size: int = 5
#    max_size: int = 1000
) -> dict:

    # trim ref_gene_lists
    ref_gene_lists = filter_gene_list(ref_gene_lists, min_size)
    query_gene_lists = filter_gene_list(query_gene_lists, min_size)
    ans = dict()
    qlist_names = list(query_gene_lists.keys())
    for qlist_name in qlist_names:
        ans[qlist_name] = gp.enrichr(gene_list=query_gene_lists[qlist_name], gene_sets=ref_gene_lists, background=background_genes, outdir=None)
    return ans


def enrichR_on_gene_modules(
    # adata: anndata,
    adata: AnnData,
    geneset: dict,
    result_name: str, # this should indicate the data source(s), but omit cell types and categories
    module_method = 'knn',
    seed: int = 3,
    min_size: int = 10,
    max_size: int = 500,
    hvg = True
) -> dict:
    # trim geneset
    geneset = filter_gene_list(geneset, min_size, max_size)
    ans = dict()
    bg_genes = adata.var_names.to_list()
    if hvg:
        bg_genes = adata.var_names[adata.var['highly_variable']].to_list()
    modname = module_method + "_modules"
    genemodules = adata.uns[modname].copy()
    for gmod, genelist in genemodules.items():
        tmp_enr = gp.enrichr(gene_list=genelist, gene_sets=geneset, background=bg_genes, outdir=None)
        ans[gmod] = tmp_enr
    return ans




def gsea_on_rank_genes_groups(
    adata,
    gene_sets,
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


def collect_enrichR_results_from_dict(
    enr_results: dict,
    adj_p_threshold = 1e-5
):
    # Initialize set of pathways. The order of these in prerank results and their composition will differ
    # so we need to get the union first
    pathways = pd.Index([])
    gene_signatures= list(enr_results.keys())
    for signature in gene_signatures:
        tmpRes = enr_results[signature].res2d.copy()
        gene_set_names = list(tmpRes['Term'])
        pathways = pathways.union(gene_set_names)
    # initialize an empty results data.frame
    enr_df = pd.DataFrame(0, columns = gene_signatures, index=pathways)
    for signature in gene_signatures:
        tmpRes = enr_results[signature].res2d.copy() 
        tmpRes.index = tmpRes['Term']
        tmpRes.loc[lambda df: df['Adjusted P-value'] > adj_p_threshold, "Odds Ratio"] = 0
        # nes_df.loc[ct_df.index,cell_type] = ct_df.loc[:,"NES"]
        enr_df[signature] = tmpRes["Odds Ratio"]
    enr_df = enr_df.apply(pd.to_numeric, errors='coerce')
    enr_df.fillna(0, inplace=True)
    return enr_df

def what_module_has_gene(
    adata,
    target_gene,
    module_method='knn'
) -> list:
    mod_slot = module_method + "_modules"
    if mod_slot not in adata.uns.keys():
        raise ValueError(mod_slot + " have not been identified.")
    genemodules = adata.uns[mod_slot]
    return [key for key, genes in genemodules.items() if target_gene in genes]




def deg(
    adata: AnnData,
    sample_obsvals: list = [],  # impacts the sign of the test statistic
    limitto_obsvals: list = [], # what cell_grps to test, if empty, test them all
    cellgrp_obsname: str = 'comb_cellgrp',  # .obs column name holding the cell sub-groups that will be iterated over
    groupby_obsname: str = 'comb_sampname',
    ncells_per_sample: int = 30, # don't test if fewer than ncells_per_sample
    test_name: str = 't-test'
    
) -> dict:
    ans = dict()
    # these are the keys for the rank_genes_groups object
    subset_keys = ['names', 'scores', 'pvals', 'pvals_adj', 'logfoldchanges']
    if len(sample_obsvals) == 0:
        sample_obsvals = adata.obs[groupby_obsname].unique().tolist()
    ans['sample_names'] = sample_obsvals # this is used later to properly order diffExp DF
    # check for mis-matches between provided cell_group names and those available in adata
    cellgroup_names_in_anndata = adata.obs[cellgrp_obsname].unique()
    if len(limitto_obsvals) > 0:
        unique_to_input = [x for x in limitto_obsvals if x not in cellgroup_names_in_anndata]
        if len(unique_to_input) > 0:
            print(f"The argument cellgrp_obsname has values that are not present in adata: {unique_to_input}")
        else:
            cellgroup_names = limitto_obsvals
    else:
        cellgroup_names = cellgroup_names_in_anndata
    tmp_dict = dict()

    mask = adata.obs[groupby_obsname].isin(sample_obsvals)
    adata = adata[mask].copy()

    for cell_group in cellgroup_names:
        print(f"cell group: {cell_group}")
        adTmp = adata[adata.obs[cellgrp_obsname] == cell_group].copy()
        vcounts = adTmp.obs[groupby_obsname].value_counts()
        if (len(vcounts) == 2) and ( (vcounts >= ncells_per_sample).all() ):
            sc.tl.rank_genes_groups(adTmp, use_raw=False, groupby=groupby_obsname, groups=[sample_obsvals[0]], reference=sample_obsvals[1], method=test_name)
            tmp_dict[cell_group] = convert_rankGeneGroup_to_df(adTmp.uns['rank_genes_groups'].copy(), subset_keys)
            #tmp_dict[cell_group] = sc.get.rank_genes_groups_df(adTmp, cell_group)
    
    ans['geneTab_dict'] = tmp_dict
    return ans



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

    



### 06-24-24: deprecate this function
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

    #adata = comp_max_min_path(adata, net, cateres, 'ct')
    adata = comp_max_min_path(adata, net, cateres, ct_name_column)

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






# 06-24-24 PC
# Deprecate this func
def class_by_threshold_old(adata_c: AnnData, thresholds: pd.DataFrame, columns_to_ignore: list = ["rand"], inplace = True ):
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







def class_by_threshold(adata_c: AnnData, thresholds: pd.DataFrame, relationships: dict = None, columns_to_ignore: list = ["rand"], inplace=True, class_obs_name='SCN_class_argmax'):
    """
    Classify cells based on SCN scores and thresholds, and annotate hybrid cells based on lineage relationships.
    
    Args:
        adata_c (AnnData): Annotated data matrix with SCN scores in `.obsm`.
        thresholds (pd.DataFrame): DataFrame containing thresholds for each cell type.
        relationships (dict, optional): Dictionary of relationships between cell types. If None, all hybrid cells are given a 'Hybrid' SCN_class_type. Defaults to None.
        columns_to_ignore (list, optional): List of columns to ignore in SCN scores. Defaults to ["rand"].
        inplace (bool, optional): If True, modifies the AnnData object in place. If False, returns a new AnnData object. Defaults to True.
    
    Returns:
        AnnData or None: Returns the modified AnnData object if inplace is False, otherwise returns None.
    """
    SCN_scores = adata_c.obsm["SCN_score"].copy()
    SCN_scores.drop(columns=columns_to_ignore, inplace=True, errors='ignore')
    
    # Create a boolean DataFrame indicating which values exceed the thresholds 
    exceeded = SCN_scores.sub(thresholds.squeeze(), axis=1) > 0 
    true_counts = exceeded.sum(axis=1)

    # Categorize the classification results based on number of passing classes
    class_type = ["None"] * len(true_counts)
    class_type = pd.DataFrame(class_type, index=true_counts.index.copy(), columns=['SCN_class_type'])
    singulars = true_counts == 1
    class_type.loc[singulars] = "Singular"
    hybrids = true_counts > 1

    # Create a list of lists of column names where values exceed the thresholds 
    result_list = [[col for col in exceeded.columns[exceeded.iloc[row].values]] for row in range(exceeded.shape[0])]
    
    if relationships is not None:
        for idx, cell_types in enumerate(result_list):
            if len(cell_types) > 1:
                # Determine the type of hybrid relationship
                hybrid_type = "Hybrid"
                for i, ct1 in enumerate(cell_types):
                    for ct2 in cell_types[i+1:]:
                        if ct1 in relationships and ct2 in relationships[ct1]:
                            if relationships[ct1][ct2] == "parent_child":
                                hybrid_type = "Parent.Child"
                            elif relationships[ct1][ct2] == "sibling":
                                hybrid_type = "Sibling"
                            elif relationships[ct1][ct2] == "grandparent_grandchild":
                                hybrid_type = "Gp.Gc"
                class_type.iloc[idx] = hybrid_type
    else:
        class_type.loc[hybrids] = "Mix"

    ans = ['_'.join(lst) for lst in result_list]
    ans = [i if len(i) != 0 else "None" for i in ans]

    adata_c.obs['SCN_class_emp'] = ans.copy()
    adata_c.obs['SCN_class_type'] = class_type.copy()
    # deal with cells classified as 'rand'
    adata_c.obs['SCN_class_emp'] = adata_c.obs.apply(lambda row: 'Rand' if row[class_obs_name] == 'rand' else row['SCN_class_emp'], axis=1)
    adata_c.obs['SCN_class_type'] = adata_c.obs.apply( lambda row: 'Rand' if row[class_obs_name] == 'rand' else row['SCN_class_type'], axis=1 )

    # adata_c.obs['SCN_class_emp'] = adata_c.obs.apply( lambda row: 'Rand' if row['SCN_class'] == 'rand', axis=1 )
    # adata_c.obs['SCN_class_type'] = adata_c.obs.apply( lambda row: 'Rand' if row['SCN_class'] == 'rand', axis=1 )

    if inplace:        
        return None
    else:
        return adata_c

    





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









import numpy as np
import pandas as pd
import igraph as ig
from anndata import AnnData
import scanpy as sc


# import warnings
# import anndata as ad
# from scipy.sparse import csr_matrix
# from .utils import *
# from .tsp_rf import *
# import os
# import copy
# from collections import defaultdict

def add_scn_class_cat(
    adata: AnnData
):
    """
    Add a new .obs column 'SCN_class_cat' to an AnnData object.
    The value is taken from .obs['SCN_class_emp'] when .obs['SCN_class_type'] is 'Singular'.
    Otherwise, it is taken from .obs['SCN_class_type'].
    
    Parameters:
        adata (AnnData): The AnnData object to modify.
    
    Returns:
        None: The function modifies the AnnData object in place.
    """
    if 'SCN_class_emp' not in adata.obs or 'SCN_class_type' not in adata.obs:
        raise ValueError("The input AnnData object must have both 'SCN_class_emp' and 'SCN_class_type' in .obs.")
    
    adata.obs['SCN_class_cat'] = adata.obs.apply(
        lambda row: row['SCN_class_emp'] if row['SCN_class_type'] == 'Singular' else row['SCN_class_type'], axis=1
    )


def categorize_classification(
    adata_c: AnnData,
    thresholds: pd.DataFrame,
    graph: ig.Graph = None,
    k: int = 3,
    columns_to_ignore: list = ["rand"],
    inplace: bool = True,
    class_obs_name: str = 'SCN_class_argmax'
):
    """
    Classify cells based on SCN scores and thresholds, then categorize 
    multi-class cells as either 'Intermediate' or 'Hybrid' by checking 
    the distance among their high-scoring cell types in a user-provided iGraph.
    
    Rules:
      - If exactly one cell type exceeds threshold => "Singular"
      - If zero cell types exceed threshold => "None"
      - If more than one => check distances among all cell types in the iGraph:
          * If ALL pairs are within k edges => "Intermediate"
          * Else => "Hybrid"
      - If cell is predicted as 'rand' => "Rand"
    
    Parameters
    ----------
    adata_c : AnnData
        Annotated data matrix with:
          - .obsm["SCN_score"]: DataFrame of SCN scores for each cell type.
          - .obs[class_obs_name]: the predicted cell type (argmax classification).
    thresholds : pd.DataFrame
        Thresholds for each cell type (column names matching the SCN score columns).
        Typically has a single row of thresholds or a Series-like shape.
    graph : ig.Graph
        An iGraph describing relationships between cell types. Must have 
        vertex names matching the cell-type columns in SCN_score. 
        If None, a ValueError is raised.
    k : int
        Maximum distance on the graph for cell types to be considered "Intermediate".
        If any pair of high-scoring cell types is >= k edges apart, the cell is "Hybrid".
    columns_to_ignore : list
        SCN score columns to ignore (e.g. ["rand"]).
    inplace : bool
        If True, modifies adata_c in place. Otherwise, returns a new AnnData object.
    class_obs_name : str
        Name of the .obs column with the argmax classification (e.g. "SCN_class_argmax").
    
    Returns
    -------
    AnnData or None
        If inplace=False, returns the modified AnnData. Otherwise returns None.
    """
    
    if graph is None:
        raise ValueError("A valid iGraph 'graph' must be provided. None was given.")
    
    # 1) Ensure SCN scores are present
    if "SCN_score" not in adata_c.obsm:
        raise ValueError("No 'SCN_score' in adata_c.obsm. Please provide SCN scores.")
    
    # 2) Prepare SCN scores and threshold checks
    SCN_scores = adata_c.obsm["SCN_score"].copy()
    SCN_scores.drop(columns=columns_to_ignore, inplace=True, errors='ignore')
    
    # Ensure that thresholds align with SCN_scores columns
    # If thresholds is a DataFrame with 1 row, or a Series, .squeeze() should map columns well.
    # Here we assume thresholds has columns matching SCN_scores.columns
    exceeded = SCN_scores.sub(thresholds.squeeze(), axis=1) > 0
    true_counts = exceeded.sum(axis=1)  # how many cell types exceed threshold
    
    # Make a list of which cell types exceed threshold for each cell
    # print("SCN_scores.shape:", SCN_scores.shape)
    result_list = [
        [col for col in exceeded.columns[exceeded.iloc[row].values]]
        for row in range(exceeded.shape[0])
    ]
    
    # 3) Initialize classification as "None", then fill in "Singular" or "Hybrid/Intermediate"
    class_type = pd.Series(["None"] * len(true_counts), index=true_counts.index, name="SCN_class_type")
    
    # Single = exactly one exceeding threshold
    singular_mask = (true_counts == 1)
    class_type.loc[singular_mask] = "Singular"
    # print("B")
    # 4) Build a mapping from cell type name to vertex index
    #    We assume graph.vs["name"] are the same cell-type labels that appear in SCN_scores columns.
    type2index = {}
    if "name" in graph.vs.attributes():
        type2index = {graph.vs[i]["name"]: i for i in range(graph.vcount())}
    else:
        raise ValueError("graph does not have a 'name' attribute for vertices.")
    
    def is_all_within_k_edges(cell_types):
        """
        Return True if *all pairs* of cell types are within k edges
        in the graph. Otherwise False.
        """
        if len(cell_types) <= 1:
            return True
        for i in range(len(cell_types)):
            for j in range(i + 1, len(cell_types)):
                ct1, ct2 = cell_types[i], cell_types[j]
                if ct1 not in type2index or ct2 not in type2index:
                    # If we can't find a cell type in the graph, consider it not close
                    return False
                idx1 = type2index[ct1]
                idx2 = type2index[ct2]
                dist = graph.shortest_paths(idx1, idx2)[0][0]
                if dist >= k:
                    return False
        return True
    
    # print("C")
    # More than one = multi_mask
    multi_mask = (true_counts > 1)
    # Get integer indices where multi_mask is True
    multi_indices = np.where(multi_mask)[0]
    
    for i in multi_indices:
        c_types = result_list[i]  
        if is_all_within_k_edges(c_types):
            class_type.iloc[i] = "Intermediate"
        else:
            class_type.iloc[i] = "Hybrid"
    
    # print("5")
    # 5) Build underscore-delimited strings of the cell types that exceeded threshold
    ans = ['_'.join(lst) if lst else 'None' for lst in result_list]
    
    # 6) Place in obs
    adata_c.obs['SCN_class_emp'] = ans
    adata_c.obs['SCN_class_type'] = class_type
    
    # 7) Overwrite with "Rand" for cells with predicted argmax = "rand"
    if class_obs_name not in adata_c.obs:
        raise ValueError(f"{class_obs_name} not found in adata_c.obs.")
    
    adata_c.obs['SCN_class_emp'] = adata_c.obs.apply(
        lambda row: 'Rand' if row[class_obs_name] == 'rand' else row['SCN_class_emp'],
        axis=1
    )
    adata_c.obs['SCN_class_type'] = adata_c.obs.apply(
        lambda row: 'Rand' if row[class_obs_name] == 'rand' else row['SCN_class_type'],
        axis=1
    )
    
    # add a new .obs column that combines SCN_class_emp and SCN_class_type
    add_scn_class_cat(adata_c)

    # 8) Return
    if inplace:
        return None
    else:
        return adata_c



def comp_ct_thresh(adata_c: AnnData, qTile: int = 0.05, obs_name = 'SCN_class_argmax') -> pd.DataFrame:

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
            templocs = sampTab[sampTab[obs_name] == ct].index
            tempscores = scnScores.loc[templocs, ct]
            thrs.loc[[ct], [0]] = np.quantile(tempscores, q = qTile)
    
        return thrs




def paga_connectivities_to_igraph(
    adata,
    threshold=0.05, 
    paga_key="paga", 
    connectivities_key="connectivities", 
    group_key="auto_cluster"
):
    """
    Convert a PAGA adjacency matrix (stored in adata.uns[paga_key][connectivities_key]) 
    to an undirected iGraph object, keeping only edges whose weight >= threshold.
    
    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object. Must have:
          - adata.uns[paga_key][connectivities_key]: the PAGA adjacency matrix (CSR).
          - adata.obs[group_key].cat.categories: the node labels (one per cluster).
    threshold : float
        Keep only edges with weight >= this threshold.
    paga_key : str
        The key in adata.uns where PAGA results are stored.
    connectivities_key : str
        The key in adata.uns[paga_key] for the connectivity/adjacency matrix.
    group_key : str
        The .obs column name whose categorical .categories define the node labels.
    
    Returns
    -------
    ig.Graph
        An undirected iGraph graph, with edges that pass the threshold, 
        vertex names set to the cluster categories, and edge weights.
    """
    # 1) Pull out the adjacency matrix (CSR)
    adjacency_csr = adata.uns[paga_key][connectivities_key]
    
    # 2) Convert it to COO for easy iteration
    adjacency_coo = adjacency_csr.tocoo()
    
    # 3) Build lists of edges + weights, thresholding
    edges = []
    weights = []
    for i, j, val in zip(adjacency_coo.row, adjacency_coo.col, adjacency_coo.data):
        # Only keep one of (i, j) or (j, i) in an undirected graph
        # and only keep edges above the threshold
        if i < j and val >= threshold:
            edges.append((i, j))
            weights.append(val)
    
    # 4) Create an undirected graph in iGraph
    g = ig.Graph(n=adjacency_csr.shape[0], edges=edges, directed=False)
    
    # 5) Assign edge weights
    g.es["weight"] = weights
    
    # 6) If the shape of the adjacency matches the # of cluster categories, assign node names
    if group_key in adata.obs:
        categories = adata.obs[group_key].cat.categories
        if len(categories) == adjacency_csr.shape[0]:
            g.vs["name"] = list(categories)
        else:
            print(
                f"Warning: adjacency matrix size ({adjacency_csr.shape[0]}) "
                f"differs from number of categories ({len(categories)}). "
                "Vertex names will not be assigned."
            )
    else:
        print(
            f"Warning: {group_key} not found in adata.obs; "
            "vertex names will not be assigned."
        )
    
    return g









def gsea_on_deg(
    deg_res: dict,
    genesets_name: str,
    genesets: dict,
    permutation_num: int = 100,
    threads: int = 4,
    seed: int = 3,
    min_size: int = 10,
    max_size: int = 500
) -> dict:
    """Performs Gene Set Enrichment Analysis (GSEA) on differential gene expression results.

    This function applies GSEA to each group of differential gene expression results 
    using the provided gene sets. It uses the `gseapy.prerank` method to identify 
    enriched gene sets across multiple cell groups or conditions.

    Args:
        deg_res (dict): 
            A dictionary where keys are cell group names and values are DataFrames 
            containing differential expression results. Each DataFrame must have 
            `'names'` (gene names) and `'scores'` (ranking scores) columns.
            
        genesets_name (str): 
            A name for the gene set collection used for identification (currently unused).
            
        genesets (dict): 
            A dictionary of gene sets where keys are gene set names and values are lists of genes.
            
        permutation_num (int, optional): 
            Number of permutations to perform during GSEA. Default is 100.
            
        threads (int, optional): 
            Number of parallel threads to use for computation. Default is 4.
            
        seed (int, optional): 
            Random seed for reproducibility. Default is 3.
            
        min_size (int, optional): 
            Minimum size of gene sets to be included in the analysis. Default is 10.
            
        max_size (int, optional): 
            Maximum size of gene sets to be included in the analysis. Default is 500.

    Returns:
        dict: 
            A dictionary where keys are cell group names and values are GSEA result objects 
            from `gseapy.prerank` for each group.

    Example:
        >>> deg_results = {
        ...     'Cluster1': pd.DataFrame({'names': ['GeneA', 'GeneB'], 'scores': [2.5, -1.3]}),
        ...     'Cluster2': pd.DataFrame({'names': ['GeneC', 'GeneD'], 'scores': [1.2, -2.1]})
        ... }
        >>> gene_sets = {'Pathway1': ['GeneA', 'GeneC'], 'Pathway2': ['GeneB', 'GeneD']}
        >>> results = gsea_on_deg(deg_results, 'ExampleGeneSets', gene_sets)
    """
    ans = dict()
    # categories = diff_gene_dict['category_names']
    # categories = list(deg_res.keys())[0]
    ### diff_gene_tables = deg_res['geneTab_dict']
    diff_gene_tables = deg_res
    cellgrp_vals = list(diff_gene_tables.keys()) # this should be an optional parameter
    for cellgrp in cellgrp_vals:
        # run_name = gene_set_name + "::" + cell_type + "_" + categories[0] + "_vs_" + categories[1]
        #  out_dir = path_to_results + "/" + genesets_name + "/" + cellgrp
        # os.makedirs(out_dir, exist_ok=True)
        # prep data
        atab = diff_gene_tables[cellgrp]
        atab = atab[['names', 'scores']]
        atab.columns = ['0', '1']
        pre_res = gp.prerank(rnk=atab, gene_sets=genesets, permutation_num = permutation_num, ascending = False, threads=threads,  no_plot = True, seed=seed, min_size = min_size, max_size=max_size)
        ans[cellgrp] = pre_res
    return ans

def convert_rankGeneGroup_to_df( rgg: dict, list_of_keys: list) -> pd.DataFrame:
# Annoying but necessary function to deal with recarray format of .uns['rank_genes_groups'] to make sorting/extracting easier

    arrays_dict = {}
    for key in list_of_keys:
        recarray = rgg[key]
        field_name = recarray.dtype.names[0]  # Get the first field name
        arrays_dict[key] = recarray[field_name]

    return pd.DataFrame(arrays_dict)

def collect_gsea_results_from_dict(
    gsea_dict2: dict,
    fdr_thr = 0.25
):
    
    # Initialize set of pathways. The order of these in prerank results and their composition will differ
    # so we need to get the union first

    gsea_dict = copy.deepcopy(gsea_dict2)
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
        ### tmp_dict[ct] = convert_rankGeneGroup_to_df(adTmp.uns['rank_genes_groups'].copy(), subset_keys)
    
    ans['geneTab_dict'] = tmp_dict
    return ans



def convert_rankGeneGroup_to_df(rgg: dict, list_of_keys: list) -> pd.DataFrame:
    """Convert a rank_genes_groups recarray to a DataFrame.

    Transforms a recarray output from `.uns['rank_genes_groups']` into a 
    pandas DataFrame for easier sorting and extraction.

    Args:
        rgg (dict): Dictionary containing recarrays from rank_genes_groups data.
        list_of_keys (list): List of keys to extract from each recarray.

    Returns:
        pd.DataFrame: DataFrame constructed from the recarray data.
    """
    arrays_dict = {}
    for key in list_of_keys:
        recarray = rgg[key]
        field_name = recarray.dtype.names[0]  # Get the first field name
        arrays_dict[key] = recarray[field_name]

    return pd.DataFrame(arrays_dict)




def make_diff_gene_dict(
    adata: AnnData,
    celltype_groupby: str = "SCN_class",
    category_groupby: str = "SCN_class_type",
    category_names: list = ["None", "Singular"],
    celltype_names: list = []
) -> dict:
    """Generate a dictionary of differential gene expression tables for each cell type.

    For each specified cell type in `adata`, compute differential gene expression 
    between two categories using Scanpy's ranking methods and store the results 
    in a dictionary.

    Args:
        adata (AnnData): Annotated data matrix containing gene expression and metadata.
        celltype_groupby (str, optional): Column in `adata.obs` used to group cells by type. Defaults to "SCN_class".
        category_groupby (str, optional): Column in `adata.obs` used to define categories 
            for differential expression. Defaults to "SCN_class_type".
        category_names (list, optional): List containing two category names for comparison. 
            Defaults to ["None", "Singular"].
        celltype_names (list, optional): Specific cell type names to process. If empty, 
            uses all unique cell types found in `adata.obs[celltype_groupby]`. Defaults to [].

    Returns:
        dict: Dictionary with differential gene expression results. The structure is intended to 
            include keys like 'category_names' and 'geneTab_dict', but note that parts of the function 
            (e.g., filling `tmp_dict`) are incomplete in the provided code.
    """
    import scanpy as sc  # Ensure Scanpy is imported
    ans = dict()

    subset_keys = ['names', 'scores', 'pvals', 'pvals_adj', 'logfoldchanges']

    celltype_names_in_anndata = adata.obs[celltype_groupby].unique()
    if len(celltype_names) > 0:
        unique_to_input = [x for x in celltype_names if x not in celltype_names_in_anndata]
        if len(unique_to_input) > 0:
            print(f"The argument celltype_names has values that are not present in adata: {unique_to_input}")
        else:
            ans['category_names'] = category_names
            tmp_dict = dict()

    if len(celltype_names) == 0:
        celltype_names = celltype_names_in_anndata

    for ct in celltype_names:
        adTmp = adata[adata.obs[celltype_groupby] == ct].copy()
        sc.tl.rank_genes_groups(
            adTmp,
            use_raw=False,
            groupby=category_groupby,
            groups=[category_names[0]],
            reference=category_names[1],
            method="wilcoxon"
        )
        # The following lines are commented out in the original code.
        # They likely should convert results and store in tmp_dict.
        # tmp_dict[ct] = convert_rankGeneGroup_to_df(adTmp.uns['rank_genes_groups'].copy(), subset_keys)
    
    ans['geneTab_dict'] = tmp_dict
    return ans









































