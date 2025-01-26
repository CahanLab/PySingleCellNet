from .qc import (
    find_knee_point,
    mito_rib,
    mito_rib_heme
)

from .adataTools import (
    generate_joint_graph,
    combine_pca_scores,
    limit_anndata_to_common_genes,
#    split_common_anndata,
    split_adata_indices,
    sort_obs_table,
    remove_singleton_groups,
    read_broken_geo_mtx,
    remove_genes,
    filter_anndata_slots,
    pull_out_genes,
    pull_out_genes_v2,
    find_elbow,
 #   convert_diffExp_to_dict,
    rank_genes_subsets,
    build_knn_graph,
    assign_optimal_cluster,
    rename_cluster_labels,
    reassign_selected_clusters
)

from .annotation import (
    write_gmt,
    read_gmt,
    filter_gene_list,
#    convert_ensembl_to_symbol,
    create_gene_structure_dict_by_stage,
    filter_genes_dict
)

# from .oldFunctions import (
#    norm_hvg_scale_pca,
#    norm_hvg_scale_pca_oct
#)

from .colors import (
    get_unique_colors
)

# API
__all__ = [
    "find_knee_point",
    "mito_rib",
    "mito_rib_heme",
    "generate_joint_graph",
    "combine_pca_scores",
    "limit_anndata_to_common_genes",
#    "split_common_anndata",
    "split_adata_indices",
    "sort_obs_table",
    "remove_singleton_groups",
    "read_broken_geo_mtx",
    "remove_genes",
    "filter_anndata_slots",
    "pull_out_genes",
    "pull_out_genes_v2",
    "find_elbow",
#    "convert_diffExp_to_dict",
    "write_gmt",
    "read_gmt",
    "filter_gene_list",
#    "convert_ensembl_to_symbol",
    "create_gene_structure_dict_by_stage",
    "filter_genes_dict",
#    "norm_hvg_scale_pca",
#    "norm_hvg_scale_pca_oct",
    "get_unique_colors",
    "rank_genes_subsets",
    "build_knn_graph",
    "assign_optimal_cluster",
    "rename_cluster_labels",
    "reassign_selected_clusters"
]

