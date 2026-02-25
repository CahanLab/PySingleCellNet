"""Tests for the categorize pipeline."""

import numpy as np
import pandas as pd
import pytest
import scanpy as sc


def test_comp_ct_thresh(classified_query):
    """comp_ct_thresh should return a DataFrame with values in [0,1]."""
    from pySingleCellNet.tools.categorize import comp_ct_thresh

    thresholds = comp_ct_thresh(classified_query)
    assert isinstance(thresholds, pd.DataFrame)
    assert thresholds.shape[0] > 0
    # All threshold values should be in [0, 1]
    assert (thresholds.values >= 0).all()
    assert (thresholds.values <= 1).all()


def test_paga_connectivities_to_igraph(pijuan):
    """paga_connectivities_to_igraph should return an igraph.Graph with vertex names."""
    from pySingleCellNet.tools.categorize import paga_connectivities_to_igraph

    ad = pijuan.copy()
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)
    sc.pp.highly_variable_genes(ad, n_top_genes=2000, flavor="seurat_v3")
    sc.pp.pca(ad, n_comps=20, mask_var="highly_variable")
    sc.pp.neighbors(ad, n_neighbors=10)

    groupby = "ct1" if "ct1" in ad.obs.columns else ad.obs.columns[0]
    ad.obs["auto_cluster"] = ad.obs[groupby].astype("category")

    import igraph as ig
    graph = paga_connectivities_to_igraph(ad, group_key="auto_cluster", n_comps=20)
    assert isinstance(graph, ig.Graph)
    assert "name" in graph.vs.attributes()
    assert graph.vcount() > 0


def test_categorize_classification(classified_query):
    """categorize_classification should add SCN_class_type to obs."""
    from pySingleCellNet.tools.categorize import (
        comp_ct_thresh,
        categorize_classification,
        paga_connectivities_to_igraph,
    )

    ad = classified_query.copy()

    # Need a paga graph - build a simple one
    sc.pp.highly_variable_genes(ad, n_top_genes=1000, flavor="seurat_v3")
    sc.pp.pca(ad, n_comps=20, mask_var="highly_variable")
    sc.pp.neighbors(ad, n_neighbors=10)

    ad.obs["auto_cluster"] = ad.obs["SCN_class_argmax"].astype(str).astype("category")
    graph = paga_connectivities_to_igraph(ad, group_key="auto_cluster", n_comps=20)

    thresholds = comp_ct_thresh(ad)
    categorize_classification(ad, thresholds=thresholds, graph=graph)

    assert "SCN_class_type" in ad.obs.columns
    valid_categories = {"Singular", "None", "Intermediate", "Hybrid", "Rand"}
    observed = set(ad.obs["SCN_class_type"].unique())
    assert observed.issubset(valid_categories)


def test_filter_adata_by_group_size(pijuan):
    """filter_adata_by_group_size should remove small groups."""
    from pySingleCellNet.utils.adataTools import filter_adata_by_group_size

    ad = pijuan.copy()
    groupby = "ct1" if "ct1" in ad.obs.columns else ad.obs.columns[0]

    # Set threshold so some groups are removed
    group_sizes = ad.obs[groupby].value_counts()
    median_size = int(group_sizes.median())

    filtered = filter_adata_by_group_size(ad, groupby=groupby, ncells=median_size)
    # Should have fewer or equal cells
    assert filtered.n_obs <= ad.n_obs
    # All remaining groups should meet threshold
    remaining_sizes = filtered.obs[groupby].value_counts()
    assert (remaining_sizes >= median_size).all()
