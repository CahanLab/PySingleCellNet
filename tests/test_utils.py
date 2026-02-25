"""Tests for utility functions."""

import numpy as np
import pandas as pd
import pytest
import scanpy as sc


def test_rename_cluster_labels(pijuan):
    """rename_cluster_labels should create a new column with short labels."""
    from pySingleCellNet.utils.adataTools import rename_cluster_labels

    ad = pijuan.copy()
    groupby = "ct1" if "ct1" in ad.obs.columns else ad.obs.columns[0]
    rename_cluster_labels(ad, old_col=groupby, new_col="short_label")
    assert "short_label" in ad.obs.columns
    assert ad.obs["short_label"].dtype.name == "category"


def test_generate_joint_graph(pijuan):
    """generate_joint_graph should create obsp keys."""
    from pySingleCellNet.utils.knn import generate_joint_graph

    ad = pijuan.copy()
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)
    sc.pp.highly_variable_genes(ad, n_top_genes=2000, flavor="seurat_v3")
    sc.pp.pca(ad, n_comps=20, mask_var="highly_variable")

    # Build two neighbor graphs with different params
    sc.pp.neighbors(ad, n_neighbors=10, key_added="nn10")
    sc.pp.neighbors(ad, n_neighbors=5, key_added="nn5")

    generate_joint_graph(
        ad,
        connectivity_keys=["nn10_connectivities", "nn5_connectivities"],
        weights=[0.5, 0.5],
        output_key="joint",
    )

    assert "joint_connectivities" in ad.obsp
    assert "joint_distances" in ad.obsp
    assert "joint" in ad.uns
