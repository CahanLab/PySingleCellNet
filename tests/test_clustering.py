"""Tests for the cluster_alot pipeline."""

import numpy as np
import pandas as pd
import pytest
import scanpy as sc


@pytest.fixture(scope="module")
def clustered_pijuan(pijuan):
    """Prepare and run cluster_alot on the Pijuan dataset."""
    from pySingleCellNet.tools.cluster import cluster_alot

    ad = pijuan.copy()
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)
    sc.pp.highly_variable_genes(ad, n_top_genes=2000, flavor="seurat_v3")
    sc.pp.pca(ad, n_comps=30, mask_var="highly_variable")

    runs_df = cluster_alot(
        ad,
        leiden_resolutions=[0.25],
        pca_params={"top_n_pcs": [20]},
        knn_params={"n_neighbors": [10]},
        verbose=False,
    )
    return ad, runs_df


def test_cluster_alot_returns_dataframe(clustered_pijuan):
    """cluster_alot should return a DataFrame with expected columns."""
    _, runs_df = clustered_pijuan
    assert isinstance(runs_df, pd.DataFrame)
    assert "obs_key" in runs_df.columns
    assert "n_clusters" in runs_df.columns
    assert len(runs_df) >= 1


def test_cluster_alot_creates_obs_columns(clustered_pijuan):
    """cluster_alot should create obs columns in adata."""
    ad, runs_df = clustered_pijuan
    for obs_key in runs_df["obs_key"]:
        assert obs_key in ad.obs.columns


def test_cluster_alot_n_clusters(clustered_pijuan):
    """Each run should produce at least 1 cluster."""
    _, runs_df = clustered_pijuan
    for _, row in runs_df.iterrows():
        if row["status"] == "ok":
            assert row["n_clusters"] >= 1


def test_cluster_subclusters(pijuan):
    """cluster_subclusters should create a subcluster column."""
    from pySingleCellNet.tools.cluster import cluster_subclusters

    ad = pijuan.copy()
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)
    sc.pp.highly_variable_genes(ad, n_top_genes=2000, flavor="seurat_v3")

    # First create a basic leiden clustering
    sc.pp.pca(ad, n_comps=20, mask_var="highly_variable")
    sc.pp.neighbors(ad, n_neighbors=10)
    sc.tl.leiden(ad, resolution=0.25, flavor="igraph", n_iterations=2)

    # Add lognorm layer for HVG detection in subclustering
    ad.layers["lognorm"] = ad.X.copy()

    clusters = ad.obs["leiden"].unique().tolist()[:1]  # subcluster just one for speed
    cluster_subclusters(
        ad,
        cluster_column="leiden",
        to_subcluster=clusters,
        layer="lognorm",
        hvg_flavor="seurat_v3",
        n_hvg=500,
        n_pcs=10,
        n_neighbors=5,
        leiden_resolution=0.1,
    )

    assert "subcluster" in ad.obs.columns
    # Subclustered labels should contain underscore pattern
    subclustered_vals = ad.obs.loc[ad.obs["original_cluster"].isin(clusters), "subcluster"]
    assert any("_" in str(v) for v in subclustered_vals)
