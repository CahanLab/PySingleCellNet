"""Tests for the quickstart classifier pipeline."""

import numpy as np
import pandas as pd
import pytest
import scanpy as sc


def test_limit_anndata_to_common_genes(pbmc_ref, pbmc_query):
    """After limiting to common genes, var_names should match."""
    from pySingleCellNet.utils.adataTools import limit_anndata_to_common_genes

    ref = pbmc_ref.copy()
    query = pbmc_query.copy()
    limit_anndata_to_common_genes([ref, query])
    assert set(ref.var_names) == set(query.var_names)
    assert len(ref.var_names) > 0


def test_split_adata_indices(pbmc_ref):
    """Split should have no overlap and full coverage."""
    from pySingleCellNet.utils.adataTools import split_adata_indices

    ad = pbmc_ref.copy()
    train_ids, val_ids = split_adata_indices(ad, n_cells=20, groupby="cell_type")

    train_set = set(train_ids)
    val_set = set(val_ids)
    # No overlap
    assert len(train_set & val_set) == 0
    # Full coverage
    assert train_set | val_set == set(ad.obs_names)


def test_train_classifier(trained_classifier):
    """Classifier dict should have expected keys."""
    clf = trained_classifier
    expected_keys = {"tpGeneArray", "topPairs", "classifier", "diffExpGenes", "ctColors", "argList"}
    assert expected_keys == set(clf.keys())
    # Classifier should be a RandomForest
    from sklearn.ensemble import RandomForestClassifier
    assert isinstance(clf["classifier"], RandomForestClassifier)


def test_classify_anndata(classified_query):
    """Classification should add SCN_score to obsm and SCN_class_argmax to obs."""
    query = classified_query
    assert "SCN_score" in query.obsm
    assert "SCN_class_argmax" in query.obs.columns

    scores = query.obsm["SCN_score"]
    assert isinstance(scores, pd.DataFrame)
    assert scores.shape[0] == query.n_obs
    assert scores.shape[1] > 0


def test_create_classifier_report(pbmc_ref, trained_classifier):
    """Report should be a DataFrame with Precision/Recall/F1 in [0,1]."""
    from pySingleCellNet.tools.classifier import classify_anndata, create_classifier_report
    from pySingleCellNet.utils.adataTools import split_adata_indices

    ad = pbmc_ref.copy()
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)
    sc.pp.highly_variable_genes(ad, n_top_genes=2000, flavor="seurat_v3")

    train_ids, val_ids = split_adata_indices(ad, n_cells=20, groupby="cell_type")
    ad_val = ad[val_ids].copy()
    classify_anndata(ad_val, trained_classifier, nrand=0)

    report = create_classifier_report(ad_val, ground_truth="cell_type", prediction="SCN_class_argmax")
    assert isinstance(report, pd.DataFrame)
    assert "Precision" in report.columns
    assert "Recall" in report.columns
    assert "F1-Score" in report.columns
    # Metrics should be in [0,1] for non-average rows
    non_avg = report[~report["Label"].isin(["micro avg", "macro avg", "weighted avg", "accuracy"])]
    assert (non_avg["Precision"].dropna() >= 0).all()
    assert (non_avg["Precision"].dropna() <= 1).all()
