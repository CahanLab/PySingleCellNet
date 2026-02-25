"""Smoke tests for plotting functions (no-error checks with Agg backend)."""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
import scanpy as sc


def test_heatmap_scores(classified_query):
    """heatmap_scores should run without raising."""
    from pySingleCellNet.plotting.heatmap import heatmap_scores

    ad = classified_query.copy()
    # heatmap_scores requires a groupby column
    heatmap_scores(ad, groupby="SCN_class_argmax")


def test_heatmap_classifier_report(pbmc_ref, trained_classifier):
    """heatmap_classifier_report should run without raising."""
    from pySingleCellNet.tools.classifier import classify_anndata, create_classifier_report
    from pySingleCellNet.plotting.heatmap import heatmap_classifier_report

    ad = pbmc_ref.copy()
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)
    sc.pp.highly_variable_genes(ad, n_top_genes=2000, flavor="seurat_v3")
    classify_anndata(ad, trained_classifier, nrand=0)

    report = create_classifier_report(ad, ground_truth="cell_type", prediction="SCN_class_argmax")
    heatmap_classifier_report(report)


def test_bar_classifier_f1(classified_query):
    """bar_classifier_f1 should run without raising when ground truth is available."""
    from pySingleCellNet.plotting.bar import bar_classifier_f1

    ad = classified_query.copy()
    # Only run if cell_type column exists for ground truth
    if "cell_type" not in ad.obs.columns:
        pytest.skip("No cell_type column in classified query")

    ad.obs["cell_type"] = ad.obs["cell_type"].astype("category")
    bar_classifier_f1(ad, ground_truth="cell_type", class_prediction="SCN_class_argmax")
