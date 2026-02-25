"""Session-scoped fixtures for PySingleCellNet tests.

Loads small test data once per session and trains classifier once.
Run `python tests/generate_test_data.py` first to create the fixtures.
"""

import os
import pytest
import scanpy as sc
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def _fixture_path(name):
    return os.path.join(DATA_DIR, name)


@pytest.fixture(scope="session")
def pbmc_ref():
    """Load the small PBMC reference dataset."""
    path = _fixture_path("pbmc_ref_small.h5ad")
    if not os.path.exists(path):
        pytest.skip(f"Test data not found: {path}. Run `python tests/generate_test_data.py` first.")
    return sc.read_h5ad(path)


@pytest.fixture(scope="session")
def pbmc_query():
    """Load the small PBMC query dataset."""
    path = _fixture_path("pbmc_query_small.h5ad")
    if not os.path.exists(path):
        pytest.skip(f"Test data not found: {path}. Run `python tests/generate_test_data.py` first.")
    return sc.read_h5ad(path)


@pytest.fixture(scope="session")
def pijuan():
    """Load the small Pijuan dataset."""
    path = _fixture_path("pijuan_small_test.h5ad")
    if not os.path.exists(path):
        pytest.skip(f"Test data not found: {path}. Run `python tests/generate_test_data.py` first.")
    return sc.read_h5ad(path)


@pytest.fixture(scope="session")
def trained_classifier(pbmc_ref):
    """Train a SCN classifier on the PBMC reference data (once per session)."""
    from pySingleCellNet.tools.classifier import train_classifier

    ad = pbmc_ref.copy()
    # Ensure lognorm and HVG are set
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)
    sc.pp.highly_variable_genes(ad, n_top_genes=2000, flavor="seurat_v3")

    clf = train_classifier(ad, groupby="cell_type", n_top_genes=15, n_top_gene_pairs=20, n_trees=500)
    return clf


@pytest.fixture(scope="session")
def classified_query(pbmc_ref, pbmc_query, trained_classifier):
    """Classify the query dataset using the trained classifier."""
    from pySingleCellNet.utils.adataTools import limit_anndata_to_common_genes
    from pySingleCellNet.tools.classifier import classify_anndata

    ref = pbmc_ref.copy()
    query = pbmc_query.copy()

    # Normalize query
    sc.pp.normalize_total(query, target_sum=1e4)
    sc.pp.log1p(query)

    # Limit to common genes
    limit_anndata_to_common_genes([ref, query])

    # Classify
    classify_anndata(query, trained_classifier, nrand=0)
    return query
