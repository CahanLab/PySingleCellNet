"""Tests for the prediction path: densification cost and score stability.

`_scn_predict` used to densify the entire query (n_cells x n_genes) before
subsetting it to the few hundred genes the classifier actually needs, so peak
memory scaled with the dataset rather than with the classifier.
"""

import numpy as np
import pandas as pd
import pytest
import scanpy as sc
import scipy.sparse as sp


def _lognorm(adata):
    ad = adata.copy()
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)
    return ad


def _pad_with_unused_genes(adata, n_extra):
    """Append `n_extra` all-zero genes the classifier never references."""
    import anndata

    extra = sp.csr_matrix((adata.n_obs, n_extra), dtype=np.float32)
    X = sp.hstack([sp.csr_matrix(adata.X), extra], format="csr")
    var = pd.DataFrame(index=list(adata.var_names) + [f"UNUSED_PAD{i}" for i in range(n_extra)])
    return anndata.AnnData(X=X, obs=adata.obs.copy(), var=var)


@pytest.fixture
def densify_spy(monkeypatch):
    """Record the column count of every sparse matrix that gets densified."""
    from scipy.sparse._compressed import _cs_matrix

    widths = []
    original = _cs_matrix.toarray

    def spy(self, *args, **kwargs):
        widths.append(self.shape[1])
        return original(self, *args, **kwargs)

    monkeypatch.setattr(_cs_matrix, "toarray", spy)
    return widths


def test_only_classifier_genes_are_densified(pbmc_ref, trained_classifier, densify_spy):
    """Densification is bounded by the classifier's gene count, not the query's."""
    from pySingleCellNet.tools.classifier import classify_anndata

    query = _pad_with_unused_genes(_lognorm(pbmc_ref), n_extra=5000)
    n_classifier_genes = len(trained_classifier["tpGeneArray"])
    assert query.n_vars > 10 * n_classifier_genes, "padding should dwarf the classifier gene set"

    classify_anndata(query, trained_classifier, nrand=0)

    assert densify_spy, "expected the query matrix to be densified at least once"
    assert max(densify_spy) <= n_classifier_genes


def test_padding_the_query_does_not_change_scores(pbmc_ref, trained_classifier):
    """Genes the classifier ignores must not perturb the scores."""
    from pySingleCellNet.tools.classifier import classify_anndata

    base = _lognorm(pbmc_ref)
    padded = _pad_with_unused_genes(base, n_extra=500)

    classify_anndata(base, trained_classifier, nrand=0)
    classify_anndata(padded, trained_classifier, nrand=0)

    pd.testing.assert_frame_equal(base.obsm["SCN_score"], padded.obsm["SCN_score"])


def test_shuffled_query_gene_order_does_not_change_scores(pbmc_ref, trained_classifier):
    """Scores depend on gene identity, not on the query's column order."""
    from pySingleCellNet.tools.classifier import classify_anndata

    base = _lognorm(pbmc_ref)
    order = list(base.var_names)
    rng = np.random.default_rng(0)
    rng.shuffle(order)
    shuffled = base[:, order].copy()

    classify_anndata(base, trained_classifier, nrand=0)
    classify_anndata(shuffled, trained_classifier, nrand=0)

    pd.testing.assert_frame_equal(base.obsm["SCN_score"], shuffled.obsm["SCN_score"])


def test_nrand_positive_still_scores_every_cell(pbmc_ref, trained_classifier):
    """The null distribution must not push extra rows into the per-cell result."""
    from pySingleCellNet.tools.classifier import classify_anndata

    query = _lognorm(pbmc_ref)
    classify_anndata(query, trained_classifier, nrand=25)

    assert query.obsm["SCN_score"].shape[0] == query.n_obs
    assert list(query.obsm["SCN_score"].index) == list(query.obs_names)
    assert query.obs["SCN_class_argmax"].notna().all()


def test_nrand_stores_null_distribution_in_uns(pbmc_ref, trained_classifier):
    """Random profiles are kept, but alongside the result rather than inside it."""
    from pySingleCellNet.tools.classifier import classify_anndata

    query = _lognorm(pbmc_ref)
    classify_anndata(query, trained_classifier, nrand=25)

    rand_scores = query.uns["SCN_random_scores"]
    assert rand_scores.shape[0] == 25
    assert list(rand_scores.columns) == list(query.obsm["SCN_score"].columns)


def test_nrand_zero_stores_no_null_distribution(pbmc_ref, trained_classifier):
    """Asking for no random profiles leaves no null-distribution entry behind."""
    from pySingleCellNet.tools.classifier import classify_anndata

    query = _lognorm(pbmc_ref)
    classify_anndata(query, trained_classifier, nrand=0)

    assert "SCN_random_scores" not in query.uns


def test_nrand_does_not_change_real_cell_scores(pbmc_ref, trained_classifier):
    """Adding a null distribution must leave the real cells' scores untouched."""
    from pySingleCellNet.tools.classifier import classify_anndata

    without = _lognorm(pbmc_ref)
    with_rand = _lognorm(pbmc_ref)

    classify_anndata(without, trained_classifier, nrand=0)
    classify_anndata(with_rand, trained_classifier, nrand=25)

    pd.testing.assert_frame_equal(without.obsm["SCN_score"], with_rand.obsm["SCN_score"])


def test_negative_nrand_raises(pbmc_ref, trained_classifier):
    """A negative count is a caller error, not something to silently ignore."""
    from pySingleCellNet.tools.classifier import classify_anndata

    query = _lognorm(pbmc_ref)
    with pytest.raises(ValueError, match="nrand"):
        classify_anndata(query, trained_classifier, nrand=-1)


def test_dense_query_still_classifies(pbmc_ref, trained_classifier):
    """A query whose .X is a plain ndarray must classify like a sparse one."""
    from pySingleCellNet.tools.classifier import classify_anndata

    sparse_query = _lognorm(pbmc_ref)
    dense_query = sparse_query.copy()
    dense_query.X = np.asarray(sparse_query.X.todense())

    classify_anndata(sparse_query, trained_classifier, nrand=0)
    classify_anndata(dense_query, trained_classifier, nrand=0)

    pd.testing.assert_frame_equal(
        sparse_query.obsm["SCN_score"], dense_query.obsm["SCN_score"]
    )
