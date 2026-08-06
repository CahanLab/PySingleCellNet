"""Tests for query/classifier gene-overlap reporting.

A classifier gene absent from the query is reindexed to a column of zeros, so
every pair feature involving it collapses to a constant. The forest still
returns confident-looking scores, so the failure is invisible without an
explicit overlap check.
"""

import pytest
import scanpy as sc


def _lognorm(adata):
    ad = adata.copy()
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)
    return ad


def _drop_classifier_genes(query, clf, fraction):
    """Remove `fraction` of the classifier's genes from `query`.

    Returns the trimmed query and the full set of classifier genes it then
    lacks -- which includes any the query was already missing.
    """
    present = set(query.var_names)
    clf_genes = [g for g in clf["tpGeneArray"] if g in present]
    dropped = set(clf_genes[: int(len(clf_genes) * fraction)])
    keep = [g for g in query.var_names if g not in dropped]
    already_absent = {g for g in clf["tpGeneArray"] if g not in present}
    return query[:, keep].copy(), dropped | already_absent


def test_full_overlap_records_no_missing_genes(pbmc_ref, trained_classifier):
    """A query carrying every classifier gene reports 100% overlap and no warning."""
    from pySingleCellNet.tools.classifier import classify_anndata

    # The classifier's genes were selected from the reference, so overlap is total.
    query = _lognorm(pbmc_ref)
    missing = set(trained_classifier["tpGeneArray"]) - set(query.var_names)
    assert not missing, "classifier genes are drawn from the reference"

    classify_anndata(query, trained_classifier, nrand=0)

    overlap = query.uns["SCN_gene_overlap"]
    assert overlap["n_missing"] == 0
    assert overlap["fraction"] == 1.0
    assert overlap["missing_genes"] == []


def test_partial_overlap_warns_and_lists_missing_genes(pbmc_query, trained_classifier):
    """Dropping some classifier genes warns and records exactly which are absent."""
    from pySingleCellNet.tools.classifier import classify_anndata

    query, dropped = _drop_classifier_genes(_lognorm(pbmc_query), trained_classifier, 0.5)

    with pytest.warns(UserWarning, match="classifier genes"):
        classify_anndata(query, trained_classifier, nrand=0)

    overlap = query.uns["SCN_gene_overlap"]
    assert overlap["n_missing"] == len(dropped)
    assert set(overlap["missing_genes"]) == dropped
    assert overlap["n_required"] == len(trained_classifier["tpGeneArray"])
    assert 0 < overlap["fraction"] < 1


def test_zero_overlap_raises(pbmc_query, trained_classifier):
    """A query sharing no genes with the classifier is refused, not scored."""
    from pySingleCellNet.tools.classifier import classify_anndata

    query = _lognorm(pbmc_query)
    # Simulate the species/case-convention mismatch: same data, unrecognisable symbols.
    query.var_names = [f"{g}-notagene" for g in query.var_names]

    with pytest.raises(ValueError, match="overlap"):
        classify_anndata(query, trained_classifier, nrand=0)


def test_min_gene_overlap_threshold_is_configurable(pbmc_query, trained_classifier):
    """Callers can demand a higher overlap floor than the default."""
    from pySingleCellNet.tools.classifier import classify_anndata

    query, _ = _drop_classifier_genes(_lognorm(pbmc_query), trained_classifier, 0.5)

    with pytest.raises(ValueError, match="overlap"):
        classify_anndata(query, trained_classifier, nrand=0, min_gene_overlap=0.9)


def test_train_classifier_restores_warning_filters(pbmc_ref):
    """Training must not leave a global warning filter that mutes later overlap warnings."""
    import warnings

    from pySingleCellNet.tools.classifier import train_classifier

    ad = _lognorm(pbmc_ref)
    sc.pp.highly_variable_genes(ad, n_top_genes=2000, flavor="seurat_v3")

    before = list(warnings.filters)
    train_classifier(ad, groupby="cell_type", n_top_genes=5, n_top_gene_pairs=5, n_trees=10)
    assert list(warnings.filters) == before


def test_min_gene_overlap_zero_disables_the_floor(pbmc_query, trained_classifier):
    """Setting the floor to 0 keeps the warning but allows the run to proceed."""
    from pySingleCellNet.tools.classifier import classify_anndata

    query = _lognorm(pbmc_query)
    query.var_names = [f"{g}-notagene" for g in query.var_names]

    with pytest.warns(UserWarning, match="classifier genes"):
        classify_anndata(query, trained_classifier, nrand=0, min_gene_overlap=0.0)

    assert query.uns["SCN_gene_overlap"]["fraction"] == 0.0
    assert "SCN_score" in query.obsm
