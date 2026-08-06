"""Tests for gene-pair encoding.

Top-scoring pairs used to be identified only by the string ``f"{a}_{b}"`` and
recovered by splitting on ``_``, so any gene symbol containing an underscore
decoded into fragments that are not genes. Spatial and module-style datasets
name genes that way routinely.
"""

import pickle

import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from scipy.sparse import csr_matrix


def _underscore_adata(n_cells=120, n_genes=200, seed=0):
    """A minimal trainable dataset whose every gene symbol contains an underscore."""
    import anndata

    rng = np.random.default_rng(seed)
    ad = anndata.AnnData(X=csr_matrix(rng.poisson(3, (n_cells, n_genes)).astype(np.float32)))
    ad.var_names = [f"mod_{i}" for i in range(n_genes)]
    ad.obs["celltype"] = pd.Categorical(["a"] * (n_cells // 2) + ["b"] * (n_cells - n_cells // 2))

    ad.layers["counts"] = ad.X.copy()
    sc.pp.normalize_total(ad)
    sc.pp.log1p(ad)
    sc.pp.highly_variable_genes(ad, n_top_genes=150, flavor="seurat_v3", layer="counts")
    return ad


def test_train_classifier_handles_underscore_gene_symbols():
    """Training on genes named `mod_0`, `mod_1`, ... must not fail."""
    from pySingleCellNet.tools.classifier import train_classifier

    ad = _underscore_adata()
    clf = train_classifier(ad, groupby="celltype", n_trees=50, n_comps=10)

    assert len(clf["topPairs"]) > 0
    # Every recorded pair member must be a real gene, not a decoded fragment.
    assert set(np.asarray(clf["topPairGenes"]).ravel()) <= set(ad.var_names)


def test_classify_roundtrips_with_underscore_gene_symbols():
    """A classifier trained on underscore genes must also predict with them."""
    from pySingleCellNet.tools.classifier import classify_anndata, train_classifier

    ad = _underscore_adata()
    clf = train_classifier(ad, groupby="celltype", n_trees=50, n_comps=10)

    query = _underscore_adata(seed=1)
    classify_anndata(query, clf, nrand=0)

    assert query.obsm["SCN_score"].shape[0] == query.n_obs
    assert query.uns["SCN_gene_overlap"]["n_missing"] == 0


def test_pair_genes_match_their_display_labels():
    """The structural pairs and the human-readable labels describe the same pairs."""
    from pySingleCellNet.tools.classifier import train_classifier

    ad = _underscore_adata()
    clf = train_classifier(ad, groupby="celltype", n_trees=50, n_comps=10)

    pair_genes = np.asarray(clf["topPairGenes"])
    assert pair_genes.shape == (len(clf["topPairs"]), 2)
    for (gene_a, gene_b), label in zip(pair_genes, clf["topPairs"]):
        assert label == f"{gene_a}_{gene_b}"


def test_pair_members_are_never_equal():
    """A pair of a gene with itself carries no signal and must not be emitted."""
    from pySingleCellNet.tools.classifier import train_classifier

    ad = _underscore_adata()
    clf = train_classifier(ad, groupby="celltype", n_trees=50, n_comps=10)

    for gene_a, gene_b in np.asarray(clf["topPairGenes"]):
        assert gene_a != gene_b


def test_query_transform_uses_structural_pairs_not_string_splitting():
    """Pair features are computed from the named genes, whatever the label looks like."""
    from pySingleCellNet.tools.classifier import _query_transform

    exp = pd.DataFrame(
        {"a_1": [5.0, 0.0], "b_2": [1.0, 4.0]},
        index=["cell1", "cell2"],
    )
    pair_genes = np.array([["a_1", "b_2"]])

    result = _query_transform(exp, pair_genes)

    assert list(result.columns) == ["a_1_b_2"]
    # cell1 has a_1 > b_2 (5 > 1); cell2 does not (0 > 4 is False).
    assert result.iloc[0, 0] == 1
    assert result.iloc[1, 0] == 0


def test_legacy_classifier_without_pair_genes_still_predicts(tmp_path, pbmc_ref, trained_classifier):
    """Classifiers pickled before the format change keep working via label splitting."""
    from pySingleCellNet.tools.classifier import classify_anndata
    from pySingleCellNet.utils import load_classifier

    legacy = {k: v for k, v in trained_classifier.items() if k not in ("topPairGenes", "format_version")}
    path = tmp_path / "legacy.pkl"
    with open(path, "wb") as handle:
        pickle.dump(legacy, handle)

    reloaded = load_classifier(path)

    expected = pbmc_ref.copy()
    sc.pp.normalize_total(expected, target_sum=1e4)
    sc.pp.log1p(expected)
    actual = expected.copy()

    classify_anndata(expected, trained_classifier, nrand=0)
    classify_anndata(actual, reloaded, nrand=0)

    pd.testing.assert_frame_equal(expected.obsm["SCN_score"], actual.obsm["SCN_score"])


def test_legacy_classifier_with_underscore_genes_is_rejected(tmp_path):
    """A pre-format-2 classifier whose genes contain underscores cannot be decoded."""
    from pySingleCellNet.utils import load_classifier
    from pySingleCellNet.tools.classifier import train_classifier

    ad = _underscore_adata()
    clf = train_classifier(ad, groupby="celltype", n_trees=50, n_comps=10)

    # Strip the structural pairs, leaving only the ambiguous joined labels.
    legacy = {k: v for k, v in clf.items() if k not in ("topPairGenes", "format_version")}
    path = tmp_path / "legacy_underscore.pkl"
    with open(path, "wb") as handle:
        pickle.dump(legacy, handle)

    with pytest.raises(ValueError, match="underscore"):
        load_classifier(path)
