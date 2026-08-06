"""Tests that training the same classifier twice gives the same classifier.

Two independent sources of drift: gene pairs were drawn from the unseeded
`random` module, and the candidate gene lists were de-duplicated through
`set()`, whose iteration order for strings varies with PYTHONHASHSEED. The
second means results differed between processes even when the RNG was seeded.
"""

import hashlib
import os
import subprocess
import sys
import textwrap

import pytest
import scanpy as sc


def _lognorm_hvg(adata):
    ad = adata.copy()
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)
    sc.pp.highly_variable_genes(ad, n_top_genes=2000, flavor="seurat_v3")
    return ad


def _train(ad, **kwargs):
    from pySingleCellNet.tools.classifier import train_classifier

    params = dict(
        groupby="cell_type",
        n_top_genes=15,
        n_top_gene_pairs=20,
        n_trees=20,
        build_graph=False,
    )
    params.update(kwargs)
    return train_classifier(ad, **params)


def test_same_random_state_gives_the_same_gene_pairs(pbmc_ref):
    """Two trainings with the same seed must select the same pairs."""
    ad = _lognorm_hvg(pbmc_ref)

    first = _train(ad, random_state=0)
    second = _train(ad, random_state=0)

    assert list(first["topPairs"]) == list(second["topPairs"])


def test_same_random_state_gives_the_same_forest(pbmc_ref):
    """Reproducibility must reach the fitted estimator, not just the features."""
    ad = _lognorm_hvg(pbmc_ref)

    first = _train(ad, random_state=0)
    second = _train(ad, random_state=0)

    tree_a = first["classifier"].estimators_[0].tree_
    tree_b = second["classifier"].estimators_[0].tree_
    assert tree_a.threshold.tolist() == tree_b.threshold.tolist()


def test_different_random_states_give_different_gene_pairs(pbmc_ref):
    """The seed has to actually steer selection, otherwise it is decoration."""
    ad = _lognorm_hvg(pbmc_ref)

    first = _train(ad, random_state=0)
    second = _train(ad, random_state=12345)

    assert list(first["topPairs"]) != list(second["topPairs"])


def test_random_state_none_still_trains(pbmc_ref):
    """Omitting the seed keeps the previous unseeded behaviour."""
    ad = _lognorm_hvg(pbmc_ref)

    clf = _train(ad)
    assert len(clf["topPairs"]) > 0


_SUBPROCESS_SCRIPT = textwrap.dedent(
    """
    import sys, hashlib, warnings
    warnings.filterwarnings("ignore")
    import scanpy as sc
    import pySingleCellNet as cn

    ad = sc.read_h5ad(sys.argv[1])
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)
    sc.pp.highly_variable_genes(ad, n_top_genes=2000, flavor="seurat_v3")
    clf = cn.tl.train_classifier(
        ad, groupby="cell_type", n_top_genes=15, n_top_gene_pairs=20,
        n_trees=10, build_graph=False, random_state=0,
    )
    digest = hashlib.md5(repr(list(clf["topPairs"])).encode()).hexdigest()
    print("PAIRHASH:" + digest)
    """
)


def _train_in_subprocess(fixture_path, hash_seed):
    env = dict(os.environ, PYTHONHASHSEED=str(hash_seed))
    result = subprocess.run(
        [sys.executable, "-c", _SUBPROCESS_SCRIPT, fixture_path],
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    lines = [ln for ln in result.stdout.splitlines() if ln.startswith("PAIRHASH:")]
    assert lines, f"no hash in output:\n{result.stdout[-2000:]}"
    return lines[0]


def test_gene_pairs_do_not_depend_on_pythonhashseed(pbmc_ref):
    """Seeded training must be reproducible across processes, not just within one.

    De-duplicating gene lists through `set()` made the candidate ordering depend
    on string hash randomisation, so a seeded run still differed run to run.
    """
    path = os.path.join(os.path.dirname(__file__), "data", "pbmc_ref_small.h5ad")
    if not os.path.exists(path):
        pytest.skip(f"Test data not found: {path}")

    assert _train_in_subprocess(path, 1) == _train_in_subprocess(path, 2)
