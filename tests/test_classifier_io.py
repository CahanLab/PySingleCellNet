"""Tests for classifier persistence.

Trained classifiers are normally shared as pickled dicts. Without a save/load
pair there is nowhere to record a format version and nowhere to validate a
loaded object, so a malformed one surfaces as an AttributeError from deep
inside prediction instead of a clear message at load time.
"""

import pickle

import pandas as pd
import pytest
import scanpy as sc
from sklearn.ensemble import RandomForestClassifier


def _lognorm(adata):
    ad = adata.copy()
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)
    return ad


def test_round_trip_preserves_classification(tmp_path, pbmc_ref, trained_classifier):
    """A saved-then-loaded classifier scores identically to the original."""
    from pySingleCellNet.tools.classifier import classify_anndata
    from pySingleCellNet.utils import load_classifier, save_classifier

    path = tmp_path / "clf.pkl"
    save_classifier(trained_classifier, path)
    reloaded = load_classifier(path)

    before = _lognorm(pbmc_ref)
    after = _lognorm(pbmc_ref)
    classify_anndata(before, trained_classifier, nrand=0)
    classify_anndata(after, reloaded, nrand=0)

    pd.testing.assert_frame_equal(before.obsm["SCN_score"], after.obsm["SCN_score"])


def test_saved_classifier_carries_a_format_version(tmp_path, trained_classifier):
    """The on-disk payload records its format so future readers can migrate it."""
    from pySingleCellNet.utils import save_classifier
    from pySingleCellNet.utils.classifier_io import SCN_CLASSIFIER_FORMAT_VERSION

    path = tmp_path / "clf.pkl"
    save_classifier(trained_classifier, path)

    with open(path, "rb") as handle:
        payload = pickle.load(handle)

    assert payload["format_version"] == SCN_CLASSIFIER_FORMAT_VERSION


def test_saving_does_not_mutate_the_classifier(tmp_path, trained_classifier):
    """Persisting a classifier must leave the in-memory dict untouched."""
    from pySingleCellNet.utils import save_classifier

    before = set(trained_classifier.keys())
    save_classifier(trained_classifier, tmp_path / "clf.pkl")
    assert set(trained_classifier.keys()) == before


def test_accepts_string_paths(tmp_path, trained_classifier):
    """Paths may be plain strings, not just Path objects."""
    from pySingleCellNet.utils import load_classifier, save_classifier

    path = str(tmp_path / "clf.pkl")
    save_classifier(trained_classifier, path)
    assert load_classifier(path)["tpGeneArray"] is not None


def test_legacy_payload_without_version_is_migrated(tmp_path, trained_classifier):
    """A hand-pickled dict from before versioning still loads."""
    from pySingleCellNet.utils import load_classifier
    from pySingleCellNet.utils.classifier_io import SCN_CLASSIFIER_FORMAT_VERSION

    legacy = {k: v for k, v in trained_classifier.items() if k != "format_version"}
    path = tmp_path / "legacy.pkl"
    with open(path, "wb") as handle:
        pickle.dump(legacy, handle)

    loaded = load_classifier(path)
    assert loaded["format_version"] == SCN_CLASSIFIER_FORMAT_VERSION
    assert set(loaded["tpGeneArray"]) == set(trained_classifier["tpGeneArray"])


def test_missing_required_key_is_reported_by_name(tmp_path, trained_classifier):
    """A truncated payload fails at load with the offending key named."""
    from pySingleCellNet.utils import load_classifier

    broken = {k: v for k, v in trained_classifier.items() if k != "topPairs"}
    path = tmp_path / "broken.pkl"
    with open(path, "wb") as handle:
        pickle.dump(broken, handle)

    with pytest.raises(ValueError, match="topPairs"):
        load_classifier(path)


def test_unfitted_estimator_is_rejected(tmp_path, trained_classifier):
    """An unfitted forest fails at load, not midway through prediction."""
    from pySingleCellNet.utils import load_classifier

    payload = dict(trained_classifier)
    payload["classifier"] = RandomForestClassifier()
    path = tmp_path / "unfitted.pkl"
    with open(path, "wb") as handle:
        pickle.dump(payload, handle)

    with pytest.raises(ValueError, match="fitted"):
        load_classifier(path)


def test_non_dict_payload_is_rejected(tmp_path):
    """A pickle of something else entirely fails clearly."""
    from pySingleCellNet.utils import load_classifier

    path = tmp_path / "notaclassifier.pkl"
    with open(path, "wb") as handle:
        pickle.dump(["not", "a", "classifier"], handle)

    with pytest.raises(ValueError, match="dict"):
        load_classifier(path)


def test_missing_file_raises_filenotfound(tmp_path):
    """A path that does not exist reports itself as such."""
    from pySingleCellNet.utils import load_classifier

    with pytest.raises(FileNotFoundError):
        load_classifier(tmp_path / "nope.pkl")
