"""Tests for the cell-type relatedness graph travelling with the classifier.

`categorize_classification` splits multi-class cells into Intermediate (the
classes are close in a cell-type graph) and Hybrid (they are not). The graph is
built from the reference, so anyone who receives only a trained classifier
could not reproduce that distinction at all.
"""

import pickle

import igraph as ig
import pytest


def test_train_classifier_ships_a_cell_type_graph(trained_classifier):
    """Training records a graph whose vertices are the training cell types."""
    graph = trained_classifier["ctGraph"]

    assert isinstance(graph, ig.Graph)
    assert "name" in graph.vs.attributes()

    trained_types = {ct for ct in trained_classifier["ctColors"] if ct != "rand"}
    assert set(graph.vs["name"]) == trained_types


def test_categorize_uses_the_classifier_graph(classified_query, trained_classifier):
    """Categorization works from the classifier alone, with no graph argument."""
    from pySingleCellNet.tools.categorize import categorize_classification, comp_ct_thresh

    ad = classified_query.copy()
    thresholds = comp_ct_thresh(ad)
    categorize_classification(ad, thresholds=thresholds, clf=trained_classifier)

    assert "SCN_class_type" in ad.obs.columns
    valid = {"Singular", "None", "Intermediate", "Hybrid", "Rand"}
    assert set(ad.obs["SCN_class_type"].unique()).issubset(valid)


def test_explicit_graph_takes_precedence_over_the_classifier(classified_query, trained_classifier):
    """An explicitly supplied graph still wins, preserving the old call style."""
    from pySingleCellNet.tools.categorize import categorize_classification, comp_ct_thresh

    ad = classified_query.copy()
    thresholds = comp_ct_thresh(ad)

    # A graph with no edges forces every multi-class cell to Hybrid.
    names = list(trained_classifier["ctGraph"].vs["name"])
    empty = ig.Graph(n=len(names), edges=[], directed=False)
    empty.vs["name"] = names

    categorize_classification(ad, thresholds=thresholds, graph=empty, clf=trained_classifier)
    assert "Intermediate" not in set(ad.obs["SCN_class_type"].unique())


def _count_distance_calls(adata, thresholds, graph):
    """Run categorization, returning how many times the graph was queried."""
    from pySingleCellNet.tools.categorize import categorize_classification

    calls = []
    original = graph.distances

    def spy(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    graph.distances = spy
    try:
        categorize_classification(adata, thresholds=thresholds, graph=graph)
    finally:
        del graph.distances
    return len(calls)


def test_distance_lookups_do_not_scale_with_cell_count(classified_query, trained_classifier):
    """Shortest paths depend only on the graph, so they must not be recomputed per cell."""
    from pySingleCellNet.tools.categorize import comp_ct_thresh

    graph = trained_classifier["ctGraph"]

    small = classified_query[: classified_query.n_obs // 4].copy()
    full = classified_query.copy()
    assert full.n_obs > small.n_obs * 2, "need a meaningful size difference"

    small_calls = _count_distance_calls(small, comp_ct_thresh(small), graph)
    full_calls = _count_distance_calls(full, comp_ct_thresh(full), graph)

    assert small_calls == full_calls


def test_distances_are_computed_at_most_once(classified_query, trained_classifier):
    """The whole distance matrix is small; one computation per call is enough."""
    from pySingleCellNet.tools.categorize import comp_ct_thresh

    ad = classified_query.copy()
    calls = _count_distance_calls(ad, comp_ct_thresh(ad), trained_classifier["ctGraph"])
    assert calls <= 1


def test_categorize_without_graph_or_classifier_raises(classified_query):
    """Neither source of a graph means a clear error, not a silent guess."""
    from pySingleCellNet.tools.categorize import categorize_classification, comp_ct_thresh

    ad = classified_query.copy()
    thresholds = comp_ct_thresh(ad)

    with pytest.raises(ValueError, match="graph"):
        categorize_classification(ad, thresholds=thresholds)


def test_graph_survives_save_and_load(tmp_path, trained_classifier):
    """The graph must survive persistence -- that is the point of shipping it."""
    from pySingleCellNet.utils import load_classifier, save_classifier

    path = tmp_path / "clf.pkl"
    save_classifier(trained_classifier, path)
    reloaded = load_classifier(path)

    original = trained_classifier["ctGraph"]
    restored = reloaded["ctGraph"]
    assert set(restored.vs["name"]) == set(original.vs["name"])
    assert restored.ecount() == original.ecount()


def test_classifier_without_graph_reports_the_gap(classified_query, trained_classifier, tmp_path):
    """A classifier predating ctGraph names the problem instead of failing obscurely."""
    from pySingleCellNet.tools.categorize import categorize_classification, comp_ct_thresh
    from pySingleCellNet.utils import load_classifier

    legacy = {k: v for k, v in trained_classifier.items() if k not in ("ctGraph", "format_version")}
    path = tmp_path / "legacy.pkl"
    with open(path, "wb") as handle:
        pickle.dump(legacy, handle)
    reloaded = load_classifier(path)

    ad = classified_query.copy()
    thresholds = comp_ct_thresh(ad)

    with pytest.raises(ValueError, match="ctGraph"):
        categorize_classification(ad, thresholds=thresholds, clf=reloaded)
