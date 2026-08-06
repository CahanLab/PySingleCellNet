"""Persistence for trained SCN classifiers.

A trained classifier is a plain dict, and the usual way to share one is to
pickle it. Routing that through `save_classifier` / `load_classifier` gives the
payload a format version -- so the on-disk layout can change without silently
mispredicting -- and gives loading a single place to validate, so a malformed
classifier is reported at load time rather than as an obscure error from inside
prediction.
"""

import os
import pickle
import warnings

import numpy as np

#: Current on-disk layout for a pickled classifier.
#:
#: 1. The original layout: gene pairs stored only as ``"geneA_geneB"`` strings.
#: 2. Adds ``topPairGenes``, an (n_pairs, 2) array of the pair members, so gene
#:    symbols containing underscores survive the round trip.
SCN_CLASSIFIER_FORMAT_VERSION = 2

#: Keys a payload must carry to be usable for prediction.
REQUIRED_CLASSIFIER_KEYS = ("tpGeneArray", "topPairs", "classifier")

#: Separator used to build a human-readable label for a gene pair. It is only a
#: display convention -- pair membership is carried structurally by
#: ``topPairGenes`` -- but format 1 classifiers depend on it being unambiguous.
PAIR_LABEL_SEP = "_"


def _pair_labels(pair_genes) -> np.ndarray:
    """Build display labels for an (n_pairs, 2) array of gene pairs.

    Args:
        pair_genes: Array-like of shape (n_pairs, 2).

    Returns:
        np.ndarray: 1-D array of ``"geneA_geneB"`` strings.
    """
    arr = np.asarray(pair_genes, dtype=object)
    if arr.size == 0:
        return np.empty(0, dtype=object)
    return np.array(
        [f"{a}{PAIR_LABEL_SEP}{b}" for a, b in arr],
        dtype=object,
    )


def _split_pair_labels(labels, genes=None) -> np.ndarray:
    """Recover pair members from format 1 ``"geneA_geneB"`` labels.

    Args:
        labels: Iterable of joined pair labels.
        genes: Optional iterable of the classifier's gene symbols. When any of
            them contains the separator the labels are ambiguous and cannot be
            decoded, so this raises instead of returning fragments.

    Returns:
        np.ndarray: (n_pairs, 2) object array of gene symbols.

    Raises:
        ValueError: If the labels cannot be decoded unambiguously.
    """
    if genes is not None:
        offenders = sorted({str(g) for g in genes if PAIR_LABEL_SEP in str(g)})
        if offenders:
            preview = ", ".join(offenders[:5])
            if len(offenders) > 5:
                preview += f", ... (+{len(offenders) - 5} more)"
            raise ValueError(
                f"This classifier predates structural gene pairs (format 1) and "
                f"{len(offenders)} of its genes contain an underscore, so pairs "
                f"labelled 'geneA_geneB' cannot be split back into genes: {preview}. "
                "Retrain it with tl.train_classifier to store pair members directly."
            )

    decoded = []
    for label in labels:
        parts = str(label).split(PAIR_LABEL_SEP)
        if len(parts) != 2:
            raise ValueError(
                f"Cannot decode gene pair {label!r}: expected exactly one "
                f"'{PAIR_LABEL_SEP}' separator. Retrain the classifier with "
                "tl.train_classifier."
            )
        decoded.append(parts)
    return np.array(decoded, dtype=object).reshape(-1, 2)


def _resolve_pair_genes(pairs, genes=None) -> np.ndarray:
    """Normalize gene pairs to an (n_pairs, 2) array, whichever form they arrive in.

    Args:
        pairs: Either an (n_pairs, 2) array of gene pairs (format 2) or a 1-D
            array of joined ``"geneA_geneB"`` labels (format 1).
        genes: Optional classifier gene symbols, used to detect labels that
            cannot be decoded.

    Returns:
        np.ndarray: (n_pairs, 2) object array of gene symbols.

    Raises:
        ValueError: If `pairs` is neither shape.
    """
    arr = np.asarray(pairs, dtype=object)
    if arr.ndim == 2 and arr.shape[1] == 2:
        return arr
    if arr.ndim == 1:
        return _split_pair_labels(arr, genes=genes)
    raise ValueError(
        f"Gene pairs must be a 1-D array of labels or an (n_pairs, 2) array of "
        f"gene names; got shape {arr.shape}."
    )


def _validate_classifier(payload) -> None:
    """Check that `payload` looks like a usable trained classifier.

    Args:
        payload: The object read back from disk.

    Raises:
        ValueError: If it is not a dict, is missing a required key, or holds an
            estimator that was never fitted.
    """
    if not isinstance(payload, dict):
        raise ValueError(
            f"Expected a classifier dict, got {type(payload).__name__}. "
            "This file was not written by save_classifier."
        )

    missing = [key for key in REQUIRED_CLASSIFIER_KEYS if key not in payload]
    if missing:
        raise ValueError(
            f"Classifier is missing required key(s): {', '.join(missing)}. "
            f"Expected at least {', '.join(REQUIRED_CLASSIFIER_KEYS)}."
        )

    estimator = payload["classifier"]
    if not hasattr(estimator, "predict_proba"):
        raise ValueError(
            f"Classifier['classifier'] is a {type(estimator).__name__}, which has no "
            "predict_proba method and cannot be used for prediction."
        )
    if not hasattr(estimator, "classes_"):
        raise ValueError(
            "Classifier['classifier'] has not been fitted (no classes_ attribute). "
            "Train it with tl.train_classifier before saving."
        )


def _migrate_classifier(payload: dict) -> dict:
    """Bring an older payload up to `SCN_CLASSIFIER_FORMAT_VERSION`.

    Args:
        payload: A validated classifier dict, possibly written by an older
            version of the package.

    Returns:
        dict: A copy stamped with the current format version.

    Raises:
        ValueError: If the payload is newer than this version of the package
            knows how to read.
    """
    migrated = dict(payload)
    # Payloads written before versioning existed are format 1 by definition.
    version = migrated.get("format_version", 1)

    if version > SCN_CLASSIFIER_FORMAT_VERSION:
        raise ValueError(
            f"Classifier format version {version} is newer than this version of "
            f"PySingleCellNet supports (up to {SCN_CLASSIFIER_FORMAT_VERSION}). "
            "Upgrade PySingleCellNet to read it."
        )

    # 1 -> 2: recover pair members from the joined labels. Raises when the gene
    # symbols make those labels ambiguous, which is the bug format 2 exists to fix.
    if version < 2 and "topPairGenes" not in migrated:
        migrated["topPairGenes"] = _resolve_pair_genes(
            migrated["topPairs"], genes=migrated["tpGeneArray"]
        )

    migrated["format_version"] = SCN_CLASSIFIER_FORMAT_VERSION
    return migrated


def save_classifier(clf: dict, filepath) -> None:
    """Write a trained classifier to disk with its format version recorded.

    Args:
        clf (dict): A classifier as returned by `tl.train_classifier`.
        filepath: Destination path (str or os.PathLike).

    Returns:
        None. Writes `filepath`.

    Raises:
        ValueError: If `clf` is not a usable trained classifier.

    Example:
        >>> clf = train_classifier(adata, groupby='celltype')
        >>> save_classifier(clf, 'pbmc_classifier.pkl')
    """
    _validate_classifier(clf)

    # Copy so the caller's dict does not acquire a format_version key.
    payload = dict(clf)
    payload["format_version"] = SCN_CLASSIFIER_FORMAT_VERSION

    with open(os.fspath(filepath), "wb") as handle:
        pickle.dump(payload, handle)


def load_classifier(filepath) -> dict:
    """Read a classifier from disk, validating and migrating it.

    Args:
        filepath: Path to a file written by `save_classifier`, or a
            hand-pickled classifier dict from before versioning existed.

    Returns:
        dict: The classifier, stamped with the current format version.

    Raises:
        FileNotFoundError: If `filepath` does not exist.
        ValueError: If the file does not hold a usable classifier, or holds a
            format newer than this package supports.

    Example:
        >>> clf = load_classifier('pbmc_classifier.pkl')
        >>> classify_anndata(query, clf)
    """
    path = os.fspath(filepath)
    if not os.path.exists(path):
        raise FileNotFoundError(f"No such classifier file: {path}")

    try:
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
    except ModuleNotFoundError as err:
        raise ValueError(
            f"Could not unpickle {path}: it references the module '{err.name}', which "
            "is not installed. The classifier was probably saved in an environment "
            f"with different packages; install '{err.name}' and retry."
        ) from err

    _validate_classifier(payload)
    migrated = _migrate_classifier(payload)

    if "format_version" not in payload:
        warnings.warn(
            f"{path} predates classifier format versioning; reading it as format 1. "
            "Re-save it with save_classifier to silence this warning.",
            UserWarning,
            stacklevel=2,
        )

    return migrated
