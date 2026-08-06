"""Package-level API surface tests."""

import importlib.metadata

import pytest


def test_version_is_exposed():
    """`__version__` is advertised in __all__, so it must actually resolve."""
    import pySingleCellNet as cn

    assert isinstance(cn.__version__, str)
    assert cn.__version__


def test_version_matches_installed_distribution():
    """The attribute should agree with the installed distribution metadata."""
    import pySingleCellNet as cn

    assert cn.__version__ == importlib.metadata.version("pySingleCellNet")


def test_public_namespaces_are_importable():
    """Everything named in __all__ is reachable as an attribute."""
    import pySingleCellNet as cn

    for name in cn.__all__:
        assert hasattr(cn, name), f"{name} is in __all__ but not bound"
