"""PySingleCellNet"""

try:
    # Written by setuptools-scm at build time; absent in a bare source checkout.
    from ._version import version as __version__
except ImportError:  # pragma: no cover - depends on how the package was obtained
    from importlib.metadata import PackageNotFoundError, version as _version

    try:
        __version__ = _version(__name__)
    except PackageNotFoundError:
        __version__ = "0.0.0+unknown"

from .config import SCN_CATEGORY_COLOR_DICT
from .config import SCN_DIFFEXP_KEY
from . import plotting as pl
from . import utils as ut
from . import tools as tl

# Public API
__all__ = [
    "__version__",
    "pl",
    "ut",
    "tl",
]    


