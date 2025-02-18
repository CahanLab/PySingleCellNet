"""PySingleCellNet"""

from .config import SCN_CATEGORY_COLOR_DICT 
from .config import SCN_DIFFEXP_KEY
from . import plotting as pl
from . import utils as ut
from . import classify as cl

# Public API
__all__ = [
    "__version__",
    "pl",
    "ut",
    "cl"
]    


