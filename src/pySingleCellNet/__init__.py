"""PySingleCellNet"""

from .config import SCN_CATEGORY_COLOR_DICT 
from .config import SCN_DIFFEXP_KEY
from . import plotting as pl
from . import utils as ut
from .stats import *
from .tsp_rf import *
from .scn_train import *
from .scn_assess import create_classifier_report
from .postclass_analysis import *
from .rank_class import *



# Public API
__all__ = [
    "__version__",
    "pl",
    "ut",
    "train_classifier",
    "scn_train",
    "classify_anndata",
    "graph_from_nodes_and_edges",
    "comp_ct_thresh",
    "class_by_threshold",
    "determine_relationships"
    "remove_xist_y_genes",
    "create_classifier_report"
]    


