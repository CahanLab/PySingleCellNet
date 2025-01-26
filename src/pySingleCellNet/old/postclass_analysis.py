import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from anndata import AnnData
from sklearn.ensemble import RandomForestClassifier
from scipy.sparse import csr_matrix
import warnings
from .utils import *
from .tsp_rf import *
import gseapy as gp
import os
# import anndata
# import pySingleCellNet as pySCN
#import pacmap
import copy
import igraph as ig
from collections import defaultdict




# Example usage:
# combine_pca_scores(adata, n_pcs=30, score_key='SCN_score')

