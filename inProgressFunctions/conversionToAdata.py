
"""
Created on Tue Jul 16 15:41:22 2019
@author: SamCrowl
"""

import os
import pandas as pd

import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter
from anndata import AnnData

def convertRDAtoAdata(expMat_file, sampTab_file, file_path):
    """Takes expMat and sampTab .rda files and converts them to one AnnData object for use in python SCN
    
    Parameters:
        expMat_file: Name of expression matrix .rda file
        sampTab_file: Name of sample table .rda file
        file_path: location of both files (need to be in the same directory)
    
    Returns:
        n_obs x n_vars AnnData object which contains gene information in columns (var) and cell information in rows (obs)
    """
    pandas2ri.activate()
    
    base = importr('base')
    matFileName = '{}/{}'.format(file_path, expMat_file)
    stFileName = '{}/{}'.format(file_path, sampTab_file)
    base.load(matFileName)
    base.load(stFileName)
    rdf = base.mget(base.ls())
    #converts r objects into pandas versions of expMat and sampTab
    with localconverter(robjects.default_converter + pandas2ri.converter):
        expMat = robjects.conversion.rpy2py(rdf[0])
        metadata = robjects.conversion.rpy2py(rdf[1])
    #Loads expMat into a AnnData object, with gene names as index for vars and cell_ids as index for obs
    adata = AnnData(expMat).T
    #Load sampTab data into obs
    for data in metadata.columns.values:
        adata.obs[data] = metadata.loc[:,data].values
    #Create var for gene_ids in vars in addition to index
    adata.var["gene_ids"] = adata.var.index.values

    return(adata)
    
