# PySingleCellNet
singleCellNet in Python
### Introduction
This is a python package for singleCellNet that accomplishes the same tasks as the original version written in R. Unlike the R version, which takes an expression matrix and sample table as inputs, PySingleCellNet requires that all information be stored in an AnnData object. The expression matrix is in the form n_obs x n_vars, where gene info is stored in variables and cell info is stored in objects. For more information on working with AnnData objects, see the documentation [here](https://anndata.readthedocs.io/en/latest/anndata.AnnData.html). 

#### Installing required packages
PySingleCellNet makes use of the following packages:
- scanpy
- rpy2
- numpy
- pandas
- sklearn

Make sure these packages are installed on your machine prior to using PySingleCellNet.

#### Using PySingleCellNet
Once all necessary packages and PySingleCellNet are installed, import the necesary packages
```python
import PySingleCellNet as pscn
import scanpy as sc
```

#### Converting from .rda expression matrix and sample table to AnnData object
If your data is stored in .rda files, you will first need to convert them to an AnnData object with the following code. Expression matrix will be stored in adata.X, gene info will be stored in adata.var, and all cell info, including meta data, will be stored in adata.obs
```python
expMat_file = 'fileNameForExpMat.rda'
sampTab_file = 'fileNameForSampTab.rda'
filepath = 'locationOfFiles'
adata = pscn.ut.convertRDAtoAdata(expMat_file, sampTab_file, filepath)
```
### Training the SCN classifier
#### Load data


#### Find genes in common between training and query datasets and limit analysis to these genes
```python
commonGenes = np.intersect1d(aTrain.var.index, aQuery.var.index, assume_unique = True)
aTrain = aTrain[: ,aTrain.var.index == commonGenes]
```

#### Split for training and assessment, transform the training data
```python
aList = pscn.ut.splitCommon(aTrain, ncells = 100, dLevel = "newAnn")
aTrain2 = aList[0]

sc.pp.normalize_per_cell(aTrain2, counts_per_cell_after = 1e5)
```

#### Find the best set of classifier genes
```python
cgenes2 = pscn.tr.findClassyGenes(aTrain2, dLevel = "newAnn", topX = 10)

#limit analysis to these genes
aTrain2 = aTrain2[:, aTrain2.var.index == cgenes2]
```

#### Find the top gene pairs and transform the data. This section of code is not yet functional. 
```python
xpairs = pscn.tr.ptGetTop(aTrain2, dLevel = "newAnn", topX = 50, sliceSize = 5000)
pdTrain = pscn.query_transform(aTrain2, xpairs)
```

#### Train the classifier. This section of code is not yet functional
Returns a RandomForest object from the sklearn.ensemble package
```python
rf_tspAll = pscn.tr.sc_makeClassifier(pdTrain, genes = xpairs, groups = "newAnn", nrand = 50, ntrees = 1000)
```

#### Apply to held out data
```python
aTest = aList[1]

aQueryTransform = pscn.qt.query_transform(aTest[:, aTest.var.index == cgenes2], xpairs)

classRes = rf_tspAll.predict_proba(aQueryTransform)
```
