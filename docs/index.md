## PySingleCellNet
### a Python package for classifying scRNA-seq data

SingleCellNet predicts the 'cell type' of query scRNA-seq data by Random forest multi-class classification. See [Tan 2019](https://pubmed.ncbi.nlm.nih.gov/31377170/) for more details. It was [originally written in R](https://github.com/pcahan1/SingleCellNet). PySingleCellNet (**PySCN**) is the Python version, and it is compatible with [Scanpy](https://scanpy.readthedocs.io/en/stable/). PySCN was crafted to aid in the analysis of engineered cell populations (i.e. cells derived via directed differentiation of pluripotent stem cells or via direct conversion), but can just as easily be used to perform _cell typing_ on data dervived from other sources as long as adequate training data is available.



