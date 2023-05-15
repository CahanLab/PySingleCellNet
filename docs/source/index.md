
# PySingleCellNet: classify scRNAseq data in Python

SingleCellNet predicts the 'cell type' of query scRNA-seq data by Random forest multi-class classification. See [Tan 2019](https://pubmed.ncbi.nlm.nih.gov/31377170/) for more details. It was originally written in R. PySCN is the Python version which includes functionality to aid in the analysis of engineered cell populations (i.e. cells derived via directed differentiation of pluripotent stem cells or via direct conversion).

[github]: https://github.com/pcahan1/PySingleCellNet
[original version]: https://github.com/pcahan1/SingleCellNet


```{toctree}
:maxdepth: 2
:caption: User guide

Getting started <install.md>
Prepare training data <notebooks/how-to_prepare_reference_data.ipynb>
Train and classify <notebooks/train_classifier.ipynb>
Explore results <notebooks/explore.ipynb>
FAQ <faq.md>
Training data <training_data.md>
References <refs.md>
```

