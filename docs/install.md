PySingleCellNet is available on PyPI. You can install it with:

```
pip install pySingleCellNet
```

You could also install it directly from GitHub with:

```shell
pip install git+https://github.com/CahanLab/PySingleCellNet.git
```

## From requirements.txt

Clone the repository and install all dependencies:

```shell
git clone https://github.com/CahanLab/PySingleCellNet.git
cd PySingleCellNet
pip install -r requirements.txt
pip install -e .
```

## From conda environment.yml

Create and activate a conda environment with all dependencies:

```shell
git clone https://github.com/CahanLab/PySingleCellNet.git
cd PySingleCellNet
conda env create -f environment.yml
conda activate pyscn
```

!!! note "Jupyter notebooks"
    To run the tutorial notebooks, install Jupyter in the environment:
    ```shell
    pip install jupyter ipykernel
    ```
