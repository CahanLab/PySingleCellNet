from setuptools import setup

setup(name='pySingleCellNet',
      version='0.1',
      description='Determine cell identity from single cell RNA-Seq data',
      url='http://github.com/pcahan1/PySingleCellNet/',
      author='Patrick Cahan',
      author_email='patrick.cahan@gmail.com',
      license='MIT',
      packages=['pySingleCellNet'],
      install_requires=[
          'pandas',
          'numpy',
          'scikit-learn',
          'scanpy',
          'statsmodels',
          'scipy',
          'matplotlib',
          'seaborn',
          'umap-learn',
          'mygene',
          'palettable',
          'gseapy'
      ],
      zip_safe=False)
