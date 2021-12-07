from setuptools import setup

setup(name='pyscn_gavehan',
      version='0.1',
      description='Determining cell identity from single cell RNA-Seq data',
      url='http://github.com/pcahan1/PySingleCellNet/',
      author='Patrick Cahan',
      author_email='patrick.cahan@gmail.com',
      license='MIT',
      packages=['pySingleCellNet'],
      install_requires=[
          'pandas',
          'numpy',
          'sklearn',
          'scanpy',
          'sklearn',
          'statsmodels',
          'scipy',
          'matplotlib',
          'seaborn',
          'umap-learn',
          'tqdm'
      ],
      zip_safe=False)
