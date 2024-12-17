from setuptools import setup, find_packages
import os

# Function to read the version from __version__.py
def get_version(package_name):
    version_file = os.path.join(os.path.dirname(__file__), 'src', package_name, '__version__.py')
    with open(version_file) as f:
        exec(f.read())
    return locals()['__version__']

setup(name='pySingleCellNet',
    version=get_version('pySingleCellNet'),  # Dynamically read the version
    description='Single cell classification and analysis, optimized for embryonic and fetal development',
    long_description=open("README.md").read(),
    long_description_content_type='text/markdown',  
    author='Patrick Cahan, Yuqi Tan',
    author_email='patrick.cahan@gmail.com',
    license='MIT',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
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
        'gseapy',
        'alive_progress',
        'python-igraph',
        'marsilea'
    ],
    project_urls={
        'Documentation': 'https://pysinglecellnet.readthedocs.io/en/latest/',
        'Source': 'https://github.com/CahanLab/PySingleCellNet'
    },
)