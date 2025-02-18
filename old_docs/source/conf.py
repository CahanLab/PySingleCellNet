import os
import sys
sys.path.insert(0, os.path.abspath('../'))
# from pathlib import Path
# from sphinx.application import Sphinx
# from sphinx.ext import autosummary
# HERE = Path(__file__).parent
# sys.path[:0] = [str(HERE.parent), str(HERE / "extensions")]
# import pySingleCellNet

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'pySingleCellNet'
copyright = '2023, Patrick Cahan, Yuqi Tan, Yuqing Han'
author = 'Patrick Cahan, Yuqi Tan, Yuqing Han'
release = 'Oct 16, 2020'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
#    'autoapi.extension',
    'sphinx.ext.autosummary',
    'sphinx_autodoc_typehints',
    'myst_nb',
    'sphinx_copybutton',
    'autodocsumm'
]

autosummary_generate = True  
autodoc_member_order = (
    "groupwise"  # Sort automatically documented members by member type
)
python_use_unqualified_type_names = True
napoleon_google_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_rtype = True  # having a separate entry generally helps readability
napoleon_use_param = True

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
    'autosummary': True,  # Generate summary tables
}

bibtex_bibfiles = ['SCN_refs.bib']
bibtex_reference_style = 'author_year'
templates_path = ['_templates']
exclude_patterns = [
    "_build",
]

# autodoc2_packages = [
#     "../pySingleCellNet"
# ]

# autodoc2_render_plugin = "myst"
# autodoc2_output_dir = "apidocs"

suppress_warnings = [
#    "autodoc2.*",  # suppress all
#    "autodoc2.config_error",  # suppress specific
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

#html_theme = 'sphinx_rtd_theme'
html_theme = 'sphinx_book_theme'
html_static_path = ['_static']
html_logo = "_static/pySCN_tree.png"
html_show_sphinx = False
html_title = "pySingleCellNet"

html_theme_options = {
    "repository_url": "https://github.com/CahanLab/pySingleCellNet",
    "use_repository_button": True,
}

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
    "html_admonition",
]

nb_execution_mode = "off"

