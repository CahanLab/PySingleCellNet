# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'PySingleCellNet'
copyright = '2023, Patrick Cahan, Yuqi Tan, Yuqing Han'
author = 'Patrick Cahan, Yuqi Tan, Yuqing Han'
release = 'Oct 16, 2020'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
 #   'sphinx.ext.autodoc',
 #   'autodoc2',
    'sphinx_copybutton',
    'myst_nb'
]

templates_path = ['_templates']
exclude_patterns = []

autodoc2_packages = [
    "../pySingleCellNet"
]

autodoc2_render_plugin = "myst"
autodoc2_output_dir = "apidocs"

suppress_warnings = [
    "autodoc2.*",  # suppress all
    "autodoc2.config_error",  # suppress specific
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

#html_theme = 'sphinx_rtd_theme'
html_theme = 'sphinx_book_theme'
html_static_path = ['_static']
html_logo = "_static/pySCN_tree.png"


# import os
# import sys
# sys.path.insert(0, os.path.abspath('../path/to/your/project'))

myst_enable_extensions = ["colon_fence"]
jupyter_execute_notebooks = "off"