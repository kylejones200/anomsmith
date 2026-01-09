"""Sphinx configuration for Anomsmith documentation."""

import os
import sys

# Add the project root to the path
sys.path.insert(0, os.path.abspath(".."))

# Project information
project = "Anomsmith"
copyright = "2024, Kyle T. Jones"
author = "Kyle T. Jones"
release = "0.0.1"

# General configuration
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
]

# Try to load myst_parser if available
try:
    import myst_parser
    extensions.append("myst_parser")
except ImportError:
    pass

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# HTML output options
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# Autodoc settings
# Note: We don't mock timesmith since it's a required dependency
autodoc_mock_imports = []
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

# Napoleon settings for Google-style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

