# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Add path to package ------------------------------------------------------
import sys
from pathlib import Path

sys.path.insert(0, Path.resolve()("../../src/"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "treeffuser"
copyright = "2024, Nicolas Beltran-Velez, Alessandro Antonio Grande, Achille Nazaret"
author = "Nicolas Beltran-Velez, Alessandro Antonio Grande, Achille Nazaret"
release = "0.1.3"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    # "recommonmark",
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # If using Google or NumPy style docstrings
    "sphinx.ext.viewcode",  # To include source code links
]

autoclass_content = "both"  # display doc both from a class docstring and its __init__ methods

templates_path = ["_templates"]
exclude_patterns = []

source_suffix = {  # use Markdown files with extensions other than .md
    ".rst": "restructuredtext",
    ".txt": "markdown",
    ".md": "markdown",
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"  # "alabaster"
html_static_path = ["_static"]
html_favicon = "_static/logo.svg"
