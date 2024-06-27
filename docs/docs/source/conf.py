# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Add path to package ------------------------------------------------------
import sys
from pathlib import Path

path_to_package = Path("../../src/")
sys.path.insert(0, path_to_package.resolve())

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "treeffuser"
copyright = "2024, Nicolas Beltran-Velez, Alessandro Antonio Grande, Achille Nazaret"
author = "Nicolas Beltran-Velez, Alessandro Antonio Grande, Achille Nazaret"
release = "0.1.3"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # If using Google or NumPy style docstrings
    "sphinx.ext.viewcode",  # To include source code links
    "sphinx.ext.githubpages",  # Creates .nojekyll making "_" folders with css styles accessible
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
html_static_path = ["static"]
html_favicon = "static/logo.svg"
html_logo = "static/logo.svg"
html_title = " "
# html_additional_pages = {"index": "getting-started.html"}  # set custom landing page

# Change colors of Furo theme
html_theme_options = {
    "light_css_variables": {
        #         "color-background-secondary": "#145c15",  # see --treffuser-color in docs/css/treeffuser.css
        "color-brand-primary": "#145c15",
        "color-brand-content": "#145c15",
        # "color-foreground-primary": "#145c15",
    },
}
