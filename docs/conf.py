# Configuration file for the Sphinx documentation builder.

from importlib import metadata

project = "bioviz-kit"
author = "Victoria Cheung"
try:
    release = metadata.version("bioviz-kit") or "0.4.3"
except metadata.PackageNotFoundError:
    release = "0.4.3"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
]

templates_path = ["_templates"]
exclude_patterns = []

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

autodoc_typehints = "description"
autosummary_generate = True
master_doc = "index"
