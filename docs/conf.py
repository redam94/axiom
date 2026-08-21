"""Sphinx configuration for the axiom API reference.

Build with ``make docs`` (``sphinx-build -W --keep-going -b html docs docs/_build/html``).
"""

from __future__ import annotations

import importlib.metadata

project = "axiom"
author = "Matthew Reda"
copyright = "2026, Matthew Reda"  # noqa: A001
try:
    release = importlib.metadata.version("axiom")
except importlib.metadata.PackageNotFoundError:  # building against an un-installed checkout
    release = "0.0.0"
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
]

# Markdown (MyST) alongside reStructuredText.
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
myst_enable_extensions = ["colon_fence", "deflist"]
myst_heading_anchors = 3

# docs/plan and docs/notes are included in the toctree (see index.md); only
# the build output and working files are excluded.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# ---- autodoc -------------------------------------------------------------
# Nothing is mocked: every dependency is installed by the dev group.
autodoc_typehints = "description"
autodoc_member_order = "bysource"
autodoc_default_options = {"members": True, "undoc-members": False, "show-inheritance": False}
autosummary_generate = False
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# Docstring markup problems that live in src/ (not fixable from docs/). Each is a
# Markdown pipe-table or a ``|x|`` absolute value inside a reStructuredText
# docstring, which docutils reads as a substitution reference / line block.
# Remove this suppression once they are rewritten:
#   src/axiom/core/entities.py:4            pipe table
#   src/axiom/adapters/marketing.py:8       pipe table
#   src/axiom/estimands/spec.py:6, :25      pipe tables
#   src/axiom/design/power.py:16            |δ| in a formula
#   src/axiom/diagnose/sensitivity.py:15    |bias| in a formula
#   src/axiom/calibrate/check.py:12         |z| in a formula
#   src/axiom/identify/frontdoor.py:195     enumerated list without blank line
suppress_warnings = ["docutils"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}

# ---- html ----------------------------------------------------------------
html_theme = "furo"
html_title = "axiom"
html_static_path: list[str] = []
