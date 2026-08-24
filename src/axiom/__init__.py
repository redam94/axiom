"""axiom — Bayesian causal decision science.

Identification, experimental design and calibration, response-surface
methodology, and evidence meta-analysis, over one domain-general vocabulary.

Version 1.0: every subpackage is implemented and demonstrated under ``nbs/``;
the plan is in ``docs/plan/`` and the decision log in ``docs/notes/``.

Import discipline: this module must never import a sampler, a plotting library,
or anything outside {numpy, scipy, pandas, pydantic}. Pinned by
``tests/contracts/test_import_weight.py``.
"""

__version__ = "1.1.0"
__all__: list[str] = []
