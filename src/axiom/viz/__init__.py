"""viz: optional plotly figures behind the ``[viz]`` extra.

Importing this package never imports plotly; each figure function returns
``Unsupported(reason="plotly not installed", missing=("viz",))`` when it is
absent. See ``nbs/viz/``.
"""

from __future__ import annotations

from axiom.viz.figures import (
    available,
    backtest_plot,
    boundary,
    causal_graph,
    convergence,
    corrections,
    coverage_plot,
    forest,
    funnel,
    intervals,
    marginal_curve,
    panel_coverage,
    recovery,
    response_curve,
    sbc_ranks,
    spec_curve_plot,
    stability,
    transfer,
    unrolled,
)

__all__ = [
    "available",
    "backtest_plot",
    "boundary",
    "causal_graph",
    "convergence",
    "corrections",
    "coverage_plot",
    "forest",
    "funnel",
    "intervals",
    "marginal_curve",
    "panel_coverage",
    "recovery",
    "response_curve",
    "sbc_ranks",
    "spec_curve_plot",
    "stability",
    "transfer",
    "unrolled",
]
