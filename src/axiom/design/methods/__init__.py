"""design.methods: the experiment-method registry and its six estimators.

Difference in differences, synthetic control, time-based regression,
cluster-based regression, switchback, and ghost exposure. Every estimator
reads a ``PanelArrays`` and returns a ``MethodEstimate`` whose ``effect`` is the
average effect per treated unit per post period, with a ``wald`` interval
whose critical value is Student-t at the method's degrees of freedom
(``wald_t``), or a typed ``Unsupported``. ``estimate(method, arrays)`` dispatches by name;
``estimate_<method>`` are the keyword conveniences. Ported from the parent's
``planning/methods``; see ``docs/plan/02-porting-ledger.md``.
"""

from axiom.design.methods.cluster_based_regression import estimate_cluster_based_regression
from axiom.design.methods.difference_in_differences import estimate_difference_in_differences
from axiom.design.methods.ghost import estimate_ghost
from axiom.design.methods.registry import (
    ASSUMPTIONS,
    METHODS,
    MethodEstimate,
    MethodName,
    MethodSpec,
    MethodStatus,
    PanelArrays,
    estimate,
    method_assumption,
    method_spec,
    panel_arrays,
    t_critical,
    wald_t,
)
from axiom.design.methods.switchback import bartlett_bandwidth, estimate_switchback
from axiom.design.methods.synthetic_control import estimate_synthetic_control, simplex_weights
from axiom.design.methods.time_based_regression import estimate_time_based_regression

__all__ = [
    "ASSUMPTIONS",
    "METHODS",
    "MethodEstimate",
    "MethodName",
    "MethodSpec",
    "MethodStatus",
    "PanelArrays",
    "bartlett_bandwidth",
    "estimate",
    "estimate_cluster_based_regression",
    "estimate_difference_in_differences",
    "estimate_ghost",
    "estimate_switchback",
    "estimate_synthetic_control",
    "estimate_time_based_regression",
    "method_assumption",
    "method_spec",
    "panel_arrays",
    "simplex_weights",
    "t_critical",
    "wald_t",
]
