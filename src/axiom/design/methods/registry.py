"""The experiment-method registry: what each estimator assumes, needs, and returns.

A ``MethodSpec`` names a method, the ``core.Assumption`` s (facet ``"method"``)
under which its number is a causal effect, what data it needs (a pre-period?
control units?), and its ``status``: ``"stable"`` methods have passed the A/A
calibration in ``design.simulate``; a method that fails is ``"experimental"``
and ``simulate.calibrate_registry`` returns a registry saying so, rather than
shipping it quietly.

Every estimator has the same shape: it reads a ``PanelArrays`` — an
``(n_units, n_periods)`` outcome array plus the treated indices and the
``pre`` / ``post`` windows (and, for designs that need them, a switchback
``assignment`` or a ghost ``exposed`` mask) — and returns a ``MethodEstimate``
or a typed ``Unsupported``. ``effect`` is always the *average effect per
treated unit per post period*, so estimates from different methods are
comparable and ``simulate`` can score bias and coverage uniformly; methods
whose natural quantity is cumulative (TBR) put the total in ``detail``. The
interval is a ``wald`` interval at the requested ``mass`` whose critical value
is Student-t at the method's residual degrees of freedom (``wald_t``): every
method's standard error is an estimated variance with finite ``df`` (as few
as ``n_pre − 2`` for time-based regression), and the normal ``z`` runs the A/A
false-positive rate several points hot at those sizes. ``detail["df"]`` and
``detail["critical_value"]`` put the critical value on the record, and
``MethodEstimate.critical`` names it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

import numpy as np
import numpy.typing as npt
from pydantic import model_validator

from axiom.core import Assumption, Interval, NonEmptyStr, Spec, Unsupported

__all__ = [
    "ASSUMPTIONS",
    "METHODS",
    "MethodEstimate",
    "MethodName",
    "MethodSpec",
    "MethodStatus",
    "PanelArrays",
    "estimate",
    "method_assumption",
    "method_spec",
    "panel_arrays",
    "t_critical",
    "wald_t",
]

Array = npt.NDArray[np.float64]
MethodStatus = Literal["stable", "experimental"]
CriticalValue = Literal["normal", "student_t"]
MethodName = Literal[
    "difference_in_differences",
    "synthetic_control",
    "time_based_regression",
    "cluster_based_regression",
    "switchback",
    "ghost",
]


# -- assumptions -----------------------------------------------------------------------


def method_assumption(name: str, statement: str, challenged_by: str) -> Assumption:
    """A method-facet ``Assumption`` in its default ``unverified`` state."""
    return Assumption(name=name, facet="method", statement=statement, challenged_by=challenged_by)


ASSUMPTIONS: Mapping[str, Assumption] = MappingProxyType(
    {
        "parallel_trends": method_assumption(
            "parallel_trends",
            "absent treatment, treated and control outcomes would have moved in parallel",
            "a pre-period event-study showing diverging pre-trends",
        ),
        "no_anticipation": method_assumption(
            "no_anticipation",
            "outcomes before the treatment window do not respond to the coming treatment",
            "a pre-period effect at the last pre periods",
        ),
        "no_interference": method_assumption(
            "no_interference",
            "a unit's outcome depends only on its own treatment (no spillover across units)",
            "an effect on control units adjacent to treated ones",
        ),
        "random_assignment": method_assumption(
            "random_assignment",
            "treatment was assigned by a known randomization, independent of potential outcomes",
            "imbalance on pre-period outcomes beyond what the randomization predicts",
        ),
        "convex_hull": method_assumption(
            "convex_hull",
            "the treated unit's pre-period trajectory lies in the convex hull of the donors",
            "pre-period RMSPE large relative to the treated series' variation",
        ),
        "stable_pre_relationship": method_assumption(
            "stable_pre_relationship",
            "the pre-period linear relation between treated and control aggregates persists",
            "a structural break in the control aggregate or a changing slope in the pre period",
        ),
        "no_carryover": method_assumption(
            "no_carryover",
            "the effect of a treated period does not persist into the following untreated one",
            "an outcome in off periods that depends on the previous period's assignment",
        ),
        "exposure_known": method_assumption(
            "exposure_known",
            "which control units would have been exposed is recorded exactly (ghost exposure)",
            "a ghost-exposure rate in control differing from the exposure rate in treatment",
        ),
        "stationary_noise": method_assumption(
            "stationary_noise",
            "within-unit noise is weakly stationary so a HAC variance is meaningful",
            "a trend or variance change in within-unit residuals",
        ),
    }
)


# -- specs -----------------------------------------------------------------------------


class MethodSpec(Spec):
    """A registered experiment method: its assumptions, requirements, and status."""

    name: NonEmptyStr
    assumptions: tuple[Assumption, ...]
    requires_pre_period: bool
    requires_controls: bool
    status: MethodStatus = "stable"
    description: str = ""

    @model_validator(mode="after")
    def _valid(self) -> MethodSpec:
        if not self.assumptions:
            raise ValueError(f"method {self.name!r} must name at least one assumption")
        if any(a.facet != "method" for a in self.assumptions):
            raise ValueError("method assumptions must have facet='method'")
        return self

    def assumption_names(self) -> tuple[str, ...]:
        return tuple(a.name for a in self.assumptions)


class MethodEstimate(Spec):
    """One method's estimate: the average effect per treated unit per post period.

    ``se_method`` says where the standard error came from; ``detail`` carries
    the method's own numbers (degrees of freedom, fit statistics, cumulative
    totals). ``interval`` is ``effect ± c · se`` at its stated mass, labelled
    ``wald``; ``critical`` says whether ``c`` is the normal quantile or the
    Student-t quantile at ``detail["df"]`` (``detail["critical_value"]`` is
    ``c`` itself).
    """

    method: NonEmptyStr
    effect: float
    se: float
    interval: Interval
    n_treated: int
    n_control: int
    n_pre: int
    n_post: int
    se_method: NonEmptyStr
    critical: CriticalValue = "normal"
    detail: dict[str, float] = {}

    @model_validator(mode="after")
    def _valid(self) -> MethodEstimate:
        if self.se < 0 or not np.isfinite(self.se):
            raise ValueError(f"se must be finite and non-negative, got {self.se}")
        if not np.isfinite(self.effect):
            raise ValueError("effect must be finite")
        if self.interval.definition != "wald":
            raise ValueError("a MethodEstimate interval is a wald confidence interval")
        return self

    @property
    def z(self) -> float:
        return self.effect / self.se if self.se > 0 else float("inf")

    def excludes_zero(self) -> bool:
        return not self.interval.contains(0.0)


# -- intervals -------------------------------------------------------------------------


def t_critical(mass: float, df: float) -> float:
    """Student-t two-sided critical value ``t_{df, (1 + mass) / 2}``.

    Every registered method's standard error is an estimated variance with a
    finite residual degree of freedom, so the interval uses the t quantile at
    that ``df`` rather than the normal ``z``. At ``df = 10`` (time-based
    regression with twelve pre periods) ``z`` turns a nominal 5 % A/A
    false-positive rate into 10 %; the t quantile returns it to nominal.
    """
    if not 0.0 < mass < 1.0:
        raise ValueError(f"mass must be in (0, 1), got {mass}")
    if not (np.isfinite(df) and df > 0):
        raise ValueError(f"df must be finite and positive, got {df}")
    from scipy.stats import t as student_t

    return float(student_t.ppf((1.0 + mass) / 2.0, df))


def wald_t(effect: float, se: float, mass: float, df: float) -> Interval:
    """``effect ± t_critical(mass, df) · se`` as a ``wald`` interval at ``mass``."""
    if se < 0 or not np.isfinite(se):
        raise ValueError(f"standard error must be finite and non-negative, got {se}")
    c = t_critical(mass, df)
    return Interval(lower=effect - c * se, upper=effect + c * se, definition="wald", mass=mass)


# -- the registry ----------------------------------------------------------------------


def _spec(
    name: str,
    assumptions: Sequence[str],
    *,
    requires_pre_period: bool,
    requires_controls: bool,
    description: str,
) -> MethodSpec:
    return MethodSpec(
        name=name,
        assumptions=tuple(ASSUMPTIONS[a] for a in assumptions),
        requires_pre_period=requires_pre_period,
        requires_controls=requires_controls,
        description=description,
    )


METHODS: Mapping[str, MethodSpec] = MappingProxyType(
    {
        "difference_in_differences": _spec(
            "difference_in_differences",
            ("parallel_trends", "no_anticipation", "no_interference"),
            requires_pre_period=True,
            requires_controls=True,
            description=(
                "Change in treated means minus change in control means; SE from the "
                "unit-level pre-to-post changes (robust to serial correlation within unit)."
            ),
        ),
        "synthetic_control": _spec(
            "synthetic_control",
            ("convex_hull", "no_anticipation", "no_interference"),
            requires_pre_period=True,
            requires_controls=True,
            description=(
                "Simplex-weighted donor combination fit on the pre period; SE from the "
                "placebo-in-space permutation over donors."
            ),
        ),
        "time_based_regression": _spec(
            "time_based_regression",
            ("stable_pre_relationship", "no_anticipation", "no_interference"),
            requires_pre_period=True,
            requires_controls=True,
            description=(
                "Regress the treated aggregate on the control aggregate in the pre period, "
                "forecast the post period; effect = mean residual, SE from the forecast error."
            ),
        ),
        "cluster_based_regression": _spec(
            "cluster_based_regression",
            ("random_assignment", "no_interference"),
            requires_pre_period=True,
            requires_controls=True,
            description=(
                "Unit-level regression of the post-period mean on the treatment indicator "
                "with the pre-period mean as covariate; classical OLS SE."
            ),
        ),
        "switchback": _spec(
            "switchback",
            ("random_assignment", "no_carryover", "stationary_noise"),
            requires_pre_period=False,
            requires_controls=False,
            description=(
                "Within-unit time-randomized on/off periods; within-unit regression with "
                "HAC (Bartlett) or block-bootstrap SE."
            ),
        ),
        "ghost": _spec(
            "ghost",
            ("random_assignment", "exposure_known", "no_interference"),
            requires_pre_period=False,
            requires_controls=True,
            description=(
                "Intent-to-treat among the exposed: treated-and-exposed versus control-and-"
                "ghost-exposed post-period means; two-sample SE."
            ),
        ),
    }
)


def method_spec(
    method: str | MethodSpec, registry: Mapping[str, MethodSpec] = METHODS
) -> MethodSpec:
    """Resolve a name through ``registry`` (a ``MethodSpec`` passes through)."""
    if isinstance(method, MethodSpec):
        return method
    try:
        return registry[method]
    except KeyError:
        raise KeyError(f"unknown method {method!r}; registered: {sorted(registry)}") from None


# -- the data contract -----------------------------------------------------------------


@dataclass(frozen=True)
class PanelArrays:
    """The arrays every method reads.

    ``outcome`` is ``(n_units, n_periods)``; ``treated`` indexes its rows;
    ``pre`` and ``post`` are half-open period windows. ``assignment`` is a
    ``(n_units, n_periods)`` 0/1 on/off schedule (switchback) and ``exposed`` an
    ``(n_units,)`` 0/1 mask of units that were, or as ghosts would have been,
    exposed (ghost). Built through ``panel_arrays`` so the shapes are checked
    once.
    """

    outcome: Array
    treated: tuple[int, ...]
    pre: slice
    post: slice
    assignment: Array | None = None
    exposed: Array | None = None

    @property
    def n_units(self) -> int:
        return int(self.outcome.shape[0])

    @property
    def n_periods(self) -> int:
        return int(self.outcome.shape[1])

    @property
    def control(self) -> tuple[int, ...]:
        t = set(self.treated)
        return tuple(i for i in range(self.n_units) if i not in t)


def _window(name: str, w: slice, n_periods: int) -> slice:
    start, stop, step = w.indices(n_periods)
    if step != 1:
        raise ValueError(f"{name} window must have unit step, got {w}")
    return slice(start, stop)


def panel_arrays(
    outcome: npt.ArrayLike,
    treated: Sequence[int],
    pre: slice,
    post: slice,
    *,
    assignment: npt.ArrayLike | None = None,
    exposed: npt.ArrayLike | None = None,
) -> PanelArrays:
    """Validate and freeze the arrays a method reads; see ``PanelArrays``."""
    y = np.asarray(outcome, dtype=np.float64)
    if y.ndim != 2 or y.shape[0] < 1 or y.shape[1] < 1:
        raise ValueError(f"outcome must be (n_units, n_periods), got shape {y.shape}")
    if not np.all(np.isfinite(y)):
        raise ValueError("outcome must be finite; methods do not impute")
    n_units, n_periods = y.shape
    t = tuple(int(i) for i in treated)
    if len(set(t)) != len(t):
        raise ValueError("treated indices must be distinct")
    if any(i < 0 or i >= n_units for i in t):
        raise ValueError(f"treated indices must lie in [0, {n_units})")
    pre_w = _window("pre", pre, n_periods)
    post_w = _window("post", post, n_periods)
    if pre_w.stop > post_w.start and pre_w.start < post_w.stop and pre_w.start < pre_w.stop:
        raise ValueError(f"pre {pre_w} and post {post_w} windows overlap")
    a: Array | None = None
    if assignment is not None:
        a = np.asarray(assignment, dtype=np.float64)
        if a.shape != y.shape:
            raise ValueError(f"assignment must have the outcome's shape {y.shape}, got {a.shape}")
        if not np.all(np.isin(a, (0.0, 1.0))):
            raise ValueError("assignment must be 0/1")
    e: Array | None = None
    if exposed is not None:
        e = np.asarray(exposed, dtype=np.float64).ravel()
        if e.shape != (n_units,):
            raise ValueError(f"exposed must have shape ({n_units},), got {e.shape}")
        if not np.all(np.isin(e, (0.0, 1.0))):
            raise ValueError("exposed must be 0/1")
    return PanelArrays(outcome=y, treated=t, pre=pre_w, post=post_w, assignment=a, exposed=e)


# -- dispatch --------------------------------------------------------------------------

Estimator = Callable[[PanelArrays, float], "MethodEstimate | Unsupported"]


def _estimator(name: str) -> Estimator:
    # Imported inside the function: the method modules import this one.
    if name == "difference_in_differences":
        from axiom.design.methods import difference_in_differences as m1

        return m1.estimate_arrays
    if name == "synthetic_control":
        from axiom.design.methods import synthetic_control as m2

        return m2.estimate_arrays
    if name == "time_based_regression":
        from axiom.design.methods import time_based_regression as m3

        return m3.estimate_arrays
    if name == "cluster_based_regression":
        from axiom.design.methods import cluster_based_regression as m4

        return m4.estimate_arrays
    if name == "switchback":
        from axiom.design.methods import switchback as m5

        return m5.estimate_arrays
    if name == "ghost":
        from axiom.design.methods import ghost as m6

        return m6.estimate_arrays
    raise KeyError(f"no estimator is wired for method {name!r}")


def estimate(
    method: str | MethodSpec,
    arrays: PanelArrays,
    *,
    mass: float = 0.95,
    registry: Mapping[str, MethodSpec] = METHODS,
) -> MethodEstimate | Unsupported:
    """Run a registered method on ``arrays`` with a ``wald`` interval at ``mass``.

    The method's data requirements are checked here so an estimator never
    sees data it cannot use: a missing pre period or missing controls is an
    ``Unsupported`` naming the requirement.
    """
    spec = method_spec(method, registry)
    if not 0.0 < mass < 1.0:
        raise ValueError(f"mass must be in (0, 1), got {mass}")
    n_pre = arrays.pre.stop - arrays.pre.start
    if spec.requires_pre_period and n_pre < 1:
        return Unsupported(
            reason=f"{spec.name} requires a pre period; got an empty pre window",
            missing=("pre_period",),
        )
    if spec.requires_controls and not arrays.control:
        return Unsupported(
            reason=f"{spec.name} requires control units; every unit is treated",
            missing=("controls",),
        )
    return _estimator(spec.name)(arrays, mass)
