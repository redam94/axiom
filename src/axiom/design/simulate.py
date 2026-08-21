"""A/A and A/B simulation of the experiment methods, and the methodology leaderboard.

Ported from the parent's ``planning/simulation.py`` by specification. The
parent simulated each method on its own data-generating process, which is
how two methods came to be "calibrated" on panels that differed in ways the
comparison never mentioned. Here one ``SimulationSpec`` describes one panel
and every method reads the same draw of it.

**The panel.** For unit ``i`` and period ``t``

    y[i, t] = mu_i + s_t + e[i, t] + effect · D[i, t]

with ``mu_i ~ N(0, unit_sd²)`` unit intercepts, ``s_t`` a stationary AR(1)
common shock (``s_t = rho · s_{t−1} + sqrt(1 − rho²) · period_sd · z_t``, so
its marginal standard deviation is ``period_sd`` whatever ``rho`` is),
``e[i, t] ~ N(0, noise_sd²)`` independent noise, and ``D`` the design's
treatment indicator. The treated set is a uniformly random subset of
``n_treated`` units (random assignment), so the cluster-based regression's
assumption holds by construction.

**Three designs, one base draw.** ``simulate_panel`` draws the intercepts,
shocks, noise, treated set, a switchback ``assignment`` and an ``exposed``
mask once, and applies the effect according to ``design``:

* ``"holdout"`` — ``D = treated_i · post_t``. The truth scored against is
  ``effect`` for every treated unit in every post period: the estimand of
  difference-in-differences, synthetic control, time-based regression and
  cluster-based regression.
* ``"switchback"`` — ``D = assignment[i, t]``, a per-unit per-period
  Bernoulli(``on_fraction``) draw, applied to every unit in every period; the
  holdout effect is *not* applied. The truth is ``effect`` per on period.
* ``"ghost"`` — ``D = treated_i · post_t · exposed_i``; ``exposed`` is a
  Bernoulli(``exposure_rate``) draw over all units, so control units carry a
  ghost-exposure flag that says whether they *would* have been exposed. The
  truth scored against is ``effect`` — the intent-to-treat effect among the
  exposed, which is what the ghost method estimates. (A holdout method run on
  a ghost panel would be estimating ``effect · exposure_rate``; the
  simulation does not do that.)

Every panel carries ``assignment`` and ``exposed`` whatever its design, so a
method that needs them never sees ``Unsupported`` for a missing array. The
design a method is served is read off its data requirements in
``design_for_method``: a method needing neither a pre period nor controls is
a within-unit switchback design; one needing controls but no pre period is a
ghost design; anything else is a holdout.

**A/A calibration** (``calibrate_method``, ``calibrate_registry``) forces
``effect = 0`` and counts the runs whose interval at mass ``1 − alpha``
excludes zero. The count is judged against the exact binomial region
``core.clopper_pearson(n, alpha, 1e-3)``; the rate is reported beside it. An
``Unsupported`` return is not a false positive — it is counted separately,
and more than ten percent of them fails the method too, because a method
that declines most panels has not been calibrated. A method that fails is
returned with ``status="experimental"`` in a *new* registry; ``METHODS`` is
never mutated. An estimator that raises is a bug and the exception
propagates.

**A/B power** (``simulated_power``) scores rejections, bias, RMSE, and the
share of intervals that contain the truth; ``leaderboard`` ranks the methods
of a registry by mean power across effect sizes at their calibrated size,
calibrated methods first.

``difference_in_differences_se`` is the exact standard error of the DiD
estimator under this DGP, for predicting power: the unit intercepts cancel
in each unit's pre-to-post change, the common shocks cancel in the
treated-minus-control contrast, and what remains is independent noise with
variance ``noise_sd² · (1 / n_pre + 1 / n_post)`` per unit change, so

    se = noise_sd · sqrt(1 / n_pre + 1 / n_post) · sqrt(1 / n_treated + 1 / n_control).

``unit_sd``, ``period_sd`` and ``rho`` do not enter.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

import numpy as np
import numpy.typing as npt
from pydantic import model_validator

from axiom.core import AcceptanceRegion, NonEmptyStr, Spec, Unsupported, clopper_pearson
from axiom.design.methods.registry import (
    METHODS,
    MethodEstimate,
    MethodSpec,
    PanelArrays,
    estimate,
    method_spec,
    panel_arrays,
)

__all__ = [
    "CalibrationResult",
    "Leaderboard",
    "LeaderboardRow",
    "PanelDesign",
    "SimulatedPower",
    "SimulationSpec",
    "calibrate_method",
    "calibrate_registry",
    "design_for_method",
    "difference_in_differences_se",
    "leaderboard",
    "simulate_panel",
    "simulated_power",
]

Array = npt.NDArray[np.float64]
PanelDesign = Literal["holdout", "switchback", "ghost"]

UNSUPPORTED_SHARE_LIMIT = 0.10
"""A method returning ``Unsupported`` on more than this share of panels fails calibration."""


# -- specs -----------------------------------------------------------------------------


class SimulationSpec(Spec):
    """One simulated panel: its shape, its variance components, its effect, and its budget.

    ``effect`` is the additive effect per treated unit per post period (``0``
    for an A/A run). ``on_fraction`` is the switchback on-probability per unit
    per period; ``exposure_rate`` the probability a unit is (ghost-)exposed.
    ``mass`` is the interval mass the estimators are asked for; an interval
    at mass ``1 − alpha`` excluding zero is a rejection at level ``alpha``.
    """

    n_units: int = 20
    n_periods: int = 16
    n_pre: int = 8
    n_treated: int = 10
    unit_sd: float = 1.0
    noise_sd: float = 1.0
    period_sd: float = 0.5
    rho: float = 0.5
    effect: float = 0.0
    n_simulations: int = 200
    seed: int = 0
    mass: float = 0.95
    on_fraction: float = 0.5
    exposure_rate: float = 0.6

    @model_validator(mode="after")
    def _valid(self) -> SimulationSpec:
        if self.n_units < 2:
            raise ValueError(f"n_units must be at least 2, got {self.n_units}")
        if self.n_periods < 2:
            raise ValueError(f"n_periods must be at least 2, got {self.n_periods}")
        if not 1 <= self.n_pre < self.n_periods:
            raise ValueError(f"n_pre must be in [1, n_periods), got {self.n_pre}")
        if not 1 <= self.n_treated < self.n_units:
            raise ValueError(
                f"n_treated must be in [1, n_units) so that controls exist, got {self.n_treated}"
            )
        for name in ("unit_sd", "period_sd"):
            v = getattr(self, name)
            if not (math.isfinite(v) and v >= 0):
                raise ValueError(f"{name} must be finite and non-negative, got {v}")
        if not (math.isfinite(self.noise_sd) and self.noise_sd > 0):
            raise ValueError(f"noise_sd must be finite and positive, got {self.noise_sd}")
        if not 0.0 <= self.rho < 1.0:
            raise ValueError(f"rho must be in [0, 1), got {self.rho}")
        if not math.isfinite(self.effect):
            raise ValueError(f"effect must be finite, got {self.effect}")
        if self.n_simulations < 1:
            raise ValueError(f"n_simulations must be positive, got {self.n_simulations}")
        if not 0.0 < self.mass < 1.0:
            raise ValueError(f"mass must be in (0, 1), got {self.mass}")
        if not 0.0 < self.on_fraction < 1.0:
            raise ValueError(f"on_fraction must be in (0, 1), got {self.on_fraction}")
        if not 0.0 < self.exposure_rate <= 1.0:
            raise ValueError(f"exposure_rate must be in (0, 1], got {self.exposure_rate}")
        return self

    @property
    def n_post(self) -> int:
        return self.n_periods - self.n_pre

    @property
    def n_control(self) -> int:
        return self.n_units - self.n_treated

    @property
    def alpha(self) -> float:
        return 1.0 - self.mass


class CalibrationResult(Spec):
    """A method's A/A false-positive count against its exact binomial acceptance region.

    ``n_evaluated = n_simulations − n_unsupported`` runs produced an
    estimate; ``false_positive_rate = false_positive_count / n_evaluated``.
    ``region`` is ``clopper_pearson(n_evaluated, alpha, 1e-3)``. ``passed``
    requires the count inside the region *and* ``n_unsupported`` at most ten
    percent of the runs; ``reason`` says which failed.
    """

    method: NonEmptyStr
    design: PanelDesign
    n_simulations: int
    n_unsupported: int
    false_positive_count: int
    false_positive_rate: float
    alpha: float
    region: AcceptanceRegion
    passed: bool
    reason: str = ""
    seed: int

    @model_validator(mode="after")
    def _valid(self) -> CalibrationResult:
        if not 0 <= self.n_unsupported <= self.n_simulations:
            raise ValueError("n_unsupported must lie in [0, n_simulations]")
        if not 0 <= self.false_positive_count <= self.n_simulations - self.n_unsupported:
            raise ValueError("false_positive_count must lie in [0, n_evaluated]")
        if not 0.0 < self.alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {self.alpha}")
        return self

    @property
    def n_evaluated(self) -> int:
        return self.n_simulations - self.n_unsupported


class SimulatedPower(Spec):
    """Realized A/B power and accuracy of a method at one effect size.

    ``power = rejections / n_evaluated``; ``bias = mean_effect − truth``;
    ``coverage`` is the share of intervals containing ``truth``. When a
    ``predicted_power`` is supplied, ``region`` is the binomial acceptance
    region for ``rejections`` under it and ``within_prediction`` says whether
    the realized count fell inside.
    """

    method: NonEmptyStr
    design: PanelDesign
    effect: float
    truth: float
    n_simulations: int
    n_unsupported: int
    rejections: int
    power: float
    alpha: float
    mean_effect: float
    bias: float
    coverage: float
    rmse: float
    predicted_power: float | None = None
    region: AcceptanceRegion | None = None
    within_prediction: bool | None = None
    seed: int

    @model_validator(mode="after")
    def _valid(self) -> SimulatedPower:
        if not 0 <= self.n_unsupported <= self.n_simulations:
            raise ValueError("n_unsupported must lie in [0, n_simulations]")
        if not 0 <= self.rejections <= self.n_simulations - self.n_unsupported:
            raise ValueError("rejections must lie in [0, n_evaluated]")
        if (self.region is None) != (self.predicted_power is None):
            raise ValueError("region is present iff predicted_power is")
        return self

    @property
    def n_evaluated(self) -> int:
        return self.n_simulations - self.n_unsupported


class LeaderboardRow(Spec):
    """One method's line: its calibrated size and its power, bias and coverage per effect."""

    method: NonEmptyStr
    design: PanelDesign
    status: Literal["stable", "experimental"]
    calibrated: bool
    false_positive_rate: float
    powers: tuple[float, ...]
    mean_power: float
    biases: tuple[float, ...]
    coverages: tuple[float, ...]
    rmses: tuple[float, ...]
    n_unsupported: int
    rank: int


class Leaderboard(Spec):
    """Methods ranked by mean power at a calibrated size; uncalibrated methods rank last.

    ``effects`` are the A/B effect sizes (columns of ``powers`` etc. in each
    row); ``rows`` are in rank order. ``calibrations`` and ``powers`` are the
    underlying results, one calibration per method and one ``SimulatedPower``
    per method per effect, in the same order as ``rows``.
    """

    spec: SimulationSpec
    alpha: float
    effects: tuple[float, ...]
    rows: tuple[LeaderboardRow, ...]
    calibrations: tuple[CalibrationResult, ...]
    powers: tuple[tuple[SimulatedPower, ...], ...]

    @model_validator(mode="after")
    def _valid(self) -> Leaderboard:
        if not self.effects:
            raise ValueError("a leaderboard needs at least one effect size")
        if len(self.calibrations) != len(self.rows) or len(self.powers) != len(self.rows):
            raise ValueError("calibrations and powers must align with rows")
        if any(len(p) != len(self.effects) for p in self.powers):
            raise ValueError("each method needs one SimulatedPower per effect")
        return self

    def row(self, method: str) -> LeaderboardRow:
        for r in self.rows:
            if r.method == method:
                return r
        raise KeyError(f"method {method!r} is not on the leaderboard")


# -- the panel -------------------------------------------------------------------------


@dataclass(frozen=True)
class _BaseDraw:
    base: Array
    treated: tuple[int, ...]
    assignment: Array
    exposed: Array


def _base_draw(spec: SimulationSpec, rng: np.random.Generator) -> _BaseDraw:
    n, t = spec.n_units, spec.n_periods
    mu = spec.unit_sd * rng.standard_normal(n)
    # Stationary AR(1): start from the marginal, innovate with sd sqrt(1 - rho^2) * period_sd.
    z = rng.standard_normal(t)
    s = np.empty(t, dtype=np.float64)
    s[0] = spec.period_sd * z[0]
    innov = math.sqrt(1.0 - spec.rho**2) * spec.period_sd
    for k in range(1, t):
        s[k] = spec.rho * s[k - 1] + innov * z[k]
    e = spec.noise_sd * rng.standard_normal((n, t))
    base = mu[:, None] + s[None, :] + e
    treated = tuple(int(i) for i in np.sort(rng.choice(n, size=spec.n_treated, replace=False)))
    assignment = (rng.random((n, t)) < spec.on_fraction).astype(np.float64)
    exposed = (rng.random(n) < spec.exposure_rate).astype(np.float64)
    return _BaseDraw(base=base, treated=treated, assignment=assignment, exposed=exposed)


def _treatment_indicator(spec: SimulationSpec, draw: _BaseDraw, design: PanelDesign) -> Array:
    n, t = spec.n_units, spec.n_periods
    if design == "switchback":
        return draw.assignment
    holdout = np.zeros((n, t), dtype=np.float64)
    holdout[list(draw.treated), spec.n_pre :] = 1.0
    if design == "ghost":
        return holdout * draw.exposed[:, None]
    if design == "holdout":
        return holdout
    raise ValueError(f"unknown design {design!r}")


def simulate_panel(
    spec: SimulationSpec, rng: np.random.Generator, *, design: PanelDesign = "holdout"
) -> PanelArrays:
    """Draw one panel from ``spec`` and apply ``spec.effect`` under ``design``.

    The outcome is ``base + effect · D`` with ``D`` the design's indicator
    (module docstring); ``treated``, the ``pre``/``post`` windows, the
    switchback ``assignment`` and the ``exposed`` mask are attached whatever
    the design, so any registered method can run on the result. The truth a
    method is scored against is ``spec.effect`` under its own design.
    """
    draw = _base_draw(spec, rng)
    d = _treatment_indicator(spec, draw, design)
    return panel_arrays(
        draw.base + spec.effect * d,
        draw.treated,
        slice(0, spec.n_pre),
        slice(spec.n_pre, spec.n_periods),
        assignment=draw.assignment,
        exposed=draw.exposed,
    )


def design_for_method(
    method: str | MethodSpec, registry: Mapping[str, MethodSpec] = METHODS
) -> PanelDesign:
    """The panel design a method's data requirements imply.

    No pre period and no controls is a within-unit time-randomized design
    (switchback); controls without a pre period is a post-period comparison
    among the exposed (ghost); anything needing a pre period is a holdout.
    """
    spec = method_spec(method, registry)
    if not spec.requires_pre_period and not spec.requires_controls:
        return "switchback"
    if not spec.requires_pre_period:
        return "ghost"
    return "holdout"


def difference_in_differences_se(spec: SimulationSpec) -> float:
    """Exact SE of the DiD estimator under the spec's DGP (derivation in the module docstring)."""
    per_unit_change = spec.noise_sd * math.sqrt(1.0 / spec.n_pre + 1.0 / spec.n_post)
    return per_unit_change * math.sqrt(1.0 / spec.n_treated + 1.0 / spec.n_control)


# -- running a method over many panels -------------------------------------------------


@dataclass(frozen=True)
class _Runs:
    estimates: tuple[MethodEstimate, ...]
    n_unsupported: int


def _run(
    method: str | MethodSpec,
    spec: SimulationSpec,
    registry: Mapping[str, MethodSpec],
    design: PanelDesign,
) -> _Runs:
    rng = np.random.default_rng(spec.seed)
    estimates: list[MethodEstimate] = []
    n_unsupported = 0
    for _ in range(spec.n_simulations):
        arrays = simulate_panel(spec, rng, design=design)
        # An estimator that raises is a bug in the estimator; it propagates (rule 5).
        out = estimate(method, arrays, mass=spec.mass, registry=registry)
        if isinstance(out, Unsupported):
            n_unsupported += 1
        else:
            estimates.append(out)
    return _Runs(estimates=tuple(estimates), n_unsupported=n_unsupported)


def _resolve_design(
    method: str | MethodSpec, registry: Mapping[str, MethodSpec], design: PanelDesign | None
) -> tuple[MethodSpec, PanelDesign]:
    spec = method_spec(method, registry)
    return spec, design if design is not None else design_for_method(spec, registry)


def calibrate_method(
    method: str | MethodSpec,
    spec: SimulationSpec,
    *,
    alpha: float = 0.05,
    registry: Mapping[str, MethodSpec] = METHODS,
    design: PanelDesign | None = None,
) -> CalibrationResult:
    """A/A calibration: ``spec`` with ``effect=0`` and intervals at mass ``1 − alpha``.

    ``design`` defaults to ``design_for_method``. The false-positive count
    is judged against ``clopper_pearson(n_evaluated, alpha, 1e-3)``.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    mspec, panel_design = _resolve_design(method, registry, design)
    aa = spec.model_copy(update={"effect": 0.0, "mass": 1.0 - alpha})
    runs = _run(mspec, aa, registry, panel_design)
    n_eval = len(runs.estimates)
    count = sum(1 for e in runs.estimates if e.excludes_zero())
    reasons: list[str] = []
    if n_eval == 0:
        region = clopper_pearson(spec.n_simulations, alpha, 1e-3)
        rate = float("nan")
        reasons.append("every run returned Unsupported")
    else:
        region = clopper_pearson(n_eval, alpha, 1e-3)
        rate = count / n_eval
        if not region.accepts(count):
            reasons.append(
                f"{count} false positives in {n_eval} A/A runs lies outside "
                f"[{region.lower}, {region.upper}] at nominal {alpha:g}"
            )
    if runs.n_unsupported > UNSUPPORTED_SHARE_LIMIT * spec.n_simulations:
        reasons.append(
            f"{runs.n_unsupported} of {spec.n_simulations} runs returned Unsupported "
            f"(limit {UNSUPPORTED_SHARE_LIMIT:.0%})"
        )
    return CalibrationResult(
        method=mspec.name,
        design=panel_design,
        n_simulations=spec.n_simulations,
        n_unsupported=runs.n_unsupported,
        false_positive_count=count,
        false_positive_rate=rate,
        alpha=alpha,
        region=region,
        passed=not reasons,
        reason="; ".join(reasons),
        seed=spec.seed,
    )


def simulated_power(
    method: str | MethodSpec,
    spec: SimulationSpec,
    *,
    registry: Mapping[str, MethodSpec] = METHODS,
    design: PanelDesign | None = None,
    predicted_power: float | None = None,
) -> SimulatedPower:
    """A/B simulation at ``spec.effect``: power, bias, RMSE and coverage at ``spec.mass``.

    Rejection is the interval at mass ``spec.mass`` excluding zero, so the
    test's level is ``1 − spec.mass``. ``predicted_power`` (e.g. from
    ``power.power_from_se``) adds a binomial acceptance region for the
    rejection count.
    """
    mspec, panel_design = _resolve_design(method, registry, design)
    runs = _run(mspec, spec, registry, panel_design)
    n_eval = len(runs.estimates)
    truth = spec.effect
    if n_eval == 0:
        raise ValueError(
            f"{mspec.name} returned Unsupported on all {spec.n_simulations} panels; "
            "no power can be simulated"
        )
    effects = np.asarray([e.effect for e in runs.estimates], dtype=np.float64)
    rejections = sum(1 for e in runs.estimates if e.excludes_zero())
    coverage = sum(1 for e in runs.estimates if e.interval.contains(truth)) / n_eval
    mean_effect = float(effects.mean())
    region: AcceptanceRegion | None = None
    within: bool | None = None
    if predicted_power is not None:
        if not 0.0 < predicted_power < 1.0:
            raise ValueError(f"predicted_power must be in (0, 1), got {predicted_power}")
        region = clopper_pearson(n_eval, predicted_power, 1e-3)
        within = region.accepts(rejections)
    return SimulatedPower(
        method=mspec.name,
        design=panel_design,
        effect=spec.effect,
        truth=truth,
        n_simulations=spec.n_simulations,
        n_unsupported=runs.n_unsupported,
        rejections=rejections,
        power=rejections / n_eval,
        alpha=spec.alpha,
        mean_effect=mean_effect,
        bias=mean_effect - truth,
        coverage=coverage,
        rmse=float(np.sqrt(np.mean((effects - truth) ** 2))),
        predicted_power=predicted_power,
        region=region,
        within_prediction=within,
        seed=spec.seed,
    )


# -- registry-level calibration and the leaderboard ------------------------------------


def calibrate_registry(
    spec: SimulationSpec,
    registry: Mapping[str, MethodSpec] = METHODS,
    *,
    alpha: float = 0.05,
) -> tuple[Mapping[str, MethodSpec], tuple[CalibrationResult, ...]]:
    """Calibrate every method in ``registry`` and return a new registry with statuses set.

    A method whose A/A count passes is ``"stable"``; one that fails is
    ``"experimental"``. The input registry (``METHODS`` by default) is not
    mutated; the result is a read-only mapping in the input's order.
    """
    results = tuple(
        calibrate_method(name, spec, alpha=alpha, registry=registry) for name in registry
    )
    updated = {
        name: registry[name].model_copy(update={"status": "stable" if r.passed else "experimental"})
        for name, r in zip(registry, results, strict=True)
    }
    return MappingProxyType(updated), results


def leaderboard(
    spec: SimulationSpec,
    effects: Sequence[float],
    registry: Mapping[str, MethodSpec] = METHODS,
    *,
    alpha: float = 0.05,
) -> Leaderboard:
    """Rank ``registry``'s methods by mean power over ``effects`` at the calibrated size.

    Each method is first A/A-calibrated at ``alpha``; power is then simulated
    at each effect with intervals at mass ``1 − alpha``. Calibrated methods
    rank above uncalibrated ones (a method's power at a size it does not
    hold is not comparable); within a group, higher mean power ranks first,
    ties broken by name.
    """
    grid = tuple(float(e) for e in effects)
    if not grid or any(not math.isfinite(e) or e == 0.0 for e in grid):
        raise ValueError("effects must be a non-empty sequence of finite non-zero values")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    calibrated_registry, calibrations = calibrate_registry(spec, registry, alpha=alpha)
    powers: list[tuple[SimulatedPower, ...]] = []
    for name in registry:
        per_effect = tuple(
            simulated_power(
                name,
                spec.model_copy(update={"effect": e, "mass": 1.0 - alpha}),
                registry=registry,
            )
            for e in grid
        )
        powers.append(per_effect)
    unranked: list[tuple[LeaderboardRow, CalibrationResult, tuple[SimulatedPower, ...]]] = []
    for name, cal, per_effect in zip(registry, calibrations, powers, strict=True):
        pw = tuple(p.power for p in per_effect)
        unranked.append(
            (
                LeaderboardRow(
                    method=name,
                    design=cal.design,
                    status=calibrated_registry[name].status,
                    calibrated=cal.passed,
                    false_positive_rate=cal.false_positive_rate,
                    powers=pw,
                    mean_power=float(np.mean(pw)),
                    biases=tuple(p.bias for p in per_effect),
                    coverages=tuple(p.coverage for p in per_effect),
                    rmses=tuple(p.rmse for p in per_effect),
                    n_unsupported=max(cal.n_unsupported, *(p.n_unsupported for p in per_effect)),
                    rank=0,
                ),
                cal,
                per_effect,
            )
        )
    unranked.sort(key=lambda item: (not item[0].calibrated, -item[0].mean_power, item[0].method))
    rows = tuple(row.model_copy(update={"rank": k + 1}) for k, (row, _, _) in enumerate(unranked))
    return Leaderboard(
        spec=spec,
        alpha=alpha,
        effects=grid,
        rows=rows,
        calibrations=tuple(c for _, c, _ in unranked),
        powers=tuple(p for _, _, p in unranked),
    )
