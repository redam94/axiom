"""The likelihood route: a randomized measurement as an in-graph soft constraint (D6.3).

The prior route (``calibrate.prior``) moves the amplitude's prior; this
route adds a term to the likelihood. A ``Measurement`` says "the estimand
``E`` was measured at ``estimate ± se`` in this experiment"; the surface
says what ``E`` *is* as a function of its parameters. ``constraint_for``
builds that function as an expression over the surface's mean tree —
``estimands.estimand_expr`` gives the per-row contrast (or ratio, or
marginal) at the intervention against the reference; the experiment's dose
paths replace the dose columns; ``Reduce`` nodes aggregate over the
window and the units to the functional decision 0002.20 defines — and
wraps it in a ``core.Constraint`` with ``family="normal"`` and ``scale =
se``. The backend then maximizes or samples ``log p(panel | θ) + log
N(estimate | E(θ), se) + log p(θ)`` through the one ``log_density``, so the
measurement pulls every parameter ``E`` depends on — the amplitude *and*
the curve shape — which is what distinguishes this route from the prior
route (gate D6.7 asserts exactly that).

Aggregation (0002.20). The per-row expression lives on the ``(n_units,
n_periods)`` grid the surface reads. A ``contrast`` is summed over the
window's periods for a ``cumulative`` basis and averaged for
``per_period``, then averaged over units at the ``individual`` level (the
population-average effect, uniform weights) and summed at the
``cluster`` / ``aggregate`` levels. A ``ratio`` is ``Δagg(Y) / Δagg(X)``;
with a row-constant dose difference (the only case ``estimand_expr``
expresses) that is the plain mean of the per-row ratio over the window
and the units, whatever the basis and level. A ``marginal`` without
carryover is ``d agg(Y) / d agg(X)`` under a common shift, likewise the
mean of the kernel's derivative over the cells. Everything else —
``area``, ``elasticity``, a marginal through carryover, strata weights,
conditioning — is a typed ``Unsupported`` naming ``estimands.realize``.

Log-scale measurements. When the estimand's ``quantity.scale`` is
``"log"`` the measurement's ``se`` is read as the uncertainty of a
positive quantity on the multiplicative scale: the expression is built
for the natural-scale quantity and the constraint is ``lognormal`` with
``observed = estimate`` and ``scale = lognormal_sigma_from_moments(estimate,
se)`` — the log-scale sigma of a lognormal with that mean and standard
deviation. ``lognormal_sigma_from_moments`` and ``lognormal_mu_from_moments``
are the parent's (``calibration/likelihood.py``, golden
``calibration.likelihood.*``).

Dose paths. ``doses`` maps a treatment name to the dose the experiment
applied: a scalar, a length-``n_periods`` path, or a grid whose rows are
identical. It replaces that treatment's ``Data`` column with a ``Const``
vector in the constraint — for the estimand's own treatment it is the
path the intervention and reference act on (``scale`` / ``shift``), for
other treatments it is their level during the experiment. A per-unit grid
is ``Unsupported``: a ``Const`` is one vector, and the backends lay out
only the columns the mean reads. Without ``doses`` the panel's observed
doses are used. A ``set`` intervention on a carryover surface is an
experiment that holds the dose at the level from the start of the
horizon: both arms are constant paths, and the window selects the
periods the measurement covers (ramp-up included).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from axiom.calibrate.evidence import Measurement
from axiom.core import (
    Add,
    Const,
    Constraint,
    Data,
    Div,
    Expr,
    ModelSpec,
    Mul,
    Posterior,
    Reduce,
    Unsupported,
    Unverified,
    data_names,
    dimensionless,
)
from axiom.data import Panel
from axiom.estimands import Estimand, estimand_expr, substitute
from axiom.infer import Backend, ConvergenceReport, diagnose, get_backend
from axiom.surface import FitResult, Surface, SurfaceSpec, prepare, resolve_conventions
from axiom.surface.forward import marginal_expr
from axiom.surface.model import DataDict

__all__ = [
    "attach",
    "constraint_for",
    "fit_calibrated",
    "lognormal_mu_from_moments",
    "lognormal_sigma_from_moments",
]

Array = npt.NDArray[np.float64]
DoseMap = Mapping[str, npt.ArrayLike]
_REALIZE = "realize with estimands.realize (the posterior check) or use the prior route"


# -- moments ------------------------------------------------------------------------------


def lognormal_sigma_from_moments(value: float, se: float) -> float:
    """``sqrt(log(1 + (se / value)^2))``: the log-scale sigma of a lognormal with that mean and sd.

    ``value`` must be positive and ``se`` non-negative and finite; otherwise
    ``ValueError``. Golden: ``(1.0, 0.5) -> 0.47238072707743883``.
    """
    if not (math.isfinite(value) and value > 0.0):
        raise ValueError(f"value must be finite and positive, got {value}")
    if not (math.isfinite(se) and se >= 0.0):
        raise ValueError(f"se must be finite and non-negative, got {se}")
    return math.sqrt(math.log1p((se / value) ** 2))


def lognormal_mu_from_moments(value: float, se: float) -> float:
    """``log(value) - sigma^2 / 2``: the log-scale location whose lognormal has mean ``value``."""
    sigma = lognormal_sigma_from_moments(value, se)
    return math.log(value) - 0.5 * sigma**2


# -- dose paths ---------------------------------------------------------------------------


def _dose_path(name: str, values: npt.ArrayLike, n_periods: int) -> tuple[float, ...] | Unsupported:
    """One dose path of length ``n_periods`` for every unit, or why it cannot be one."""
    a = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(a)):
        raise ValueError(f"doses for {name!r} must be finite")
    if np.any(a < 0.0):
        raise ValueError(f"doses for {name!r} must be non-negative")
    if a.ndim == 0:
        return (float(a),) * n_periods
    if a.ndim == 2:
        if a.shape[1] != n_periods:
            raise ValueError(
                f"doses for {name!r} have {a.shape[1]} periods; the panel has {n_periods}"
            )
        if not np.all(a == a[0]):
            return Unsupported(
                reason=(
                    f"doses for {name!r} differ between units; a constraint expression holds "
                    "one dose path as a Const vector (the backends lay out only the columns "
                    f"the mean reads) — pass a common path, or {_REALIZE}"
                ),
                detail={"treatment": name, "shape": str(a.shape)},
                missing=("per_unit_dose_grid",),
            )
        a = a[0]
    if a.ndim != 1:
        raise ValueError(f"doses for {name!r} must be a scalar, a path, or a grid; got {a.shape}")
    if a.shape[0] != n_periods:
        raise ValueError(f"doses for {name!r} have length {a.shape[0]}; the panel has {n_periods}")
    return tuple(float(x) for x in a)


def _grid_shape(surface: Surface, data: Mapping[str, Any]) -> tuple[int, int]:
    column = surface.spec.treatment_names[0]
    if column not in data:
        raise KeyError(f"data has no column {column!r}; pass what surface.prepare returned")
    a = np.asarray(data[column])
    if a.ndim != 2:
        raise ValueError(
            f"column {column!r} must be laid out (n_units, n_periods); got ndim={a.ndim}"
        )
    return int(a.shape[0]), int(a.shape[1])


# -- aggregation ----------------------------------------------------------------------------


def _num(v: float) -> Const:
    return Const(value=float(v), dimension=dimensionless())


def _on_grid(expr: Expr, column: Data) -> Expr:
    """``expr`` broadcast to the ``(n_units, n_periods)`` grid of ``column``.

    A contrast whose dose paths are constants has no time axis of its own
    (a single-treatment surface with a shared intercept gives a scalar
    per draw); multiplying by ``0 · column + 1`` — a ones grid with zero
    gradient — restores the grid so that a *sum* over units or periods
    counts every cell. ``column`` is a column the mean reads, so every
    backend lays it out.
    """
    scaled = Div(numerator=column, denominator=Const(value=1.0, dimension=column.dimension))
    ones = Add(terms=(Mul(factors=(_num(0.0), scaled)), _num(1.0)))
    return Mul(factors=(expr, ones))


def _window_mask(start: int, stop: int, n_periods: int) -> Const:
    mask = tuple(1.0 if start <= t < stop else 0.0 for t in range(n_periods))
    return Const(value=mask, dimension=dimensionless())


def _aggregate(
    expr: Expr,
    estimand: Estimand,
    *,
    column: Data,
    n_periods: int,
    periods: Literal["basis", "mean"],
    units: Literal["level", "mean"],
) -> Expr:
    """Reduce a per-cell expression to the estimand's aggregate functional (0002.20).

    ``periods="basis"`` follows the window's basis (sum for ``cumulative``,
    mean for ``per_period``) and ``units="level"`` the level (mean for
    ``individual``, sum otherwise); ``"mean"`` averages regardless — the
    ratio and the marginal, which are basis- and level-invariant.
    """
    window = estimand.window
    start, stop = window.start, window.stop
    cell = _on_grid(expr, column)
    full = start == 0 and stop == n_periods
    length = stop - start
    sum_over_window: Expr = (
        Reduce(op="sum", arg=cell)
        if full
        else Reduce(op="sum", arg=Mul(factors=(cell, _window_mask(start, stop, n_periods))))
    )
    take_mean = periods == "mean" or window.basis == "per_period"
    over_t: Expr
    if take_mean:
        over_t = (
            Reduce(op="mean", arg=cell)
            if full
            else Div(numerator=sum_over_window, denominator=_num(length))
        )
    else:
        over_t = sum_over_window
    if units == "mean" or estimand.level.unit == "individual":
        return Reduce(op="mean", arg=over_t)
    return Reduce(op="sum", arg=over_t)


def _constant_arms(
    mean: Expr, column: Data, estimand: Estimand, n_periods: int
) -> Expr | Unsupported:
    """Contrast / ratio arms for a ``set`` intervention as constant dose paths.

    ``estimand_expr`` refuses a scalar ``set`` on a carryover surface (a
    scalar collapses the time axis). An experiment that holds the dose at
    the level is a constant *path*, which the convolution accepts; both
    arms start from zero history at period 0 and the window picks the
    periods the measurement covers.
    """
    ref = estimand.reference
    assert ref is not None
    treatment = estimand.treatment.name
    hi, lo = float(estimand.intervention.doses[treatment]), float(ref.doses[treatment])
    if hi < 0.0 or lo < 0.0:
        raise ValueError(f"estimand {estimand.name!r}: set levels must be non-negative")
    path = {
        "iv": Const(value=(hi,) * n_periods, dimension=column.dimension),
        "ref": Const(value=(lo,) * n_periods, dimension=column.dimension),
    }
    contrast = Add(
        terms=(
            substitute(mean, {column.name: path["iv"]}),
            Mul(factors=(_num(-1.0), substitute(mean, {column.name: path["ref"]}))),
        )
    )
    if estimand.quantity.kind == "contrast":
        return contrast
    if hi == lo:
        raise ValueError(f"estimand {estimand.name!r}: the dose difference is zero")
    return Div(numerator=contrast, denominator=Const(value=hi - lo, dimension=column.dimension))


def _both_set(estimand: Estimand) -> bool:
    arms = [estimand.intervention] + ([estimand.reference] if estimand.reference else [])
    return all(iv.mode == "set" and iv.window is None for iv in arms)


def _check_units(measurement: Measurement, surface: Surface) -> Unsupported | None:
    estimand = measurement.estimand
    spec = surface.spec
    pairs = (
        ("outcome", estimand.outcome.unit, spec.outcome.unit),
        ("dose", estimand.treatment.unit, spec.treatment(estimand.treatment.name).unit),
    )
    for what, theirs, ours in pairs:
        if theirs is not None and ours is not None and theirs != ours:
            return Unsupported(
                reason=(
                    f"measurement {measurement.source!r}: estimand {estimand.name!r} declares "
                    f"{what} unit {theirs!r} but the surface was declared in {ours!r}; the "
                    "likelihood route does not convert units — declare the measurement in the "
                    f"surface's units, or {_REALIZE}"
                ),
                detail={"what": what, "estimand_unit": theirs, "surface_unit": ours},
                missing=("unit_conversion",),
            )
    return None


# -- the constraint -----------------------------------------------------------------------


def constraint_for(
    measurement: Measurement,
    surface: Surface,
    data: Mapping[str, npt.ArrayLike],
    *,
    doses: DoseMap | None = None,
) -> Constraint | Unsupported:
    """The soft constraint that says the surface predicts ``measurement`` at its experiment.

    ``data`` is what ``surface.prepare`` returned for the panel (it fixes
    the grid shape and, without ``doses``, the dose paths). The estimand's
    treatment must be one of the surface's and carry its unit; its
    intervention and reference must be expressible per row
    (``estimands.estimand_expr``) or be ``set`` levels; its window must lie
    within the horizon (``ValueError`` otherwise, as in ``realize``).
    Returns ``Unsupported`` — never an approximation — for an ``area``, an
    ``elasticity``, a ``marginal`` through carryover, strata weights,
    conditioning, a unit mismatch, or a per-unit dose grid.
    """
    estimand = measurement.estimand
    spec = surface.spec
    treatment = estimand.treatment.name
    name = estimand.name
    detail = {
        "estimand": name,
        "estimand_hash": estimand.content_hash(),
        "measurement": measurement.source,
        "quantity": estimand.quantity.kind,
    }
    if treatment not in spec.treatment_names:
        raise ValueError(
            f"estimand {name!r} treats {treatment!r}; the surface has {list(spec.treatment_names)}"
        )
    if (bad := _check_units(measurement, surface)) is not None:
        return bad
    n_units, n_periods = _grid_shape(surface, data)
    window = estimand.window
    if window.stop > n_periods:
        raise ValueError(
            f"estimand {name!r}: window [{window.start}, {window.stop}) runs past the panel's "
            f"horizon of {n_periods} periods"
        )
    if n_periods > 4096:
        return Unsupported(
            reason=f"the panel has {n_periods} periods; a Const path holds at most 4096",
            detail=detail,
        )
    if estimand.population.strata:
        return Unsupported(
            reason=(
                f"estimand {name!r} declares strata weights over "
                f"{sorted(estimand.population.strata)}; the likelihood route aggregates units "
                f"uniformly — {_REALIZE}"
            ),
            detail=detail,
            missing=("strata_weights",),
        )
    kind = estimand.quantity.kind
    if kind in ("area", "elasticity"):
        return Unsupported(
            reason=(
                f"estimand {name!r} is an {kind}: not an expression whose cell mean is the "
                f"aggregate functional (0002.20) — {_REALIZE}"
            ),
            detail=detail,
        )
    derivative: Expr | None = None
    if kind == "marginal":
        if treatment in spec.carried:
            return Unsupported(
                reason=(
                    f"estimand {name!r} is a marginal of {treatment!r}, which declares carryover "
                    f"{spec.carryover_of(treatment).name!r}; the windowed marginal through the "
                    f"carryover is not a per-row expression — {_REALIZE}"
                ),
                detail=detail,
                missing=("time_window",),
            )
        derivative = marginal_expr(surface, treatment)

    # The expression is built for the natural-scale quantity; a log-scale measurement
    # becomes a lognormal constraint on that same quantity (module docstring).
    natural = estimand
    if estimand.quantity.scale != "natural":
        natural = estimand.model_copy(
            update={"quantity": estimand.quantity.model_copy(update={"scale": "natural"})}
        )

    column = Data(name=treatment, dimension=spec.dose_dimension(treatment))
    mean = surface.expr
    per_row = estimand_expr(natural, mean=mean, treatment_data=column, derivative=derivative)
    if isinstance(per_row, Unsupported):
        if (
            "time_window" in per_row.missing
            and kind in ("contrast", "ratio")
            and _both_set(natural)
        ):
            per_row = _constant_arms(mean, column, natural, n_periods)
        else:
            return per_row
    if isinstance(per_row, Unsupported):
        return per_row

    # The experiment's dose paths replace the dose columns.
    replacements: dict[str, Expr] = {}
    for other, values in (doses or {}).items():
        if other not in spec.treatment_names:
            raise KeyError(f"doses name {other!r}, not a treatment of {spec.name!r}")
        path = _dose_path(other, values, n_periods)
        if isinstance(path, Unsupported):
            return path
        replacements[other] = Const(value=path, dimension=spec.dose_dimension(other))
    expr = substitute(per_row, replacements) if replacements else per_row
    expr = _aggregate(
        expr,
        natural,
        column=column,
        n_periods=n_periods,
        periods="basis" if kind == "contrast" else "mean",
        units="level" if kind == "contrast" else "mean",
    )
    extra = sorted(set(data_names(expr)) - set(data_names(mean)))
    if extra:  # pragma: no cover - every column comes from the mean by construction
        raise RuntimeError(f"constraint reads columns the mean does not: {extra}")

    detail.update(
        {
            "doses": "supplied:" + ",".join(sorted(replacements)) if replacements else "observed",
            "window": f"[{window.start},{window.stop}):{window.basis}",
            "level": estimand.level.unit,
            "n_units": str(n_units),
            "n_periods": str(n_periods),
        }
    )
    label = f"{name}@{measurement.source}"
    if estimand.quantity.scale == "log":
        if measurement.estimate <= 0.0:
            return Unsupported(
                reason=(
                    f"measurement {measurement.source!r} of log-scale estimand {name!r} has a "
                    f"non-positive estimate {measurement.estimate}; a lognormal constraint needs "
                    "a positive observed value"
                ),
                detail=detail,
            )
        sigma = lognormal_sigma_from_moments(measurement.estimate, measurement.se)
        detail["se"] = repr(measurement.se)
        return Constraint(
            name=label,
            expr=expr,
            family="lognormal",
            observed=measurement.estimate,
            scale=sigma,
            detail=detail,
        )
    return Constraint(
        name=label,
        expr=expr,
        family="normal",
        observed=measurement.estimate,
        scale=measurement.se,
        detail=detail,
    )


def _with_constraints(model: ModelSpec, constraints: Sequence[Constraint]) -> ModelSpec:
    """A re-validated ``ModelSpec`` with ``constraints`` appended."""
    return ModelSpec(
        name=model.name,
        mean=model.mean,
        outcome=model.outcome,
        likelihood=model.likelihood,
        parameters=model.parameters,
        constraints=(*model.constraints, *constraints),
    )


def attach(
    measurement: Measurement,
    surface: Surface,
    data: Mapping[str, npt.ArrayLike],
    *,
    doses: DoseMap | None = None,
) -> ModelSpec | Unsupported:
    """``surface.model`` with the measurement's constraint appended, or why it cannot be."""
    constraint = constraint_for(measurement, surface, data, doses=doses)
    if isinstance(constraint, Unsupported):
        return constraint
    return _with_constraints(surface.model, (constraint,))


def fit_calibrated(
    spec: SurfaceSpec,
    panel: Panel,
    measurements: Sequence[Measurement],
    *,
    backend: Backend | str = "laplace",
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    seed: int | None = None,
    doses: DoseMap | None = None,
) -> FitResult | Unsupported:
    """``surface.fit`` with every measurement attached as a soft constraint.

    The same steps as ``surface.fit`` — ``resolve_conventions``, ``prepare``,
    the backend by name or instance, ``diagnose`` for sampled draws — run
    on the constrained ``ModelSpec``. The returned ``FitResult`` holds the
    unconstrained ``Surface`` (its ``model`` is the mean the predictions
    use; the constraint only shapes the posterior); ``provenance`` records
    ``constrained_model_hash``, one entry per constraint under
    ``constraints`` (name, hash, family, observed, scale) and the
    measurements' sources and hashes. A measurement that cannot be
    expressed returns its ``Unsupported`` before anything is fit.
    """
    surface = Surface(spec)
    conventions = resolve_conventions(spec, panel)
    data: DataDict = prepare(spec, panel, conventions=conventions)
    constraints: list[Constraint] = []
    for m in measurements:
        c = constraint_for(m, surface, data, doses=doses)
        if isinstance(c, Unsupported):
            return c
        constraints.append(c)
    model = _with_constraints(surface.model, constraints)
    completeness = panel.completeness()
    provenance: dict[str, Any] = {
        "spec_hash": spec.content_hash(),
        "model_hash": surface.model.content_hash(),
        "constrained_model_hash": model.content_hash(),
        "constraints": [
            {
                "name": c.name,
                "hash": c.content_hash(),
                "family": c.family,
                "observed": c.observed,
                "scale": c.scale,
            }
            for c in constraints
        ],
        "measurement_sources": [m.source for m in measurements],
        "measurement_hashes": [m.content_hash() for m in measurements],
        "panel_hash": panel.content_hash(),
        "n_units": completeness.n_units,
        "n_periods": completeness.n_periods,
        "unit_labels": list(spec.unit_labels or panel.units),
        "draws": draws,
        "tune": tune,
        "chains": chains,
        "seed": seed,
        "nuisance_conventions": conventions,
        "route": "likelihood",
    }
    if isinstance(backend, str):
        resolved = get_backend(backend)
        if isinstance(resolved, Unsupported):
            provenance["backend"] = backend
            return FitResult(surface, resolved, data, None, provenance)
        backend = resolved
    provenance["backend"] = backend.name
    posterior: Posterior | Unverified | Unsupported = backend.sample(
        model, data, draws=draws, tune=tune, chains=chains, seed=seed
    )
    report: ConvergenceReport | None = None
    if isinstance(posterior, Posterior) and posterior.provenance.get("method") != "laplace":
        report = diagnose(posterior)
    return FitResult(surface, posterior, data, report, provenance)
