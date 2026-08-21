"""Realization: an ``Estimand`` against anything that ``SupportsEstimands``.

``realize`` turns a declared estimand into an ``EstimandResult`` — a
``Summary`` with a provenance-carrying ``Interval``, the status of its
identification, the assumptions it rests on, and the ledger of every
evidence transfer it took — or into a typed failure. It is written against
the core protocol, so a fitted surface, a stored analysis, and a hand-built
test double are realized by the same code.

What is computed, over posterior draws
--------------------------------------
Every functional is taken over the *aggregated* outcome ``agg(Y)`` and dose
``agg(X)`` that the ``window`` (basis) and ``level`` facets define
(decision 0002.20; the ``Quantity`` docstring is the reference). ``agg`` is
the window's basis (``cumulative``: sum over the window's periods;
``per_period``: mean), then the level (``individual``: a weighted mean over
units — uniform weights, the population's stratum weights, or
``unit_weights`` — the population-average individual effect; ``cluster`` /
``aggregate``: the sum over units, stratum weights rescaling the sum to the
declared composition; ``unit_weights`` are refused at these levels, a sum
has no weights). ``Y(iv)`` is the producer's posterior of the mean outcome
under ``iv``, ``X(iv)`` the dose it realized, ``M(iv)`` its closed-form
``∂Y/∂δ`` under a common shift ``δ`` of the doses in ``iv``'s support
(``marginal_under``), and ``1_S`` the indicator of that support.

* ``contrast`` — ``agg(Y(iv)) − agg(Y(ref))``; outcome dimension; scales
  with the basis (a cumulative contrast is ``T`` times a per-period one).
* ``ratio`` — ``(agg(Y(iv)) − agg(Y(ref))) / (agg(X(iv)) − agg(X(ref)))``:
  the outcome change per unit of dose change, both aggregated identically,
  so the ratio is *invariant* to the basis.
* ``marginal`` — ``agg(M(iv)) / agg(1_S)``: the derivative of the
  aggregated outcome with respect to the aggregated dose under a common
  shift of the support's doses, ``d agg(Y) / d agg(X)``. Basis-invariant;
  the ``ratio`` of two nearby interventions tends to it, which is what makes
  the chord-versus-marginal correction coherent.
* ``elasticity`` — ``marginal · agg(X(iv)) / agg(Y(iv))``, dimensionless
  and basis-invariant.
* ``area`` — ``∫₀¹ agg(Y(x(s))) · Δx̄ ds`` along the straight path
  ``x(s) = x_ref + s · (x_iv − x_ref)`` (``Δx̄`` the mean dose change per
  cell), by 16-node Gauss–Legendre quadrature; outcome × dose; scales with
  the basis. **Cost:** sixteen counterfactual predictions, each one tree
  evaluation per draw.

Units
-----
Intervention dose levels are in the **estimand's** unit. Before the producer
is asked, ``set`` / ``shift`` levels (and every node of the ``area`` path)
are converted to the producer's dose unit with ``UNITS.factor`` when both
units are known and differ (``scale`` levels are dimensionless); the
realized draws are converted back — outcome factor for a contrast, the dose
factor for the per-dose and dose-weighted quantities — and every conversion
is a ledger line. ``detail`` reports every dose (levels, differences, the
aggregated dose) in the estimand's unit. An unregistered conversion
raises ``UnitConversionError``, never a silent number. When either side's
unit is ``None`` no conversion is possible: the result carries the
unverified assumption ``units_assumed_equal`` and, because an identified
result cannot rest on an unverified assumption, is ``downgraded``.

Licensing
---------
Every interval is a ``core.Interval`` via ``core.summarize``; the result
holds no draws (``RealizedDraws`` carries them when asked). A producer
lacking a capability yields ``Unsupported`` naming it; an identification
verdict that is not ``identified`` yields ``Blocked`` unless
``assume_identified=True``, which proceeds as ``downgraded`` under an
asserted assumption with its ledger line. No verdict at all is never a
silent ``identified``: the result is ``downgraded`` under the unverified
assumption ``identification_not_checked``.

The window is the producer's business: when it declares ``TIME_WINDOW`` the
estimand's window is passed to ``predict_under`` / ``marginal_under`` and
the returned period axis must have the window's length; without the
capability the producer is asked for the whole horizon, which is accepted
only when the estimand's window *is* the whole horizon.

Not realized in this phase (typed ``Unsupported``, with the reason):
conditional estimands (Phase 6, with the transfer machinery), quantities
on the ``log`` scale, populations stratified on more than one covariate.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
from pydantic import model_validator

from axiom.core import (
    UNITS,
    Assumption,
    Blocked,
    Capability,
    Dimension,
    DimensionError,
    Intervention,
    LedgerLine,
    Outcome,
    PredictiveDraws,
    Spec,
    Summary,
    SupportsEstimands,
    TimeWindow,
    Treatment,
    Unsupported,
    Verdict,
    missing_capabilities,
    summarize,
)
from axiom.core.intervals import IntervalDefinition
from axiom.estimands.spec import Estimand, QuantityKind, derived_dimension

__all__ = [
    "QUADRATURE_NODES",
    "EstimandResult",
    "RealizedDraws",
    "ResultStatus",
    "evaluate",
    "realize",
]

Array = npt.NDArray[np.float64]
ResultStatus = Literal["identified", "downgraded", "blocked", "unsupported"]
QUADRATURE_NODES = 16
"""Gauss–Legendre nodes used for ``area``: as many counterfactual predictions per draw."""


# -- protocols the producer may additionally satisfy ----------------------------------------


@runtime_checkable
class _SupportsOutcome(Protocol):
    @property
    def outcome(self) -> Outcome: ...


@runtime_checkable
class _SupportsProvenance(Protocol):
    @property
    def provenance(self) -> Mapping[str, Any]: ...


# -- results -------------------------------------------------------------------------------


class EstimandResult(Spec):
    """A realized estimand: the summary, its dimension and unit, and how it was licensed.

    ``status`` is ``identified`` only when an identification verdict said
    so and no unverified assumption was needed; ``downgraded`` carries at
    least one named assumption (an unchecked identification or an unknown
    unit counts). ``ledger`` records every evidence transfer: unit
    conversions (of the intervention levels going in and of the draws
    coming out) and asserted identification. ``producer_hash`` is the
    producer's model hash when it records one, else a hash of its
    provenance. ``detail`` holds the dose levels (in the estimand's unit),
    the aggregation basis and level, and the panel sizes the number was
    computed with. No draws live here (``RealizedDraws``).
    """

    estimand_hash: str
    estimand_name: str
    kind: QuantityKind
    summary: Summary
    dimension: Dimension
    unit: str | None = None
    status: ResultStatus
    identification: Verdict | None = None
    assumptions: tuple[Assumption, ...] = ()
    ledger: tuple[LedgerLine, ...] = ()
    n_draws: int
    producer_hash: str = ""
    detail: dict[str, float | str] = {}

    @model_validator(mode="after")
    def _licensed_honestly(self) -> EstimandResult:
        if self.status == "identified":
            if self.identification is None or self.identification.status != "identified":
                raise ValueError(
                    "an 'identified' result needs an identification verdict that says so"
                )
            bad = [a.name for a in self.assumptions if a.state in ("unverified", "violated")]
            if bad:
                raise ValueError(
                    f"an 'identified' result cannot rest on unverified/violated assumptions: {bad}"
                )
        if self.status == "downgraded" and not self.assumptions:
            raise ValueError("a 'downgraded' result must name at least one assumption")
        return self


@dataclass(frozen=True)
class RealizedDraws:
    """``realize(..., keep_draws=True)``: the result plus the draws it summarizes, flat."""

    result: EstimandResult
    draws: Array


# -- internals -----------------------------------------------------------------------------


@dataclass(frozen=True)
class _Grid:
    """One producer answer, flattened: ``values`` is ``(n_draws, n_units, n_periods)``."""

    values: Array
    doses: dict[str, Array]
    labels: tuple[str, ...]

    @property
    def n_units(self) -> int:
        return int(self.values.shape[1])

    @property
    def n_periods(self) -> int:
        return int(self.values.shape[2])


@dataclass(frozen=True)
class _Facets:
    """The resolved aggregation: window basis, level, and unit weights summing to one."""

    basis: Literal["per_period", "cumulative"]
    level: Literal["individual", "cluster", "aggregate"]
    weights: Array

    def aggregate(self, grid: Array) -> Array:
        """``(..., n_units, n_periods) -> (...)``: basis over periods, then level over units."""
        per_unit = grid.sum(axis=-1) if self.basis == "cumulative" else grid.mean(axis=-1)
        w = self.weights if self.level == "individual" else self.weights * self.weights.size
        return np.asarray(per_unit @ w, dtype=np.float64)


@dataclass(frozen=True)
class _Licence:
    status: ResultStatus
    assumptions: tuple[Assumption, ...]
    ledger: tuple[LedgerLine, ...]


@dataclass(frozen=True)
class _Units:
    """How the estimand's units relate to the producer's, resolved once per realization.

    ``level_factor`` takes a dose level in the estimand's unit to the
    producer's (``None``: no conversion); ``dose_out`` / ``outcome_out``
    take the producer's realized quantities back to the estimand's units.
    ``assumed`` is the unverified ``units_assumed_equal`` assumption when a
    comparison was needed but one side's unit is unknown.
    """

    level_factor: float | None
    level_line: LedgerLine | None
    dose_back: float | None
    dose_out: tuple[float, LedgerLine] | None
    outcome_out: tuple[float, LedgerLine] | None
    assumed: Assumption | None

    def to_estimand_dose(self, dose: float) -> float:
        """A dose the producer realized, in the estimand's unit (``dose_back`` is the factor)."""
        return dose * self.dose_back if self.dose_back is not None else dose


def _to_grid(draws: PredictiveDraws, what: str) -> _Grid:
    values = np.asarray(draws.values, dtype=np.float64)
    if values.ndim != 4:
        raise ValueError(
            f"{what}: producer returned values of shape {values.shape}; realization needs "
            "(chain, draw, n_units, n_periods)"
        )
    chains, per_chain, n_units, n_periods = (int(s) for s in values.shape)
    flat = values.reshape(chains * per_chain, n_units, n_periods)
    coords = draws.coords
    labels = tuple(str(u) for u in coords["unit"]) if "unit" in coords else ()
    if labels and len(labels) != n_units:
        raise ValueError(f"{what}: producer reports {len(labels)} unit labels for {n_units} units")
    if not labels:
        labels = tuple(str(i) for i in range(n_units))
    doses: dict[str, Array] = {}
    for name in draws.intervention.doses:
        if name in coords:
            d = np.asarray(coords[name], dtype=np.float64).reshape(-1)
            if d.size != n_units * n_periods:
                raise ValueError(
                    f"{what}: producer reports {d.size} realized doses for {name!r}, "
                    f"expected {n_units * n_periods}"
                )
            doses[name] = d.reshape(n_units, n_periods)
    return _Grid(values=flat, doses=doses, labels=labels)


def _predict(
    producer: SupportsEstimands,
    iv: Intervention,
    window: TimeWindow | None,
    seed: int | None,
    what: str,
) -> _Grid | Unsupported:
    out: object = producer.predict_under(iv, window, seed)
    if isinstance(out, Unsupported):
        return out
    if not isinstance(out, PredictiveDraws):
        raise TypeError(f"{what}: predict_under returned {type(out).__name__}, not PredictiveDraws")
    return _to_grid(out, what)


def _marginal(
    producer: SupportsEstimands,
    iv: Intervention,
    treatment: str,
    window: TimeWindow | None,
    seed: int | None,
    what: str,
) -> _Grid | Unsupported:
    out: object = producer.marginal_under(iv, treatment, window, seed)
    if isinstance(out, Unsupported):
        return out
    if not isinstance(out, PredictiveDraws):
        raise TypeError(
            f"{what}: marginal_under returned {type(out).__name__}, not PredictiveDraws"
        )
    return _to_grid(out, what)


def _producer_treatment(producer: SupportsEstimands, estimand: Estimand) -> Treatment:
    name = estimand.treatment.name
    for t in producer.treatments:
        if t.name == name:
            return t
    raise ValueError(
        f"estimand {estimand.name!r} is about treatment {name!r}, which the producer does not "
        f"have; its treatments are {[t.name for t in producer.treatments]}"
    )


def _check_dimensions(
    estimand: Estimand, producer: SupportsEstimands, treatment: Treatment
) -> Dimension:
    """The dimension the producer's entities derive to; must equal the declaration."""
    dose_dim = treatment.dim
    if dose_dim != estimand.treatment.dim:
        raise DimensionError(
            f"estimand {estimand.name!r}: treatment {treatment.name!r} is declared in "
            f"{estimand.treatment.dim} but the producer fit it in {dose_dim}"
        )
    outcome_dim = estimand.outcome.dim
    if isinstance(producer, _SupportsOutcome):
        outcome_dim = producer.outcome.dim
        if outcome_dim != estimand.outcome.dim:
            raise DimensionError(
                f"estimand {estimand.name!r}: outcome {estimand.outcome.name!r} is declared in "
                f"{estimand.outcome.dim} but the producer fit {producer.outcome.name!r} in "
                f"{outcome_dim}"
            )
    derived = derived_dimension(estimand.quantity.kind, outcome_dim, dose_dim)
    if derived != estimand.dimension:
        raise DimensionError(
            f"estimand {estimand.name!r} declares dimension {estimand.dimension} but a "
            f"{estimand.quantity.kind} of {outcome_dim} over {dose_dim} derives to {derived}"
        )
    return derived


def _required(estimand: Estimand) -> set[Capability]:
    kind = estimand.quantity.kind
    required: set[Capability] = set()
    if kind in ("contrast", "ratio", "area", "elasticity"):
        required.add(Capability.COUNTERFACTUAL)
    if kind in ("marginal", "elasticity"):
        required.add(Capability.MARGINAL)
    if estimand.level.unit == "individual":
        required.add(Capability.PER_UNIT)
    return required


def _unsupported(estimand: Estimand, missing: Sequence[str], why: str) -> Unsupported:
    return Unsupported(
        reason=f"estimand {estimand.name!r} ({estimand.quantity.kind}) {why}",
        missing=tuple(missing),
        detail={"estimand": estimand.name, "estimand_hash": estimand.content_hash()},
    )


def _licence(
    estimand: Estimand, verdict: Verdict | None, assume_identified: bool
) -> _Licence | Blocked:
    if verdict is not None and verdict.status == "identified":
        return _Licence("identified", verdict.assumptions, ())
    assumptions: list[Assumption] = []
    ledger: list[LedgerLine] = []
    if verdict is None:
        assumptions.append(
            Assumption(
                name="identification_not_checked",
                facet="identification",
                statement=(
                    f"no identification verdict was supplied for {estimand.name!r}; the "
                    "quantity is reported as if identified, which has not been checked"
                ),
                challenged_by="identify.identify on the causal graph",
                state="unverified",
            )
        )
    else:
        assumptions.extend(verdict.assumptions)
        if not assume_identified:
            named = ", ".join(a.name for a in verdict.assumptions) or "none named"
            return Blocked(
                reason=(
                    f"identification verdict for {estimand.name!r} is {verdict.status!r}"
                    f"{': ' + verdict.reason if verdict.reason else ''} (assumptions: {named}); "
                    "pass assume_identified=True to proceed under an asserted assumption"
                ),
                detail={"verdict": verdict.status, "estimand": estimand.name},
            )
    if assume_identified:
        asserted = Assumption(
            name="assumed_identified",
            facet="identification",
            statement=f"the caller asserts that {estimand.name!r} is identified",
            challenged_by="the identification verdict it overrides",
            state="asserted",
            detail={"verdict": verdict.status if verdict is not None else "none"},
        )
        assumptions.append(asserted)
        ledger.append(
            LedgerLine(
                kind="assume_identified",
                statement=(
                    f"proceeded with {estimand.name!r} under assume_identified=True; the "
                    f"verdict was {verdict.status if verdict is not None else 'not supplied'}"
                ),
                assumption=asserted,
                detail={"verdict": verdict.status if verdict is not None else "none"},
                source=estimand.content_hash(),
            )
        )
    return _Licence("downgraded", tuple(assumptions), tuple(ledger))


def _unit_weights(
    estimand: Estimand,
    labels: Sequence[str],
    unit_weights: Mapping[str, float] | None,
    unit_strata: Mapping[str, str] | None,
) -> Array | Blocked | Unsupported:
    """Weights over units summing to one: explicit, from the population's strata, or uniform.

    ``unit_weights`` are a weighted mean's weights and so belong to the
    ``individual`` level only; at ``cluster`` / ``aggregate`` the level sums
    over units and weights are a ``ValueError``.
    """
    n = len(labels)
    strata = estimand.population.strata
    if unit_weights is not None and estimand.level.unit != "individual":
        raise ValueError(
            f"estimand {estimand.name!r} is at the {estimand.level.unit!r} level, which sums "
            "over units; unit_weights apply to the 'individual' level only"
        )
    if unit_weights is not None and strata:
        raise ValueError(
            f"estimand {estimand.name!r}: give unit_weights or a stratified population, not both"
        )
    if unit_weights is not None:
        missing = [u for u in labels if u not in unit_weights]
        if missing:
            raise ValueError(f"unit_weights lack entries for units {missing}")
        w = np.asarray([float(unit_weights[u]) for u in labels], dtype=np.float64)
        if np.any(w < 0) or not np.all(np.isfinite(w)) or w.sum() <= 0:
            raise ValueError("unit_weights must be finite, non-negative, and not all zero")
        return np.asarray(w / w.sum(), dtype=np.float64)
    if not strata:
        return np.full(n, 1.0 / n)
    if len(strata) > 1:
        return _unsupported(
            estimand,
            (),
            f"is stratified on {sorted(strata)}; populations stratified on more than one "
            "covariate are realized in Phase 6 with the transfer machinery",
        )
    ((covariate, shares),) = strata.items()
    if unit_strata is None:
        return Blocked(
            reason=(
                f"population {estimand.population.name!r} is stratified on {covariate!r} but "
                "unit_strata (unit -> stratum) was not given; the strata weights cannot be "
                "applied to units whose stratum is unknown"
            ),
            detail={"covariate": covariate, "estimand": estimand.name},
        )
    unknown_units = [u for u in labels if u not in unit_strata]
    if unknown_units:
        return Blocked(
            reason=(
                f"unit_strata has no stratum for units {unknown_units} "
                f"(covariate {covariate!r})"
            ),
            detail={"covariate": covariate, "estimand": estimand.name},
        )
    unknown_strata = sorted({unit_strata[u] for u in labels} - set(shares))
    if unknown_strata:
        return Blocked(
            reason=(
                f"units fall in strata {unknown_strata} that population "
                f"{estimand.population.name!r} does not declare for {covariate!r}"
            ),
            detail={"covariate": covariate, "estimand": estimand.name},
        )
    counts = {s: sum(1 for u in labels if unit_strata[u] == s) for s in shares}
    empty = sorted(s for s, c in counts.items() if c == 0)
    if empty:
        return Blocked(
            reason=(
                f"strata {empty} of {covariate!r} carry weight in population "
                f"{estimand.population.name!r} but no unit falls in them"
            ),
            detail={"covariate": covariate, "estimand": estimand.name},
        )
    w = np.asarray([shares[unit_strata[u]] / counts[unit_strata[u]] for u in labels])
    return np.asarray(w / w.sum(), dtype=np.float64)


def _check_window(estimand: Estimand, grid: _Grid, windowed: bool) -> Unsupported | None:
    """The producer's period axis must be the estimand's window (or, unwindowed, the horizon)."""
    window = estimand.window
    if windowed:
        if grid.n_periods != window.length:
            raise ValueError(
                f"producer returned {grid.n_periods} periods for window "
                f"[{window.start}, {window.stop}) of length {window.length}"
            )
        return None
    if window.stop > grid.n_periods:
        raise ValueError(
            f"estimand window [{window.start}, {window.stop}) runs past the producer's horizon "
            f"of {grid.n_periods} periods"
        )
    if (window.start, window.stop) != (0, grid.n_periods):
        return _unsupported(
            estimand,
            (Capability.TIME_WINDOW.value,),
            f"asks for window [{window.start}, {window.stop}) of a {grid.n_periods}-period "
            f"horizon, which needs the {Capability.TIME_WINDOW.value!r} capability",
        )
    return None


def _dose_grid(
    estimand: Estimand, grid: _Grid, iv: Intervention, treatment: str
) -> Array | Unsupported:
    """The dose grid ``iv`` realized for ``treatment`` (in the producer's unit): reported by
    the producer, else reconstructed for a ``set`` intervention over the whole horizon."""
    if treatment in grid.doses:
        return grid.doses[treatment]
    if iv.mode == "set" and iv.window is None:
        return np.full((grid.n_units, grid.n_periods), float(iv.doses[treatment]))
    return _unsupported(
        estimand,
        (),
        f"needs the dose the producer realized for {treatment!r} under a {iv.mode!r} "
        "intervention, which the producer does not report in its coords",
    )


def _support_grid(estimand: Estimand, grid: _Grid, iv: Intervention) -> Array:
    """``1`` on the cells of the reporting window that ``iv``'s support reaches, ``0`` else.

    The grid's period axis is the estimand's window (``_check_window``), so
    period ``j`` of the grid is panel period ``window.start + j``.
    """
    out = np.ones((grid.n_units, grid.n_periods))
    if iv.window is None:
        return out
    start = estimand.window.start
    for j in range(grid.n_periods):
        if not (iv.window.start <= start + j < iv.window.stop):
            out[:, j] = 0.0
    return out


def _gauss_legendre_unit(n: int) -> tuple[Array, Array]:
    """Nodes and weights on ``[0, 1]``."""
    x, w = np.polynomial.legendre.leggauss(n)
    return np.asarray((x + 1.0) / 2.0, dtype=np.float64), np.asarray(w / 2.0, dtype=np.float64)


def _path(iv: Intervention, ref: Intervention, s: float) -> Intervention:
    """The intervention ``s`` of the way from ``ref`` to ``iv`` (same mode, support, version)."""
    doses = {t: ref.doses[t] + s * (iv.doses[t] - ref.doses[t]) for t in iv.doses}
    return Intervention(doses=doses, mode=iv.mode, version=iv.version, window=iv.window)


def _unit_label(kind: QuantityKind, outcome_unit: str | None, dose_unit: str | None) -> str | None:
    match kind:
        case "contrast":
            return outcome_unit
        case "marginal" | "ratio":
            if outcome_unit is None or dose_unit is None:
                return None
            return f"{outcome_unit}/{dose_unit}"
        case "area":
            if outcome_unit is None or dose_unit is None:
                return None
            return f"{outcome_unit}·{dose_unit}"
        case "elasticity":
            return None
    raise ValueError(f"unknown quantity kind {kind!r}")  # pragma: no cover


def _conversion(src: str | None, dst: str | None) -> tuple[float, LedgerLine] | None:
    """The factor taking ``src`` to ``dst``, with its ledger line; ``None`` when nothing to do."""
    if src is None or dst is None or src == dst:
        return None
    return UNITS.convert(1.0, src, dst)


def _levels_in_dose_units(*interventions: Intervention | None) -> bool:
    """Whether any intervention carries a level in dose units (``set`` / ``shift``)."""
    return any(iv is not None and iv.mode in ("set", "shift") for iv in interventions)


def _resolve_units(estimand: Estimand, producer: SupportsEstimands, treatment: str) -> _Units:
    """Every unit comparison the estimand needs, made once; unknown units become an assumption."""
    kind = estimand.quantity.kind
    iv, ref = estimand.intervention, estimand.reference
    est_dose, prod_dose = estimand.treatment.unit, producer.dose_unit(treatment)
    est_out, prod_out = estimand.outcome.unit, producer.outcome_unit
    unresolved: dict[str, str] = {}

    level_factor: float | None = None
    level_line: LedgerLine | None = None
    dose_back: float | None = None
    dose_out: tuple[float, LedgerLine] | None = None
    outcome_out: tuple[float, LedgerLine] | None = None

    dose_needed = _levels_in_dose_units(iv, ref) or kind in ("marginal", "ratio", "area")
    if dose_needed and (est_dose is None or prod_dose is None):
        unresolved[f"treatment {treatment}"] = (
            f"estimand {est_dose or 'unknown'}, producer {prod_dose or 'unknown'}"
        )
    elif _levels_in_dose_units(iv, ref):
        conv = _conversion(est_dose, prod_dose)
        if conv is not None:
            level_factor, line = conv
            level_line = line.model_copy(
                update={
                    "statement": (
                        f"converted the intervention dose levels of {treatment!r} from "
                        f"{est_dose} (the estimand's unit) to {prod_dose} (the producer's) by "
                        f"factor {line.detail['factor']}"
                    ),
                    "detail": {**line.detail, "applies_to": "dose_levels"},
                }
            )
    # ``detail`` reports every dose figure in the estimand's unit, so the reporting conversion
    # is resolved whenever both units are known and a dose figure is reported — a scale-mode
    # elasticity crosses no unit boundary in its draws but still reports ``dose_aggregated``.
    reports_dose = kind in ("ratio", "elasticity", "area")
    if (dose_needed or reports_dose) and est_dose is not None and prod_dose is not None:
        back = _conversion(prod_dose, est_dose)
        dose_back = back[0] if back is not None else None
        # Only the per-dose and dose-weighted quantities carry the dose unit out.
        dose_out = back if kind in ("marginal", "ratio", "area") else None
    others = sorted(
        {
            t
            for x in (iv, ref)
            if x is not None and x.mode in ("set", "shift")
            for t in x.doses
            if t != treatment
        }
    )
    if others:
        named = ", ".join(repr(t) for t in others)
        unresolved["other treatments"] = (
            f"the levels of {named} are taken in the producer's units; the estimand declares a "
            f"unit for {treatment!r} only"
        )
    if kind != "elasticity":
        if est_out is None or prod_out is None:
            unresolved[f"outcome {estimand.outcome.name}"] = (
                f"estimand {est_out or 'unknown'}, producer {prod_out or 'unknown'}"
            )
        else:
            outcome_out = _conversion(prod_out, est_out)

    assumed: Assumption | None = None
    if unresolved:
        # Every key but the outcome's is about a treatment's dose unit (its own or another's).
        assumed = Assumption(
            name="units_assumed_equal",
            facet="outcome" if all(k.startswith("outcome") for k in unresolved) else "treatment",
            statement=(
                f"estimand {estimand.name!r}: a unit is unknown on one side, so the producer's "
                f"and the estimand's units are taken as equal without conversion: "
                + "; ".join(f"{k}: {v}" for k, v in unresolved.items())
            ),
            challenged_by="declaring units on both the estimand's entities and the producer's",
            state="unverified",
            detail=unresolved,
        )
    return _Units(level_factor, level_line, dose_back, dose_out, outcome_out, assumed)


def _in_producer_units(iv: Intervention, units: _Units, treatment: str) -> Intervention:
    """``iv`` with its ``set`` / ``shift`` level for ``treatment`` in the producer's dose unit."""
    if units.level_factor is None or iv.mode == "scale":
        return iv
    doses = dict(iv.doses)
    doses[treatment] = doses[treatment] * units.level_factor
    return Intervention(doses=doses, mode=iv.mode, version=iv.version, window=iv.window)


def _producer_hash(producer: SupportsEstimands) -> str:
    if not isinstance(producer, _SupportsProvenance):
        return ""
    prov = producer.provenance
    recorded = prov.get("model_hash")
    if isinstance(recorded, str) and recorded:
        return recorded
    text = json.dumps(dict(prov), sort_keys=True, default=str)
    return hashlib.blake2b(text.encode("utf-8"), digest_size=32).hexdigest()


# -- realize -------------------------------------------------------------------------------


def realize(
    estimand: Estimand,
    producer: SupportsEstimands,
    *,
    verdict: Verdict | None = None,
    assume_identified: bool = False,
    definition: IntervalDefinition = "hdi",
    mass: float = 0.9,
    seed: int | None = None,
    unit_weights: Mapping[str, float] | None = None,
    unit_strata: Mapping[str, str] | None = None,
    keep_draws: bool = False,
) -> EstimandResult | RealizedDraws | Unsupported | Blocked:
    """Realize one estimand against a producer; see the module docstring for the arithmetic.

    ``verdict`` is the identification verdict (``IdentificationVerdict.verdict``
    from ``identify``; ``estimands`` cannot import it). ``unit_weights`` maps
    unit label to weight for an ``individual``-level estimand (normalized;
    uniform when omitted; a ``ValueError`` at the ``cluster`` / ``aggregate``
    levels, which sum over units); ``unit_strata`` maps unit label to
    stratum when the population declares strata. ``keep_draws=True``
    returns ``RealizedDraws`` (result plus the flat draws).

    Raises ``DimensionError`` when the producer's entities do not derive to
    the declared dimension, ``UnitConversionError`` when a needed unit
    conversion is not registered, and ``ValueError`` for a treatment the
    producer does not have, a window past its horizon, coinciding doses in
    a ``ratio``, a ``marginal`` whose intervention support does not reach
    the window, or draws that are not finite.
    """
    kind = estimand.quantity.kind
    if estimand.quantity.scale != "natural":
        return _unsupported(
            estimand, (), f"is on the {estimand.quantity.scale!r} scale, not realized in this phase"
        )
    if estimand.conditioning:
        return _unsupported(
            estimand,
            (),
            "is conditional; conditional estimands are realized in Phase 6 with the transfer "
            "machinery",
        )
    treatment = _producer_treatment(producer, estimand)
    dimension = _check_dimensions(estimand, producer, treatment)
    missing = missing_capabilities(producer, _required(estimand))
    if missing:
        return _unsupported(
            estimand, missing, f"needs capabilities the producer lacks: {list(missing)}"
        )
    licence = _licence(estimand, verdict, assume_identified)
    if isinstance(licence, Blocked):
        return licence
    name = treatment.name
    units = _resolve_units(estimand, producer, name)

    windowed = Capability.TIME_WINDOW in producer.capabilities()
    window = estimand.window if windowed else None
    # Levels are declared in the estimand's unit; the producer is asked in its own.
    iv = _in_producer_units(estimand.intervention, units, name)
    ref = (
        _in_producer_units(estimand.reference, units, name)
        if estimand.reference is not None
        else None
    )
    detail: dict[str, float | str] = {
        "dose_iv": float(estimand.intervention.doses[name]),
        "intervention_mode": estimand.intervention.mode,
        "window_start": float(estimand.window.start),
        "window_stop": float(estimand.window.stop),
        "basis": estimand.window.basis,
        "level": estimand.level.unit,
    }
    if estimand.reference is not None:
        detail["dose_ref"] = float(estimand.reference.doses[name])
    if units.level_factor is not None:
        detail["level_factor"] = units.level_factor

    # The first answer fixes the panel's shape and labels; facets resolve against it.
    if kind == "marginal":
        first = _marginal(producer, iv, name, window, seed, f"{estimand.name}: marginal at iv")
    else:
        first = _predict(producer, iv, window, seed, f"{estimand.name}: outcome at iv")
    if isinstance(first, Unsupported):
        return first
    unsupported = _check_window(estimand, first, windowed)
    if unsupported is not None:
        return unsupported
    weights = _unit_weights(estimand, first.labels, unit_weights, unit_strata)
    if isinstance(weights, Blocked | Unsupported):
        return weights
    facets = _Facets(estimand.window.basis, estimand.level.unit, weights)
    detail["n_units"] = float(first.n_units)
    detail["n_periods"] = float(first.n_periods)
    ones = np.ones((first.n_units, first.n_periods))

    def dose_change(kind_: str) -> float:
        """``agg(1_S)``: the aggregated dose change per unit of common shift on ``iv``'s support."""
        reached = float(facets.aggregate(_support_grid(estimand, first, iv)))
        if reached == 0.0:
            raise ValueError(
                f"estimand {estimand.name!r}: the intervention's support "
                f"[{iv.window.start if iv.window else 0}, "
                f"{iv.window.stop if iv.window else first.n_periods}) does not reach the "
                f"window [{estimand.window.start}, {estimand.window.stop}); a {kind_} needs a "
                "dose change inside the window"
            )
        return reached

    draws: Array
    match kind:
        case "contrast" | "ratio":
            assert ref is not None
            at_ref = _predict(producer, ref, window, seed, f"{estimand.name}: outcome at ref")
            if isinstance(at_ref, Unsupported):
                return at_ref
            draws = facets.aggregate(first.values) - facets.aggregate(at_ref.values)
            if kind == "ratio":
                x_iv = _dose_grid(estimand, first, iv, name)
                x_ref = _dose_grid(estimand, at_ref, ref, name)
                if isinstance(x_iv, Unsupported):
                    return x_iv
                if isinstance(x_ref, Unsupported):
                    return x_ref
                delta = float(facets.aggregate(x_iv) - facets.aggregate(x_ref))
                if delta == 0.0:
                    raise ValueError(
                        f"estimand {estimand.name!r}: the intervention and reference realize the "
                        f"same aggregated dose for {name!r}; a ratio needs a dose difference"
                    )
                detail["dose_difference"] = units.to_estimand_dose(delta)
                draws = draws / delta
        case "marginal":
            draws = facets.aggregate(first.values) / dose_change("marginal")
        case "elasticity":
            m = _marginal(producer, iv, name, window, seed, f"{estimand.name}: marginal at iv")
            if isinstance(m, Unsupported):
                return m
            x_iv = _dose_grid(estimand, first, iv, name)
            if isinstance(x_iv, Unsupported):
                return x_iv
            marginal = facets.aggregate(m.values) / dose_change("elasticity")
            dose = float(facets.aggregate(x_iv))
            outcome = facets.aggregate(first.values)
            detail["dose_aggregated"] = units.to_estimand_dose(dose)
            with np.errstate(divide="ignore", invalid="ignore"):
                draws = np.asarray(marginal * dose / outcome, dtype=np.float64)
        case "area":
            assert ref is not None
            if iv.mode != ref.mode or set(iv.doses) != set(ref.doses) or iv.window != ref.window:
                return _unsupported(
                    estimand,
                    (),
                    "needs an intervention and a reference of the same mode, support, and "
                    "treatment set to integrate along a straight path",
                )
            x_iv = _dose_grid(estimand, first, iv, name)
            if isinstance(x_iv, Unsupported):
                return x_iv
            at_ref = _predict(producer, ref, window, seed, f"{estimand.name}: outcome at ref")
            if isinstance(at_ref, Unsupported):
                return at_ref
            x_ref = _dose_grid(estimand, at_ref, ref, name)
            if isinstance(x_ref, Unsupported):
                return x_ref
            delta_bar = float(
                (facets.aggregate(x_iv) - facets.aggregate(x_ref)) / facets.aggregate(ones)
            )
            nodes, node_weights = _gauss_legendre_unit(QUADRATURE_NODES)
            total = np.zeros(first.values.shape[0])
            for s, w in zip(nodes, node_weights, strict=True):
                at_s = _predict(
                    producer, _path(iv, ref, float(s)), window, seed, f"{estimand.name}: node"
                )
                if isinstance(at_s, Unsupported):
                    return at_s
                total = total + w * facets.aggregate(at_s.values)
            draws = np.asarray(total * delta_bar, dtype=np.float64)
            detail["dose_difference"] = units.to_estimand_dose(delta_bar)
            detail["n_nodes"] = float(QUADRATURE_NODES)

    # Units: the producer's to the estimand's, each conversion a ledger line.
    ledger = list(licence.ledger)
    if units.level_line is not None:
        ledger.append(units.level_line)
    if units.outcome_out is not None:
        factor, line = units.outcome_out
        draws = draws * factor
        detail["outcome_factor"] = factor
        ledger.append(line)
    if units.dose_out is not None:
        factor, line = units.dose_out
        draws = draws * factor if kind == "area" else draws / factor
        detail["dose_factor"] = factor
        ledger.append(line)
    status = licence.status
    assumptions = licence.assumptions
    if units.assumed is not None:
        assumptions = (*assumptions, units.assumed)
        status = "downgraded"

    bad = int(np.count_nonzero(~np.isfinite(draws)))
    if bad:
        raise ValueError(
            f"estimand {estimand.name!r}: {bad} of {draws.size} realized draws are not finite"
            + (
                "; the aggregated outcome is zero where the elasticity divides by it"
                if kind == "elasticity"
                else ""
            )
        )

    result = EstimandResult(
        estimand_hash=estimand.content_hash(),
        estimand_name=estimand.name,
        kind=kind,
        summary=summarize(draws, definition=definition, mass=mass),
        dimension=dimension,
        unit=_unit_label(kind, estimand.outcome.unit, estimand.treatment.unit),
        status=status,
        identification=verdict,
        assumptions=assumptions,
        ledger=tuple(ledger),
        n_draws=int(draws.size),
        producer_hash=_producer_hash(producer),
        detail=detail,
    )
    return RealizedDraws(result=result, draws=draws) if keep_draws else result


def evaluate(
    estimands: Sequence[Estimand],
    producer: SupportsEstimands,
    *,
    verdicts: Mapping[str, Verdict] | None = None,
    assume_identified: bool = False,
    definition: IntervalDefinition = "hdi",
    mass: float = 0.9,
    seed: int | None = None,
    unit_weights: Mapping[str, float] | None = None,
    unit_strata: Mapping[str, str] | None = None,
) -> dict[str, EstimandResult | Unsupported | Blocked]:
    """``realize`` each estimand, keyed by name; ``verdicts`` maps estimand name to its verdict.

    Estimand names must be distinct (``ValueError``). Every other argument
    is passed to ``realize`` unchanged; a typed failure for one estimand
    does not stop the others.
    """
    names = [e.name for e in estimands]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ValueError(f"estimand names must be distinct; duplicated: {dupes}")
    out: dict[str, EstimandResult | Unsupported | Blocked] = {}
    for estimand in estimands:
        realized = realize(
            estimand,
            producer,
            verdict=(verdicts or {}).get(estimand.name),
            assume_identified=assume_identified,
            definition=definition,
            mass=mass,
            seed=seed,
            unit_weights=unit_weights,
            unit_strata=unit_strata,
            keep_draws=False,
        )
        assert not isinstance(realized, RealizedDraws)
        out[estimand.name] = realized
    return out
