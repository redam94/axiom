"""Specification curve: realize one estimand under every combination of analyst choices.

Port of the parent's ``validation/spec_curve.py`` (ledger row "diagnose/spec_curve"):
the *multiverse* / researcher-degrees-of-freedom idea of Simonsohn, Simmons &
Nelson (2020). An analyst makes choices that the data do not force — which
saturation kernel, whether to model carryover, what intercept structure,
which nuisance baseline — and a single reported number hides how much the
conclusion depends on them. A specification curve makes every combination
explicit: one ``SpecificationAxis`` per choice, one ``SpecOption`` per
alternative, the cartesian product fitted, and the same estimand realized
under each, so the reader sees the *distribution* of the estimate across the
choices rather than one point.

Each option is a declarative change to the base ``SurfaceSpec`` (the fields
to override) and / or to the fit call (backend, draws). Options are data,
not callables, so a curve is itself a ``Spec`` that serializes and hashes
(rule 4). Per-treatment mapping fields (``kernels``, ``carryover``) are
merged with the base's mapping so an option can change one treatment's
kernel; every other field is replaced wholesale. An option may not change
the treatments or the outcome — the estimand must remain realizable on every
branch of the curve.

Every row is realized through ``axiom.estimands.realize`` against the
``FitResult`` (one forward, rule 3); the point is the posterior mean and the
interval carries its definition and mass. A refit that fails — the backend
returned a typed failure, or the estimand could not be realized — is a row
with ``failure`` set, never a silently dropped specification (rule 5). The
product is capped at ``max_specs`` in product order and the number dropped
is reported.
"""

from __future__ import annotations

import itertools
import logging
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from pydantic import Field, field_validator, model_validator

from axiom.core import (
    Interval,
    Intervention,
    NonEmptyStr,
    Population,
    Posterior,
    Spec,
    TimeWindow,
    Unsupported,
    is_failure,
)
from axiom.core.intervals import IntervalDefinition
from axiom.data import Panel
from axiom.estimands import (
    Estimand,
    Level,
    Quantity,
    RealizedDraws,
    derived_dimension,
    realize,
)
from axiom.surface import FitResult, SurfaceSpec, fit

__all__ = [
    "FitSettings",
    "SpecCurve",
    "SpecCurveSummary",
    "SpecOption",
    "SpecRow",
    "SpecificationAxis",
    "apply_option",
    "default_estimand",
    "realized_point",
    "specification_curve",
]

log = logging.getLogger(__name__)

_MERGED_FIELDS = ("kernels", "carryover")
"""Per-treatment mapping fields of ``SurfaceSpec`` that an option merges rather than replaces."""

_FIXED_FIELDS = ("treatments", "outcome")
"""Fields an option may not change: the estimand must stay realizable on every branch."""


# -- axes --------------------------------------------------------------------------------


def _jsonable(value: Any) -> Any:
    """``Spec`` instances (possibly nested in dicts / sequences) as their JSON payloads."""
    if isinstance(value, Spec):
        return value.to_dict()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    return value


class FitSettings(Spec):
    """How each specification is fitted: backend name and sampler sizes."""

    backend: NonEmptyStr = "laplace"
    draws: int = Field(default=200, ge=1)
    tune: int = Field(default=200, ge=0)
    chains: int = Field(default=1, ge=1)


class SpecOption(Spec):
    """One alternative on an axis: a label plus the spec fields and fit settings it changes.

    ``spec_update`` holds ``SurfaceSpec`` fields to override, as JSON
    payloads (``Spec`` values are converted on construction). ``fit_update``
    holds ``FitSettings`` fields to override.
    """

    label: NonEmptyStr
    spec_update: dict[str, Any] = {}
    fit_update: dict[str, int | str] = {}

    @field_validator("spec_update", mode="before")
    @classmethod
    def _payloads(cls, v: Any) -> Any:
        if isinstance(v, Mapping):
            return {str(k): _jsonable(val) for k, val in v.items()}
        return v

    @model_validator(mode="after")
    def _known_fields(self) -> SpecOption:
        unknown = sorted(set(self.spec_update) - set(SurfaceSpec.model_fields))
        if unknown:
            raise ValueError(f"spec_update names fields SurfaceSpec does not have: {unknown}")
        fixed = sorted(set(self.spec_update) & set(_FIXED_FIELDS))
        if fixed:
            raise ValueError(
                f"an option may not change {fixed}; the estimand must remain realizable on "
                "every specification"
            )
        unknown_fit = sorted(set(self.fit_update) - set(FitSettings.model_fields))
        if unknown_fit:
            raise ValueError(f"fit_update names fields FitSettings does not have: {unknown_fit}")
        return self


class SpecificationAxis(Spec):
    """One analyst choice: a name and at least one labelled option."""

    name: NonEmptyStr
    options: tuple[SpecOption, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _distinct_labels(self) -> SpecificationAxis:
        labels = [o.label for o in self.options]
        dupes = sorted({x for x in labels if labels.count(x) > 1})
        if dupes:
            raise ValueError(f"axis {self.name!r} has duplicate option labels: {dupes}")
        return self


def apply_option(
    base: SurfaceSpec, settings: FitSettings, option: SpecOption
) -> tuple[SurfaceSpec, FitSettings]:
    """``base`` and ``settings`` with one option applied (a fresh, validated ``SurfaceSpec``).

    Mapping fields (``kernels``, ``carryover``) are merged per treatment;
    every other field is replaced. The result is re-validated, so an option
    that produces an inconsistent spec raises ``ValueError`` here, before
    any fitting.
    """
    payload = base.to_dict()
    for key, value in option.spec_update.items():
        if key in _MERGED_FIELDS and isinstance(value, Mapping):
            payload[key] = {**payload.get(key, {}), **value}
        else:
            payload[key] = value
    spec = SurfaceSpec.from_dict(payload)
    fit_settings = FitSettings.from_dict({**settings.to_dict(), **option.fit_update})
    return spec, fit_settings


# -- the estimand and its realization -------------------------------------------------------


def default_estimand(
    spec: SurfaceSpec,
    panel: Panel,
    treatment: str | None = None,
    *,
    dose: float | None = None,
) -> Estimand:
    """The contrast ``E[Y(dose)] − E[Y(0)]`` of one treatment: the per-unit average
    (``Level(unit="individual")``, uniform unit weights), cumulative over the whole horizon.

    The per-unit average — not the aggregate sum — so the number is
    invariant to how many units a refit sees (``refute.random_subset``
    fits subsets of the units and must compare like with like).

    ``treatment`` defaults to the spec's first; ``dose`` to the mean of the
    treatment's non-zero observed doses (``ValueError`` when every observed
    dose is zero and no dose is given). This is the estimand every
    refutation and specification curve uses when none is declared.
    """
    name = treatment if treatment is not None else spec.treatment_names[0]
    entity = spec.treatment(name)
    if dose is None:
        observed = np.asarray(panel.array(name), dtype=np.float64)
        positive = observed[np.isfinite(observed) & (observed > 0.0)]
        if positive.size == 0:
            raise ValueError(
                f"every observed dose of {name!r} is zero; pass `dose` for the contrast"
            )
        dose = float(positive.mean())
    if not (np.isfinite(dose) and dose > 0.0):
        raise ValueError(f"dose must be positive and finite, got {dose}")
    n_periods = len(panel.periods)
    if entity.dimension is None or spec.outcome.dimension is None:
        raise ValueError("the treatment and the outcome must carry dimensions")
    return Estimand(
        name=f"contrast_{name}",
        quantity=Quantity(kind="contrast"),
        treatment=entity,
        intervention=Intervention(doses={name: dose}, mode="set"),
        reference=Intervention(doses={name: 0.0}, mode="set"),
        outcome=spec.outcome,
        population=Population(name="all"),
        window=TimeWindow(start=0, stop=n_periods, basis="cumulative"),
        level=Level(unit="individual"),
        dimension=derived_dimension("contrast", spec.outcome.dimension, entity.dimension),
        description=(
            f"per-unit average cumulative contrast of {name!r} at dose {dose:g} versus zero "
            f"over all {n_periods} periods"
        ),
    )


def realized_point(
    estimand: Estimand,
    result: FitResult,
    *,
    definition: IntervalDefinition,
    mass: float,
    seed: int | None = None,
) -> RealizedDraws | Unsupported:
    """``realize(..., keep_draws=True)`` with every non-result outcome as ``Unsupported``.

    A fit without a posterior, a ``Blocked`` licence, or an ``Unsupported``
    realization all come back as one typed failure naming the cause so a
    caller building a table of estimates records the row as failed.
    """
    if not isinstance(result.posterior, Posterior):
        failure = result.posterior
        return Unsupported(
            reason=f"fit has no posterior ({type(failure).__name__}): {failure.reason}",
            missing=("posterior",),
            detail={"spec": result.surface.spec.name},
        )
    out = realize(estimand, result, definition=definition, mass=mass, seed=seed, keep_draws=True)
    if isinstance(out, RealizedDraws):
        return out
    if is_failure(out):
        return Unsupported(
            reason=f"{type(out).__name__} while realizing {estimand.name!r}: {out.reason}",
            detail={"estimand": estimand.name, **out.detail},
        )
    raise TypeError(  # pragma: no cover - realize's keep_draws contract
        f"realize returned {type(out).__name__} with keep_draws=True"
    )


# -- the curve -----------------------------------------------------------------------------


class SpecRow(Spec):
    """One fitted specification: its option labels, the realized point, and its fate."""

    index: int = Field(ge=0)
    labels: dict[str, str]
    spec_hash: str
    spec_name: str
    estimate: float | None = None
    interval: Interval | None = None
    sd: float | None = None
    n_draws: int = Field(default=0, ge=0)
    converged: bool = False
    failure: str | None = None

    @model_validator(mode="after")
    def _point_or_failure(self) -> SpecRow:
        has_point = self.estimate is not None and self.interval is not None
        if has_point == (self.failure is not None):
            raise ValueError("a row carries either an estimate with its interval or a failure")
        return self

    @property
    def excludes_zero(self) -> bool | None:
        """Whether the interval excludes zero (``None`` for a failed row)."""
        if self.interval is None:
            return None
        return self.interval.lower > 0.0 or self.interval.upper < 0.0


class SpecCurveSummary(Spec):
    """Median / IQR of the estimates and the share of specifications excluding zero."""

    n: int = Field(ge=0)
    n_failed: int = Field(ge=0)
    n_converged: int = Field(ge=0)
    median: float | None = None
    iqr_lower: float | None = None
    iqr_upper: float | None = None
    minimum: float | None = None
    maximum: float | None = None
    share_excluding_zero: float | None = None
    share_positive: float | None = None


class SpecCurve(Spec):
    """Every specification's realized estimand, the cap applied, and the estimand realized.

    ``n_total`` is the size of the cartesian product; ``n_dropped`` how many
    the ``max_specs`` cap left unfitted (taken in product order: the last
    axis varies fastest).
    """

    estimand_name: str
    estimand_hash: str
    base_spec_hash: str
    axes: tuple[SpecificationAxis, ...]
    rows: tuple[SpecRow, ...]
    n_total: int = Field(ge=1)
    n_dropped: int = Field(ge=0)
    max_specs: int = Field(ge=1)
    definition: IntervalDefinition
    mass: float = Field(gt=0, lt=1)
    settings: FitSettings
    seed: int | None = None

    @model_validator(mode="after")
    def _counts(self) -> SpecCurve:
        if len(self.rows) + self.n_dropped != self.n_total:
            raise ValueError(
                f"rows ({len(self.rows)}) + n_dropped ({self.n_dropped}) must equal "
                f"n_total ({self.n_total})"
            )
        return self

    @property
    def estimates(self) -> tuple[float, ...]:
        """Point estimates of the rows that produced one, in row order."""
        return tuple(r.estimate for r in self.rows if r.estimate is not None)

    def summary(self) -> SpecCurveSummary:
        """Median, interquartile range, extremes, share of rows whose interval excludes zero."""
        points = np.asarray(self.estimates, dtype=np.float64)
        n_failed = sum(1 for r in self.rows if r.failure is not None)
        n_converged = sum(1 for r in self.rows if r.converged)
        if points.size == 0:
            return SpecCurveSummary(n=len(self.rows), n_failed=n_failed, n_converged=n_converged)
        q1, med, q3 = (float(x) for x in np.quantile(points, [0.25, 0.5, 0.75]))
        excluding = [r.excludes_zero for r in self.rows if r.excludes_zero is not None]
        return SpecCurveSummary(
            n=len(self.rows),
            n_failed=n_failed,
            n_converged=n_converged,
            median=med,
            iqr_lower=q1,
            iqr_upper=q3,
            minimum=float(points.min()),
            maximum=float(points.max()),
            share_excluding_zero=float(np.mean(excluding)) if excluding else None,
            share_positive=float(np.mean(points > 0.0)),
        )


def _combinations(
    axes: Sequence[SpecificationAxis],
) -> list[tuple[tuple[str, SpecOption], ...]]:
    return list(itertools.product(*[[(a.name, o) for o in a.options] for a in axes]))


def specification_curve(
    base: SurfaceSpec,
    panel: Panel,
    axes: Sequence[SpecificationAxis],
    *,
    estimand: Estimand | None = None,
    backend: str = "laplace",
    draws: int = 200,
    tune: int = 200,
    chains: int = 1,
    seed: int | None = 0,
    max_specs: int = 64,
    definition: IntervalDefinition = "eti",
    mass: float = 0.9,
) -> SpecCurve:
    """Fit every combination of the axes' options and realize ``estimand`` under each.

    ``estimand`` defaults to ``default_estimand(base, panel)``. Each
    specification is fitted with ``axiom.surface.fit`` (``backend``,
    ``draws``, ``tune``, ``chains``, overridable per option) and seeded
    ``seed + index`` so rows are reproducible one at a time. The cartesian
    product is capped at ``max_specs`` (``ValueError`` if below 1) and the
    drop is reported on the curve. Axis names must be distinct.
    """
    if not axes:
        raise ValueError("a specification curve needs at least one axis")
    names = [a.name for a in axes]
    if len(set(names)) != len(names):
        raise ValueError(f"axis names must be distinct: {names}")
    if max_specs < 1:
        raise ValueError(f"max_specs must be at least 1, got {max_specs}")
    settings = FitSettings(backend=backend, draws=draws, tune=tune, chains=chains)
    target = estimand if estimand is not None else default_estimand(base, panel)
    combos = _combinations(axes)
    n_total = len(combos)
    kept = combos[:max_specs]
    if len(kept) < n_total:
        log.warning(
            "specification_curve: %d of %d specifications dropped by max_specs=%d",
            n_total - len(kept),
            n_total,
            max_specs,
        )
    rows: list[SpecRow] = []
    for index, combo in enumerate(kept):
        spec, fit_settings = base, settings
        for _, option in combo:
            spec, fit_settings = apply_option(spec, fit_settings, option)
        labels = {axis: option.label for axis, option in combo}
        row_seed = None if seed is None else seed + index
        result = fit(
            spec,
            panel,
            backend=fit_settings.backend,
            draws=fit_settings.draws,
            tune=fit_settings.tune,
            chains=fit_settings.chains,
            seed=row_seed,
        )
        point = realized_point(target, result, definition=definition, mass=mass, seed=row_seed)
        if isinstance(point, Unsupported):
            log.warning("specification_curve: row %d %s failed: %s", index, labels, point.reason)
            rows.append(
                SpecRow(
                    index=index,
                    labels=labels,
                    spec_hash=spec.content_hash(),
                    spec_name=spec.name,
                    converged=result.converged,
                    failure=point.reason,
                )
            )
            continue
        summary = point.result.summary
        rows.append(
            SpecRow(
                index=index,
                labels=labels,
                spec_hash=spec.content_hash(),
                spec_name=spec.name,
                estimate=summary.mean,
                interval=summary.interval,
                sd=summary.sd,
                n_draws=summary.n,
                converged=result.converged,
            )
        )
    return SpecCurve(
        estimand_name=target.name,
        estimand_hash=target.content_hash(),
        base_spec_hash=base.content_hash(),
        axes=tuple(axes),
        rows=tuple(rows),
        n_total=n_total,
        n_dropped=n_total - len(kept),
        max_specs=max_specs,
        definition=definition,
        mass=mass,
        settings=settings,
        seed=seed,
    )
