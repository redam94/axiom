"""The marketing adapter: channel / spend / geo / KPI over the general core.

This is the **only** module in ``axiom`` where marketing vocabulary appears
in identifiers (gate 3 exempts ``adapters/``). Everything here is an alias
or a thin preset over the general machinery:

| marketing            | axiom                                             |
|----------------------|---------------------------------------------------|
| channel              | ``Treatment`` (currency dose)                     |
| spend                | the dose, ``D.currency``                          |
| geo / DMA            | the panel unit (``Unit``, kind ``"cluster"``)     |
| KPI                  | ``Outcome``                                       |
| impressions          | ``Covariate`` on the ``exposure_count`` base      |
| ROAS                 | the ``ratio`` estimand, observed spend vs zero    |
| ROI                  | ROAS − 1                                          |
| contribution         | the ``contrast`` estimand, observed spend vs zero |
| marginal ROAS        | the ``marginal`` estimand at the observed spend   |

Ported from the parent's ``data_loader.py`` (ledger: DROP/EXTRACT — only
the ~150 LOC of ``MFFLoader`` that read the long "MFF" format survive, as
``panel_from_mff``) and ``builders/model.py`` presets (REWRITE). The
return-on-spend quantities are ``estimands.Estimand`` objects realized by
``estimands.realize`` against a ``surface.FitResult``; every number is an
``EstimandResult`` with its interval, dimension, unit, and ledger. ROAS is
dimensionless only when the KPI is currency-valued (revenue); for a
count-valued KPI the ratio has dimension ``outcome / currency`` and
``roas`` returns ``Unsupported`` rather than a mislabelled number.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import model_validator

from axiom.build import SurfaceBuilder
from axiom.core import (
    BASES,
    Blocked,
    Covariate,
    D,
    Dimension,
    Dose,
    Intervention,
    NonEmptyStr,
    Outcome,
    Population,
    Spec,
    TimeWindow,
    Treatment,
    Unit,
    Unsupported,
    summarize,
)
from axiom.data import Panel, RoleMap
from axiom.estimands import Estimand, EstimandResult, Level, Quantity, QuantityKind, realize
from axiom.estimands.evaluate import RealizedDraws
from axiom.surface import FitResult, InterceptKind, SurfaceSpec

__all__ = [
    "EXPOSURE",
    "Channel",
    "Geo",
    "KPI",
    "MarketingRoles",
    "contribution",
    "impressions",
    "marginal_roas",
    "marketing_spec",
    "panel_from_marketing_frame",
    "panel_from_mff",
    "roas",
    "roi",
    "role_map",
    "spend",
]

Channel = Treatment
"""A marketing channel is a treatment whose dose is spend."""
Geo = Unit
"""A geo / DMA is the panel unit (a cluster of people)."""
KPI = Outcome
"""A KPI is the outcome."""

EXPOSURE: Dimension = BASES.declare("exposure_count", symbol="imp")
"""The adapter's own base dimension for impressions / exposures."""

IntervalDefinition = Literal["eti", "hdi"]


def spend(channel: str, *, currency: str = "USD") -> Dose:
    """The spend dose of ``channel``: currency-dimensioned, costed in its own currency."""
    return Dose(name=channel, dimension=D.currency, unit=currency, numeraire=currency)


def impressions(name: str) -> Covariate:
    """An impressions / exposures covariate on the ``exposure_count`` base."""
    return Covariate(name=name, dimension=EXPOSURE, unit="impressions")


# -- roles ---------------------------------------------------------------------------------


class MarketingRoles(Spec):
    """Which columns of a marketing frame play which role.

    ``geo`` → unit, ``date`` → time, ``kpi`` → outcome, each of ``channels`` →
    a treatment with a currency dose in ``currency``, each of
    ``impressions`` → an ``exposure_count`` covariate, each of ``controls``
    → a dimensionless covariate. ``kpi_dimension`` is ``"currency"`` for a
    revenue KPI (then ROAS is dimensionless) or ``"outcome"`` for a count
    (conversions, units sold); ``kpi_unit`` is its unit string.
    """

    kpi: NonEmptyStr
    channels: tuple[NonEmptyStr, ...]
    geo: NonEmptyStr = "geo"
    date: NonEmptyStr = "date"
    impressions: tuple[NonEmptyStr, ...] = ()
    controls: tuple[NonEmptyStr, ...] = ()
    currency: NonEmptyStr = "USD"
    kpi_dimension: Literal["currency", "outcome"] = "outcome"
    kpi_unit: str | None = None

    @model_validator(mode="after")
    def _distinct(self) -> MarketingRoles:
        if not self.channels:
            raise ValueError("MarketingRoles needs at least one channel")
        cols = [self.geo, self.date, self.kpi, *self.channels, *self.impressions, *self.controls]
        dupes = sorted({c for c in cols if cols.count(c) > 1})
        if dupes:
            raise ValueError(f"columns assigned more than one role: {dupes}")
        return self

    @property
    def columns(self) -> tuple[str, ...]:
        return (
            self.geo,
            self.date,
            self.kpi,
            *self.channels,
            *self.impressions,
            *self.controls,
        )

    @property
    def kpi_entity(self) -> Outcome:
        dim = D.currency if self.kpi_dimension == "currency" else D.outcome
        unit = (
            self.kpi_unit
            if self.kpi_unit is not None
            else (self.currency if self.kpi_dimension == "currency" else None)
        )
        return Outcome(name=self.kpi, dimension=dim, unit=unit)


def role_map(roles: MarketingRoles) -> RoleMap:
    """The general ``RoleMap`` the marketing roles translate to."""
    covariates: dict[str, Covariate] = {c: impressions(c) for c in roles.impressions}
    for c in roles.controls:
        covariates[c] = Covariate(name=c, dimension=D.outcome / D.outcome)
    return RoleMap(
        unit=roles.geo,
        time=roles.date,
        outcome=(roles.kpi, roles.kpi_entity),
        treatments={
            c: Channel(name=c, dimension=D.currency, unit=roles.currency) for c in roles.channels
        },
        covariates=covariates,
    )


def panel_from_marketing_frame(df: pd.DataFrame, roles: MarketingRoles) -> Panel:
    """A ``Panel`` from a wide geo × date frame; columns with no role are dropped.

    Missing role columns are a ``ValueError`` naming them. Nothing is
    imputed: the ``Panel`` reports completeness and refuses nulls in
    measured columns downstream.
    """
    missing = [c for c in roles.columns if c not in df.columns]
    if missing:
        raise ValueError(f"frame lacks columns the marketing roles name: {missing}")
    return Panel(df.loc[:, list(roles.columns)], role_map(roles))


def panel_from_mff(
    df: pd.DataFrame,
    roles: MarketingRoles,
    *,
    variable: str = "variable",
    value: str = "value",
) -> Panel:
    """The MFF long format (one row per ``geo × date × variable``) → ``Panel``.

    The long frame is pivoted to wide on ``(geo, date)``; a ``(geo, date,
    variable)`` triple appearing twice is a ``ValueError`` (the MFF loader's
    silent "last wins" is not reproduced). Variables the roles do not name
    are dropped; a role variable absent from the frame is a ``ValueError``.
    """
    for col in (roles.geo, roles.date, variable, value):
        if col not in df.columns:
            raise ValueError(f"MFF frame lacks column {col!r}")
    key = [roles.geo, roles.date, variable]
    dup = int(df.duplicated(key).sum())
    if dup:
        raise ValueError(f"MFF frame has {dup} duplicate (geo, date, variable) rows")
    wide = df.pivot(index=[roles.geo, roles.date], columns=variable, values=value).reset_index()
    wide.columns.name = None
    return panel_from_marketing_frame(wide, roles)


# -- the surface preset ----------------------------------------------------------------------


def _reference_spend(panel: Panel, column: str) -> float:
    x = panel.column(column)
    positive = x[np.isfinite(x) & (x > 0)]
    return float(positive.mean()) if positive.size else 1.0


def marketing_spec(
    panel: Panel,
    channels: Sequence[str] | None = None,
    *,
    name: str = "marketing",
    kernel: str = "hill",
    carryover: str | None = "geometric",
    max_lag: int = 4,
    intercept: InterceptKind = "hierarchical",
    seasonality: tuple[float, int] | None = None,
    trend: bool = False,
    amplitude_scale: float = 1.0,
) -> SurfaceSpec:
    """A ``SurfaceSpec`` for a marketing panel through ``build.SurfaceBuilder``.

    Every channel gets the named saturation ``kernel`` with
    ``reference_dose`` at its mean positive spend and, unless ``carryover``
    is ``None``, the named carryover with ``max_lag``. The intercept is
    hierarchical over geos by default; ``seasonality=(period, order)`` adds
    Fourier terms and ``trend=True`` a linear trend. The treatment and
    outcome entities are the panel's own, so ``prepare`` finds the units it
    expects.
    """
    roles = panel.roles
    chosen = tuple(channels) if channels is not None else tuple(roles.treatments)
    unknown = [c for c in chosen if c not in roles.treatments]
    if unknown:
        raise ValueError(f"channels {unknown} are not treatment columns of the panel")
    b = SurfaceBuilder().name(name).outcome(roles.outcome[1])
    for c in chosen:
        b = b.treatment(
            roles.treatments[c],
            kernel=kernel,
            carryover=carryover,
            max_lag=max_lag if carryover is not None else None,
            reference_dose=_reference_spend(panel, c),
            amplitude_scale=amplitude_scale,
        )
    b = b.intercept(
        intercept, units=panel.units if intercept in ("per_unit", "hierarchical") else ()
    )
    if seasonality is not None or trend:
        b = b.nuisance(fourier=seasonality, trend=trend)
    return b.build()


# -- return-on-spend estimands ----------------------------------------------------------------


def _window(result: FitResult, window: TimeWindow | tuple[int, int] | None) -> TimeWindow:
    if window is None:
        return TimeWindow(start=0, stop=result.n_periods)
    if isinstance(window, TimeWindow):
        return window
    start, stop = window
    return TimeWindow(start=int(start), stop=int(stop))


def _estimand(
    result: FitResult, channel: str, kind: QuantityKind, window: TimeWindow, name: str
) -> Estimand:
    treatment = result.surface.spec.treatment(channel)
    outcome = result.outcome
    at = Intervention(doses={channel: 1.0}, mode="scale", version="observed")
    zero = Intervention(doses={channel: 0.0}, mode="set", version="observed")
    from axiom.estimands import derived_dimension

    if treatment.dimension is None or outcome.dimension is None:  # pragma: no cover - spec
        raise ValueError("fitted entities carry dimensions")
    return Estimand(
        name=name,
        quantity=Quantity(kind=kind),
        treatment=treatment,
        intervention=at,
        reference=zero if kind in ("contrast", "ratio", "area") else None,
        outcome=outcome,
        population=Population(name="all_geos"),
        window=window,
        level=Level(unit="aggregate"),
        dimension=derived_dimension(kind, outcome.dimension, treatment.dimension),
        description=f"{name}: observed {channel} spend against zero spend over the window",
    )


def _currency_kpi(result: FitResult, what: str) -> Unsupported | None:
    dim = result.outcome.dimension
    if dim != D.currency:
        return Unsupported(
            reason=(
                f"{what} is a dimensionless return only for a currency-valued KPI; "
                f"{result.outcome.name!r} has dimension {dim}. Use contribution() or "
                "realize the ratio estimand directly for outcome-per-currency."
            ),
            missing=("currency_outcome",),
            detail={"outcome": result.outcome.name, "dimension": str(dim)},
        )
    return None


def _realize(
    result: FitResult,
    channel: str,
    kind: QuantityKind,
    window: TimeWindow | tuple[int, int] | None,
    name: str,
    *,
    mass: float,
    definition: IntervalDefinition,
    seed: int | None,
    keep_draws: bool = False,
) -> EstimandResult | RealizedDraws | Unsupported | Blocked:
    estimand = _estimand(result, channel, kind, _window(result, window), f"{name}_{channel}")
    return realize(
        estimand,
        result,
        assume_identified=True,
        definition=definition,
        mass=mass,
        seed=seed,
        keep_draws=keep_draws,
    )


def roas(
    result: FitResult,
    channel: str,
    window: TimeWindow | tuple[int, int] | None = None,
    *,
    mass: float = 0.9,
    definition: IntervalDefinition = "hdi",
    seed: int | None = None,
) -> EstimandResult | Unsupported | Blocked:
    """Return on ad spend: ``(KPI at observed spend − KPI at zero) / observed spend``.

    The ``ratio`` estimand over the window, aggregated over geos. Needs a
    currency-valued KPI (``Unsupported`` otherwise). Identification is
    *asserted* (``assume_identified=True``) — the surface is an
    observational model — and the result says so in its assumptions.
    """
    bad = _currency_kpi(result, "ROAS")
    if bad is not None:
        return bad
    out = _realize(
        result, channel, "ratio", window, "roas", mass=mass, definition=definition, seed=seed
    )
    assert not isinstance(out, RealizedDraws)
    return out


def roi(
    result: FitResult,
    channel: str,
    window: TimeWindow | tuple[int, int] | None = None,
    *,
    mass: float = 0.9,
    definition: IntervalDefinition = "hdi",
    seed: int | None = None,
) -> EstimandResult | Unsupported | Blocked:
    """Return on investment ``ROAS − 1``: the same draws shifted, re-summarized."""
    bad = _currency_kpi(result, "ROI")
    if bad is not None:
        return bad
    out = _realize(
        result,
        channel,
        "ratio",
        window,
        "roas",
        mass=mass,
        definition=definition,
        seed=seed,
        keep_draws=True,
    )
    if not isinstance(out, RealizedDraws):
        return out
    shifted = summarize(out.draws - 1.0, definition=definition, mass=mass)
    return out.result.model_copy(
        update={
            "estimand_name": f"roi_{channel}",
            "summary": shifted,
            "detail": {
                **out.result.detail,
                "derived_from": out.result.estimand_name,
                "shift": -1.0,
            },
        }
    )


def contribution(
    result: FitResult,
    channel: str,
    window: TimeWindow | tuple[int, int] | None = None,
    *,
    mass: float = 0.9,
    definition: IntervalDefinition = "hdi",
    seed: int | None = None,
) -> EstimandResult | Unsupported | Blocked:
    """The channel's contribution to the KPI: ``KPI at observed spend − KPI at zero spend``,
    summed over geos and the window (the ``contrast`` estimand, KPI units)."""
    out = _realize(
        result,
        channel,
        "contrast",
        window,
        "contribution",
        mass=mass,
        definition=definition,
        seed=seed,
    )
    assert not isinstance(out, RealizedDraws)
    return out


def marginal_roas(
    result: FitResult,
    channel: str,
    window: TimeWindow | tuple[int, int] | None = None,
    *,
    mass: float = 0.9,
    definition: IntervalDefinition = "hdi",
    seed: int | None = None,
) -> EstimandResult | Unsupported | Blocked:
    """The derivative of the window's KPI with respect to a common shift of the channel's
    observed spend (the ``marginal`` estimand). Needs a currency-valued KPI."""
    bad = _currency_kpi(result, "marginal ROAS")
    if bad is not None:
        return bad
    out = _realize(
        result,
        channel,
        "marginal",
        window,
        "marginal_roas",
        mass=mass,
        definition=definition,
        seed=seed,
    )
    assert not isinstance(out, RealizedDraws)
    return out
