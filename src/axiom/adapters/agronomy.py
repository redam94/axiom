"""The agronomy adapter: plot / rate / yield / optimum over the general core.

A second worked domain, written to be read as a **template**. Everything here
is an alias, a thin preset, or a domain quantity assembled from
``axiom.estimands`` and ``axiom.surface``; nothing re-implements any
mathematics, and the one place it goes beyond ``marketing`` — the economic
optimum — is built out of ``surface.marginal_band`` rather than out of a new
optimizer.

| agronomy                    | axiom                                             |
|-----------------------------|---------------------------------------------------|
| plot / strip / field        | the panel unit (``Unit``, kind ``"cluster"``)     |
| season                      | the time column                                   |
| nitrogen rate (kg N/ha)     | ``Treatment``, dose on ``mass / area``            |
| grain yield (t/ha)          | ``Outcome``, also on ``mass / area``              |
| soil test, cultivar, rain   | ``Covariate``                                     |
| Mitscherlich response       | ``surface.ExponentialKernel``                     |
| yield response to N         | the ``contrast`` estimand, observed rate vs zero  |
| agronomic efficiency        | the ``ratio`` estimand — dimensionless, see below |
| marginal product of N       | the ``marginal`` estimand                         |
| N-response elasticity       | the ``elasticity`` estimand                       |
| economic optimum rate       | where the marginal product meets the price ratio  |

Three things worth reading before copying this file into a third domain.

**The response family already existed under another name.** The classical
nitrogen-response curve is Mitscherlich's ``Y = A(1 − exp(−kN))``, and that is
exactly ``surface.ExponentialKernel``. The adapter's job was to *find* it, not
to add it. Reach for a new kernel in ``axiom.surface`` only when the shape is
genuinely absent, and then it belongs there rather than here — a saturating
curve is not an agronomic idea.

**The dose and the outcome share a dimension**, both being a mass per unit
area, so ``derived_dimension("ratio", ...)`` makes agronomic efficiency
*dimensionless*. That is correct and it is the reason the number is
comparable across crops and countries; it is also the reason ``roas`` in the
marketing adapter needs a currency-valued outcome and this one needs nothing.
``axiom`` has no notion of substance, so kilograms of grain and kilograms of
nitrogen are the same base; the docstring of ``agronomic_efficiency`` says so
out loud rather than leaving a reader to discover it from a unit string.

**The economic optimum is a quantity ``estimands`` cannot express.** It is not
a functional of the response at a stated intervention — it is the *dose* at
which a functional takes a stated value, which is an inversion. So it is not
forced into an ``Estimand``; it is its own ``Spec``, it carries its interval
definition and the prices it was computed at, and it returns ``Unsupported``
rather than a number when the trial's dose range does not bracket it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import Field, model_validator

from axiom.build import SurfaceBuilder
from axiom.core import (
    BASES,
    Blocked,
    Covariate,
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
    dimensionless,
)
from axiom.data import Panel, RoleMap
from axiom.estimands import Estimand, EstimandResult, Level, Quantity, QuantityKind, realize
from axiom.estimands.evaluate import RealizedDraws
from axiom.estimands.spec import derived_dimension
from axiom.surface import (
    FitResult,
    InterceptKind,
    ResponseBand,
    SurfaceSpec,
    marginal_band,
    response_band,
)

__all__ = [
    "AREA",
    "MASS",
    "MASS_PER_AREA",
    "EconomicOptimum",
    "Nutrient",
    "Plot",
    "Prices",
    "TrialRoles",
    "Yield",
    "agronomic_efficiency",
    "economic_optimum",
    "marginal_product",
    "nutrient_elasticity",
    "nutrient_rate",
    "panel_from_trial",
    "response_to",
    "role_map",
    "soil_test",
    "trial_spec",
]

Array = npt.NDArray[np.float64]
IntervalDefinition = Literal["eti", "hdi"]

# -- vocabulary -------------------------------------------------------------------------------

Plot = Unit
"""A plot, strip or field is the panel unit — a cluster of plants, not an individual."""
Nutrient = Treatment
"""A fertilizer nutrient is a treatment whose dose is an application rate."""
Yield = Outcome
"""Yield is the outcome."""

MASS: Dimension = BASES.declare("mass", symbol="M")
"""Declared, not invented: ``BASES.declare`` is idempotent, so a session that already
has ``mass`` from another domain keeps the one it had."""
AREA: Dimension = BASES.declare("area", symbol="A")
MASS_PER_AREA: Dimension = MASS / AREA
"""The dimension of both an application rate and a yield. See the module docstring."""


def nutrient_rate(
    name: str, *, unit: str = "kg/ha", price: float | None = None, currency: str = "USD"
) -> Dose:
    """The application-rate dose of ``name``, in mass per unit area.

    ``price`` is currency per unit of ``unit`` and only sets the ``numeraire``
    string, which is what lets value-of-information arithmetic dimension-check;
    the number itself belongs in ``Prices``, where the economic optimum reads it.
    """
    if price is not None and not (math.isfinite(price) and price > 0.0):
        raise ValueError(f"price must be positive and finite, got {price}")
    return Dose(
        name=name,
        dimension=MASS_PER_AREA,
        unit=unit,
        numeraire=currency if price is not None else None,
    )


def soil_test(name: str, *, unit: str | None = None) -> Covariate:
    """A measured soil property — organic carbon, pH, residual N — as a covariate.

    Dimensionless unless ``unit`` says otherwise, because the soil tests that
    matter to a response curve are indices and percentages, and a covariate
    that claims a dimension it cannot support fails gate 10 downstream.
    """
    return Covariate(name=name, dimension=dimensionless(), unit=unit)


# -- roles ------------------------------------------------------------------------------------


class TrialRoles(Spec):
    """Which columns of a trial frame play which role.

    ``plot`` → unit, ``season`` → time, ``harvest`` → outcome, each of
    ``nutrients`` → a treatment dosed in ``rate_unit``, each of ``soil_tests``
    and ``controls`` → a dimensionless covariate.
    """

    harvest: NonEmptyStr
    nutrients: tuple[NonEmptyStr, ...] = Field(min_length=1)
    plot: NonEmptyStr = "plot"
    season: NonEmptyStr = "season"
    soil_tests: tuple[NonEmptyStr, ...] = ()
    controls: tuple[NonEmptyStr, ...] = ()
    rate_unit: NonEmptyStr = "kg/ha"
    yield_unit: NonEmptyStr = "t/ha"

    @model_validator(mode="after")
    def _distinct(self) -> TrialRoles:
        cols = list(self.columns)
        dupes = sorted({c for c in cols if cols.count(c) > 1})
        if dupes:
            raise ValueError(f"columns assigned more than one role: {dupes}")
        return self

    @property
    def columns(self) -> tuple[str, ...]:
        return (
            self.plot,
            self.season,
            self.harvest,
            *self.nutrients,
            *self.soil_tests,
            *self.controls,
        )

    @property
    def harvest_entity(self) -> Yield:
        return Yield(name=self.harvest, dimension=MASS_PER_AREA, unit=self.yield_unit)


def role_map(roles: TrialRoles) -> RoleMap:
    """The general ``RoleMap`` the trial roles translate to."""
    covariates = {c: soil_test(c) for c in (*roles.soil_tests, *roles.controls)}
    return RoleMap(
        unit=roles.plot,
        time=roles.season,
        outcome=(roles.harvest, roles.harvest_entity),
        treatments={
            n: Nutrient(name=n, dimension=MASS_PER_AREA, unit=roles.rate_unit)
            for n in roles.nutrients
        },
        covariates=covariates,
    )


def panel_from_trial(df: pd.DataFrame, roles: TrialRoles) -> Panel:
    """A ``Panel`` from a plot × season frame; columns with no role are dropped.

    Missing role columns are a ``ValueError`` naming them, and nothing is
    imputed — an unharvested plot is a missing cell the ``Panel`` will report,
    not a zero yield.
    """
    missing = [c for c in roles.columns if c not in df.columns]
    if missing:
        raise ValueError(f"frame lacks columns the trial roles name: {missing}")
    return Panel(df.loc[:, list(roles.columns)], role_map(roles))


# -- the surface preset ------------------------------------------------------------------------


def _reference_rate(panel: Panel, column: str) -> float:
    """The mean applied rate over the plots that got any, which is the scale the curve bends on."""
    x = panel.column(column)
    applied = x[np.isfinite(x) & (x > 0)]
    return float(applied.mean()) if applied.size else 1.0


def trial_spec(
    panel: Panel,
    nutrients: Sequence[str] | None = None,
    *,
    name: str = "nutrient_response",
    kernel: str = "exponential",
    intercept: InterceptKind = "shared",
    residual_carryover: bool = False,
    max_lag: int = 2,
    amplitude_scale: float = 1.0,
    interactions: Sequence[tuple[str, str]] = (),
) -> SurfaceSpec:
    """A ``SurfaceSpec`` for a fertilizer trial through ``build.SurfaceBuilder``.

    Every nutrient gets the named ``kernel`` with its ``reference_dose`` at the
    mean applied rate. ``"exponential"`` is Mitscherlich and is the default;
    ``"polynomial"`` is the quadratic that agronomists fit when they expect
    yield to *fall* at high rates, which no saturating family can express.

    ``residual_carryover`` turns on a geometric carryover, which for a
    fertilizer trial is not a modelling nicety — unused nitrogen is still in
    the soil next season, and a trial that re-randomizes rates within a plot
    across seasons has that confound whether or not it is modelled. It is off
    by default because it is only identified when the rates *change* within a
    plot; ``design.contrast_score`` on the applied series says whether they do.

    The intercept defaults to ``"shared"`` and not to ``"hierarchical"``,
    which is the statistically better description of a field — plots differ,
    and that is why they are blocked. A field trial is usually many plots and
    few seasons, and a hierarchical intercept over a hundred plots with three
    harvests each is thin enough that the Laplace mode search returns
    ``Unverified`` rather than a posterior (the same failure ``docs/notes/
    0009`` records for a school trial). ``"shared"`` puts the plot-to-plot
    variation in the residual, which is honest and is what the design was
    already paying for; move to ``"hierarchical"`` with a sampler when there
    are enough seasons to support it.
    """
    roles = panel.roles
    chosen = tuple(nutrients) if nutrients is not None else tuple(roles.treatments)
    unknown = [n for n in chosen if n not in roles.treatments]
    if unknown:
        raise ValueError(f"nutrients {unknown} are not treatment columns of the panel")
    builder = SurfaceBuilder().name(name).outcome(roles.outcome[1])
    for n in chosen:
        builder = builder.treatment(
            roles.treatments[n],
            kernel=kernel,
            carryover="geometric" if residual_carryover else None,
            max_lag=max_lag if residual_carryover else None,
            reference_dose=_reference_rate(panel, n),
            amplitude_scale=amplitude_scale,
        )
    builder = builder.intercept(
        intercept, units=panel.units if intercept in ("per_unit", "hierarchical") else ()
    )
    for first, second in interactions:
        builder = builder.interaction(first, second)
    return builder.build()


# -- domain estimands --------------------------------------------------------------------------


def _window(result: FitResult, window: TimeWindow | tuple[int, int] | None) -> TimeWindow:
    """The window, defaulting to every season on a **per-period** basis.

    ``TimeWindow`` defaults to ``basis="cumulative"``, which is right for a
    quantity that adds up over time (revenue over a campaign). A yield does
    not add up over seasons in any useful way — three harvests of four tonnes
    a hectare is four tonnes a hectare a year, not twelve — so the adapter
    overrides the default rather than letting a caller discover it.
    """
    if window is None:
        return TimeWindow(start=0, stop=result.n_periods, basis="per_period")
    if isinstance(window, TimeWindow):
        return window
    start, stop = window
    return TimeWindow(start=int(start), stop=int(stop), basis="per_period")


def _estimand(
    result: FitResult, nutrient: str, kind: QuantityKind, window: TimeWindow, name: str
) -> Estimand:
    treatment = result.surface.spec.treatment(nutrient)
    outcome = result.outcome
    if treatment.dimension is None or outcome.dimension is None:  # pragma: no cover - spec
        raise ValueError("fitted entities carry dimensions")
    return Estimand(
        name=name,
        quantity=Quantity(kind=kind),
        treatment=treatment,
        intervention=Intervention(doses={nutrient: 1.0}, mode="scale", version="observed"),
        reference=(
            Intervention(doses={nutrient: 0.0}, mode="set", version="observed")
            if kind in ("contrast", "ratio", "area")
            else None
        ),
        outcome=outcome,
        population=Population(name="all_plots"),
        window=window,
        # "individual" averages over plots; "aggregate" would sum them. A yield is an
        # *intensive* quantity -- tonnes per hectare -- so summing twenty-eight of them
        # gives twenty-eight times the answer. The marketing adapter aggregates because
        # revenue is extensive. Which one an outcome is, is the adapter's to know.
        level=Level(unit="individual"),
        dimension=derived_dimension(kind, outcome.dimension, treatment.dimension),
        description=f"{name}: the applied {nutrient} rate against an unfertilized plot",
    )


def _realize(
    result: FitResult,
    nutrient: str,
    kind: QuantityKind,
    window: TimeWindow | tuple[int, int] | None,
    name: str,
    *,
    mass: float,
    definition: IntervalDefinition,
    seed: int | None,
) -> EstimandResult | Unsupported | Blocked:
    out = realize(
        _estimand(result, nutrient, kind, _window(result, window), f"{name}_{nutrient}"),
        result,
        assume_identified=True,
        definition=definition,
        mass=mass,
        seed=seed,
    )
    assert not isinstance(out, RealizedDraws)
    return out


def response_to(
    result: FitResult,
    nutrient: str,
    window: TimeWindow | tuple[int, int] | None = None,
    *,
    mass: float = 0.9,
    definition: IntervalDefinition = "hdi",
    seed: int | None = None,
) -> EstimandResult | Unsupported | Blocked:
    """Yield response: ``yield at the applied rate − yield at zero``, in yield units.

    The ``contrast`` estimand over the window. Identification is *asserted*
    (an observational surface), and the result says so in its assumptions.
    """
    return _realize(
        result,
        nutrient,
        "contrast",
        window,
        "response",
        mass=mass,
        definition=definition,
        seed=seed,
    )


def agronomic_efficiency(
    result: FitResult,
    nutrient: str,
    window: TimeWindow | tuple[int, int] | None = None,
    *,
    mass: float = 0.9,
    definition: IntervalDefinition = "hdi",
    seed: int | None = None,
) -> EstimandResult | Unsupported | Blocked:
    """Extra yield per unit of nutrient applied — the ``ratio`` estimand.

    **Dimensionless**, because a rate and a yield are both a mass per unit
    area and ``axiom`` does not distinguish kilograms of grain from kilograms
    of nitrogen. That is the convention agronomy already uses (kg grain per kg
    N), and it is what makes the number comparable across crops; read the unit
    from the entities if the two masses matter.
    """
    return _realize(
        result,
        nutrient,
        "ratio",
        window,
        "agronomic_efficiency",
        mass=mass,
        definition=definition,
        seed=seed,
    )


def marginal_product(
    result: FitResult,
    nutrient: str,
    window: TimeWindow | tuple[int, int] | None = None,
    *,
    mass: float = 0.9,
    definition: IntervalDefinition = "hdi",
    seed: int | None = None,
) -> EstimandResult | Unsupported | Blocked:
    """``d yield / d rate`` at the applied rate — the last kilogram's worth, not the average."""
    return _realize(
        result,
        nutrient,
        "marginal",
        window,
        "marginal_product",
        mass=mass,
        definition=definition,
        seed=seed,
    )


def nutrient_elasticity(
    result: FitResult,
    nutrient: str,
    window: TimeWindow | tuple[int, int] | None = None,
    *,
    mass: float = 0.9,
    definition: IntervalDefinition = "hdi",
    seed: int | None = None,
) -> EstimandResult | Unsupported | Blocked:
    """Percentage yield per percentage rate at the applied rate — the ``elasticity`` estimand."""
    return _realize(
        result,
        nutrient,
        "elasticity",
        window,
        "elasticity",
        mass=mass,
        definition=definition,
        seed=seed,
    )


# -- the economic optimum ----------------------------------------------------------------------


class Prices(Spec):
    """What a tonne of the harvest sells for and what a kilogram of the nutrient costs.

    Both in ``currency`` per unit of the entities' own units, so the ratio
    ``nutrient / harvest`` is in yield units per rate unit — the same units the
    marginal product is in, which is what makes the tangency a comparison of
    like with like rather than a coincidence of scaling.
    """

    harvest: float = Field(gt=0)
    nutrient: float = Field(gt=0)
    currency: NonEmptyStr = "USD"

    @property
    def ratio(self) -> float:
        """``nutrient price / harvest price``: the marginal product that just breaks even."""
        return self.nutrient / self.harvest


class EconomicOptimum(Spec):
    """The rate at which the last unit of nutrient pays for itself, and its interval.

    ``rate`` is where the posterior-mean marginal product crosses
    ``prices.ratio``. ``lower``/``upper`` are the *inversion* interval: the
    rates at which the marginal product's ``mass`` interval still contains the
    price ratio, so they are the rates this trial cannot rule out rather than
    a symmetric error bar. ``definition`` and ``mass`` say which interval that
    was, and ``searched`` records the dose range it was found in — a bound
    that sits at the edge of the searched range is a bound the trial did not
    establish, and ``bracketed`` says so.
    """

    nutrient: NonEmptyStr
    rate: float = Field(ge=0)
    lower: float = Field(ge=0)
    upper: float = Field(ge=0)
    expected_gain: float
    """Yield at ``rate`` less yield at zero, in the harvest's own units."""
    prices: Prices
    price_ratio: float = Field(gt=0)
    definition: IntervalDefinition
    mass: float = Field(gt=0, lt=1)
    searched: tuple[float, float]
    bracketed: bool
    detail: dict[str, str] = {}

    @model_validator(mode="after")
    def _ordered(self) -> EconomicOptimum:
        if self.lower > self.upper:
            raise ValueError(f"lower {self.lower} exceeds upper {self.upper}")
        low, high = self.searched
        if not low < high:
            raise ValueError(f"searched range must ascend, got {self.searched}")
        return self

    @property
    def profit_over_zero(self) -> float:
        """Currency per unit area gained over an unfertilized plot at ``rate``.

        ``harvest price x expected yield gain − nutrient price x rate``. It is
        a *partial* budget: it counts the fertilizer and the grain and nothing
        else, which is the number agronomy calls a partial budget and is the
        only one this trial licenses.
        """
        return self.prices.harvest * self.expected_gain - self.prices.nutrient * self.rate


def _first_crossing(doses: Array, values: Array, target: float) -> float | None:
    """The first dose at which a falling curve reaches ``target``; ``None`` if it never does."""
    excess = values - target
    below = np.where(excess <= 0.0)[0]
    if below.size == 0:
        return None
    i = int(below[0])
    if i == 0:
        return float(doses[0])
    return float(np.interp(0.0, [excess[i], excess[i - 1]], [doses[i], doses[i - 1]]))


def economic_optimum(
    result: FitResult,
    nutrient: str,
    prices: Prices,
    *,
    rates: Sequence[float] | None = None,
    n_grid: int = 81,
    mass: float = 0.9,
    definition: IntervalDefinition = "eti",
    window: TimeWindow | None = None,
    seed: int | None = None,
) -> EconomicOptimum | Unsupported:
    """The rate where the marginal product of ``nutrient`` falls to the price ratio.

    The marginal product comes from ``surface.marginal_band``, so every draw
    goes through the same ``forward`` the likelihood used and the curve is the
    one the model believes rather than the one at the posterior mean. The
    optimum is the first crossing of ``prices.ratio``; the interval is found by
    **inversion** — the upper marginal curve crosses later than the mean and
    the lower crosses earlier, and those two crossings are the rates the trial
    cannot separate from the optimum.

    Returns ``Unsupported`` when the marginal product never falls to the price
    ratio inside the searched range. That is not a numerical failure: it means
    the trial did not apply enough nutrient to see the optimum, and the honest
    report is the range that was searched rather than an extrapolated number.
    """
    band = marginal_band(
        result,
        nutrient,
        doses=rates,
        n_grid=n_grid,
        mass=mass,
        definition=definition,
        window=window,
        seed=seed,
    )
    if isinstance(band, Unsupported):
        return band
    assert isinstance(band, ResponseBand)
    doses = np.asarray(band.doses, dtype=np.float64)
    target = prices.ratio
    at = _first_crossing(doses, np.asarray(band.mean, dtype=np.float64), target)
    searched = (float(doses[0]), float(doses[-1]))
    if at is None:
        return Unsupported(
            reason=(
                f"the marginal product of {nutrient!r} never falls to the price ratio "
                f"{target:.6g} {result.outcome.unit or 'yield'} per "
                f"{result.surface.spec.treatment(nutrient).unit or 'rate'} inside "
                f"[{searched[0]:g}, {searched[1]:g}]: this trial did not apply enough to "
                "see the optimum. Extend the rates, or report the range that was searched."
            ),
            missing=("dose_range",),
            detail={
                "searched_low": f"{searched[0]:g}",
                "searched_high": f"{searched[1]:g}",
                "price_ratio": f"{target:.6g}",
                "marginal_at_top": f"{band.mean[-1]:.6g}",
            },
        )
    # Inversion: the lower marginal curve reaches the ratio soonest, the upper latest.
    early = _first_crossing(doses, np.asarray(band.lower, dtype=np.float64), target)
    late = _first_crossing(doses, np.asarray(band.upper, dtype=np.float64), target)
    bracketed = late is not None
    # The yield gain at the optimum comes from the *response* curve on the same grid --
    # the marginal band knows the slope and nothing about the height.
    grid = sorted({0.0, *(float(d) for d in doses)})
    curve = response_band(
        result, nutrient, doses=grid, mass=mass, definition=definition, window=window, seed=seed
    )
    if isinstance(curve, Unsupported):
        return curve
    assert isinstance(curve, ResponseBand)
    heights = np.asarray(curve.mean, dtype=np.float64)
    at_zero = float(heights[0])
    return EconomicOptimum(
        nutrient=nutrient,
        rate=at,
        lower=searched[0] if early is None else early,
        upper=searched[1] if late is None else late,
        expected_gain=float(np.interp(at, np.asarray(curve.doses, dtype=np.float64), heights))
        - at_zero,
        prices=prices,
        price_ratio=target,
        definition=definition,
        mass=mass,
        searched=searched,
        bracketed=bracketed,
        detail={
            "n_draws": str(band.n_draws),
            "interval": "inversion of the marginal-product band, not a symmetric error bar",
            **(
                {}
                if bracketed
                else {
                    "upper_is_censored": (
                        "the upper marginal curve does not reach the price ratio inside the "
                        "searched range; the upper bound is the top of that range"
                    )
                }
            ),
        },
    )
