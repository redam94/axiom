"""One example instance per ``Spec`` subclass, for gate 4.

Every concrete ``Spec`` shipped in ``src/axiom`` must have an entry here.
``test_spec_roundtrip.py`` fails naming the class if one is missing, which is
the point: a spec nobody can build an example of is a spec nobody has
round-tripped.
"""

from __future__ import annotations

from collections.abc import Callable
from fractions import Fraction

from axiom.core import (
    AcceptanceRegion,
    Add,
    Apply,
    Assumption,
    Blocked,
    Const,
    Convolve,
    Covariate,
    D,
    Data,
    Dimension,
    Div,
    Dose,
    Equation,
    Interval,
    Intervention,
    LedgerLine,
    Link,
    Mul,
    ODESystem,
    Opaque,
    Outcome,
    Param,
    Population,
    Pow,
    Prior,
    Spec,
    Summary,
    System,
    TimeWindow,
    Treatment,
    Unit,
    Unsupported,
    Unverified,
    Verdict,
    dimensionless,
)
from axiom.data import ColumnScaling, Completeness, RoleMap, ScalingParameters
from axiom.estimands import Estimand, FacetDiff, Level, Quantity, TransferPlan
from axiom.io import Provenance

_DOSE = Data(name="dose", dimension=D.currency)
_K = Param(
    name="k", dimension=D.currency, prior=Prior(family="lognormal", hyper={"mu": 3.0, "sigma": 1.0})
)
_S = Param(name="s", dimension=dimensionless())
_X = Div(numerator=_DOSE, denominator=_K)


def _hill() -> Mul:
    return Mul(
        factors=(
            Param(name="beta", dimension=D.outcome),
            Div(
                numerator=Pow(base=_X, exponent=_S),
                denominator=Add(
                    terms=(Const(value=1.0, dimension=dimensionless()), Pow(base=_X, exponent=_S))
                ),
            ),
        )
    )


def _estimand(**over: object) -> Estimand:
    base: dict[str, object] = dict(
        name="lift_at_100",
        quantity=Quantity(kind="contrast"),
        treatment=Treatment(name="fertilizer", dimension=D.currency, unit="USD"),
        intervention=Intervention(doses={"fertilizer": 100.0}, version="granular"),
        reference=Intervention(doses={"fertilizer": 0.0}, version="granular"),
        outcome=Outcome(name="yield_total", dimension=D.outcome, unit="kg"),
        population=Population(name="north", strata={"soil": {"clay": 0.3, "loam": 0.7}}),
        window=TimeWindow(start=0, stop=8),
        level=Level(unit="cluster"),
        conditioning=(),
        dimension=D.outcome,
    )
    base.update(over)
    return Estimand(**base)  # type: ignore[arg-type]


def _assumption() -> Assumption:
    return Assumption(
        name="stationary_dynamics",
        facet="window",
        statement="carryover is contained in the window",
        challenged_by="half-life against window length",
        detail={"halflife_over_window": "0.4"},
    )


def _interval() -> Interval:
    return Interval(lower=-0.5, upper=1.25, definition="hdi", mass=0.9)


def _rolemap() -> RoleMap:
    return RoleMap(
        unit="unit",
        time="t",
        outcome=("y", Outcome(name="yield_total", dimension=D.outcome, unit="kg")),
        treatments={"x": Treatment(name="fertilizer", dimension=D.currency, unit="USD")},
        covariates={"rain": Covariate(name="rain", dimension=dimensionless())},
    )


EXAMPLES: dict[type[Spec], Callable[[], Spec]] = {
    Dimension: lambda: D.outcome / D.currency ** Fraction(1, 2),
    Treatment: lambda: Treatment(name="fertilizer", dimension=D.currency, unit="USD"),
    Dose: lambda: Dose(name="dose", dimension=D.currency, unit="USD", numeraire="USD"),
    Unit: lambda: Unit(name="plot", dimension=D.entity, kind="cluster"),
    Outcome: lambda: Outcome(name="yield_total", dimension=D.outcome, aggregation="mean"),
    Covariate: lambda: Covariate(name="rain", dimension=dimensionless()),
    Population: lambda: Population(name="north", strata={"soil": {"clay": 0.25, "loam": 0.75}}),
    TimeWindow: lambda: TimeWindow(start=2, stop=10, basis="per_period"),
    Intervention: lambda: Intervention(
        doses={"fertilizer": 120.0, "water": 3.5}, mode="scale", version="v2"
    ),
    Interval: _interval,
    Summary: lambda: Summary(mean=0.3, median=0.28, sd=0.5, interval=_interval(), n=4000),
    Unsupported: lambda: Unsupported(
        reason="no counterfactual capability", missing=("counterfactual",)
    ),
    Blocked: lambda: Blocked(reason="no admissible adjustment set", detail={"node": "U"}),
    Unverified: lambda: Unverified(reason="overlap not checked"),
    AcceptanceRegion: lambda: AcceptanceRegion(n=500, p=0.05, alpha=0.001, lower=10, upper=45),
    Assumption: _assumption,
    Verdict: lambda: Verdict(status="downgraded", assumptions=(_assumption(),), route="backdoor"),
    LedgerLine: lambda: LedgerLine(
        kind="facet:window",
        statement="window differs; assumed stationary dynamics",
        assumption=_assumption(),
        source="a" * 64,
        target="b" * 64,
    ),
    RoleMap: _rolemap,
    Completeness: lambda: Completeness(
        n_units=3,
        n_periods=10,
        n_rows=29,
        balanced=False,
        missing_cells=1,
        null_cells=0,
        gaps={"b": 1},
    ),
    ColumnScaling: lambda: ColumnScaling(method="standardize", loc=2.0, scale=0.5),
    ScalingParameters: lambda: ScalingParameters(
        columns={"y": ColumnScaling(method="max", scale=10.0), "x": ColumnScaling(method="none")}
    ),
    Prior: lambda: Prior(family="normal", hyper={"mu": 0.0, "sigma": 1.0}),
    Const: lambda: Const(value=2.5, dimension=D.time),
    Data: lambda: _DOSE,
    Param: lambda: _K,
    Add: lambda: Add(terms=(_DOSE, Const(value=1.0, dimension=D.currency))),
    Mul: _hill,
    Div: lambda: _X,
    Pow: lambda: Pow(base=_DOSE, exponent=Fraction(1, 2)),
    Apply: lambda: Apply(fn="log1p", arg=_X),
    Convolve: lambda: Convolve(
        signal=_DOSE,
        kernel=Opaque(
            name="geometric",
            inputs=(Param(name="lam", dimension=dimensionless()),),
            dimension=dimensionless(),
        ),
    ),
    Link: lambda: Link(fn="log", arg=_X),
    Opaque: lambda: Opaque(name="user_kernel", inputs=(_DOSE, _K), dimension=D.outcome),
    Equation: lambda: Equation(
        lhs=Data(name="y", dimension=D.outcome), rhs=_hill(), name="response"
    ),
    System: lambda: System(
        equations=(
            Equation(lhs=Data(name="y", dimension=D.outcome), rhs=_hill()),
            Equation(lhs=Data(name="z", dimension=D.currency), rhs=Mul(factors=(_K, _S))),
        )
    ),
    ODESystem: lambda: ODESystem(
        states=(Data(name="S", dimension=D.outcome),),
        rhs=(
            Mul(
                factors=(Param(name="r", dimension=D.time**-1), Data(name="S", dimension=D.outcome))
            ),
        ),
        time=Data(name="t", dimension=D.time),
    ),
    Quantity: lambda: Quantity(kind="marginal", scale="log"),
    Level: lambda: Level(
        unit="aggregate", interference="declared", interference_model="spatial lag 1"
    ),
    Estimand: _estimand,
    FacetDiff: lambda: _estimand()
    .transfer_to(_estimand(window=TimeWindow(start=0, stop=12)))
    .entries[0],
    TransferPlan: lambda: _estimand().transfer_to(_estimand(population=Population(name="all"))),
    Provenance: lambda: Provenance(
        axiom_version="0.0.0",
        created="2026-08-21T00:00:00+00:00",
        hashes={"spec:roles": "c" * 64},
        seed=7,
        environment={"python": "3.12"},
    ),
}


def example(cls: type[Spec]) -> Spec:
    return EXAMPLES[cls]()
