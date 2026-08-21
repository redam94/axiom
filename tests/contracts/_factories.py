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
    Assumption,
    Blocked,
    Covariate,
    D,
    Dimension,
    Dose,
    Interval,
    Intervention,
    LedgerLine,
    Outcome,
    Population,
    Spec,
    Summary,
    TimeWindow,
    Treatment,
    Unit,
    Unsupported,
    Unverified,
    Verdict,
    dimensionless,
)
from axiom.data import ColumnScaling, Completeness, RoleMap, ScalingParameters
from axiom.io import Provenance


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
