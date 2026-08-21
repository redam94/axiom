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
    Gather,
    Interval,
    Intervention,
    LedgerLine,
    Likelihood,
    Link,
    ModelSpec,
    Mul,
    ODESystem,
    Opaque,
    Outcome,
    Param,
    Population,
    Pow,
    Prior,
    Reduce,
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
from axiom.estimands import Estimand, EstimandResult, FacetDiff, Level, Quantity, TransferPlan
from axiom.identify import (
    CausalGraph,
    EndogeneityTest,
    FrontDoorRoute,
    InstrumentRoute,
    LinearEstimate,
    RoleAssignment,
    assign_roles,
    identify,
    transport_verdict,
)
from axiom.identify.transport import TransportVerdict
from axiom.identify.verdict import IdentificationVerdict
from axiom.infer import (
    ConvergenceReport,
    ConvergenceThresholds,
    ParameterDiagnostics,
    PointEstimate,
    SampleSettings,
)
from axiom.io import Provenance
from axiom.sim import DosePlan, LinearSCM
from axiom.surface import (
    Allocation,
    AscentPath,
    Bounds,
    DelayedCarryover,
    Design,
    EventIndicators,
    ExponentialKernel,
    FourierSeasonality,
    Frontier,
    GeometricCarryover,
    HillKernel,
    LinearKernel,
    LinearTrend,
    LogisticKernel,
    NoCarryover,
    NuisanceSet,
    PowerKernel,
    StationaryPoint,
    SurfaceSpec,
    WeibullCarryover,
)

_G = CausalGraph.from_edges("Z -> X, Z -> Y, X -> Y, X <-> W, W -> Y", unmeasured=["W"], name="toy")

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
    Reduce: lambda: Reduce(
        op="sum", arg=Pow(base=_S, exponent=Const(value=(0.0, 1.0, 2.0), dimension=dimensionless()))
    ),
    Gather: lambda: Gather(
        source=Param(name="alpha", dimension=D.outcome, shape=(3,)),
        index=Data(name="unit", dimension=dimensionless()),
    ),
    Likelihood: lambda: Likelihood(family="student_t", scale="sigma", df=4.0),
    ModelSpec: lambda: ModelSpec(
        name="toy",
        mean=Add(terms=(Param(name="a", dimension=D.outcome), _hill())),
        outcome=Data(name="y", dimension=D.outcome),
        likelihood=Likelihood(family="normal", scale="sigma"),
        parameters=(
            Param(
                name="a",
                dimension=D.outcome,
                prior=Prior(family="normal", hyper={"mu": 0.0, "sigma": 10.0}),
            ),
            Param(
                name="beta",
                dimension=D.outcome,
                prior=Prior(family="halfnormal", hyper={"sigma": 10.0}),
            ),
            Param(
                name="k",
                dimension=D.currency,
                prior=Prior(family="lognormal", hyper={"mu": 3.0, "sigma": 1.0}),
            ),
            Param(
                name="s",
                dimension=dimensionless(),
                prior=Prior(family="gamma", hyper={"alpha": 4.0, "beta": 2.0}),
            ),
            Param(
                name="sigma",
                dimension=D.outcome,
                prior=Prior(family="halfnormal", hyper={"sigma": 5.0}),
            ),
        ),
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
    EstimandResult: lambda: EstimandResult(
        estimand_hash="a" * 64,
        estimand_name="lift_at_100",
        kind="contrast",
        summary=Summary(mean=0.3, median=0.28, sd=0.5, interval=_interval(), n=4000),
        dimension=D.outcome,
        unit="kg",
        status="downgraded",
        assumptions=(_assumption(),),
        ledger=(),
        n_draws=4000,
        producer_hash="b" * 64,
        detail={"dose_iv": 100.0, "dose_ref": 0.0},
    ),
    CausalGraph: lambda: _G.with_selection("Z").model_copy(update={"feedback": True}),
    RoleAssignment: lambda: assign_roles(_G, "X", "Y"),
    FrontDoorRoute: lambda: FrontDoorRoute(mediators=("M",), treatment="X", outcome="Y"),
    InstrumentRoute: lambda: InstrumentRoute(
        instrument="Z", conditioning=("W",), treatment="X", outcome="Y"
    ),
    TransportVerdict: lambda: transport_verdict(
        CausalGraph.from_edges("Z -> X, Z -> Y, X -> Y", selection=["Z"]), "X", "Y"
    ),
    IdentificationVerdict: lambda: identify(
        CausalGraph.from_edges("Z -> X, Z -> Y, X -> Y", selection=["Z"]), "X", "Y"
    ),
    LinearEstimate: lambda: LinearEstimate(
        estimate=2.0,
        se=0.1,
        n=100,
        method="2sls",
        treatment="X",
        outcome="Y",
        covariates=("Z",),
        detail={"first_stage_f": 50.0},
    ),
    EndogeneityTest: lambda: EndogeneityTest(
        statistic=3.2,
        p_value=0.001,
        df=97,
        method="durbin_wu_hausman_control_function",
        conclusion="endogenous",
        treatment="X",
        outcome="Y",
        detail={"first_stage_f": 50.0},
    ),
    LinearSCM: lambda: LinearSCM.from_text(
        "Z -> X: 0.8, Z -> Y: 1.5, X -> Y: 2.0, X <-> Y: 0.7",
        unmeasured=["Z"],
        selection=["Z"],
        intercepts={"Z": 1.5},
        noise_sd={"Y": 0.5},
    ),
    HillKernel: lambda: HillKernel(reference_dose=50.0, amplitude_scale=2.0),
    LogisticKernel: lambda: LogisticKernel(reference_dose=50.0),
    ExponentialKernel: lambda: ExponentialKernel(reference_dose=50.0),
    PowerKernel: lambda: PowerKernel(reference_dose=50.0),
    LinearKernel: lambda: LinearKernel(reference_dose=50.0),
    GeometricCarryover: lambda: GeometricCarryover(max_lag=8),
    DelayedCarryover: lambda: DelayedCarryover(max_lag=8),
    WeibullCarryover: lambda: WeibullCarryover(max_lag=8),
    NoCarryover: lambda: NoCarryover(),
    FourierSeasonality: lambda: FourierSeasonality(period=52.0, order=2),
    LinearTrend: lambda: LinearTrend(origin=0.0, scale=52.0),
    EventIndicators: lambda: EventIndicators(events=("holiday", "outage")),
    NuisanceSet: lambda: NuisanceSet(
        terms=(
            FourierSeasonality(period=52.0, order=2),
            LinearTrend(),
            EventIndicators(events=("holiday",)),
        )
    ),
    Bounds: lambda: Bounds(treatments=("x1", "x2"), low=(0.0, 10.0), high=(4.0, 30.0)),
    Design: lambda: Design(
        treatments=("x1", "x2"),
        points=((0.0, 10.0), (4.0, 30.0)),
        kind="full_factorial",
        detail={"levels[x1]": 2.0, "levels[x2]": 2.0},
    ),
    AscentPath: lambda: AscentPath(
        treatments=("x1",), points=((0.0,), (0.5,)), values=(1.0, 2.0), stop="decrease"
    ),
    StationaryPoint: lambda: StationaryPoint(
        treatments=("x1", "x2"),
        origin=(0.0, 0.0),
        point=(1.0, -1.0),
        value=3.0,
        kind="maximum",
        gradient=(2.0, -2.0),
        eigenvalues=(-2.0, -1.0),
        eigenvectors=((1.0, 0.0), (0.0, 1.0)),
    ),
    SurfaceSpec: lambda: SurfaceSpec(
        name="demo",
        treatments=(Treatment(name="a", dimension=D.currency, unit="USD"),),
        outcome=Outcome(name="y", dimension=D.outcome),
        kernels={"a": HillKernel(reference_dose=2.0)},
        carryover={"a": GeometricCarryover(max_lag=3)},
        intercept="hierarchical",
        unit_labels=("u0", "u1", "u2"),
    ),
    DosePlan: lambda: DosePlan(distribution="lognormal", scale=50.0, spread=0.5, zero_fraction=0.1),
    Allocation: lambda: Allocation(
        doses={"x1": 3.3, "x2": 2.7},
        expected_outcome=8.2,
        budget=6.0,
        objective="mean",
        method="slsqp",
    ),
    Frontier: lambda: Frontier(
        budgets=(2.0,),
        outcomes=(5.0,),
        allocations=(
            Allocation(
                doses={"x1": 1.0, "x2": 1.0},
                expected_outcome=5.0,
                budget=2.0,
                objective="mean",
                method="slsqp",
            ),
        ),
    ),
    PointEstimate: lambda: PointEstimate(
        theta={"mu": 1.0, "alpha": (0.5, 1.5)},
        log_density=-12.3,
        converged=True,
        method="trust-ncg",
        n_iter=7,
    ),
    SampleSettings: lambda: SampleSettings(draws=500, tune=500, chains=2, target_accept=0.95),
    ConvergenceThresholds: lambda: ConvergenceThresholds(
        rhat_max=1.01, ess_min=400.0, divergences_max=0
    ),
    ParameterDiagnostics: lambda: ParameterDiagnostics(
        name="mu", rhat=1.001, ess_bulk=900.0, ess_tail=850.0, mcse_mean=0.03, mean=1.2, sd=0.4
    ),
    ConvergenceReport: lambda: ConvergenceReport(
        rows=(
            ParameterDiagnostics(
                name="mu",
                rhat=1.001,
                ess_bulk=900.0,
                ess_tail=850.0,
                mcse_mean=0.03,
                mean=1.2,
                sd=0.4,
            ),
        ),
        divergences=0,
        thresholds=ConvergenceThresholds(),
        converged=True,
        n_chains=4,
        n_draws=1000,
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
