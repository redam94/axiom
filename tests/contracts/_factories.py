"""One example instance per ``Spec`` subclass, for gate 4.

Every concrete ``Spec`` shipped in ``src/axiom`` must have an entry here.
``test_spec_roundtrip.py`` fails naming the class if one is missing, which is
the point: a spec nobody can build an example of is a spec nobody has
round-tripped.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from fractions import Fraction

import pandas as pd

from axiom.adapters import MarketingRoles
from axiom.adapters.agronomy import EconomicOptimum, Prices, TrialRoles
from axiom.calibrate import (
    Agreement,
    CalibratedSpec,
    Correction,
    Ledger,
    Measurement,
    ResolvedTransfer,
    aggregation_level,
    carryover_window_factor,
    derive_prior,
    resolve,
    variance_reweight,
)
from axiom.core import (
    AcceptanceRegion,
    Add,
    Apply,
    Assumption,
    Blocked,
    Const,
    Constraint,
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
    LatentSelection,
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
    clopper_pearson,
    dimensionless,
    wald,
)
from axiom.data import ColumnScaling, Completeness, RoleMap, ScalingParameters
from axiom.design import (
    MDE,
    AnchoredEffect,
    AnytimeLook,
    ArmAllocation,
    ArmAssignment,
    Assignment,
    BalanceRow,
    Boundary,
    CalibrationResult,
    CandidateScore,
    ClusterDesign,
    Collision,
    ConfidenceSequence,
    CostPerOutcomeInterval,
    CostPerOutcomePower,
    CrossingProbabilities,
    DecisionSpec,
    DesignCandidate,
    EconomicInputs,
    EIGEstimate,
    EVOIResult,
    ExperimentValue,
    Factorial,
    FactorialCell,
    FisherInformation,
    HoldoutTradeoff,
    IdentifiabilityRidge,
    IdentifyingDesign,
    Leaderboard,
    LeaderboardRow,
    LearningPriority,
    LookSchedule,
    MethodEstimate,
    MethodSpec,
    Occupancy,
    OnlineDecision,
    OnlineProgram,
    OperatingCharacteristics,
    OpportunityCost,
    PowerCurve,
    PowerResult,
    ProgramDecision,
    ProgramReport,
    ProgramSchedule,
    Readout,
    Recommendation,
    ReExperimentTiming,
    SampleSize,
    Schedule,
    ScheduledExperiment,
    SensitivityTable,
    SimulatedPower,
    SimulationSpec,
    StoppedEstimate,
    StoppingRule,
    StudySummary,
    TreatmentCandidate,
    ValuePerOutcome,
    alpha_spending,
    anchor_draws,
    assign,
    collisions,
    confidence_sequence,
    cost_per_outcome_interval,
    cost_per_outcome_power,
    crossing_probabilities,
    eig_monte_carlo,
    evaluate_candidate,
    evoi_gaussian,
    experiment_value,
    factorial,
    harm_boundary,
    holdout_tradeoff,
    match_clusters,
    mde,
    method_spec,
    monitor,
    online_decisions,
    operating_characteristics,
    opportunity_cost,
    perturb,
    pocock,
    power,
    power_curve,
    program_decisions,
    pulse,
    rank_treatments,
    recommend,
    sample_size,
    schedule_with_cooldown,
    stopped_estimate,
    time_to_re_experiment,
)
from axiom.design.identifiability import (
    Combination,
    EstimabilityReport,
    Prescription,
    ProfileReport,
)
from axiom.diagnose import (
    ArmCount,
    Attrition,
    AttritionRow,
    Backtest,
    BalanceCheck,
    BalanceTest,
    Benchmark,
    BiasBounds,
    CoverageResult,
    Delivery,
    DeliveryReport,
    DeliveryRow,
    EstimandCoverage,
    EstimandCoverageResult,
    FitSettings,
    HorizonScore,
    LearningReport,
    OriginFailure,
    OriginForecast,
    ParameterCoverage,
    ParameterLearning,
    ParameterRanks,
    PPCResult,
    PriorPredictive,
    Refutation,
    ResidualReport,
    ResidualTest,
    RobustnessValue,
    SampleRatio,
    SBCResult,
    SBCSpec,
    SpecCurve,
    SpecCurveSummary,
    SpecificationAxis,
    SpecOption,
    SpecRow,
    StatisticCheck,
    TippingPoint,
    UnitResiduals,
    WeakIdReport,
    attrition,
    benchmark,
    bias_bounds,
    check_delivery,
    rank_uniformity,
    robustness_value,
    tipping_point,
)
from axiom.diagnose.structure import (
    ImpliedIndependence,
    IndependenceCheck,
    StructureRefutation,
    refute_structure,
)
from axiom.discover import Dataset as DiscoveryDataset
from axiom.discover import EssentialGraph, GaussianBIC
from axiom.discover.fci import PAG, PagEdge
from axiom.discover.independence import IndependenceResult
from axiom.discover.search import DiscoveryResult, ges
from axiom.discover.stability import EdgeSupport, StabilityReport
from axiom.dynamics import (
    Block,
    BlockOrder,
    BlockSolution,
    DynamicEquation,
    DynamicSystem,
    Unrolled,
    Variable,
    block_order,
    conditional_form,
    parse_system,
    unroll,
)
from axiom.estimands import Estimand, EstimandResult, FacetDiff, Level, Quantity, TransferPlan
from axiom.identify import (
    CausalGraph,
    ComplianceReport,
    ComplianceTable,
    DerivativeReport,
    EndogeneityTest,
    FirstStage,
    FrontDoorRoute,
    InstrumentRoute,
    LeeBounds,
    LinearEstimate,
    RoleAssignment,
    assign_roles,
    compliance,
    identify,
    lee_bounds,
    response_to_dose,
    transport_verdict,
)
from axiom.identify.cluster import ClusterDAG
from axiom.identify.cyclic import MixedGraph
from axiom.identify.dynamic import SequentialPlan, sequential_plan
from axiom.identify.formula import Density, Marginal, Product, Ratio
from axiom.identify.id_algorithm import Hedge, IdentifiedEffect, identify_effect
from axiom.identify.transport import TransportVerdict
from axiom.identify.verdict import IdentificationVerdict
from axiom.infer import (
    ConvergenceReport,
    ConvergenceThresholds,
    ParameterDiagnostics,
    PointEstimate,
    SampleSettings,
)
from axiom.io import (
    CatalogEntry,
    Change,
    Consensus,
    Definition,
    Deviation,
    ExperimentRun,
    Program,
    Provenance,
    Transition,
)
from axiom.meta import (
    BaujatData,
    Cell,
    Commensurability,
    Corpus,
    EffectShrinkage,
    EggerTest,
    EpsilonCharge,
    EpsilonLedger,
    EpsilonSplit,
    ForestData,
    ForestRow,
    FunnelContour,
    FunnelData,
    Heterogeneity,
    Incompatibility,
    LeaveOneOut,
    ParameterSummary,
    Pooled,
    PooledEstimate,
    PoolPriors,
    PoolResult,
    PoolSpec,
    PrivacyPolicy,
    Release,
    StudyRecord,
    TauEstimate,
    baujat,
    charge,
    commensurable,
    egger,
    fixed_effect,
    forest_data,
    funnel_data,
    heterogeneity,
    leave_one_out,
    orthogonal_split,
    pool,
    random_effects,
    release,
    tau_dersimonian_laird,
)
from axiom.report import (
    Divider,
    Heading,
    LedgerBlock,
    Metric,
    PageBreak,
    Paragraph,
    Report,
    Section,
    Theme,
)
from axiom.report import (
    Figure as ReportFigure,
)
from axiom.report import (
    Table as ReportTable,
)
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
    GaussianProcessKernel,
    GeometricCarryover,
    HillKernel,
    LinearKernel,
    LinearTrend,
    LogisticKernel,
    NoCarryover,
    NuisanceSet,
    PiecewiseLinearKernel,
    PolynomialKernel,
    PowerKernel,
    ResponseBand,
    SplineKernel,
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


def _constraint(family: str = "normal") -> Constraint:
    """A soft constraint on the mean Hill response over the dose column (D6.3)."""
    if family == "lognormal":
        return Constraint(
            name="lift_log@study-2",
            expr=Reduce(op="mean", arg=Add(terms=(Param(name="a", dimension=D.outcome), _hill()))),
            family="lognormal",
            observed=2.0,
            scale=0.25,
            detail={"measurement": "study-2"},
        )
    return Constraint(
        name="lift@study-1",
        expr=Reduce(op="mean", arg=_hill()),
        family="normal",
        observed=1.2,
        scale=0.3,
        detail={"measurement": "study-1"},
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


def _measurement() -> Measurement:
    return Measurement(
        estimand=_estimand(),
        estimate=1.2,
        se=0.3,
        definition="wald",
        mass=0.95,
        method="difference_in_differences",
        n_units=40,
        n_periods=8,
        source="study-1",
    )


def _plan() -> TransferPlan:
    """A source/target pair differing in window and level (the ledger gate's shape)."""
    source = _estimand(
        name="experiment",
        window=TimeWindow(start=0, stop=2, basis="cumulative"),
        level=Level(unit="individual"),
    )
    target = _estimand(
        name="decision",
        window=TimeWindow(start=0, stop=8, basis="cumulative"),
        level=Level(unit="cluster"),
    )
    return source.transfer_to(target)


def _resolved() -> ResolvedTransfer:
    window = carryover_window_factor(
        GeometricCarryover(max_lag=6), {"lam_fertilizer": 0.7}, 2, treatment="fertilizer"
    )
    level = aggregation_level(
        Level(unit="individual"), Level(unit="cluster"), cluster_size=25, icc=0.05
    )
    assert isinstance(window, Correction) and isinstance(level, Correction)
    return resolve(_plan(), corrections=[window, level])


def _calibrated_spec() -> CalibratedSpec:
    spec = SurfaceSpec(
        name="calibrated_demo",
        treatments=(Treatment(name="fertilizer", dimension=D.currency, unit="USD"),),
        outcome=Outcome(name="yield_total", dimension=D.outcome, unit="kg"),
        kernels={"fertilizer": HillKernel(reference_dose=50.0)},
    )
    out = derive_prior(
        [_measurement()],
        spec,
        "fertilizer",
        beta_draws=(1.0, 1.1, 0.9, 1.05),
        contribution_draws=(10.0, 11.2, 8.9, 10.4),
    )
    assert isinstance(out, CalibratedSpec)
    return out


def _estimand_result() -> EstimandResult:
    return EstimandResult(
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
    )


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


_CLUSTER = Unit(name="region", dimension=D.entity, kind="cluster")


def _cluster_design() -> ClusterDesign:
    return ClusterDesign(unit=_CLUSTER, n_clusters=20, cluster_size=50, icc=0.05, allocation=0.5)


def _decision() -> DecisionSpec:
    return DecisionSpec(
        name="scale_up", threshold=0.5, value_per_outcome_unit=100.0, numeraire="USD"
    )


def _value_per_outcome() -> ValuePerOutcome:
    return ValuePerOutcome(
        value=3.0, outcome_unit="kg", numeraire="USD", source="contract price", assumption=None
    )


def _opportunity_cost() -> OpportunityCost:
    return opportunity_cost(
        0.2, 8, 100.0, (1.5, 2.0, 2.5), _value_per_outcome(), 0.01, dose_cost_per_unit=1.0
    )


def _candidate(name: str = "holdout_a", se: float = 0.2, cost: float = 50.0) -> DesignCandidate:
    return DesignCandidate(
        name=name,
        method="difference_in_differences",
        n_units=40,
        n_periods=8,
        holdout_fraction=0.25,
        experiment_se=se,
        cost=cost,
        cooldown_periods=2,
    )


def _economics() -> EconomicInputs:
    return EconomicInputs(
        value_per_outcome=_value_per_outcome(),
        dose_per_period=100.0,
        discount_rate=0.01,
        dose_cost_per_unit=1.0,
        marginal_value_ratio=1.2,
    )


def _score(name: str = "holdout_a", se: float = 0.2, cost: float = 50.0) -> CandidateScore:
    return evaluate_candidate(_candidate(name, se, cost), _decision(), 1.0, 0.8, _economics())


def _treatment_candidate(name: str = "fertilizer", se: float = 0.3) -> TreatmentCandidate:
    return TreatmentCandidate(
        name=name,
        prior_mean=1.0,
        prior_sd=0.8,
        experiment_se=se,
        decision=_decision(),
        opportunity_cost=2.0,
        fixed_cost=4.0,
    )


def _calibration() -> CalibrationResult:
    return CalibrationResult(
        method="difference_in_differences",
        design="holdout",
        n_simulations=100,
        n_unsupported=0,
        false_positive_count=4,
        false_positive_rate=0.04,
        alpha=0.05,
        region=clopper_pearson(100, 0.05, 1e-3),
        passed=True,
        seed=0,
    )


def _simulated_power() -> SimulatedPower:
    return SimulatedPower(
        method="difference_in_differences",
        design="holdout",
        effect=0.5,
        truth=0.5,
        n_simulations=100,
        n_unsupported=0,
        rejections=81,
        power=0.81,
        alpha=0.05,
        mean_effect=0.49,
        bias=-0.01,
        coverage=0.95,
        rmse=0.2,
        predicted_power=0.8,
        region=clopper_pearson(100, 0.8, 1e-3),
        within_prediction=True,
        seed=0,
    )


def _leaderboard_row() -> LeaderboardRow:
    return LeaderboardRow(
        method="difference_in_differences",
        design="holdout",
        status="stable",
        calibrated=True,
        false_positive_rate=0.04,
        powers=(0.81,),
        mean_power=0.81,
        biases=(-0.01,),
        coverages=(0.95,),
        rmses=(0.2,),
        n_unsupported=0,
        rank=1,
    )


def _fisher() -> FisherInformation:
    return FisherInformation(
        parameters=("alpha", "beta_a", "k_a"),
        matrix=((4.0, 1.0, 0.5), (1.0, 3.0, 0.2), (0.5, 0.2, 2.0)),
        noise_sd=0.5,
        n_observations=24,
        det=19.37,
        min_eigenvalue=1.76,
        singular=False,
        method="finite",
        detail={"round_off_columns": ""},
    )


# -- meta ------------------------------------------------------------------------------------

_META_Y = (0.42, 0.55, 0.31, 0.67, 0.48, 0.39)
_META_SE = (0.10, 0.15, 0.12, 0.20, 0.11, 0.14)


def _study(i: int) -> StudyRecord:
    return StudyRecord(
        study=f"s{i}",
        contributor=f"c{i % 3}",
        quantity="elasticity",
        estimate=_META_Y[i],
        se=_META_SE[i],
        read="experiment" if i % 2 else "model",
        family="fertilizer",
        n=40 + 10 * i,
        moderators={"follow_up": float(4 + 2 * i)},
        source=f"report-{i}",
    )


def _corpus() -> Corpus:
    return Corpus(records=tuple(_study(i) for i in range(6)), name="demo")


def _pool_result() -> PoolResult:
    spec = PoolSpec(family="fertilizer", bias_term=True, priors=PoolPriors(mu_scale=2.0))
    out = pool(spec, _corpus(), draws=200, seed=0)
    assert isinstance(out, Pooled), out
    return out.result


def _cell() -> Cell:
    return Cell(records=(("c0", 0.42), ("c1", 0.55), ("c2", 0.31), ("c3", 0.67)), name="fertilizer")


def _ledger() -> EpsilonLedger:
    out = charge(EpsilonLedger(budget=1.0), release_id="r1", epsilon=0.25, mechanism="laplace")
    assert isinstance(out, EpsilonLedger), out
    return out


def _release() -> Release:
    out = release(
        _cell(),
        PrivacyPolicy(k=3, epsilon_total=1.0),
        EpsilonLedger(budget=1.0),
        release_id="r1",
        epsilon=0.5,
        clip=(0.0, 1.0),
        seed=0,
    )
    assert isinstance(out, tuple), out
    return out[0]


# -- diagnose (Phase 8) --------------------------------------------------------------------------

# Cinelli & Hazlett (2020), Table 1: the Darfur "directly harmed" estimate.
_DARFUR = (0.0973, 0.0232, 783)


def _ranks() -> ParameterRanks:
    """Sixty ranks cycling through 0..19: exactly uniform, so the check passes."""
    return rank_uniformity("beta", [i % 20 for i in range(60)], n_ranks=19, alpha=0.05)


def _sbc_result() -> SBCResult:
    ranks = _ranks()
    return SBCResult(
        spec=SBCSpec(n_simulations=60, draws=100, rank_draws=19, seed=1, parameters=("beta",)),
        model_hash="a" * 64,
        refit_model_hash="a" * 64,
        alpha_per_parameter=0.05,
        n_simulations=60,
        n_fitted=60,
        n_failed_fits=0,
        parameters=(ranks,),
        failed_parameters=(),
        passed=True,
    )


_REGION_10 = clopper_pearson(10, 0.9, 0.01)


def _parameter_coverage(name: str = "beta_a", covered: int = 9) -> ParameterCoverage:
    return ParameterCoverage(
        name=name,
        n=10,
        covered=covered,
        mass=0.9,
        definition="eti",
        region=_REGION_10,
        passed=_REGION_10.accepts(covered),
    )


def _coverage_result() -> CoverageResult:
    rows = (_parameter_coverage("alpha", 10), _parameter_coverage("beta_a", 9))
    return CoverageResult(
        n=10,
        mass=0.9,
        definition="eti",
        alpha=0.01,
        n_fitted=10,
        n_failed_fits=0,
        nominal_region=_REGION_10,
        parameters=rows,
        failed_parameters=(),
        passed=True,
        provenance={"backend": "laplace", "draws": 100},
    )


def _estimand_coverage(covered: int = 9) -> EstimandCoverage:
    return EstimandCoverage(
        name="contrast_at_dose",
        estimand_hash="e" * 64,
        n=10,
        covered=covered,
        mass=0.9,
        definition="eti",
        region=_REGION_10,
        passed=_REGION_10.accepts(covered),
        true_values=tuple(12.0 + 0.1 * i for i in range(10)),
    )


def _estimand_coverage_result() -> EstimandCoverageResult:
    row = _estimand_coverage()
    return EstimandCoverageResult(
        n=10,
        mass=0.9,
        definition="eti",
        alpha=0.01,
        n_fitted=10,
        n_failed_fits=0,
        nominal_region=_REGION_10,
        estimands=(row,),
        failed_estimands=(),
        passed=True,
    )


def _parameter_learning(name: str = "beta_a", dominated: bool = False) -> ParameterLearning:
    return ParameterLearning(
        name=name,
        prior_mean=0.0,
        prior_sd=2.0,
        posterior_mean=1.0 if not dominated else 0.05,
        posterior_sd=0.2 if not dominated else 1.95,
        contraction=0.99 if not dominated else 0.05,
        overlap=0.3 if not dominated else 0.99,
        shift=0.5 if not dominated else 0.025,
        prior_dominated=dominated,
    )


def _learning_report() -> LearningReport:
    return LearningReport(
        parameters=(_parameter_learning(), _parameter_learning("k_a", dominated=True)),
        n_prior=4000,
        n_posterior=400,
        seed=0,
        threshold=0.1,
        prior_dominated=("k_a",),
        passed=False,
    )


def _statistic_check(name: str = "mean", p: float = 0.4) -> StatisticCheck:
    two = min(1.0, 2.0 * min(p, 1.0 - p))
    return StatisticCheck(
        name=name,
        observed=1.2,
        interval=Interval(lower=0.8, upper=1.6, definition="eti", mass=0.9),
        p_value=p,
        p_two_sided=two,
        n=200,
        alpha=0.05,
        extreme=two < 0.05,
    )


def _ppc_result() -> PPCResult:
    checks = (_statistic_check("mean", 0.4), _statistic_check("max", 0.01))
    return PPCResult(
        n_draws=200,
        seed=0,
        mass=0.9,
        alpha=0.05,
        statistics=checks,
        skipped=("nan_statistic",),
        extreme_statistics=("max",),
        provenance={"spec_hash": "a" * 64},
    )


def _refutation() -> Refutation:
    return Refutation(
        kind="placebo_treatment",
        estimand_name="contrast_a",
        estimand_hash="e" * 64,
        treatment="a",
        original=3.2,
        original_interval=Interval(lower=2.4, upper=4.0, definition="eti", mass=0.9),
        refuted_mean=0.1,
        refuted_sd=0.3,
        refuted_interval=Interval(lower=-0.4, upper=0.6, definition="eti", mass=0.9),
        refuted_estimates=(0.1,),
        p_value=0.0,
        n=200,
        n_refits=1,
        n_failed=0,
        alpha=0.1,
        rule="share of N = 200 placebo draws at or beyond the original is below alpha",
        passed=True,
        seed=0,
        detail={"placebo_ratio": 0.03125},
    )


def _residual_report() -> ResidualReport:
    tests = (
        ResidualTest(name="durbin_watson", statistic=1.9, n=24, note="no p-value"),
        ResidualTest(name="ljung_box[1]", statistic=0.4, p_value=0.52, n=24, lag=1),
        ResidualTest(name="shapiro_wilk", statistic=0.97, p_value=0.02, n=24),
    )
    return ResidualReport(
        n_units=3,
        n_periods=8,
        n=24,
        alpha=0.05,
        residual_sd=0.31,
        units=tuple(UnitResiduals(unit=f"u{i}", n=8, mean=0.01 * i, sd=0.3) for i in range(3)),
        tests=tests,
        skipped=("ljung_box[10]: lag must be below n_periods=8",),
        flagged=("shapiro_wilk",),
    )


def _spec_axes() -> tuple[SpecificationAxis, ...]:
    return (
        SpecificationAxis(
            name="kernel",
            options=(
                SpecOption(label="hill"),
                SpecOption(label="logistic", spec_update={"kernels": {"a": LogisticKernel()}}),
            ),
        ),
        SpecificationAxis(
            name="draws",
            options=(SpecOption(label="few", fit_update={"draws": 50}),),
        ),
    )


def _spec_row(index: int = 0) -> SpecRow:
    return SpecRow(
        index=index,
        labels={"kernel": "hill", "draws": "few"},
        spec_hash="s" * 64,
        spec_name="surface",
        estimate=1.1,
        interval=Interval(lower=0.6, upper=1.6, definition="eti", mass=0.9),
        sd=0.3,
        n_draws=50,
        converged=True,
    )


def _spec_curve() -> SpecCurve:
    failed = SpecRow(
        index=1,
        labels={"kernel": "logistic", "draws": "few"},
        spec_hash="t" * 64,
        spec_name="surface",
        failure="backend declined",
    )
    return SpecCurve(
        estimand_name="contrast_a",
        estimand_hash="e" * 64,
        base_spec_hash="b" * 64,
        axes=_spec_axes(),
        rows=(_spec_row(0), failed),
        n_total=2,
        n_dropped=0,
        max_specs=64,
        definition="eti",
        mass=0.9,
        settings=FitSettings(draws=50),
        seed=0,
    )


def _prior_predictive() -> PriorPredictive:
    def summary(mean: float, sd: float) -> Summary:
        return Summary(
            mean=mean,
            median=mean,
            sd=sd,
            interval=Interval(
                lower=mean - 1.6 * sd, upper=mean + 1.6 * sd, definition="eti", mass=0.9
            ),
            n=200,
        )

    return PriorPredictive(
        spec_hash="a" * 64,
        panel_hash="p" * 64,
        n=200,
        seed=0,
        n_units=3,
        n_periods=8,
        treatments=("a",),
        outcome_sd=1.4,
        expected_sign="positive",
        magnitude_factor=10.0,
        tolerance=0.2,
        mass=0.9,
        response=summary(2.0, 1.5),
        response_range=Interval(lower=-1.0, upper=5.0, definition="eti", mass=0.9),
        contribution={"a": summary(1.0, 0.8)},
        share_wrong_sign={"a": 0.0},
        share_implausible_magnitude={"a": 0.05},
        share_flagged=0.05,
        passed=True,
    )


def _weak_id_report() -> WeakIdReport:
    return WeakIdReport(
        parameters=("beta_a", "k_a"),
        n_draws=400,
        correlation=((1.0, 0.95), (0.95, 1.0)),
        condition_number=40.0,
        rho_threshold=0.9,
        condition_threshold=1e3,
        ratio_threshold=0.9,
        high_pairs=(("beta_a", "k_a", 0.95),),
        prior_sd={"beta_a": 1.0, "k_a": 2.0},
        posterior_sd={"beta_a": 0.3, "k_a": 1.9},
        sd_ratio={"beta_a": 0.3, "k_a": 0.95},
        unlearned=("k_a",),
        saturated=(),
        saturation_share=0.1,
        saturation_eps=1e-3,
        resolved_hypers={},
        passed=False,
    )


_REGION_6 = clopper_pearson(6, 0.9, 0.01)


def _horizon_score(step: int = 1) -> HorizonScore:
    covered = 6 if step == 1 else 5
    return HorizonScore(
        step=step,
        n=6,
        mae=0.1 * step,
        rmse=0.12 * step,
        crps=0.05 * step,
        coverage=covered / 6,
        coverage_region=_REGION_6,
        passed=_REGION_6.accepts(covered),
        bias=0.01 * step,
    )


def _origin_forecast() -> OriginForecast:
    return OriginForecast(
        origin=8,
        periods=(8.0, 9.0),
        units=("u0", "u1", "u2"),
        observed=((1.0, 1.1), (0.9, 1.0), (1.2, 1.3)),
        mean=((1.0, 1.1), (0.95, 1.05), (1.15, 1.25)),
        lower=((0.8, 0.9), (0.75, 0.85), (0.95, 1.05)),
        upper=((1.2, 1.3), (1.15, 1.25), (1.35, 1.45)),
        converged=True,
    )


def _backtest() -> Backtest:
    return Backtest(
        spec_hash="a" * 64,
        spec_name="surface",
        panel_hash="p" * 64,
        origins=(8, 10),
        horizon=2,
        n_units=3,
        mass=0.9,
        definition="eti",
        alpha=0.01,
        scores=(_horizon_score(1), _horizon_score(2)),
        forecasts=(_origin_forecast(),),
        failures=(OriginFailure(origin=10, reason="backend declined"),),
        n_failed_fits=1,
        backend="laplace",
        draws=100,
        seed=0,
    )


_LOOKS = (0.25, 0.5, 0.75, 1.0)


def _stopping_rule() -> StoppingRule:
    return StoppingRule(
        name="hyper3_primary",
        looks=LookSchedule(labels=("week_6", "week_12", "week_18", "week_24"), information=_LOOKS),
        boundaries=(
            alpha_spending(0.025, _LOOKS, side="upper", kind="efficacy"),
            harm_boundary(0.95, _LOOKS, margin=2.0, se_at_full_information=1.5),
        ),
    )


def _report_section() -> Section:
    return Section(
        title="Readout",
        summary="Every number carries its interval.",
        blocks=(
            Heading(text="Headline", level=2),
            Paragraph(text="The effect is {effect:.1f} units."),
            Metric(source="contrast", label="Effect at 50", unit="mmHg"),
            ReportFigure(source="response", caption="With its 90 % band"),
            ReportTable(source="arms", caption="Arm means"),
            LedgerBlock(source="ledger"),
            Divider(),
            PageBreak(),
        ),
    )


# -- dynamics ---------------------------------------------------------------------------


def _dynamic_variables() -> tuple[Variable, ...]:
    return (
        Variable(name="stock", dimension=D.outcome, initial=0.0),
        Variable(name="inflow", dimension=D.outcome, role="exogenous"),
    )


def _dynamic_system() -> DynamicSystem:
    return parse_system(
        "stock = decay * stock[t-1] + inflow",
        variables=_dynamic_variables(),
        parameters=(Param(name="decay", dimension=Dimension(exponents={})),),
        name="one-compartment",
    )


def _dynamic_equation() -> DynamicEquation:
    return _dynamic_system().equation("stock")


def _block_order() -> BlockOrder:
    return block_order(_dynamic_system())


def _block() -> Block:
    return _block_order().blocks[0]


def _block_solution() -> BlockSolution:
    compiled = conditional_form(_dynamic_system())
    assert isinstance(compiled, Unrolled)
    return compiled.solutions[0]


def _unrolled() -> Unrolled:
    compiled = unroll(_dynamic_system(), periods=3)
    assert isinstance(compiled, Unrolled)
    return compiled


def _sequential_plan() -> SequentialPlan:
    graph = CausalGraph.from_edges(
        "dose.t0 -> outcome.t0, outcome.t0 -> dose.t1, dose.t1 -> outcome.t1, "
        "outcome.t0 -> outcome.t1"
    )
    return sequential_plan(graph, ["dose.t0", "dose.t1"], "outcome.t1")


# -- identifiability --------------------------------------------------------------------


def _combination() -> Combination:
    return Combination(
        exponents={"alpha": 1, "k": -1},
        kind="estimable",
        scaling="log",
        score=1.4,
        label="alpha / k",
    )


def _estimability_report() -> EstimabilityReport:
    return EstimabilityReport(
        parameters=("alpha", "k"),
        observations=("low dose",),
        scaling="log",
        n_points=2,
        rank=1,
        deficiency=1,
        persistent_deficiency=0,
        singular_values=(6.8, 0.02),
        condition_number=336.0,
        symmetries=(
            Combination(
                exponents={"alpha": 1, "k": 1},
                kind="symmetry",
                score=0.01,
                label="alpha -> alpha*c, k -> k*c",
            ),
        ),
        estimable=(_combination(),),
        null_basis=((0.7071, 0.7071),),
        tolerance=0.03,
    )


def _profile_report() -> ProfileReport:
    return ProfileReport(
        targets=("alpha", "alpha / k"),
        truth={"alpha": 4.0, "alpha / k": 0.4},
        recovered={"alpha": 0.38, "alpha / k": 0.4},
        interval={"alpha": (0.17, float("inf")), "alpha / k": (0.39, 0.41)},
        level=0.95,
        threshold=1.92,
        flat=("alpha",),
        n_observations=24,
        seed=11,
    )


def _prescription() -> Prescription:
    return Prescription(
        added=("wide dose",),
        rank_before=1,
        rank_after=2,
        n_parameters=2,
        broken=(("alpha * k",),),
        considered=("wide dose",),
        complete=True,
    )


# -- graphs with cycles, and identification formulas -------------------------------------


def _mixed_graph() -> MixedGraph:
    return MixedGraph.from_edges("x -> a, a -> b, b -> a, b -> y", name="a market at equilibrium")


def _density() -> Density:
    return Density(outcomes=("Y",), given=("X", "Z"))


def _product() -> Product:
    return Product(factors=(_density(), Density(outcomes=("Z",))))


def _marginal() -> Marginal:
    return Marginal(over=("Z",), term=_product())


def _ratio() -> Ratio:
    return Ratio(numerator=Density(outcomes=("X", "Y")), denominator=Density(outcomes=("X",)))


def _hedge() -> Hedge:
    return Hedge(root=("X", "Y"), subset=("Y",), variables=("X", "Y"))


def _identified_effect() -> IdentifiedEffect:
    return identify_effect(CausalGraph.from_edges("Z -> X, Z -> Y, X -> Y"), "X", "Y")


# -- discovery ---------------------------------------------------------------------------


def _cluster_dag() -> ClusterDAG:
    return ClusterDAG(
        clusters={"demand": ("price", "quantity"), "cost": ("wage",)},
        edges=(("cost", "demand"),),
        name="a two-block market",
    )


def _essential_graph() -> EssentialGraph:
    from axiom.discover import cpdag

    return cpdag(CausalGraph.from_edges("a -> c, b -> c, c -> d"))


def _discovery_dataset() -> DiscoveryDataset:
    import numpy as np

    rng = np.random.default_rng(0)
    a = rng.normal(size=200)
    b = 1.5 * a + rng.normal(size=200)
    return DiscoveryDataset(np.column_stack([a, b]), ("a", "b"), (frozenset(),) * 200)


def _discovery_result() -> DiscoveryResult:
    return ges(GaussianBIC(_discovery_dataset()))


# -- refutation and discovery under latents ----------------------------------------------


def _independence_result() -> IndependenceResult:
    return IndependenceResult(
        x="heat",
        y="growth",
        given=("light",),
        correlation=0.65,
        p_value=1e-60,
        n=2000,
        statistic=38.5,
    )


def _implied_independence() -> ImpliedIndependence:
    return ImpliedIndependence(x="heat", y="growth", given=("light",))


def _structure_refutation() -> StructureRefutation:
    import numpy as np

    rng = np.random.default_rng(0)
    rows = 400
    a = rng.normal(size=rows)
    b = 1.4 * a + rng.normal(size=rows)
    c = 0.9 * b + rng.normal(size=rows)
    data = DiscoveryDataset(np.column_stack([a, b, c]), ("a", "b", "c"), (frozenset(),) * rows)
    report = refute_structure(CausalGraph.from_edges("a -> b, b -> c"), data)
    assert isinstance(report, StructureRefutation)
    return report


def _independence_check() -> IndependenceCheck:
    return _structure_refutation().checks[0]


def _pag_edge() -> PagEdge:
    return PagEdge(a="sprout", b="harvest", mark_a="arrow", mark_b="arrow")


def _pag() -> PAG:
    return PAG(
        nodes=("harvest", "seed", "sprout"),
        edges=(
            _pag_edge(),
            PagEdge(a="seed", b="sprout", mark_a="circle", mark_b="arrow"),
        ),
        limits_hit=("Zhang's rules R4 and R5-R10 are not implemented",),
    )


def _edge_support() -> EdgeSupport:
    return EdgeSupport(
        a="heat", b="light", adjacent=0.97, forward=0.10, backward=0.02, undirected=0.85
    )


def _stability_report() -> StabilityReport:
    return StabilityReport(
        variables=("heat", "light"),
        edges=(_edge_support(),),
        n_bootstrap=60,
        n_rows=2000,
        penalty=1.0,
        seed=7,
    )


def _deviation() -> Deviation:
    return Deviation(
        role="window",
        planned="a" * 64,
        realized="b" * 64,
        reason="the field ran four weeks long",
        at="2026-05-04T09:00:00+00:00",
        stage="running",
        state="asserted",
    )


def _experiment_run() -> ExperimentRun:
    return ExperimentRun(
        experiment="NW-14",
        scope="northwind/dose-response",
        stage="read",
        roles={"window": "b" * 64, "schedule": "c" * 64},
        estimand_hash="d" * 64,
        plan_hash="e" * 64,
        readout_hash="f" * 64,
        readout_estimand="d" * 64,
        deviations=(_deviation(),),
        ledger=(LedgerLine(kind="note", statement="the field window slipped four weeks"),),
        history=(
            Transition(stage="designed", at="2026-03-02T09:00:00+00:00"),
            Transition(stage="committed", at="2026-03-04T09:00:00+00:00", note="plan frozen"),
        ),
    )


def _stopped_estimate() -> StoppedEstimate:
    looks = LookSchedule(labels=("L1", "L2", "L3"), information=(0.4, 0.7, 1.0))
    rule = StoppingRule(name="Pocock-3", looks=looks, boundaries=(pocock(0.05, looks),))
    result = stopped_estimate(monitor(rule, [0.6, 0.9, 2.6], ses=[0.9, 0.7, 0.5]))
    assert isinstance(result, StoppedEstimate)
    return result


_ASSIGN_UNITS = tuple(f"u{i:04d}" for i in range(120))
_ASSIGN_ALLOCATION = ArmAllocation(arms=("control", "low", "high"), shares=(0.5, 0.25, 0.25))


def _assigned():  # type: ignore[no-untyped-def]
    n = len(_ASSIGN_UNITS)
    covariates = {
        "age": [20.0 + 40.0 * i / (n - 1) for i in range(n)],
        "pre_outcome": [math.cos(float(i)) for i in range(n)],
    }
    return assign(_ASSIGN_UNITS, _ASSIGN_ALLOCATION, method="block", seed=4, covariates=covariates)


def _arm_assignment() -> ArmAssignment:
    return _assigned().spec


def _balance_row() -> BalanceRow:
    return _arm_assignment().balance[0]


def _delivery_report() -> DeliveryReport:
    assigned = _assigned()
    return check_delivery(
        assigned,
        exposed={"control": 55, "low": 28, "high": 24},
        covariates=dict(assigned.covariates or {}),
    )


def _compliance_frame():  # type: ignore[no-untyped-def]
    n = 400
    assigned = [float(i % 2) for i in range(n)]
    # compliers take it when assigned; a fifth are never-takers, a twentieth always-takers
    exposed = [1.0 if i % 20 == 0 else (0.0 if i % 5 == 0 else a) for i, a in enumerate(assigned)]
    y = [10.0 + 2.0 * e + math.sin(float(i)) for i, e in enumerate(exposed)]
    return pd.DataFrame({"y": y, "assigned": assigned, "exposed": exposed})


def _compliance_report() -> ComplianceReport:
    return compliance(_compliance_frame(), "y", "assigned", "exposed")


_LATENT = LatentSelection(kind="complier", instrument="letter", exposure="attended", share=0.61)


def _commensurability() -> Commensurability:
    everybody = _estimand(name="itt", population=Population(name="enrolled"))
    compliers = _estimand(name="cace", population=Population(name="enrolled", latent=_LATENT))
    corpus = Corpus(records=(_study(0), _study(1)), name="two")
    return commensurable(
        corpus, {corpus.records[0].study: everybody, corpus.records[1].study: compliers}
    )


_OCCUPANCIES = (
    Occupancy(
        experiment="NW-14",
        units=("london", "leeds"),
        window=TimeWindow(start=0, stop=8),
        treatments=("price",),
    ),
    Occupancy(
        experiment="NW-15",
        units=("leeds", "york"),
        window=TimeWindow(start=4, stop=12),
        treatments=("banner",),
    ),
)


def _confidence_sequence() -> ConfidenceSequence:
    return confidence_sequence(
        (0.4, 1.2, 2.1, 2.9), (0.25, 0.5, 0.75, 1.0), alpha=0.05, name="NW-14"
    )


_READOUTS = (
    Readout(experiment="E1", party="acme", metric="primary", p_value=0.001, evalue=40.0),
    Readout(experiment="E1", party="acme", metric="churn", p_value=0.4, evalue=1.1),
    Readout(experiment="E2", party="northwind", metric="primary", p_value=0.02, evalue=8.0),
)


def _program_report() -> ProgramReport:
    return program_decisions(_READOUTS, alpha=0.05, method="e_bh", period="2026-Q3")


def _attrition() -> Attrition:
    return attrition({"treated": 3000, "control": 3000}, {"treated": 2670, "control": 2050})


def _lee_bounds() -> LeeBounds:
    n = 400
    assigned = [float(i % 2) for i in range(n)]
    # every fourth control unit goes silent, so the treated arm over-selects
    reported = [1.0 if (a == 1.0 or i % 4) else 0.0 for i, a in enumerate(assigned)]
    y = [10.0 + 1.5 * a + math.sin(float(i)) for i, a in enumerate(assigned)]
    frame = pd.DataFrame({"y": y, "assigned": assigned, "reported": reported})
    result = lee_bounds(frame, "y", "assigned", "reported")
    if not isinstance(result, LeeBounds):
        raise RuntimeError(f"the factory world must be boundable: {result}")
    return result


def _dose_report() -> DerivativeReport:
    n = 400
    assigned = [float(i % 2) for i in range(n)]
    dose = [1.0 + 0.5 * a + 0.25 * math.cos(float(i)) for i, a in enumerate(assigned)]
    y = [10.0 + 0.8 * d + math.sin(float(i)) for i, d in enumerate(dose)]
    frame = pd.DataFrame({"y": y, "assigned": assigned, "dose": dose})
    return response_to_dose(frame, "y", "assigned", "dose")


def _factorial() -> Factorial:
    n = 400
    left = [float(i % 2) for i in range(n)]
    right = [float((i // 2) % 2) for i in range(n)]
    y = [
        10.0 + 2.0 * a + 1.0 * b + 1.5 * a * b + math.sin(float(i))
        for i, (a, b) in enumerate(zip(left, right, strict=True))
    ]
    frame = pd.DataFrame({"y": y, "price": left, "banner": right})
    return factorial(frame, "y", "price", "banner")


def _online_program() -> OnlineProgram:
    ps = [0.001, 0.02, 0.4, 0.6, 0.003, 0.9]
    stream = [Readout(experiment=f"E{i}", p_value=p) for i, p in enumerate(ps)]
    return online_decisions(stream, alpha=0.05, stream="2026-Q3")


EXAMPLES: dict[type[Spec], Callable[[], Spec]] = {
    IndependenceResult: _independence_result,
    ImpliedIndependence: _implied_independence,
    IndependenceCheck: _independence_check,
    StructureRefutation: _structure_refutation,
    PagEdge: _pag_edge,
    PAG: _pag,
    EdgeSupport: _edge_support,
    StabilityReport: _stability_report,
    ClusterDAG: _cluster_dag,
    EssentialGraph: _essential_graph,
    DiscoveryResult: _discovery_result,
    MixedGraph: _mixed_graph,
    Density: _density,
    Product: _product,
    Marginal: _marginal,
    Ratio: _ratio,
    Hedge: _hedge,
    IdentifiedEffect: _identified_effect,
    Variable: lambda: _dynamic_variables()[0],
    DynamicEquation: _dynamic_equation,
    DynamicSystem: _dynamic_system,
    Block: _block,
    BlockOrder: _block_order,
    BlockSolution: _block_solution,
    Unrolled: _unrolled,
    SequentialPlan: _sequential_plan,
    Combination: _combination,
    EstimabilityReport: _estimability_report,
    ProfileReport: _profile_report,
    Prescription: _prescription,
    ResponseBand: lambda: ResponseBand(
        treatment="a",
        outcome="y",
        kind="response",
        doses=(0.0, 25.0, 50.0),
        mean=(0.0, 4.1, 6.2),
        median=(0.0, 4.0, 6.1),
        lower=(-0.3, 3.4, 5.1),
        upper=(0.3, 4.9, 7.4),
        definition="eti",
        mass=0.9,
        n_draws=200,
        dimension=D.outcome,
        dose_unit="USD",
        outcome_unit="units",
    ),
    Theme: lambda: Theme(name="house", accent_color="#2f7fd1"),
    Heading: lambda: Heading(text="Headline", level=2),
    Paragraph: lambda: Paragraph(text="The effect is {effect:.1f} units."),
    ReportFigure: lambda: ReportFigure(source="response", caption="With its 90 % band"),
    ReportTable: lambda: ReportTable(source="arms", caption="Arm means"),
    Metric: lambda: Metric(source="contrast", label="Effect at 50", unit="mmHg"),
    LedgerBlock: lambda: LedgerBlock(source="ledger", show_detail=True),
    Divider: Divider,
    PageBreak: PageBreak,
    Section: _report_section,
    Report: lambda: Report(
        name="readout", title="HYPER-3", subtitle="as of {as_of}", sections=(_report_section(),)
    ),
    LookSchedule: lambda: LookSchedule(
        labels=("week_6", "week_12", "week_18", "week_24"), information=_LOOKS
    ),
    Boundary: lambda: alpha_spending(0.025, _LOOKS, side="upper", kind="efficacy"),
    StoppingRule: _stopping_rule,
    CrossingProbabilities: lambda: crossing_probabilities(_stopping_rule(), 0.0),
    OperatingCharacteristics: lambda: operating_characteristics(_stopping_rule(), 2.8),
    PowerResult: lambda: power(100, 1.0, 2.0, allocation=0.4),
    MDE: lambda: mde(100, 2.0, power=0.9),
    SampleSize: lambda: sample_size(1.0, 2.0, two_sided=False),
    PowerCurve: lambda: power_curve(100, 2.0, (0.0, 0.5, 1.0, 1.5)),
    ClusterDesign: _cluster_design,
    Assignment: lambda: match_clusters(
        ((1.0, 1.2), (3.0, 3.1), (1.1, 0.9), (2.9, 3.3), (5.0, 5.0)), metric="trajectory", seed=1
    ),
    HoldoutTradeoff: lambda: holdout_tradeoff(_cluster_design(), 2.0, (0.2, 0.3, 0.5)),
    AnchoredEffect: lambda: anchor_draws(
        (0.2, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 1.0, 1.3), "beta", 0.5, credence=0.8
    ),
    CostPerOutcomeInterval: lambda: cost_per_outcome_interval(100.0, 5.0, 1.5, alpha=0.1),
    CostPerOutcomePower: lambda: cost_per_outcome_power(100.0, 5.0, 1.5, threshold=30.0),
    EIGEstimate: lambda: eig_monte_carlo((-1.0, -0.5, 0.0, 0.5, 1.0), 0.5, n_sims=16, seed=3),
    ReExperimentTiming: lambda: time_to_re_experiment(0.2, 10.0, 0.3, 0.1, design_kind="ghost"),
    DecisionSpec: _decision,
    EVOIResult: lambda: evoi_gaussian(_decision(), 1.0, 0.8, 0.4),
    MethodSpec: lambda: method_spec("switchback").model_copy(update={"status": "experimental"}),
    MethodEstimate: lambda: MethodEstimate(
        method="difference_in_differences",
        effect=0.42,
        se=0.1,
        interval=wald(0.42, 0.1, 0.95),
        n_treated=10,
        n_control=10,
        n_pre=8,
        n_post=8,
        se_method="unit_changes",
        detail={"df": 18.0},
    ),
    ValuePerOutcome: _value_per_outcome,
    OpportunityCost: _opportunity_cost,
    ExperimentValue: lambda: experiment_value(120.0, _opportunity_cost(), 25.0),
    StudySummary: lambda: StudySummary(
        treatment="fertilizer", estimate=1.2, se=0.3, periods_ago=4.0, definition="wald"
    ),
    TreatmentCandidate: _treatment_candidate,
    LearningPriority: lambda: rank_treatments(
        (_treatment_candidate(), _treatment_candidate("water", 0.5))
    )[0],
    Recommendation: lambda: recommend(
        (_treatment_candidate(), _treatment_candidate("water", 0.5)), budget=8.0
    ),
    DesignCandidate: _candidate,
    EconomicInputs: _economics,
    CandidateScore: _score,
    ScheduledExperiment: lambda: ScheduledExperiment(
        name="holdout_a", start=0, end=8, free_at=10, net_value=12.5
    ),
    ProgramSchedule: lambda: schedule_with_cooldown(
        (_score(), _score("holdout_b", 0.1, 80.0)), horizon_periods=24
    ),
    SensitivityTable: lambda: perturb(
        (_candidate(), _candidate("holdout_b", 0.1, 80.0)),
        _decision(),
        1.0,
        0.8,
        _economics(),
        "value_per_outcome",
        (1.0, 3.0, 6.0),
    ),
    Schedule: lambda: pulse(8, 80.0, 20.0, on=2, off=2, treatment="a"),
    SimulationSpec: lambda: SimulationSpec(n_units=12, n_periods=10, n_pre=5, n_treated=6, seed=3),
    CalibrationResult: _calibration,
    SimulatedPower: _simulated_power,
    LeaderboardRow: _leaderboard_row,
    Leaderboard: lambda: Leaderboard(
        spec=SimulationSpec(n_units=12, n_periods=10, n_pre=5, n_treated=6, n_simulations=100),
        alpha=0.05,
        effects=(0.5,),
        rows=(_leaderboard_row(),),
        calibrations=(_calibration(),),
        powers=((_simulated_power(),),),
    ),
    FisherInformation: _fisher,
    IdentifiabilityRidge: lambda: IdentifiabilityRidge(
        parameters=("alpha", "beta_a", "k_a"),
        direction={"alpha": 0.1, "beta_a": 0.7, "k_a": 0.707},
        min_eigenvalue=0.02,
        max_eigenvalue=1.9,
        condition_number=95.0,
        pairs=(("beta_a", "k_a"),),
        correlations=(0.85,),
        detail={"noise_sd": "0.5"},
    ),
    IdentifyingDesign: lambda: IdentifyingDesign(
        target="k_a",
        design=Design(
            treatments=("a",), points=((0.0,), (50.0,), (50.0,), (200.0,)), kind="identify:k_a"
        ),
        indices=(0, 2, 2, 4),
        expected_sd=7.5,
        expected_sds={"alpha": 0.4, "beta_a": 1.1, "k_a": 7.5},
        prior_sds={"alpha": 5.0, "k_a": 20.0},
        noise_sd=0.5,
        seed=0,
        n_restarts=3,
        passes=4,
    ),
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
        constraints=(_constraint("normal"), _constraint("lognormal")),
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
    EstimandResult: _estimand_result,
    Measurement: _measurement,
    Constraint: _constraint,
    Ledger: lambda: Ledger.from_plan(_plan()),
    Correction: lambda: variance_reweight(0.3, 40, 80, icc=0.05, cluster_size=10),
    ResolvedTransfer: _resolved,
    CalibratedSpec: _calibrated_spec,
    Agreement: lambda: Agreement(
        estimand_name="lift_at_100",
        estimand_hash="a" * 64,
        measurement_hash="c" * 64,
        source="study-1",
        estimate=1.2,
        se=0.3,
        posterior_mean=0.3,
        posterior_sd=0.5,
        interval=_interval(),
        z=1.5434,
        p=0.1227,
        inside=True,
        verdict="tension",
        tension_at=1.0,
        disagrees_at=2.0,
        n_draws=4000,
        realized=_estimand_result(),
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
    PolynomialKernel: lambda: PolynomialKernel(reference_dose=50.0, degree=3),
    SplineKernel: lambda: SplineKernel(reference_dose=50.0, knots=(10.0, 25.0, 40.0)),
    PiecewiseLinearKernel: lambda: PiecewiseLinearKernel(reference_dose=50.0, knots=(15.0, 35.0)),
    GaussianProcessKernel: lambda: GaussianProcessKernel(reference_dose=50.0, n_basis=6),
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
    StudyRecord: lambda: _study(0),
    Corpus: _corpus,
    Heterogeneity: lambda: heterogeneity(_META_Y, _META_SE),
    TauEstimate: lambda: tau_dersimonian_laird(_META_Y, _META_SE),
    PooledEstimate: lambda: random_effects(
        _META_Y, _META_SE, tau_method="reml", knapp_hartung=True
    ),
    PoolPriors: lambda: PoolPriors(mu_scale=2.0, tau_scale=0.5, tau_fixed=None),
    PoolSpec: lambda: PoolSpec(
        family="fertilizer", moderators=("follow_up",), bias_term=True, mass=0.9
    ),
    ParameterSummary: lambda: ParameterSummary(
        name="mu_fertilizer", mean=0.47, sd=0.06, interval=wald(0.47, 0.06, 0.95)
    ),
    EffectShrinkage: lambda: EffectShrinkage(
        effect="c0",
        studies=("s0", "s3"),
        estimate=0.47,
        se=0.09,
        theta=0.465,
        interval=wald(0.465, 0.08, 0.95),
        analytic=0.45,
        empirical=0.44,
    ),
    PoolResult: _pool_result,
    LeaveOneOut: lambda: leave_one_out(_META_Y, _META_SE, method="dl"),
    EggerTest: lambda: egger(_META_Y, _META_SE),
    FunnelContour: lambda: funnel_data(
        _META_Y, _META_SE, fixed_effect(_META_Y, _META_SE), n_grid=5
    ).contours[0],
    FunnelData: lambda: funnel_data(_META_Y, _META_SE, fixed_effect(_META_Y, _META_SE), n_grid=5),
    ForestRow: lambda: ForestRow(label="s0", estimate=0.42, se=0.1, interval=wald(0.42, 0.1, 0.95)),
    ForestData: lambda: forest_data(_corpus(), None, random_effects(_META_Y, _META_SE)),
    BaujatData: lambda: baujat(_META_Y, _META_SE),
    Cell: _cell,
    PrivacyPolicy: lambda: PrivacyPolicy(k=3, dominance_p=0.5, epsilon_total=1.0),
    EpsilonCharge: lambda: EpsilonCharge(release_id="r1", epsilon=0.25, mechanism="laplace"),
    EpsilonLedger: _ledger,
    Release: _release,
    EpsilonSplit: lambda: orthogonal_split(1.0, 3),
    StoppedEstimate: _stopped_estimate,
    ArmAllocation: lambda: _ASSIGN_ALLOCATION,
    ArmAssignment: _arm_assignment,
    BalanceRow: _balance_row,
    ArmCount: lambda: _delivery_report().ratio.arms[0],
    SampleRatio: lambda: _delivery_report().ratio,
    DeliveryRow: lambda: _delivery_report().exposure.arms[0],  # type: ignore[union-attr]
    Delivery: lambda: _delivery_report().exposure,  # type: ignore[return-value]
    BalanceTest: lambda: _delivery_report().covariates.tests[0],  # type: ignore[union-attr]
    BalanceCheck: lambda: _delivery_report().covariates,  # type: ignore[return-value]
    DeliveryReport: _delivery_report,
    Attrition: _attrition,
    AttritionRow: lambda: _attrition().arms[0],
    LeeBounds: _lee_bounds,
    ComplianceTable: lambda: _compliance_report().table,
    ComplianceReport: _compliance_report,
    DerivativeReport: _dose_report,
    FirstStage: lambda: _dose_report().stage,
    LatentSelection: lambda: _LATENT,
    Commensurability: _commensurability,
    Incompatibility: lambda: _commensurability().entries[0],
    Occupancy: lambda: _OCCUPANCIES[0],
    Collision: lambda: collisions(_OCCUPANCIES)[0],
    Factorial: _factorial,
    FactorialCell: lambda: _factorial().cells[0],
    ConfidenceSequence: _confidence_sequence,
    AnytimeLook: lambda: _confidence_sequence().looks[-1],
    Readout: lambda: _READOUTS[0],
    ProgramReport: _program_report,
    ProgramDecision: lambda: _program_report().decisions[0],
    OnlineProgram: _online_program,
    OnlineDecision: lambda: _online_program().decisions[0],
    Provenance: lambda: Provenance(
        axiom_version="0.0.0",
        created="2026-08-21T00:00:00+00:00",
        hashes={"spec:roles": "c" * 64},
        seed=7,
        environment={"python": "3.12"},
    ),
    Program: lambda: Program(
        party="northwind",
        program="dose-response",
        description="fertilizer dose-response across the north region",
        started="2026-01-05",
    ),
    CatalogEntry: lambda: CatalogEntry(
        digest="a" * 64,
        type_name="axiom.core.entities:TimeWindow",
        scope="northwind/dose-response",
        label="window",
        created="2026-03-02T09:00:00+00:00",
        derived_from="b" * 64,
        tags={"role": "plan"},
    ),
    Definition: lambda: Definition(
        scope="northwind/growth",
        name="conversion",
        version=2,
        digest="c" * 64,
        type_name="axiom.core.entities:Outcome",
        registered="2026-03-02T09:00:00+00:00",
        supersedes="b" * 64,
        note="the growth team switched to a rate",
    ),
    Change: lambda: Change(
        scope="northwind/growth",
        name="conversion",
        from_version=1,
        to_version=2,
        at="2026-03-02T09:00:00+00:00",
        changed={"aggregation": "'sum' -> 'mean'"},
        digests=("b" * 64, "c" * 64),
    ),
    Consensus: lambda: Consensus(
        name="conversion",
        scopes=("northwind/growth", "acme/growth"),
        by_digest={"b" * 64: ("acme/growth",), "c" * 64: ("northwind/growth",)},
        differences={"northwind/growth": {"aggregation": "'sum' -> 'mean'"}},
    ),
    Transition: lambda: Transition(
        stage="committed", at="2026-03-04T09:00:00+00:00", note="plan frozen"
    ),
    Deviation: _deviation,
    ExperimentRun: _experiment_run,
    TrialRoles: lambda: TrialRoles(
        harvest="grain",
        nutrients=("nitrogen", "phosphorus"),
        plot="plot",
        season="season",
        soil_tests=("soil_carbon",),
        controls=("irrigated",),
    ),
    Prices: lambda: Prices(harvest=220.0, nutrient=1.1, currency="USD"),
    EconomicOptimum: lambda: EconomicOptimum(
        nutrient="nitrogen",
        rate=138.0,
        lower=131.0,
        upper=145.0,
        expected_gain=4.45,
        prices=Prices(harvest=220.0, nutrient=2.2),
        price_ratio=0.01,
        definition="eti",
        mass=0.9,
        searched=(0.0, 270.0),
        bracketed=True,
        detail={"n_draws": "800"},
    ),
    MarketingRoles: lambda: MarketingRoles(
        kpi="revenue",
        channels=("tv", "search"),
        geo="region",
        date="week",
        impressions=("tv_impressions",),
        controls=("holiday",),
        kpi_dimension="currency",
    ),
    RobustnessValue: lambda: robustness_value(*_DARFUR, q=1.0, alpha=0.05),
    BiasBounds: lambda: bias_bounds(*_DARFUR, r2_yz_dx=0.1246, r2_dz_x=0.0092),
    Benchmark: lambda: benchmark(
        *_DARFUR, covariate="female", r2_dxj_x=0.00916, r2_yxj_dx=0.11, k_d=1.0, k_y=1.0
    ),
    TippingPoint: lambda: tipping_point(
        [0.8 + 0.01 * i for i in range(41)], 0.5, [0.0, 0.25, 0.5, 0.75], certainty=0.5
    ),
    SBCSpec: lambda: SBCSpec(n_simulations=40, draws=100, rank_draws=19, bins=4, seed=1),
    ParameterRanks: _ranks,
    SBCResult: _sbc_result,
    ParameterCoverage: _parameter_coverage,
    CoverageResult: _coverage_result,
    EstimandCoverage: _estimand_coverage,
    EstimandCoverageResult: _estimand_coverage_result,
    ParameterLearning: _parameter_learning,
    LearningReport: _learning_report,
    StatisticCheck: _statistic_check,
    PPCResult: _ppc_result,
    Refutation: _refutation,
    ResidualTest: lambda: ResidualTest(
        name="ljung_box[1]", statistic=0.4, p_value=0.52, n=24, lag=1
    ),
    UnitResiduals: lambda: UnitResiduals(unit="u0", n=8, mean=0.01, sd=0.3),
    ResidualReport: _residual_report,
    FitSettings: lambda: FitSettings(backend="laplace", draws=50, tune=50, chains=1),
    SpecOption: lambda: SpecOption(
        label="logistic", spec_update={"kernels": {"a": LogisticKernel()}}, fit_update={"draws": 50}
    ),
    SpecificationAxis: lambda: _spec_axes()[0],
    SpecRow: _spec_row,
    SpecCurveSummary: lambda: SpecCurveSummary(
        n=2,
        n_failed=1,
        n_converged=1,
        median=1.1,
        iqr_lower=1.1,
        iqr_upper=1.1,
        minimum=1.1,
        maximum=1.1,
        share_excluding_zero=1.0,
        share_positive=1.0,
    ),
    SpecCurve: _spec_curve,
    PriorPredictive: _prior_predictive,
    WeakIdReport: _weak_id_report,
    HorizonScore: _horizon_score,
    OriginForecast: _origin_forecast,
    OriginFailure: lambda: OriginFailure(origin=10, reason="backend declined"),
    Backtest: _backtest,
}


def example(cls: type[Spec]) -> Spec:
    return EXAMPLES[cls]()
