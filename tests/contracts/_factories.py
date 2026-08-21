"""One example instance per ``Spec`` subclass, for gate 4.

Every concrete ``Spec`` shipped in ``src/axiom`` must have an entry here.
``test_spec_roundtrip.py`` fails naming the class if one is missing, which is
the point: a spec nobody can build an example of is a spec nobody has
round-tripped.
"""

from __future__ import annotations

from collections.abc import Callable
from fractions import Fraction

from axiom.adapters import MarketingRoles
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
    Assignment,
    CalibrationResult,
    CandidateScore,
    ClusterDesign,
    CostPerOutcomeInterval,
    CostPerOutcomePower,
    DecisionSpec,
    DesignCandidate,
    EconomicInputs,
    EIGEstimate,
    EVOIResult,
    ExperimentValue,
    FisherInformation,
    HoldoutTradeoff,
    IdentifiabilityRidge,
    IdentifyingDesign,
    Leaderboard,
    LeaderboardRow,
    LearningPriority,
    MethodEstimate,
    MethodSpec,
    OpportunityCost,
    PowerCurve,
    PowerResult,
    ProgramSchedule,
    Recommendation,
    ReExperimentTiming,
    SampleSize,
    Schedule,
    ScheduledExperiment,
    SensitivityTable,
    SimulatedPower,
    SimulationSpec,
    StudySummary,
    TreatmentCandidate,
    ValuePerOutcome,
    anchor_draws,
    cost_per_outcome_interval,
    cost_per_outcome_power,
    eig_monte_carlo,
    evaluate_candidate,
    evoi_gaussian,
    experiment_value,
    holdout_tradeoff,
    match_clusters,
    mde,
    method_spec,
    opportunity_cost,
    perturb,
    power,
    power_curve,
    pulse,
    rank_treatments,
    recommend,
    sample_size,
    schedule_with_cooldown,
    time_to_re_experiment,
)
from axiom.diagnose import (
    Backtest,
    Benchmark,
    BiasBounds,
    CoverageResult,
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
    benchmark,
    bias_bounds,
    rank_uniformity,
    robustness_value,
    tipping_point,
)
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
from axiom.meta import (
    BaujatData,
    Cell,
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


EXAMPLES: dict[type[Spec], Callable[[], Spec]] = {
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
    Provenance: lambda: Provenance(
        axiom_version="0.0.0",
        created="2026-08-21T00:00:00+00:00",
        hashes={"spec:roles": "c" * 64},
        seed=7,
        environment={"python": "3.12"},
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
