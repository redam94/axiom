"""``axiom.build``: every builder's ``build()`` round-trips through ``load_spec`` and equals the
hand-built spec; builders are immutable; missing inputs are ``BuildError``."""

from __future__ import annotations

import pandas as pd
import pytest

from axiom.build import (
    Builder,
    BuildError,
    Fields,
    GraphBuilder,
    MetaBuilder,
    PriorBuilder,
    StudyBuilder,
    SurfaceBuilder,
    VariableBuilder,
)
from axiom.calibrate import Measurement, amplitude_prior
from axiom.core import (
    Covariate,
    D,
    Dose,
    Intervention,
    Likelihood,
    Outcome,
    Population,
    Prior,
    Spec,
    TimeWindow,
    Treatment,
    Unit,
    load_spec,
)
from axiom.design import DesignCandidate, Schedule, SimulationSpec, pulse
from axiom.estimands import Estimand, Level, Quantity
from axiom.identify import CausalGraph
from axiom.meta import Corpus, PoolPriors, PoolSpec, StudyRecord
from axiom.surface import (
    FourierSeasonality,
    GeometricCarryover,
    HillKernel,
    LinearTrend,
    LogisticKernel,
    NuisanceSet,
    SurfaceSpec,
)


def _roundtrips(spec: Spec) -> None:
    back = load_spec(spec.to_json())
    assert back == spec
    assert type(back) is type(spec)
    assert back.content_hash() == spec.content_hash()


# -- base ----------------------------------------------------------------------------------------


def test_fields_are_immutable_and_copy_on_write() -> None:
    f = Fields({"a": 1})
    g = f.with_(b=2)
    assert "b" not in f and g.get("b") == 2 and g.get("a") == 1
    with pytest.raises(TypeError):
        f.values["c"] = 3  # type: ignore[index]
    assert g.with_(a=None).to_dict() == {"b": 2}
    with pytest.raises(BuildError, match=r"set \['z'\]"):
        f.require("z", builder="x")


def test_builders_satisfy_the_protocol() -> None:
    assert isinstance(SurfaceBuilder(), Builder)
    assert isinstance(PriorBuilder(), Builder)


# -- prior ---------------------------------------------------------------------------------------


def test_prior_builder_matches_hand_built_and_roundtrips() -> None:
    p = PriorBuilder().normal(mu=0.0, sigma=2.0).build()
    assert p == Prior(family="normal", hyper={"mu": 0.0, "sigma": 2.0})
    _roundtrips(p)
    h = PriorBuilder().normal(mu="alpha_mean", sigma="alpha_sd").build()
    assert h.parents == ("alpha_mean", "alpha_sd")
    assert PriorBuilder().family("gamma").hyper(alpha=2.0, beta=1.0).build() == Prior(
        family="gamma", hyper={"alpha": 2.0, "beta": 1.0}
    )
    assert PriorBuilder().fixed(3.0).build().family == "fixed"


def test_prior_from_moments_agrees_with_calibrate() -> None:
    assert PriorBuilder.from_moments(2.0, 0.5).build() == amplitude_prior(2.0, 0.5)
    assert PriorBuilder.from_moments(2.0, 0.5, "gamma").build() == amplitude_prior(
        2.0, 0.5, "gamma"
    )
    n = PriorBuilder.from_moments(1.0, 0.3, "normal").build()
    assert n == Prior(family="normal", hyper={"mu": 1.0, "sigma": 0.3})


def test_prior_builder_is_immutable_and_refuses_incomplete() -> None:
    base = PriorBuilder().family("normal")
    full = base.hyper(mu=0.0, sigma=1.0)
    assert "hyper" not in base.fields and "hyper" in full.fields
    with pytest.raises(BuildError, match="missing"):
        base.build()
    with pytest.raises(BuildError, match="family"):
        PriorBuilder().build()
    with pytest.raises(BuildError):
        PriorBuilder().family("cauchy")  # type: ignore[arg-type]


# -- variable ------------------------------------------------------------------------------------


def test_variable_builder_builds_every_kind() -> None:
    t = VariableBuilder().treatment("x").dimension(D.currency).measured_in("USD").build()
    assert t == Treatment(name="x", dimension=D.currency, unit="USD")
    _roundtrips(t)
    o = VariableBuilder().outcome("y").dimension(D.outcome).aggregation("mean").build()
    assert o == Outcome(name="y", dimension=D.outcome, aggregation="mean")
    d = VariableBuilder().dose("x").dimension(D.currency).numeraire("USD").build()
    assert d == Dose(name="x", dimension=D.currency, numeraire="USD")
    u = VariableBuilder().unit("region").dimension(D.entity).cluster().build()
    assert u == Unit(name="region", dimension=D.entity, kind="cluster")
    c = VariableBuilder().covariate("w").dimension(D.entity).describe("weight").build()
    assert c == Covariate(name="w", dimension=D.entity, description="weight")


def test_variable_builder_refuses_misapplied_fields() -> None:
    with pytest.raises(BuildError, match="numeraire"):
        VariableBuilder().treatment("x").dimension(D.currency).numeraire("USD").build()
    with pytest.raises(BuildError, match="kind"):
        VariableBuilder().name("x").build()


# -- surface -------------------------------------------------------------------------------------


def test_surface_builder_equals_hand_built_spec_and_roundtrips() -> None:
    built = (
        SurfaceBuilder()
        .name("s")
        .treatment("a", kernel="hill", reference_dose=50.0, carryover="geometric", max_lag=3)
        .treatment("b", unit="USD", kernel="logistic", amplitude_scale=2.0)
        .outcome("y", unit="units")
        .intercept("hierarchical", units=("u0", "u1"), scale=0.5)
        .interaction("a", "b", scale=0.3)
        .nuisance(fourier=(12.0, 2), trend=True)
        .likelihood("student_t", noise_scale=0.7, df=5.0)
        .build()
    )
    by_hand = SurfaceSpec(
        name="s",
        treatments=(
            Treatment(name="a", dimension=D.currency),
            Treatment(name="b", dimension=D.currency, unit="USD"),
        ),
        outcome=Outcome(name="y", dimension=D.outcome, unit="units"),
        kernels={
            "a": HillKernel(reference_dose=50.0),
            "b": LogisticKernel(amplitude_scale=2.0),
        },
        carryover={"a": GeometricCarryover(max_lag=3)},
        nuisance=NuisanceSet(terms=(FourierSeasonality(period=12.0, order=2), LinearTrend())),
        intercept="hierarchical",
        interactions=(("a", "b"),),
        likelihood=Likelihood(family="student_t", scale="sigma", df=5.0),
        unit_labels=("u0", "u1"),
        intercept_scale=0.5,
        interaction_scale=0.3,
        noise_scale=0.7,
    )
    assert built == by_hand
    _roundtrips(built)


def test_surface_builder_defaults_and_entities() -> None:
    t = Treatment(name="z", dimension=D.outcome, unit="mg")
    spec = (
        SurfaceBuilder()
        .treatment(t, max_lag=2)
        .outcome(Outcome(name="y", dimension=D.outcome))
        .build()
    )
    assert spec.name == "surface"
    assert spec.treatments == (t,)
    assert spec.kernels == {} and spec.carryover == {"z": GeometricCarryover(max_lag=2)}
    assert spec.kernel_of("z") == HillKernel()
    # kernel fields alone pick the default family
    spec2 = SurfaceBuilder().treatment("a", reference_dose=5.0).outcome("y").build()
    assert spec2.kernels == {"a": HillKernel(reference_dose=5.0)}


def test_surface_builder_is_immutable_and_refuses_bad_input() -> None:
    base = SurfaceBuilder().treatment("a")
    more = base.treatment("b")
    assert len(base.fields.get("treatments")) == 1 and len(more.fields.get("treatments")) == 2
    with pytest.raises(BuildError, match="already added"):
        more.treatment("a")
    with pytest.raises(BuildError, match="outcome"):
        base.build()
    with pytest.raises(BuildError, match="entity"):
        SurfaceBuilder().treatment(Treatment(name="a", dimension=D.currency), unit="USD")
    with pytest.raises(BuildError, match="family name"):
        SurfaceBuilder().treatment("a", kernel=HillKernel(), reference_dose=2.0)
    with pytest.raises(ValueError, match="unknown kernel"):
        SurfaceBuilder().treatment("a", kernel="sigmoid")


# -- study ---------------------------------------------------------------------------------------


def _estimand() -> Estimand:
    t = Treatment(name="x", dimension=D.currency, unit="USD")
    o = Outcome(name="y", dimension=D.outcome)
    return Estimand(
        name="contrast_x",
        quantity=Quantity(kind="contrast"),
        treatment=t,
        intervention=Intervention(doses={"x": 10.0}),
        reference=Intervention(doses={"x": 0.0}),
        outcome=o,
        population=Population(name="all"),
        window=TimeWindow(start=0, stop=4),
        level=Level(unit="individual"),
        dimension=D.outcome,
    )


def test_study_builder_builds_every_design_object() -> None:
    study = (
        StudyBuilder()
        .name("holdout_a")
        .method("difference_in_differences")
        .size(n_units=12, n_periods=8, n_pre=4, n_treated=6)
        .variance(unit_sd=1.0, noise_sd=0.5, period_sd=0.2, rho=0.3)
        .effect(0.4)
        .simulations(50, seed=3)
        .mass(0.9)
        .holdout(0.5)
        .precision(0.2)
        .cost(100.0, cooldown_periods=2)
        .schedule("pulse", high=1.0, low=0.0, on=2, off=2, treatment="x")
    )
    sim = study.build()
    assert sim == SimulationSpec(
        n_units=12,
        n_periods=8,
        n_pre=4,
        n_treated=6,
        unit_sd=1.0,
        noise_sd=0.5,
        period_sd=0.2,
        rho=0.3,
        effect=0.4,
        n_simulations=50,
        seed=3,
        mass=0.9,
    )
    _roundtrips(sim)
    cand = study.build_candidate()
    assert cand == DesignCandidate(
        name="holdout_a",
        method="difference_in_differences",
        n_units=12,
        n_periods=8,
        holdout_fraction=0.5,
        experiment_se=0.2,
        cost=100.0,
        cooldown_periods=2,
    )
    _roundtrips(cand)
    sched = study.build_schedule()
    assert sched == pulse(8, 1.0, 0.0, on=2, off=2, treatment="x")
    assert isinstance(sched, Schedule)
    _roundtrips(sched)
    m = study.measured(_estimand(), 0.3, 0.1, source="study:holdout_a").build_measurement()
    assert m == Measurement(
        estimand=_estimand(),
        estimate=0.3,
        se=0.1,
        mass=0.9,
        method="difference_in_differences",
        n_units=12,
        n_periods=8,
        source="study:holdout_a",
    )
    _roundtrips(m)


def test_study_builder_errors_name_what_is_missing() -> None:
    with pytest.raises(BuildError, match="unknown method"):
        StudyBuilder().method("magic")
    with pytest.raises(BuildError, match="build_candidate"):
        StudyBuilder().name("a").build_candidate()
    with pytest.raises(BuildError, match="needs parameter 'high'"):
        StudyBuilder().size(n_periods=4).schedule("alternating").build_schedule()
    with pytest.raises(BuildError, match="n_periods"):
        StudyBuilder().schedule("constant", dose=1.0).build_schedule()
    with pytest.raises(BuildError, match="build_measurement"):
        StudyBuilder().build_measurement()
    assert StudyBuilder().build() == SimulationSpec()


# -- meta ----------------------------------------------------------------------------------------


def test_meta_builder_builds_corpus_and_pool_spec() -> None:
    b = (
        MetaBuilder()
        .name("evidence")
        .family("fertilizer")
        .study("s1", estimate=0.4, se=0.1, read="experiment", contributor="lab_a")
        .study("s2", estimate=0.5, se=0.2, read="model", contributor="lab_a", n=30)
        .moderators("season")
        .bias_term()
        .priors(mu_scale=2.0, tau_fixed=0.1)
        .interval(mass=0.9, definition="hdi")
    )
    corpus = b.build()
    assert corpus == Corpus(
        name="evidence",
        records=(
            StudyRecord(
                study="s1",
                contributor="lab_a",
                quantity="elasticity",
                estimate=0.4,
                se=0.1,
                read="experiment",
                family="fertilizer",
            ),
            StudyRecord(
                study="s2",
                contributor="lab_a",
                quantity="elasticity",
                estimate=0.5,
                se=0.2,
                read="model",
                family="fertilizer",
                n=30,
            ),
        ),
    )
    _roundtrips(corpus)
    spec = b.build_pool_spec()
    assert spec == PoolSpec(
        family="fertilizer",
        moderators=("season",),
        bias_term=True,
        priors=PoolPriors(mu_scale=2.0, tau_fixed=0.1),
        mass=0.9,
        definition="hdi",
    )
    _roundtrips(spec)


def test_meta_builder_from_frame_and_refusals() -> None:
    df = pd.DataFrame(
        {
            "study": ["a", "b"],
            "contributor": ["x", "y"],
            "quantity": ["elasticity", "elasticity"],
            "estimate": [0.1, 0.2],
            "se": [0.05, 0.06],
            "read": ["experiment", "model"],
            "family": ["f", "f"],
        }
    )
    corpus = MetaBuilder().from_frame(df).build()
    assert [r.study for r in corpus.records] == ["a", "b"]
    with pytest.raises(BuildError, match="already added"):
        MetaBuilder().from_frame(df).from_frame(df)
    with pytest.raises(BuildError, match="family"):
        MetaBuilder().study("s", estimate=0.1, se=0.1, read="model")
    with pytest.raises(BuildError, match="build_corpus"):
        MetaBuilder().build()
    with pytest.raises(BuildError, match="build_pool_spec"):
        MetaBuilder().build_pool_spec()


# -- graph ---------------------------------------------------------------------------------------


def test_graph_builder_equals_hand_built_graph_and_roundtrips() -> None:
    g = (
        GraphBuilder()
        .name("toy")
        .edge("Z", "X")
        .edge("X", "Y")
        .bidirected("X", "W")
        .edge("W", "Y")
        .unmeasured("W")
        .node("I")
        .build()
    )
    assert g == CausalGraph.from_edges(
        "Z -> X, X -> Y, X <-> W, W -> Y", nodes=["I"], unmeasured=["W"], name="toy"
    )
    _roundtrips(g)
    assert GraphBuilder().edges("Z -> X, X -> Y").build() == CausalGraph.from_edges(
        "Z -> X, X -> Y"
    )


def test_graph_builder_estimand_on_graph_nodes() -> None:
    b = GraphBuilder().edge("Z", "X").edge("X", "Y")
    t = Treatment(name="X", dimension=D.currency, unit="USD")
    y = Outcome(name="Y", dimension=D.outcome)
    e = b.estimand(t, y, dose=10.0, window=TimeWindow(start=0, stop=4))
    assert e == Estimand(
        name="contrast_X_Y",
        quantity=Quantity(kind="contrast"),
        treatment=t,
        intervention=Intervention(doses={"X": 10.0}),
        reference=Intervention(doses={"X": 0.0}),
        outcome=y,
        population=Population(name="all"),
        window=TimeWindow(start=0, stop=4),
        level=Level(unit="individual"),
        dimension=D.outcome,
    )
    _roundtrips(e)
    m = b.estimand(t, y, kind="marginal", dose=10.0, name="m")
    assert m.reference is None and m.dimension == D.outcome / D.currency
    with pytest.raises(BuildError, match="not a node"):
        b.estimand(Treatment(name="Q", dimension=D.currency), y, dose=1.0)
    with pytest.raises(BuildError, match="reference_dose"):
        b.estimand(t, y, dose=1.0, reference_dose=None)
    with pytest.raises(BuildError, match="at least one"):
        GraphBuilder().build()
