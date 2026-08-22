"""``SurfaceSpec`` / ``Surface`` / ``prepare`` / ``fit`` / ``predict`` / marginals."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from axiom.core import (
    Covariate,
    D,
    DimensionError,
    Likelihood,
    ModelSpec,
    Outcome,
    Param,
    Posterior,
    SupportsForward,
    Treatment,
    Unsupported,
    Unverified,
    dimension,
    dimensionless,
    load_spec,
    value,
    walk,
)
from axiom.data import Panel, PanelError, RoleMap
from axiom.surface.carryover import GeometricCarryover, NoCarryover
from axiom.surface.forward import (
    GRID_VERSION,
    forward,
    marginal,
    marginal_expr,
    marginal_total,
    predict,
)
from axiom.surface.kernels import HillKernel, LinearKernel
from axiom.surface.linearize import (
    check_linearization,
    column_names,
    design_matrix,
    linear_coefficients,
)
from axiom.surface.model import (
    FitResult,
    Surface,
    SurfaceSpec,
    build,
    fit,
    interaction_name,
    parameter_roles,
    prepare,
    resolve_conventions,
)
from axiom.surface.nuisance import EventIndicators, LinearTrend, NuisanceSet

USD = D.currency
A = Treatment(name="a", dimension=USD, unit="USD")
B = Treatment(name="b", dimension=USD, unit="USD")
Y = Outcome(name="y", dimension=D.outcome)
UNITS = ("u0", "u1", "u2")
N_PERIODS = 8

PANEL_THETA: dict[str, object] = {
    "alpha_unit": np.array([0.5, 1.0, 1.5]),
    "alpha_mean": 1.0,
    "alpha_sd": 0.5,
    "k_a": 2.0,
    "s_a": 1.5,
    "beta_a": 2.0,
    "lam_a": 0.6,
    "beta_rate_b": 0.3,
    "gamma_a_b": 0.4,
    "n0_coef_trend": 0.7,
    "sigma": 0.3,
}


def arms_spec(**over: object) -> SurfaceSpec:
    base: dict[str, object] = dict(
        name="arms",
        treatments=(A,),
        outcome=Y,
        kernels={"a": HillKernel(reference_dose=2.0)},
    )
    base.update(over)
    return SurfaceSpec(**base)  # type: ignore[arg-type]


def panel_spec(**over: object) -> SurfaceSpec:
    base: dict[str, object] = dict(
        name="panel",
        treatments=(A, B),
        outcome=Y,
        kernels={"a": HillKernel(reference_dose=2.0), "b": LinearKernel(reference_dose=2.0)},
        carryover={"a": GeometricCarryover(max_lag=3)},
        interactions=(("a", "b"),),
        intercept="hierarchical",
        unit_labels=UNITS,
        nuisance=NuisanceSet(terms=(LinearTrend(scale=10.0),)),
    )
    base.update(over)
    return SurfaceSpec(**base)  # type: ignore[arg-type]


def roles(outcome: Outcome = Y, **columns: Treatment) -> RoleMap:
    return RoleMap(
        unit="unit", time="t", outcome=("y", outcome), treatments=columns or {"a": A, "b": B}
    )


def frame(
    seed: int = 0, units: tuple[str, ...] = UNITS, n_periods: int = N_PERIODS
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = [
        {"unit": u, "t": t, "y": 0.0, "a": rng.uniform(0.0, 5.0), "b": rng.uniform(0.0, 5.0)}
        for u in units
        for t in range(n_periods)
    ]
    return pd.DataFrame(rows)


def hill(x: npt.ArrayLike, k: float, s: float) -> np.ndarray:
    u = (np.asarray(x, dtype=float) / k) ** s
    return np.asarray(u / (1.0 + u))


@pytest.fixture
def panel() -> Panel:
    return Panel(frame(), roles())


# -- spec ------------------------------------------------------------------------------


def test_spec_rejects_inconsistent_declarations() -> None:
    with pytest.raises(ValueError, match="do not exist"):
        arms_spec(kernels={"zzz": HillKernel()})
    with pytest.raises(ValueError, match="do not exist"):
        arms_spec(carryover={"zzz": NoCarryover()})
    with pytest.raises(ValueError, match="two distinct treatments"):
        panel_spec(interactions=(("a", "a"),))
    with pytest.raises(ValueError, match="declared twice"):
        panel_spec(interactions=(("a", "b"), ("b", "a")))
    with pytest.raises(ValueError, match="needs unit_labels"):
        arms_spec(intercept="per_unit")
    with pytest.raises(ValueError, match="distinct"):
        arms_spec(intercept="per_unit", unit_labels=("u0", "u0"))
    with pytest.raises(ValueError, match="collide"):
        arms_spec(treatments=(Treatment(name="t", dimension=USD, unit="USD"),), kernels={})
    with pytest.raises(ValueError, match="distinct"):
        arms_spec(treatments=(A, A))


def test_spec_round_trips_and_hashes() -> None:
    spec = panel_spec()
    back = SurfaceSpec.from_json(spec.to_json())
    assert back == spec
    assert load_spec(spec.to_json()) == spec
    assert back.content_hash() == spec.content_hash()
    assert spec.diff(back).is_empty
    assert isinstance(back.kernels["b"], LinearKernel)
    assert isinstance(back.carryover["a"], GeometricCarryover)


def test_defaults_fill_in_hill_and_no_carryover() -> None:
    spec = SurfaceSpec(name="bare", treatments=(A,), outcome=Y)
    assert isinstance(spec.kernel_of("a"), HillKernel)
    assert isinstance(spec.carryover_of("a"), NoCarryover)
    assert spec.intercept == "shared"
    assert spec.likelihood.scale == "sigma"
    assert spec.data_columns == ("unit", "t", "y", "a")
    assert spec.carried == ()
    assert panel_spec().carried == ("a",)
    with pytest.raises(KeyError):
        spec.kernel_of("zzz")


# -- build -----------------------------------------------------------------------------


def test_build_declares_every_parameter_with_its_role() -> None:
    spec = panel_spec()
    model = build(spec)
    assert model.name == "panel"
    assert dimension(model.mean) == D.outcome
    assert parameter_roles(spec) == {
        "alpha_mean": "auxiliary",
        "alpha_sd": "auxiliary",
        "alpha_unit": "linear",
        "k_a": "nonlinear",
        "s_a": "nonlinear",
        "beta_a": "linear",
        "lam_a": "nonlinear",
        "beta_rate_b": "linear",
        "gamma_a_b": "linear",
        "n0_coef_trend": "linear",
        "sigma": "auxiliary",
    }
    alpha_unit = model.parameter("alpha_unit")
    assert alpha_unit.shape == (3,)
    assert alpha_unit.prior is not None
    assert alpha_unit.prior.hyper == {"mu": "alpha_mean", "sigma": "alpha_sd"}
    gamma = model.parameter(interaction_name("a", "b"))
    assert gamma.dimension == D.outcome
    assert model.parameter("beta_rate_b").dimension == D.outcome / USD
    assert model.parameter("sigma").dimension == D.outcome


def test_build_intercept_variants_and_likelihood_scale_dimension() -> None:
    assert "alpha" not in {p.name for p in build(arms_spec(intercept="none")).parameters}
    shared = build(arms_spec())
    assert shared.parameter("alpha").shape == ()
    per_unit = build(arms_spec(intercept="per_unit", unit_labels=UNITS))
    prior = per_unit.parameter("alpha_unit").prior
    assert prior is not None and prior.family == "normal" and prior.hyper["mu"] == 0.0
    assert "alpha_mean" not in {p.name for p in per_unit.parameters}
    lognormal = build(arms_spec(likelihood=Likelihood(family="lognormal", scale="sigma")))
    assert lognormal.parameter("sigma").dimension == dimensionless()
    poisson = build(arms_spec(likelihood=Likelihood(family="poisson")))
    assert "sigma" not in {p.name for p in poisson.parameters}


# -- surface / forward -----------------------------------------------------------------


def test_surface_satisfies_the_protocol_and_forward_is_the_tree() -> None:
    s = Surface(arms_spec())
    assert isinstance(s, SupportsForward)
    assert s.expr == s.model.mean
    assert s.linear == ("alpha", "beta_a")
    assert s.nonlinear == ("k_a", "s_a")
    assert s.auxiliary == ("sigma",)
    assert s.provenance == {"steady_state": False}
    x = np.linspace(0.0, 5.0, 7)
    theta = {"alpha": 0.3, "k_a": 2.0, "s_a": 1.5, "beta_a": 2.0}
    expected = 0.3 + 2.0 * hill(x, 2.0, 1.5)
    np.testing.assert_allclose(s.forward({"a": x}, theta), expected, rtol=1e-12)
    np.testing.assert_allclose(forward(s, {"a": x}, theta), expected, rtol=1e-12)
    with pytest.raises(KeyError, match="k_a"):
        s.forward({"a": x}, {"alpha": 0.3, "s_a": 1.5, "beta_a": 2.0})
    assert "arms" in repr(s)


def test_carryover_refuses_a_1d_dose_grid_everywhere(panel: Panel) -> None:
    spec = panel_spec(intercept="shared", unit_labels=(), nuisance=NuisanceSet(terms=()))
    s = Surface(spec)
    theta = {**PANEL_THETA, "alpha": 0.1}
    grid = {"a": np.array([4.0, 0.5, 4.0, 0.5]), "b": np.array([1.0, 1.0, 1.0, 1.0])}
    with pytest.raises(ValueError, match="'a'.*steady_state") as info:
        s.forward(grid, theta)
    assert "geometric" in str(info.value) and "ndim=1" in str(info.value)
    with pytest.raises(ValueError, match="linearize: treatment 'a'"):
        s.linearize(grid, theta)
    with pytest.raises(ValueError, match="forward: treatment 'a'"):
        forward(s, grid, theta)
    with pytest.raises(ValueError, match="marginal: treatment 'a'"):
        marginal(s, theta, grid, "b")
    with pytest.raises(ValueError, match="marginal_total: treatment 'a'"):
        marginal_total(s, theta, grid, "a")
    with pytest.raises(ValueError, match="predict: treatment 'a'"):
        predict(s, _hand_posterior(), grid)
    # a scalar dose is also refused; the treatment without carryover is not the problem
    with pytest.raises(ValueError, match="'a'"):
        s.forward({"a": 2.0, "b": 1.0}, theta)
    # the (n_units, n_periods) layout is accepted, as is a treatment absent from dose
    # (left to the interpreter, which names it)
    data = prepare(spec, panel)
    assert s.forward(data, theta).shape == (3, N_PERIODS)
    with pytest.raises(KeyError, match="a"):
        s.forward({"b": grid["b"]}, theta)


def test_steady_state_is_the_no_carryover_surface() -> None:
    spec = panel_spec(intercept="shared", unit_labels=(), nuisance=NuisanceSet(terms=()))
    s = Surface(spec)
    steady = s.steady_state()
    theta = {**PANEL_THETA, "alpha": 0.1}
    x = np.array([4.0, 0.5, 4.0, 0.5])
    b = np.array([1.0, 2.0, 0.0, 3.0])
    grid = {"a": x, "b": b}
    got = steady.forward(grid, theta)
    plain = Surface(
        panel_spec(intercept="shared", unit_labels=(), carryover={}, nuisance=NuisanceSet(terms=()))
    )
    np.testing.assert_array_equal(got, plain.forward(grid, theta))
    f = hill(x, 2.0, 1.5)
    expected = 0.1 + 2.0 * f + 0.3 * b + 0.4 * f * (b / 2.0)
    np.testing.assert_allclose(got, expected, rtol=1e-12)
    # a constant dose held for max_lag periods reaches the steady state: the last period of
    # the carryover surface equals the steady-state surface on the same rows
    held = {"a": np.tile(x[:, None], (1, 3)), "b": np.tile(b[:, None], (1, 3))}
    np.testing.assert_allclose(s.forward(held, theta)[:, -1], got, rtol=1e-12)
    assert not np.allclose(s.forward(held, theta)[:, 0], got)  # period 0 is still ramping up
    # the record: name, provenance, the dropped carryover parameters, the roles
    assert steady.spec.name == "panel:steady_state"
    assert steady.spec.carried == ()
    assert all(isinstance(c, NoCarryover) for c in steady.spec.carryover.values())
    assert set(steady.spec.carryover) == {"a", "b"}
    assert steady.provenance == {
        "steady_state": True,
        "source_spec": "panel",
        "source_spec_hash": spec.content_hash(),
        "dropped_parameters": ("lam_a",),
    }
    assert "lam_a" not in steady.nonlinear and "lam_a" in s.nonlinear
    assert steady.linear == s.linear
    assert check_linearization(steady, grid, theta) < 1e-12
    # everything else is unchanged: kernels, interactions, likelihood, intercept
    assert steady.spec.kernels == spec.kernels
    assert steady.spec.interactions == spec.interactions
    assert steady.spec.likelihood == spec.likelihood
    assert steady.spec.model_copy(update={"name": spec.name, "carryover": spec.carryover}) == spec


# -- prepare ---------------------------------------------------------------------------


def test_prepare_lays_out_units_by_time(panel: Panel) -> None:
    spec = panel_spec(unit_labels=("u2", "u0", "u1"))
    data = prepare(spec, panel)
    assert set(data) == {"unit", "t", "y", "a", "b", "n0_trend"}
    assert data["unit"].shape == (3, 1) and data["unit"].dtype.kind == "i"
    np.testing.assert_array_equal(data["unit"][:, 0], [0, 1, 2])
    for col in ("t", "y", "a", "b", "n0_trend"):
        assert data[col].shape == (3, N_PERIODS)
    np.testing.assert_array_equal(data["t"][0], np.arange(N_PERIODS))
    # rows follow spec.unit_labels, not the panel's sorted order
    df = panel.frame
    np.testing.assert_array_equal(data["a"][0], df.loc[df["unit"] == "u2", "a"].to_numpy())
    np.testing.assert_array_equal(data["a"][1], df.loc[df["unit"] == "u0", "a"].to_numpy())
    np.testing.assert_allclose(data["n0_trend"], data["t"] / 10.0)


def test_prepare_finds_a_treatment_by_entity_name() -> None:
    df = frame().rename(columns={"a": "dose_a"})
    panel = Panel(df, roles(dose_a=A, b=B))
    data = prepare(panel_spec(), panel)
    assert data["a"].shape == (3, N_PERIODS)


def test_prepare_refuses_what_it_cannot_lay_out(panel: Panel) -> None:
    with pytest.raises(PanelError, match="balanced"):
        prepare(panel_spec(), Panel(frame().iloc[1:], roles()))
    with pytest.raises(ValueError, match="do not match"):
        prepare(panel_spec(unit_labels=("u0", "u1", "u9")), panel)
    df = frame()
    df.loc[3, "a"] = np.nan
    with pytest.raises(ValueError, match="'a' has missing"):
        prepare(panel_spec(), Panel(df, roles()))
    other = Treatment(name="a", dimension=D.time, unit="day")
    with pytest.raises(DimensionError, match="treatment 'a'"):
        prepare(panel_spec(), Panel(frame(), roles(a=other, b=B)))
    with pytest.raises(ValueError, match="neither the panel's outcome"):
        prepare(panel_spec(outcome=Outcome(name="sales_total", dimension=D.outcome)), panel)


def test_prepare_refuses_a_unit_mismatch_even_at_equal_dimension() -> None:
    cents = Treatment(name="a", dimension=USD, unit="cents")
    with pytest.raises(DimensionError, match="treatment 'a'.*'cents'.*'USD'"):
        prepare(panel_spec(), Panel(frame(), roles(a=cents, b=B)))
    counted = Outcome(name="y", dimension=D.outcome, unit="count")
    with pytest.raises(DimensionError, match="outcome 'y'.*'count'.*None"):
        prepare(panel_spec(), Panel(frame(), roles(outcome=counted)))
    # the same unit string on both sides is what is required, not a declared unit
    spec = panel_spec(outcome=counted)
    assert prepare(spec, Panel(frame(), roles(outcome=counted)))["y"].shape == (3, N_PERIODS)


def test_prepare_requires_equispaced_periods_when_carryover_is_declared() -> None:
    df = frame()
    df["t"] = df["t"].map(dict(enumerate([0, 1, 2, 10, 11, 12, 13, 14])))
    gapped = Panel(df, roles())
    with pytest.raises(PanelError, match="equispaced.*\\[1.0, 8.0\\]"):
        prepare(panel_spec(), gapped)
    # without carryover the gaps are the caller's business
    spec = panel_spec(carryover={})
    np.testing.assert_array_equal(
        prepare(spec, gapped)["t"][0], [0.0, 1.0, 2.0, 10.0, 11.0, 12.0, 13.0, 14.0]
    )


def test_prepare_accepts_a_datetime_time_column() -> None:
    df = frame()
    weekly = pd.date_range("2024-01-01", periods=N_PERIODS, freq="7D")
    df["t"] = np.tile(weekly, len(UNITS))
    data = prepare(panel_spec(), Panel(df, roles()))
    days = (weekly - pd.Timestamp("1970-01-01")) / pd.Timedelta(days=1)
    np.testing.assert_array_equal(data["t"][1], days.to_numpy())
    np.testing.assert_allclose(np.diff(data["t"][0]), 7.0)
    np.testing.assert_allclose(data["n0_trend"], (data["t"] - data["t"][0, 0]) / 10.0)
    monthly = pd.date_range("2024-01-01", periods=N_PERIODS, freq="MS")
    df["t"] = np.tile(monthly, len(UNITS))
    with pytest.raises(PanelError, match="equispaced"):
        prepare(panel_spec(), Panel(df, roles()))
    assert prepare(panel_spec(carryover={}), Panel(df, roles()))["t"].shape == (3, N_PERIODS)


def test_prepare_reuses_the_fitted_trend_conventions_for_a_forecast_panel(panel: Panel) -> None:
    spec = panel_spec()
    res = fit(spec, panel, backend="no_such_backend")
    conventions = res.provenance["nuisance_conventions"]
    assert conventions == {"n0_trend": {"origin": 0.0, "scale": 10.0}}
    assert conventions == resolve_conventions(spec, panel)
    future = Panel(frame(seed=1).assign(t=lambda d: d["t"] + N_PERIODS), roles())
    fresh = prepare(spec, future)
    same = prepare(spec, future, conventions=conventions)
    np.testing.assert_allclose(fresh["n0_trend"], (same["t"] - N_PERIODS) / 10.0)
    np.testing.assert_allclose(same["n0_trend"], same["t"] / 10.0)
    assert resolve_conventions(spec, future, conventions) == conventions
    # a fully explicit trend resolves to its own fields wherever it is laid out
    fixed = panel_spec(nuisance=NuisanceSet(terms=(LinearTrend(origin=-2.0, scale=4.0),)))
    assert resolve_conventions(fixed, future) == {"n0_trend": {"origin": -2.0, "scale": 4.0}}
    # what conventions may not do: name an unknown column, disagree with the spec, be partial
    with pytest.raises(ValueError, match="not trend columns.*\\['zzz'\\]"):
        prepare(spec, future, conventions={**conventions, "zzz": {"origin": 0.0, "scale": 1.0}})
    with pytest.raises(ValueError, match="spec fixes scale=10.0"):
        prepare(spec, future, conventions={"n0_trend": {"origin": 0.0, "scale": 5.0}})
    with pytest.raises(ValueError, match="lack \\['scale'\\]"):
        prepare(spec, future, conventions={"n0_trend": {"origin": 0.0}})
    with pytest.raises(ValueError, match="scale > 0"):
        prepare(spec, future, conventions={"n0_trend": {"origin": 0.0, "scale": -1.0}})
    assert resolve_conventions(panel_spec(nuisance=NuisanceSet(terms=())), panel) == {}


def test_event_indicators_flow_through_prepare_as_0_1_columns() -> None:
    df = frame()
    df["holiday"] = (df["t"] == 3).astype(float)
    holiday = Covariate(name="holiday", dimension=dimensionless())
    rm = RoleMap(
        unit="unit",
        time="t",
        outcome=("y", Y),
        treatments={"a": A, "b": B},
        covariates={"holiday": holiday},
    )
    spec = panel_spec(nuisance=NuisanceSet(terms=(EventIndicators(events=("holiday",)),)))
    assert spec.data_columns[-1] == "holiday"
    data = prepare(spec, Panel(df, rm))
    assert data["holiday"].shape == (3, N_PERIODS)
    np.testing.assert_array_equal(data["holiday"], np.tile(np.arange(N_PERIODS) == 3, (3, 1)))
    s = Surface(spec)
    assert "n0_coef_holiday" in s.linear
    theta = {k: v for k, v in PANEL_THETA.items() if k != "n0_coef_trend"}
    dm = s.linearize(data, theta)
    np.testing.assert_array_equal(
        dm.X[:, dm.columns.index("n0_coef_holiday")], data["holiday"].ravel()
    )
    df.loc[0, "holiday"] = 0.5
    with pytest.raises(ValueError, match="must be 0/1"):
        prepare(spec, Panel(df, rm))


def test_carryover_convolves_within_unit_only(panel: Panel) -> None:
    spec = panel_spec(intercept="none", interactions=(), nuisance=NuisanceSet(terms=()))
    s = Surface(spec)
    data = prepare(spec, panel)
    theta = dict(PANEL_THETA)
    mu = s.forward(data, theta)
    lam, k, sh, beta = 0.6, 2.0, 1.5, 2.0
    w = lam ** np.arange(3)
    w = w / w.sum()
    for i in range(3):
        x = data["a"][i]
        carried = np.array([sum(w[j] * x[t - j] for j in range(3) if t - j >= 0) for t in range(8)])
        expected = beta * hill(carried, k, sh) + 0.3 * data["b"][i]
        np.testing.assert_allclose(mu[i], expected, rtol=1e-12)
    bumped = {**data, "a": data["a"] + np.array([[10.0], [0.0], [0.0]])}
    mu2 = s.forward(bumped, theta)
    assert np.all(mu2[0] != mu[0])
    np.testing.assert_array_equal(mu2[1:], mu[1:])


# -- linearize -------------------------------------------------------------------------


def test_linearize_columns_offset_and_invariant(panel: Panel) -> None:
    spec = panel_spec()
    s = Surface(spec)
    data = prepare(spec, panel)
    dm = s.linearize(data, PANEL_THETA)
    assert dm.columns == (
        "alpha_unit[0]",
        "alpha_unit[1]",
        "alpha_unit[2]",
        "beta_a",
        "beta_rate_b",
        "gamma_a_b",
        "n0_coef_trend",
    )
    assert dm.columns == column_names(s.model, s.linear)
    assert dm.X.shape == (24, 7) and dm.offset.shape == (24,)
    np.testing.assert_array_equal(dm.offset, 0.0)
    assert set(dm.at) == {"k_a", "s_a", "lam_a"}
    # the per-unit intercept columns are unit indicators down the time axis
    np.testing.assert_array_equal(dm.X[:, 0], np.repeat([1.0, 0.0, 0.0], N_PERIODS))
    assert check_linearization(s, data, PANEL_THETA) < 1e-12
    coef = linear_coefficients(dm.columns, PANEL_THETA)
    np.testing.assert_array_equal(coef[:3], [0.5, 1.0, 1.5])
    np.testing.assert_allclose(dm.offset + dm.X @ coef, s.forward(data, PANEL_THETA).ravel())
    with pytest.raises(KeyError, match="beta_a"):
        linear_coefficients(dm.columns, {k: v for k, v in PANEL_THETA.items() if k != "beta_a"})


def test_linear_coefficients_refuses_to_truncate_an_array() -> None:
    columns = ("alpha", "beta_a", "alpha_unit[1]")
    theta: dict[str, object] = {
        "alpha": 0.5,
        "beta_a": np.array([2.0]),
        "alpha_unit": [1.0, 2.0, 3.0],
    }
    np.testing.assert_array_equal(linear_coefficients(columns, theta), [0.5, 2.0, 2.0])
    with pytest.raises(ValueError, match="'beta_a'.*shape \\(2,\\).*'beta_a\\[i\\]'"):
        linear_coefficients(columns, {**theta, "beta_a": np.array([2.0, 3.0])})
    with pytest.raises(ValueError, match="'alpha'"):
        linear_coefficients(columns, {**theta, "alpha": np.zeros((2, 2))})


def test_linearize_ignores_linear_values_and_needs_nonlinear_ones(panel: Panel) -> None:
    spec = panel_spec()
    s = Surface(spec)
    data = prepare(spec, panel)
    at = {k: v for k, v in PANEL_THETA.items() if k in s.nonlinear}
    dm = design_matrix(s.model, s.linear, data, at)
    other = {**PANEL_THETA, "beta_a": 99.0, "gamma_a_b": -5.0}
    np.testing.assert_array_equal(dm.X, s.linearize(data, other).X)
    with pytest.raises(KeyError, match="lam_a"):
        s.linearize(data, {"k_a": 2.0, "s_a": 1.5})


def test_linearize_on_an_arms_grid_has_one_row_per_run() -> None:
    s = Surface(arms_spec())
    dose = {"a": np.linspace(0.5, 4.0, 6)}
    dm = s.linearize(dose, {"k_a": 2.0, "s_a": 1.5})
    assert dm.X.shape == (6, 2) and dm.columns == ("alpha", "beta_a")
    np.testing.assert_array_equal(dm.X[:, 0], 1.0)
    theta = {"alpha": -0.2, "beta_a": 1.7, "k_a": 2.0, "s_a": 1.5}
    np.testing.assert_allclose(dm.predict(theta), s.forward(dose, theta), atol=1e-13)


# -- fit -------------------------------------------------------------------------------


def _arms_panel(spec: SurfaceSpec, truth: dict[str, object], seed: int = 9) -> Panel:
    s = Surface(spec)
    df = frame(seed=5, n_periods=12).drop(columns=["b"])
    panel = Panel(df, roles(a=A))
    mu = s.forward(prepare(spec, panel), truth)
    rng = np.random.default_rng(seed)
    df["y"] = mu.ravel() + rng.normal(0.0, 0.2, mu.size)
    return Panel(df, roles(a=A))


def test_fit_laplace_returns_a_posterior_with_provenance() -> None:
    spec = arms_spec(kernels={"a": HillKernel(reference_dose=2.0, amplitude_scale=3.0)})
    panel = _arms_panel(spec, {"alpha": 0.5, "k_a": 2.0, "s_a": 1.5, "beta_a": 2.0})
    res = fit(spec, panel, draws=150, chains=2, seed=11)
    assert isinstance(res, FitResult)
    assert isinstance(res.posterior, Posterior)
    assert res.report is None  # laplace draws are independent; no MCMC report
    assert res.converged
    assert res.posterior.names() == {"alpha", "k_a", "s_a", "beta_a", "sigma"}
    assert res.posterior.n_draws() == 300 and res.posterior.n_chains == 2
    assert res.posterior.provenance["seed"] == 11
    assert res.provenance["backend"] == "laplace"
    assert res.provenance["spec_hash"] == spec.content_hash()
    assert res.provenance["model_hash"] == res.surface.model.content_hash()
    assert res.provenance["panel_hash"] == panel.content_hash()
    assert res.provenance["nuisance_conventions"] == {}
    assert (res.provenance["n_units"], res.provenance["n_periods"]) == (3, 12)
    assert res.data["y"].shape == (3, 12)
    beta = res.posterior.summary("beta_a", mass=0.95)
    assert beta.interval.lower < 2.0 < beta.interval.upper


def test_fit_per_unit_intercept_through_prepare_and_laplace() -> None:
    spec = arms_spec(intercept="per_unit", unit_labels=UNITS, intercept_scale=3.0)
    truth = {"alpha_unit": np.array([-1.0, 0.0, 1.0]), "k_a": 2.0, "s_a": 1.5, "beta_a": 2.0}
    panel = _arms_panel(spec, truth)
    res = fit(spec, panel, draws=60, chains=1, seed=3)
    assert isinstance(res.posterior, Posterior)
    assert res.converged
    assert res.data["unit"].shape == (3, 1)
    assert res.posterior.draws("alpha_unit").shape == (1, 60, 3)
    means = res.posterior.draws("alpha_unit").mean(axis=(0, 1))
    assert means[0] < means[1] < means[2]
    assert abs(float(means[2] - means[0]) - 2.0) < 0.6


def test_fit_with_an_unknown_backend_is_unsupported(panel: Panel) -> None:
    res = fit(panel_spec(), panel, backend="no_such_backend")
    assert isinstance(res.posterior, Unsupported)
    assert res.report is None
    assert not res.converged
    assert res.provenance["backend"] == "no_such_backend"


class _DecliningBackend:
    """A backend that will not certify its draws: ``fit`` must pass that through untouched."""

    name = "declining"

    def __init__(self) -> None:
        self.calls: list[tuple[ModelSpec, int, int, int, int | None]] = []

    def sample(
        self,
        model: ModelSpec,
        data: Mapping[str, npt.ArrayLike],
        *,
        draws: int,
        tune: int,
        chains: int,
        seed: int | None,
    ) -> Posterior | Unverified:
        self.calls.append((model, draws, tune, chains, seed))
        return Unverified(reason="mode search did not converge", detail={"n_iter": "200"})


def test_fit_propagates_unverified_from_the_backend(panel: Panel) -> None:
    spec = panel_spec()
    backend = _DecliningBackend()
    res = fit(spec, panel, backend=backend, draws=10, tune=5, chains=1, seed=2)
    assert isinstance(res.posterior, Unverified)
    assert res.posterior.reason == "mode search did not converge"
    assert res.report is None
    assert not res.converged
    assert res.provenance["backend"] == "declining"
    assert res.provenance["nuisance_conventions"] == {"n0_trend": {"origin": 0.0, "scale": 10.0}}
    assert backend.calls == [(res.surface.model, 10, 5, 1, 2)]
    assert res.data["a"].shape == (3, N_PERIODS)


# -- predict ---------------------------------------------------------------------------


def _hand_posterior(n: int = 6, chains: int = 2, **over: object) -> Posterior:
    rng = np.random.default_rng(3)
    draws: dict[str, np.ndarray] = {}
    for name, v in {**PANEL_THETA, **over}.items():
        base = np.asarray(v, dtype=float)
        jitter = rng.uniform(0.9, 1.1, size=(chains, n, *base.shape))
        draws[name] = base * jitter
    return Posterior(draws)


def test_predict_pushes_every_draw_through_forward(panel: Panel) -> None:
    spec = panel_spec()
    s = Surface(spec)
    data = prepare(spec, panel)
    post = _hand_posterior()
    pred = predict(s, post, data)
    assert pred.values.shape == (2, 6, 3, N_PERIODS)
    theta_0 = {name: post.draws(name)[1, 4] for name in post.names()}
    np.testing.assert_allclose(pred.values[1, 4], s.forward(data, theta_0), rtol=1e-12)
    assert pred.seed is None
    # the rows differ, so the intervention says "grid": first cell in doses, all of them in coords
    assert pred.intervention.version == GRID_VERSION == "grid"
    assert pred.intervention.mode == "set"
    assert pred.intervention.doses == {"a": float(data["a"][0, 0]), "b": float(data["b"][0, 0])}
    assert pred.coords["a"] == [float(x) for x in data["a"].ravel()]
    assert pred.coords["b"] == [float(x) for x in data["b"].ravel()]


def test_predict_intervention_is_a_set_level_only_when_every_row_shares_it() -> None:
    s = Surface(arms_spec())
    post = _hand_posterior(alpha=0.3)
    level = predict(s, post, {"a": np.full(5, 2.5)})
    assert level.intervention.doses == {"a": 2.5}
    assert level.intervention.mode == "set"
    assert level.intervention.version == "unspecified"
    assert level.coords["a"] == [2.5] * 5
    grid = predict(s, post, {"a": np.array([2.5, 2.5, 3.0])})
    assert grid.intervention.version == "grid" and grid.intervention.doses == {"a": 2.5}
    with pytest.raises(KeyError, match="'a'"):  # the interpreter names the missing column
        predict(s, post, {"zzz": np.ones(3)})
    with pytest.raises(ValueError, match="empty.*\\['a'\\]"):
        predict(s, post, {"a": np.zeros(0)})


def test_predict_noise_is_seeded_and_on_the_outcome_scale(panel: Panel) -> None:
    spec = panel_spec()
    s = Surface(spec)
    data = prepare(spec, panel)
    post = _hand_posterior()
    clean = predict(s, post, data)
    noisy_1 = predict(s, post, data, seed=7, noise=True)
    noisy_2 = predict(s, post, data, seed=7, noise=True)
    np.testing.assert_array_equal(noisy_1.values, noisy_2.values)
    assert noisy_1.seed == 7
    assert np.all(noisy_1.values != clean.values)
    resid = noisy_1.values - clean.values
    assert abs(float(resid.std()) - 0.3) < 0.1


def test_predict_noise_for_the_other_likelihood_families() -> None:
    x = np.linspace(0.5, 4.0, 6)
    post = _hand_posterior(n=40, alpha=1.5)
    student = Surface(arms_spec(likelihood=Likelihood(family="student_t", scale="sigma", df=5.0)))
    clean = predict(student, post, {"a": x})
    heavy = predict(student, post, {"a": x}, seed=1, noise=True)
    resid = heavy.values - clean.values
    assert abs(float(resid.std()) - 0.3 * np.sqrt(5.0 / 3.0)) < 0.1
    lognormal = Surface(arms_spec(likelihood=Likelihood(family="lognormal", scale="sigma")))
    ln = predict(lognormal, post, {"a": x}, seed=2, noise=True)
    assert np.all(ln.values > 0.0) and np.all(np.isfinite(ln.values))
    log_resid = np.log(ln.values) - np.log(clean.values)
    assert abs(float(log_resid.std()) - 0.3) < 0.1
    poisson = Surface(arms_spec(likelihood=Likelihood(family="poisson")))
    counts = predict(poisson, post, {"a": x}, seed=3, noise=True)
    assert np.all(counts.values == np.round(counts.values)) and np.all(counts.values >= 0.0)
    assert abs(float(counts.values.mean() - clean.values.mean())) < 0.2
    np.testing.assert_array_equal(predict(poisson, post, {"a": x}).values, clean.values)


def test_predict_noise_refuses_a_non_positive_mean_under_lognormal_and_poisson() -> None:
    x = np.array([0.0, 1.0, 2.0])
    negative = _hand_posterior(n=2, chains=1, alpha=-0.5)
    for family, lik in (
        ("lognormal", Likelihood(family="lognormal", scale="sigma")),
        ("poisson", Likelihood(family="poisson")),
    ):
        s = Surface(arms_spec(likelihood=lik))
        assert np.any(predict(s, negative, {"a": x}).values <= 0.0)
        with pytest.raises(ValueError, match=f"{family}.*draw 0.*\\[\\(0,\\)") as info:
            predict(s, negative, {"a": x}, noise=True, seed=0)
        assert "non-positive" in str(info.value)
    # a mean that is positive everywhere is fine; the zero-dose row is the Hill floor alpha > 0
    positive = _hand_posterior(n=2, chains=1, alpha=0.5)
    s = Surface(arms_spec(likelihood=Likelihood(family="poisson")))
    assert predict(s, positive, {"a": x}, noise=True, seed=0).values.shape == (1, 2, 3)


# -- marginals -------------------------------------------------------------------------


def _central_difference(
    s: Surface, data: dict[str, np.ndarray], theta: dict[str, object], name: str, i: int, t: int
) -> np.ndarray:
    h = 1e-5
    up = {**data, name: data[name].copy()}
    down = {**data, name: data[name].copy()}
    up[name][i, t] += h
    down[name][i, t] -= h
    return np.asarray((s.forward(up, theta) - s.forward(down, theta)) / (2 * h))


@pytest.mark.parametrize("name", ["a", "b"])
def test_marginal_and_total_match_finite_differences(panel: Panel, name: str) -> None:
    spec = panel_spec()
    s = Surface(spec)
    data = prepare(spec, panel)
    theta = dict(PANEL_THETA)
    expr = marginal_expr(s, name)
    assert dimension(expr) == D.outcome / USD
    same = marginal(s, theta, data, name)
    total = marginal_total(s, theta, data, name)
    assert same.shape == total.shape == (3, N_PERIODS)
    i, t = 1, 2
    fd = _central_difference(s, data, theta, name, i, t)
    assert same[i, t] == pytest.approx(fd[i, t], rel=1e-6)
    assert total[i, t] == pytest.approx(fd[i].sum(), rel=1e-6)
    assert fd[[0, 2]].max() == 0.0  # nothing leaks to the other units


@pytest.mark.parametrize("name", ["a", "b"])
def test_marginal_expr_declares_each_parameter_with_one_dimension(name: str) -> None:
    s = Surface(panel_spec())
    expr = marginal_expr(s, name)
    seen: dict[str, set[Any]] = {}
    for _, node in walk(expr):
        if isinstance(node, Param):
            seen.setdefault(node.name, set()).add(node.dimension)
    amplitude = "beta_a" if name == "a" else "beta_rate_b"
    assert "gamma_a_b" in seen and amplitude in seen
    if name == "a":  # the interaction term divides the Hill derivative by the model's beta_a
        assert seen["beta_a"] == {D.outcome}
    for pname, dims in seen.items():
        assert dims == {s.model.parameter(pname).dimension}, (pname, dims)


def test_marginal_total_equals_marginal_without_carryover_and_at_steady_state(
    panel: Panel,
) -> None:
    spec = panel_spec()
    s = Surface(spec)
    data = prepare(spec, panel)
    theta = dict(PANEL_THETA)
    # b has no carryover: the total is the same-period effect
    np.testing.assert_array_equal(
        marginal_total(s, theta, data, "b"), marginal(s, theta, data, "b")
    )
    # a at a constant dose: the carried dose is at steady state from t = max_lag - 1 on
    # (zero history before t = 0), and Σ_l w_l g_{t+l} = g_t while t + max_lag <= T
    flat = {**data, "a": np.full_like(data["a"], 2.5), "b": np.full_like(data["b"], 1.0)}
    g = marginal_expr(s, "a")
    g_val = np.asarray(value(g, data=flat, params=theta))
    total = marginal_total(s, theta, flat, "a")
    steady = slice(2, N_PERIODS - 2)
    np.testing.assert_allclose(
        g_val[:, steady], np.broadcast_to(g_val[:, 2:3], g_val[:, steady].shape), rtol=1e-12
    )
    np.testing.assert_allclose(total[:, steady], g_val[:, steady], rtol=1e-12)
    # before steady state the total is the lag-weighted sum of the rising derivative
    w = 0.6 ** np.arange(3)
    w = w / w.sum()
    assert total[0, 0] == pytest.approx(float(w @ g_val[0, :3]), rel=1e-12)
    same = marginal(s, theta, flat, "a")
    w0 = 1.0 / sum(0.6**j for j in range(3))
    np.testing.assert_allclose(same, w0 * g_val, rtol=1e-12)
    with pytest.raises(KeyError):
        marginal(s, theta, data, "zzz")


def test_marginal_total_truncates_lags_beyond_the_horizon(panel: Panel) -> None:
    max_lag = N_PERIODS + 4
    spec = panel_spec(carryover={"a": GeometricCarryover(max_lag=max_lag)})
    s = Surface(spec)
    data = prepare(spec, panel)
    theta = dict(PANEL_THETA)
    g = np.asarray(value(marginal_expr(s, "a"), data=data, params=theta))
    total = marginal_total(s, theta, data, "a")
    assert total.shape == (3, N_PERIODS) and np.all(np.isfinite(total))
    w = 0.6 ** np.arange(max_lag)
    w = w / w.sum()
    expected = np.array(
        [
            [sum(w[lag] * g[i, t + lag] for lag in range(N_PERIODS - t)) for t in range(N_PERIODS)]
            for i in range(3)
        ]
    )
    np.testing.assert_allclose(total, expected, rtol=1e-12)
    # the in-horizon weights sum to less than one, so the total under-counts the full effect
    assert np.all(total[:, -1] == pytest.approx(w[0] * g[:, -1]))
    np.testing.assert_allclose(marginal(s, theta, data, "a"), w[0] * g, rtol=1e-12)
    fd = _central_difference(s, data, theta, "a", 2, 5)
    assert total[2, 5] == pytest.approx(fd[2].sum(), rel=1e-6)
