"""Tests for ``axiom.sim.panel`` and ``axiom.sim.surface_world``."""

from __future__ import annotations

import math

import numpy as np
import pytest
from pydantic import ValidationError

from axiom.core import D, Likelihood, Outcome, Treatment, Unsupported
from axiom.data import Panel
from axiom.sim.panel import DosePlan, draw_doses, long_frame, panel_from_arrays, unit_labels
from axiom.sim.surface_world import SurfaceWorld, arms_world, surface_world, true_parameters
from axiom.surface.carryover import DelayedCarryover, GeometricCarryover, WeibullCarryover
from axiom.surface.forward import marginal as forward_marginal
from axiom.surface.forward import marginal_total as forward_marginal_total
from axiom.surface.kernels import ExponentialKernel, HillKernel, LinearKernel, PowerKernel
from axiom.surface.linearize import check_linearization
from axiom.surface.model import INTERCEPT, UNIT_INTERCEPT, Surface, build, prepare
from axiom.surface.nuisance import FourierSeasonality, LinearTrend, NuisanceSet

# -- panel primitives ----------------------------------------------------------------


def test_dose_plan_validates_uniform_spread() -> None:
    DosePlan(distribution="uniform", spread=1.0)
    with pytest.raises(ValidationError):
        DosePlan(distribution="uniform", spread=1.5)
    with pytest.raises(ValidationError):
        DosePlan(zero_fraction=1.0)
    with pytest.raises(ValidationError):
        DosePlan(scale=0.0)


def test_draw_doses_lognormal_is_positive_and_seeded() -> None:
    plan = DosePlan(scale=50.0, spread=0.3)
    a = draw_doses(plan, 5, 7, seed=3)
    b = draw_doses(plan, 5, 7, seed=3)
    c = draw_doses(plan, 5, 7, seed=4)
    assert a.shape == (5, 7)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert np.all(a > 0)
    # median dose is the scale: the log-doses are centred on log(scale)
    big = draw_doses(plan, 200, 200, seed=0)
    assert abs(np.median(np.log(big)) - np.log(50.0)) < 0.02


def test_draw_doses_uniform_respects_bounds_and_zero_fraction() -> None:
    plan = DosePlan(distribution="uniform", scale=10.0, spread=0.5, zero_fraction=0.2)
    d = draw_doses(plan, 100, 100, seed=1)
    nonzero = d[d > 0]
    assert np.all(nonzero >= 5.0) and np.all(nonzero <= 15.0)
    share = float(np.mean(d == 0.0))
    assert 0.17 < share < 0.23
    with pytest.raises(ValueError):
        draw_doses(plan, 0, 3)


def test_unit_labels_are_zero_padded_and_sorted() -> None:
    labels = unit_labels(12)
    assert labels[:3] == ("u00", "u01", "u02") and labels[-1] == "u11"
    assert list(labels) == sorted(labels)
    assert unit_labels(3, prefix="site") == ("site0", "site1", "site2")
    with pytest.raises(ValueError):
        unit_labels(0)


def test_long_frame_is_unit_major_period_minor() -> None:
    x = np.arange(6, dtype=float).reshape(2, 3)
    frame = long_frame({"x": x}, units=("a", "b"), periods=(10, 11, 12))
    assert list(frame.columns) == ["unit", "t", "x"]
    assert frame["unit"].tolist() == ["a", "a", "a", "b", "b", "b"]
    assert frame["t"].tolist() == [10, 11, 12, 10, 11, 12]
    assert frame["x"].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]


def test_long_frame_rejects_bad_inputs() -> None:
    x = np.zeros((2, 3))
    with pytest.raises(ValueError):
        long_frame({}, units=("a", "b"))
    with pytest.raises(ValueError):
        long_frame({"x": x}, units=("a",))
    with pytest.raises(ValueError):
        long_frame({"x": x}, units=("a", "a"))
    with pytest.raises(ValueError):
        long_frame({"x": x, "y": np.zeros((3, 2))}, units=("a", "b"))
    with pytest.raises(ValueError):
        long_frame({"x": x}, units=("a", "b"), periods=(0, 1))
    with pytest.raises(ValueError):
        long_frame({"t": x}, units=("a", "b"))
    bad = x.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        long_frame({"x": bad}, units=("a", "b"))


_TREATMENTS = [
    Treatment(name="a", dimension=D.currency, unit="USD"),
    Treatment(name="b", dimension=D.currency, unit="USD"),
]
_OUTCOME = Outcome(name="y", dimension=D.outcome)


def test_panel_from_arrays_is_balanced_and_round_trips() -> None:
    rng = np.random.default_rng(0)
    a = rng.random((4, 6))
    b = rng.random((4, 6))
    y = rng.random((4, 6))
    panel = panel_from_arrays(
        doses={"a": a, "b": b},
        outcome=y,
        treatments=_TREATMENTS,
        outcome_entity=_OUTCOME,
        units=unit_labels(4),
    )
    assert isinstance(panel, Panel)
    assert panel.completeness().balanced
    assert panel.units == unit_labels(4)
    assert panel.periods == (0, 1, 2, 3, 4, 5)
    assert np.array_equal(panel.array("a"), a)
    assert np.array_equal(panel.array("b"), b)
    assert np.array_equal(panel.array("y"), y)
    assert panel.roles.dimension_of("a") == D.currency
    assert panel.roles.dimension_of("y") == D.outcome


def test_panel_from_arrays_requires_matching_treatments() -> None:
    with pytest.raises(ValueError):
        panel_from_arrays(
            doses={"a": np.zeros((2, 2))},
            outcome=np.zeros((2, 2)),
            treatments=[Treatment(name="b", dimension=D.currency)],
            outcome_entity=_OUTCOME,
            units=("u0", "u1"),
        )


def test_panel_from_arrays_rejects_unsorted_units_and_periods() -> None:
    # Panel sorts by label, so unsorted labels would silently permute the rows.
    doses = {"a": np.arange(4.0).reshape(2, 2)}
    with pytest.raises(ValueError, match="sorted"):
        panel_from_arrays(
            doses=doses,
            outcome=np.zeros((2, 2)),
            treatments=_TREATMENTS[:1],
            outcome_entity=_OUTCOME,
            units=("u1", "u0"),
        )
    # "u10" sorts before "u2" as a string: the numeric order is not the string order
    with pytest.raises(ValueError, match="sorted"):
        panel_from_arrays(
            doses=doses,
            outcome=np.zeros((2, 2)),
            treatments=_TREATMENTS[:1],
            outcome_entity=_OUTCOME,
            units=("u2", "u10"),
        )
    with pytest.raises(ValueError, match="ascending"):
        panel_from_arrays(
            doses=doses,
            outcome=np.zeros((2, 2)),
            treatments=_TREATMENTS[:1],
            outcome_entity=_OUTCOME,
            units=("u0", "u1"),
            periods=(1, 0),
        )
    ok = panel_from_arrays(
        doses=doses,
        outcome=np.zeros((2, 2)),
        treatments=_TREATMENTS[:1],
        outcome_entity=_OUTCOME,
        units=("u0", "u1"),
        periods=(3, 7),
    )
    assert np.array_equal(ok.array("a"), doses["a"])


# -- surface worlds ------------------------------------------------------------------


def _world(seed: int = 7, **kw: object) -> SurfaceWorld:
    defaults: dict[str, object] = dict(
        n_units=3,
        n_periods=10,
        carryover={"a": GeometricCarryover(max_lag=4)},
        doses=DosePlan(scale=50.0, zero_fraction=0.1),
        seed=seed,
    )
    defaults.update(kw)
    return surface_world(**defaults)  # type: ignore[arg-type]


def test_surface_world_panel_is_balanced_and_shaped() -> None:
    w = _world()
    assert isinstance(w, SurfaceWorld)
    assert w.panel.completeness().balanced
    assert w.panel.units == unit_labels(3)
    assert w.panel.periods == tuple(range(10))
    assert w.mean.shape == (3, 10) and w.n_units == 3 and w.n_periods == 10
    assert w.treatments == ("a", "b")  # default treatment names carry no marketing nouns
    assert set(w.spec.unit_labels) == set(w.panel.units)
    assert w.provenance["panel_hash"] == w.panel.content_hash()
    assert w.provenance["seed"] == 7


def test_surface_world_is_deterministic_by_seed() -> None:
    a, b, c = _world(seed=11), _world(seed=11), _world(seed=12)
    assert a.panel.content_hash() == b.panel.content_hash()
    assert a.theta.keys() == b.theta.keys()
    assert all(np.array_equal(a.theta[k], b.theta[k]) for k in a.theta)
    assert a.panel.content_hash() != c.panel.content_hash()


def test_forward_at_truth_equals_noise_free_outcome() -> None:
    w = _world(noise_sd=1.0)
    # The world's own forward, the Surface's forward, and core.value on the built model
    # are the same number: one forward().
    surface = Surface(w.spec)
    mu = surface.forward(w.data, w.theta)
    assert np.allclose(mu, w.mean, atol=1e-12)
    assert np.allclose(w.forward(), w.mean, atol=1e-12)
    observed = w.panel.array("y")
    assert np.allclose(observed - w.mean, w.noise, atol=1e-12)
    assert not np.allclose(observed, w.mean)
    # the noise is what the likelihood says it is
    assert w.noise_sd == 1.0
    assert abs(np.std(w.noise) - w.noise_sd) < 0.5
    assert np.allclose(prepare(w.spec, w.panel)["a"], w.panel.array("a"))


def test_linearize_invariant_holds_on_world_data() -> None:
    # gate 9 on the world's own (n_units, n_periods) layout, with carryover and a nuisance
    nuisance = NuisanceSet(terms=(LinearTrend(), FourierSeasonality(period=10.0, order=1)))
    for w in (_world(), _world(nuisance=nuisance, interactions=(("a", "b"),))):
        assert check_linearization(w.surface, w.data, w.theta) < 1e-12


def test_truth_centre_mode_pins_structural_parameters() -> None:
    w = _world()
    assert w.theta["k_a"] == pytest.approx(50.0)  # reference dose = dose plan scale
    assert w.theta["s_a"] == pytest.approx(2.0)  # gamma(4, 2) mean
    assert w.theta["lam_a"] == pytest.approx(0.5)  # beta(2, 2) mean
    assert w.theta[UNIT_INTERCEPT].shape == (3,)
    assert set(w.structural) == {"k_a", "s_a", "beta_a", "lam_a", "k_b", "s_b", "beta_b"}
    assert set(w.theta) == {p.name for p in w.model.parameters}
    assert w.model == build(w.spec)
    assert w.provenance["structural"] == list(w.structural)


def test_delayed_and_weibull_carryover_centre_values() -> None:
    w = _world(carryover={"a": DelayedCarryover(max_lag=5), "b": WeibullCarryover(max_lag=4)})
    assert w.theta["lam_a"] == pytest.approx(0.5)  # beta(2, 2) mean
    assert w.theta["theta_a"] == pytest.approx(2.0)  # uniform(0, max_lag - 1) mean
    assert w.theta["lam_b"] == pytest.approx(2.0)  # gamma(2, 4 / max_lag) mean = max_lag / 2
    assert w.theta["kappa_b"] == pytest.approx(2.0)  # gamma(2, 1) mean
    assert {"lam_a", "theta_a", "lam_b", "kappa_b"} <= set(w.structural)
    # the weights at the centre are a proper distribution and the panel is finite
    assert np.all(np.isfinite(w.mean))
    m = w.marginal("b", horizon="total")
    assert not isinstance(m, Unsupported)


def test_noise_sd_resolution_and_conflict() -> None:
    # noise_sd given: it is the likelihood scale
    assert _world(noise_sd=0.2).theta["sigma"] == pytest.approx(0.2)
    # truth alone: the truth's scale is not overwritten
    w = _world(truth={"sigma": 7.0})
    assert w.theta["sigma"] == pytest.approx(7.0)
    assert w.noise_sd == pytest.approx(7.0) and w.provenance["noise_sd"] == pytest.approx(7.0)
    # both, agreeing: fine; disagreeing: an error, not a silent overwrite
    assert _world(truth={"sigma": 0.3}, noise_sd=0.3).noise_sd == pytest.approx(0.3)
    with pytest.raises(ValueError, match="disagrees"):
        _world(truth={"sigma": 7.0}, noise_sd=1.0)
    # neither: the prior centre (halfnormal(noise_scale) mean), never a draw
    centre = _world()
    assert centre.noise_sd == pytest.approx(math.sqrt(2.0 / math.pi))
    assert _world(truth_mode="prior").noise_sd == pytest.approx(math.sqrt(2.0 / math.pi))
    with pytest.raises(ValueError):
        _world(noise_sd=0.0)
    with pytest.raises(ValueError):
        _world(noise_sd=float("nan"))


def test_truth_overrides_and_validation() -> None:
    w = _world(truth={"beta_a": 3.0, "alpha_mean": 10.0, "alpha_sd": 0.01}, noise_sd=0.2)
    assert w.theta["beta_a"] == pytest.approx(3.0)
    assert w.theta["sigma"] == pytest.approx(0.2)
    assert np.all(np.abs(w.theta[UNIT_INTERCEPT] - 10.0) < 0.1)
    assert w.provenance["truth"]["names"] == ["alpha_mean", "alpha_sd", "beta_a"]
    assert len(w.provenance["truth"]["hash"]) == 64
    assert w.provenance["truth"]["hash"] != _world().provenance["truth"]["hash"]
    with pytest.raises(KeyError):
        _world(truth={"not_a_parameter": 1.0})
    # a vector truth of the wrong length names the parameter and its declared shape
    with pytest.raises(ValueError, match=r"alpha_unit.*\(3,\)"):
        _world(truth={UNIT_INTERCEPT: np.zeros(4)})
    with pytest.raises(ValueError, match="beta_a"):
        _world(truth={"beta_a": np.zeros(2)})
    assert np.allclose(_world(truth={UNIT_INTERCEPT: 2.0}).theta[UNIT_INTERCEPT], 2.0)
    with pytest.raises(ValueError):
        surface_world(n_units=0, n_periods=3)


def test_prior_mode_is_seeded_and_differs_only_in_structural_names() -> None:
    p1, p2 = _world(seed=5, truth_mode="prior"), _world(seed=5, truth_mode="prior")
    assert p1.panel.content_hash() == p2.panel.content_hash()
    assert all(np.array_equal(p1.theta[k], p2.theta[k]) for k in p1.theta)
    assert p1.provenance["truth_mode"] == "prior"
    c = _world(seed=5)
    structural = set(c.structural)
    for name in c.theta:
        if name not in structural:
            assert np.array_equal(c.theta[name], p1.theta[name]), name
    assert any(not np.allclose(c.theta[n], p1.theta[n]) for n in structural)
    # the doses are the same too: only the truth moved
    assert np.array_equal(c.panel.array("a"), p1.panel.array("a"))
    assert p1.theta["s_a"] > 0 and 0 < p1.theta["lam_a"] < 1


def test_true_parameters_resolves_hierarchy_and_fixed() -> None:
    w = _world()
    theta = true_parameters(
        w.model,
        structural=w.structural,
        truth={"alpha_mean": 5.0, "alpha_sd": 1e-9},
        seed=0,
    )
    assert np.allclose(theta[UNIT_INTERCEPT], 5.0, atol=1e-6)
    again = true_parameters(w.model, structural=w.structural, truth={"alpha_mean": 5.0}, seed=0)
    assert again["alpha_mean"] == pytest.approx(5.0)
    assert again["k_a"] == pytest.approx(50.0)
    with pytest.raises(KeyError, match="structural"):
        true_parameters(w.model, structural=("not_declared",), seed=0)


def test_total_response_is_zero_at_zero_dose_and_positive_otherwise() -> None:
    w = _world()
    zero = {"a": 0.0, "b": 0.0}
    assert np.allclose(w.total_response(zero), 0.0, atol=1e-12)
    assert np.all(w.total_response({"a": 50.0, "b": 50.0}) > 0)
    # at the half-saturation dose each Hill response is beta / 2
    expected = 0.5 * (w.theta["beta_a"] + w.theta["beta_b"])
    assert np.allclose(w.total_response({"a": 50.0, "b": 50.0})[:, 5:], expected)
    with pytest.raises(KeyError):
        w.forward({"c": 1.0})


def test_dose_grid_shapes_are_unambiguous() -> None:
    w = _world()
    full = np.full((3, 10), 5.0)
    assert np.allclose(w.forward({"a": full}), w.forward({"a": 5.0}))
    for bad in (np.ones(7), np.ones(10), np.ones(3), np.ones((1, 10)), np.ones((3, 1))):
        with pytest.raises(ValueError, match="shape"):
            w.forward({"a": bad})
    with pytest.raises(ValueError, match="finite"):
        w.forward({"a": np.full((3, 10), np.nan)})
    # square grid: a 1-D vector could be per-unit or per-period, so it is rejected
    sq = _world(n_units=4, n_periods=4)
    with pytest.raises(ValueError, match="shape"):
        sq.forward({"a": np.ones(4)})
    # a one-period world accepts a per-unit vector (and nothing else 1-D)
    one = _world(n_units=4, n_periods=1, carryover=None)
    assert np.allclose(one.forward({"a": np.full(4, 2.0)}), one.forward({"a": 2.0}))
    with pytest.raises(ValueError, match="shape"):
        one.forward({"a": np.ones(3)})


def test_marginal_matches_single_cell_finite_differences() -> None:
    w = _world(carryover={"a": GeometricCarryover(max_lag=4)})
    a = w.panel.array("a")
    h = 1e-4
    m_period = w.marginal("a")
    m_total = w.marginal("a", horizon="total")
    assert not isinstance(m_period, Unsupported) and not isinstance(m_total, Unsupported)
    assert m_period.shape == m_total.shape == (3, 10)
    for unit, t in ((0, 0), (1, 5), (2, 9), (1, 7)):
        up, dn = a.copy(), a.copy()
        up[unit, t] += h
        dn[unit, t] -= h
        fd = (w.forward({"a": up}) - w.forward({"a": dn})) / (2 * h)
        # same-period: the perturbed cell; total: the whole of the unit's horizon
        assert fd[unit, t] == pytest.approx(float(m_period[unit, t]), rel=1e-5, abs=1e-9)
        assert fd[unit].sum() == pytest.approx(float(m_total[unit, t]), rel=1e-5, abs=1e-9)
        # carryover never crosses units: the other rows do not move at all
        assert np.array_equal(np.delete(fd, unit, axis=0), np.zeros((2, 10)))
    # the last period has no future inside the horizon: total == period there
    assert np.allclose(m_total[:, -1], m_period[:, -1])
    # the world's marginals are surface.forward's marginals at the truth
    assert np.allclose(m_period, forward_marginal(w.surface, w.theta, w.data, "a"))
    assert np.allclose(m_total, forward_marginal_total(w.surface, w.theta, w.data, "a"))
    # no carryover: period and total marginals coincide
    m_b = w.marginal("b")
    assert not isinstance(m_b, Unsupported)
    assert np.allclose(m_b, w.marginal("b", horizon="total"))
    with pytest.raises(KeyError):
        w.marginal("c")


def test_marginal_includes_interaction_cross_terms() -> None:
    w = _world(interactions=(("a", "b"),))
    assert "gamma_a_b" in w.theta
    a = w.panel.array("a")
    h = 1e-4
    for horizon, reduce_fd in (("period", False), ("total", True)):
        m = w.marginal("a", horizon=horizon)  # type: ignore[arg-type]
        assert not isinstance(m, Unsupported)
        for unit, t in ((0, 0), (1, 5), (2, 8)):
            up, dn = a.copy(), a.copy()
            up[unit, t] += h
            dn[unit, t] -= h
            fd = (w.forward({"a": up}) - w.forward({"a": dn})) / (2 * h)
            got = fd[unit].sum() if reduce_fd else fd[unit, t]
            assert got == pytest.approx(float(m[unit, t]), rel=1e-5, abs=1e-9)


def test_marginal_is_unsupported_where_the_slope_is_infinite() -> None:
    # PowerKernel's default prior puts s in (0, 1): the slope at zero dose is infinite
    w = arms_world(n_units=4, doses={"dose": [0.0, 1.0, 2.0, 3.0]}, kernels=PowerKernel(), seed=0)
    assert 0 < w.theta["s_dose"] < 1
    m = w.marginal("dose")
    assert isinstance(m, Unsupported)
    assert "u0" in m.reason and m.detail["n_cells"] == "1"
    # away from zero the marginal is a number
    ok = w.marginal("dose", dose={"dose": [1.0, 1.0, 2.0, 3.0]})
    assert not isinstance(ok, Unsupported) and np.all(np.isfinite(ok))


def test_nuisance_and_intercept_variants() -> None:
    nuisance = NuisanceSet(terms=(FourierSeasonality(period=10.0, order=1), LinearTrend()))
    w = _world(nuisance=nuisance, intercept="per_unit")
    assert {"n0_sin_1", "n0_cos_1", "n1_trend"} <= set(w.data)
    assert "n0_coef_sin_1" in w.theta and "alpha_mean" not in w.theta
    assert w.theta[UNIT_INTERCEPT].shape == (3,)
    shared = _world(intercept="shared")
    assert INTERCEPT in shared.theta and UNIT_INTERCEPT not in shared.theta
    none = _world(intercept="none")
    assert np.allclose(none.total_response(), none.mean)


def test_kernel_and_treatment_variants() -> None:
    w = surface_world(
        n_units=2,
        n_periods=6,
        treatments=(Treatment(name="x", dimension=D.currency, unit="USD"), "z"),
        kernels={"x": ExponentialKernel(reference_dose=5.0), "z": LinearKernel()},
        doses={"x": DosePlan(scale=5.0), "z": DosePlan(distribution="uniform", scale=2.0)},
        seed=3,
    )
    assert w.theta["k_x"] == pytest.approx(5.0)
    assert "beta_rate_z" in w.theta
    m = w.marginal("z")
    assert not isinstance(m, Unsupported)
    assert m.shape == (2, 6) and np.allclose(m, w.theta["beta_rate_z"])
    with pytest.raises(KeyError):
        surface_world(n_units=2, n_periods=2, kernels={"zzz": HillKernel()})
    with pytest.raises(ValueError):
        surface_world(n_units=2, n_periods=2, treatments=("a", "a"))


def test_arms_world_is_single_period_with_shared_intercept() -> None:
    grid = np.linspace(0.0, 100.0, 8)
    w = arms_world(n_units=8, doses={"dose": grid}, truth={"beta_dose": 2.0}, seed=5)
    assert w.n_periods == 1 and w.n_units == 8
    assert w.panel.completeness().balanced and w.panel.periods == (0,)
    assert np.array_equal(w.panel.array("dose").ravel(), grid)
    assert w.spec.intercept == "shared" and INTERCEPT in w.theta
    assert w.spec.carryover_of("dose").name == "none"
    assert w.theta["k_dose"] == pytest.approx(float(grid.mean()))
    assert np.allclose(w.forward(), w.mean)
    # a 1-D per-unit dose vector is accepted for a one-period world
    assert np.allclose(w.forward({"dose": grid}), w.mean)
    resp = w.total_response().ravel()
    assert resp[0] == pytest.approx(0.0) and np.all(np.diff(resp) > 0)
    assert resp[-1] < 2.0


def test_arms_world_random_doses_and_validation() -> None:
    w = arms_world(n_units=5, treatments=("x", "z"), doses=DosePlan(scale=3.0), seed=1)
    assert w.panel.array("x").shape == (5, 1)
    assert (
        arms_world(
            n_units=5, treatments=("x", "z"), doses=DosePlan(scale=3.0), seed=1
        ).panel.content_hash()
        == w.panel.content_hash()
    )
    with pytest.raises(ValueError):
        arms_world(n_units=4, doses={"dose": np.ones(3)})
    with pytest.raises(KeyError):
        arms_world(n_units=4, doses={"other": np.ones(4)})
    with pytest.raises(TypeError):
        arms_world(n_units=4, treatments=("a", "b"), doses={"a": np.ones(4), "b": DosePlan()})
    with pytest.raises(ValueError):
        arms_world(n_units=0)
    with pytest.raises(ValueError, match="disagrees"):
        arms_world(n_units=4, truth={"sigma": 2.0}, noise_sd=1.0)


def test_student_t_likelihood_noise_uses_the_declared_scale() -> None:
    lik = Likelihood(family="student_t", scale="sigma", df=5.0)
    w = _world(likelihood=lik, noise_sd=0.5, n_units=20, n_periods=30)
    assert w.spec.likelihood == lik and w.model.likelihood == lik
    assert w.theta["sigma"] == pytest.approx(0.5)
    # t_5 has variance df / (df - 2) = 5/3 times the scale squared
    assert abs(np.std(w.noise) - 0.5 * np.sqrt(5.0 / 3.0)) < 0.1


def test_lognormal_world_needs_a_positive_mean() -> None:
    lik = Likelihood(family="lognormal", scale="sigma")
    w = _world(likelihood=lik, intercept="shared", truth={"alpha": 5.0}, noise_sd=0.1, n_units=10)
    y = w.panel.array("y")
    assert np.all(w.mean > 0) and np.all(y > 0)
    # outcome = mean · exp(eps), eps ~ N(0, sigma): the log-ratio has sd sigma
    assert abs(np.std(np.log(y) - np.log(w.mean)) - 0.1) < 0.03
    assert np.allclose(w.noise, y - w.mean)
    with pytest.raises(ValueError, match="lognormal.*positive.*u0"):
        _world(likelihood=lik, intercept="shared", truth={"alpha": -50.0}, noise_sd=0.1)


def test_poisson_world_needs_a_positive_mean_and_no_scale() -> None:
    lik = Likelihood(family="poisson")
    w = _world(likelihood=lik, intercept="shared", truth={"alpha": 20.0}, n_units=10)
    y = w.panel.array("y")
    assert "sigma" not in w.theta
    assert w.noise_sd is None and w.provenance["noise_sd"] is None
    assert np.all(y >= 0) and np.all(np.mod(y, 1.0) == 0.0)
    assert abs(np.mean(y) - np.mean(w.mean)) < 1.0
    with pytest.raises(ValueError, match="poisson.*positive"):
        _world(likelihood=lik, intercept="shared", truth={"alpha": -50.0})
    with pytest.raises(ValueError, match="no scale"):
        _world(likelihood=lik, intercept="shared", truth={"alpha": 20.0}, noise_sd=1.0)
