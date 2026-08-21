"""Backtest: forecasts through the one forward with carried state, scoring, freeze-and-replay."""

from __future__ import annotations

import numpy as np
import pytest

from axiom.core import Posterior, PredictiveDraws, Unsupported
from axiom.diagnose.backtest import (
    Backtest,
    FrozenPredictor,
    crps,
    forecast,
    freeze,
    rolling_origin,
    training_panel,
)
from axiom.sim import DosePlan, surface_world
from axiom.sim.surface_world import SurfaceWorld
from axiom.surface import FitResult, fit, prepare, resolve_conventions
from axiom.surface.carryover import GeometricCarryover
from axiom.surface.forward import predict
from axiom.surface.nuisance import LinearTrend, NuisanceSet

N_UNITS, N_PERIODS, HORIZON = 3, 14, 3


@pytest.fixture(scope="module")
def carry() -> SurfaceWorld:
    return surface_world(
        n_units=N_UNITS,
        n_periods=N_PERIODS,
        treatments=("a",),
        carryover={"a": GeometricCarryover(max_lag=4)},
        intercept="shared",
        doses=DosePlan(scale=50.0, zero_fraction=0.2),
        noise_sd=0.1,
        seed=31,
    )


@pytest.fixture(scope="module")
def plain() -> SurfaceWorld:
    return surface_world(
        n_units=N_UNITS,
        n_periods=N_PERIODS,
        treatments=("a",),
        intercept="shared",
        doses=DosePlan(scale=50.0, zero_fraction=0.2),
        noise_sd=0.1,
        seed=32,
    )


def _truth_posterior(world: SurfaceWorld) -> Posterior:
    """The world's true parameters as a one-draw posterior."""
    return Posterior({k: np.asarray(v)[None, None, ...] for k, v in world.theta.items()})


# -- training panels ---------------------------------------------------------------------------


def test_training_panel_is_a_prefix(carry: SurfaceWorld) -> None:
    p = training_panel(carry.panel, 5)
    assert p.periods == carry.panel.periods[:5]
    assert p.units == carry.panel.units
    assert np.array_equal(p.array("a"), carry.panel.array("a")[:, :5])
    with pytest.raises(ValueError):
        training_panel(carry.panel, 0)
    with pytest.raises(ValueError):
        training_panel(carry.panel, N_PERIODS + 1)


# -- the forecast carries the carryover state ---------------------------------------------------


def test_forecast_on_a_carryover_world_equals_the_world_forward(carry: SurfaceWorld) -> None:
    origin = 8
    post = _truth_posterior(carry)
    out = forecast(carry.spec, post, carry.panel, origin=origin, horizon=HORIZON)
    assert isinstance(out, PredictiveDraws)
    assert out.values.shape == (1, 1, N_UNITS, HORIZON)
    truth = carry.forward()[:, origin : origin + HORIZON]
    # exact: the same forward over the same full dose path at the same parameters
    assert np.allclose(out.values[0, 0], truth, rtol=1e-12, atol=1e-12)
    assert list(out.coords["period"]) == list(carry.panel.periods[origin : origin + HORIZON])
    assert np.allclose(
        np.asarray(out.coords["a"]).reshape(N_UNITS, HORIZON),
        carry.panel.array("a")[:, origin : origin + HORIZON],
    )
    # and the observed outcome differs from it only by the noise that was added
    observed = carry.panel.array("y")[:, origin : origin + HORIZON]
    assert np.max(np.abs(observed - out.values[0, 0])) < 5 * (carry.noise_sd or 1.0)


def test_forecast_is_not_the_horizon_only_evaluation(carry: SurfaceWorld) -> None:
    """Evaluating only the horizon's panel drops the dose history — the parent's bug."""
    origin = 8
    post = _truth_posterior(carry)
    carried = forecast(carry.spec, post, carry.panel, origin=origin, horizon=HORIZON)
    frame = carry.panel.frame
    roles = carry.panel.roles
    tail = frame[frame[roles.time].isin(carry.panel.periods[origin : origin + HORIZON])]
    from axiom.data import Panel

    horizon_only = prepare(carry.spec, Panel(tail, roles))
    naive = predict(carry.surface, post, horizon_only)
    assert naive.values.shape == carried.values.shape
    assert not np.allclose(naive.values, carried.values, rtol=1e-3)


def test_forecast_on_a_no_carryover_world_matches_a_direct_predict(plain: SurfaceWorld) -> None:
    origin = 8
    post = _truth_posterior(plain)
    out = forecast(plain.spec, post, plain.panel, origin=origin, horizon=HORIZON)
    data = prepare(plain.spec, plain.panel)
    direct = predict(plain.surface, post, data)
    assert np.allclose(out.values[0, 0], direct.values[0, 0][:, origin : origin + HORIZON])
    assert np.allclose(out.values[0, 0], plain.forward()[:, origin : origin + HORIZON])


def test_forecast_argument_validation(carry: SurfaceWorld) -> None:
    post = _truth_posterior(carry)
    with pytest.raises(ValueError, match="horizon"):
        forecast(carry.spec, post, carry.panel, origin=3, horizon=0)
    with pytest.raises(ValueError, match="origin"):
        forecast(carry.spec, post, carry.panel, origin=0, horizon=2)
    with pytest.raises(ValueError, match="runs past"):
        forecast(carry.spec, post, carry.panel, origin=N_PERIODS - 1, horizon=2)


def test_forecast_with_noise_scatters_around_the_mean(carry: SurfaceWorld) -> None:
    post = _truth_posterior(carry)
    draws = Posterior({k: np.repeat(post.draws(k), 200, axis=1) for k in post.names()})
    noisy = forecast(carry.spec, draws, carry.panel, origin=6, horizon=2, seed=0, noise=True)
    mean = forecast(carry.spec, post, carry.panel, origin=6, horizon=2)
    assert noisy.values.shape == (1, 200, N_UNITS, 2)
    assert np.allclose(noisy.values.mean(axis=1), mean.values[0], atol=0.05)
    assert noisy.values.std(axis=1).mean() == pytest.approx(carry.noise_sd, rel=0.2)


# -- crps ----------------------------------------------------------------------------------------


def test_crps_matches_the_pairwise_definition_and_degenerates_to_mae() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(50, 4))
    y = rng.normal(size=4)
    got = crps(x, y)
    pairwise = np.mean(np.abs(x - y), axis=0) - 0.5 * np.mean(
        np.abs(x[:, None, :] - x[None, :, :]), axis=(0, 1)
    )
    assert np.allclose(got, pairwise)
    # a point forecast's CRPS is its absolute error
    assert np.allclose(crps(np.full((1, 4), 2.0), np.full(4, 0.5)), 1.5)
    with pytest.raises(ValueError):
        crps(np.empty((0, 3)), np.zeros(3))


# -- rolling origin -----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def backtest(carry: SurfaceWorld) -> Backtest:
    out = rolling_origin(
        carry.spec, carry.panel, origins=(8, 10), horizon=HORIZON, draws=40, chains=1, seed=0
    )
    assert isinstance(out, Backtest)
    return out


def test_rolling_origin_scores_every_step_with_provenance(
    carry: SurfaceWorld, backtest: Backtest
) -> None:
    bt = backtest
    assert bt.origins == (8, 10) and bt.horizon == HORIZON and bt.n_units == N_UNITS
    assert bt.n_failed_fits == 0 and bt.failures == ()
    assert len(bt.forecasts) == 2 and len(bt.scores) == HORIZON
    assert bt.spec_hash == carry.spec.content_hash()
    assert bt.panel_hash == carry.panel.content_hash()
    for step, s in enumerate(bt.scores, start=1):
        assert s.step == step
        assert s.n == 2 * N_UNITS
        assert s.coverage_region.n == s.n and s.coverage_region.p == 0.9
        assert s.coverage_region.alpha == 0.01
        assert 0.0 <= s.coverage <= 1.0
        assert s.rmse >= s.mae >= 0.0 and s.crps >= 0.0
        assert s.passed == s.coverage_region.accepts(round(s.coverage * s.n))
    # a well-fitted carryover world forecasts close to the truth: errors at the noise scale
    noise = carry.noise_sd or 1.0
    assert all(s.mae < 4 * noise for s in bt.scores)
    assert all(s.crps < 4 * noise for s in bt.scores)
    assert bt.passed
    back = Backtest.from_json(bt.to_json())
    assert back == bt


def test_rolling_origin_forecasts_record_observed_and_mean_paths(
    carry: SurfaceWorld, backtest: Backtest
) -> None:
    f = backtest.forecasts[0]
    assert f.origin == 8 and f.converged
    assert f.units == carry.spec.unit_labels
    assert f.periods == tuple(float(p) for p in carry.panel.periods[8 : 8 + HORIZON])
    observed = np.asarray(f.observed)
    assert observed.shape == (N_UNITS, HORIZON)
    assert np.allclose(observed, carry.panel.array("y")[:, 8 : 8 + HORIZON])
    mean = np.asarray(f.mean)
    truth = carry.forward()[:, 8 : 8 + HORIZON]
    assert np.max(np.abs(mean - truth)) < 4 * (carry.noise_sd or 1.0)
    assert np.all(np.asarray(f.lower) <= np.asarray(f.upper))


def test_rolling_origin_with_a_trend_lays_the_horizon_out_on_the_fitted_basis() -> None:
    world = surface_world(
        n_units=2,
        n_periods=12,
        treatments=("a",),
        intercept="shared",
        nuisance=NuisanceSet(terms=(LinearTrend(),)),
        doses=DosePlan(scale=50.0, zero_fraction=0.2),
        noise_sd=0.1,
        seed=5,
    )
    post = _truth_posterior(world)
    res = fit(world.spec, training_panel(world.panel, 8), draws=5, chains=1, seed=0)
    conventions = res.provenance["nuisance_conventions"]
    out = forecast(world.spec, post, world.panel, origin=8, horizon=2, conventions=conventions)
    # the training conventions resolve the trend against 8 periods; the truth used all 12,
    # so the forecast reproduces the truth only on the basis the truth was generated with
    full = forecast(
        world.spec,
        post,
        world.panel,
        origin=8,
        horizon=2,
        conventions=resolve_conventions(world.spec, world.panel),
    )
    assert np.allclose(full.values[0, 0], world.forward()[:, 8:10])
    assert not np.allclose(out.values[0, 0], full.values[0, 0])
    # and rolling_origin, which lays the horizon out on the fitted basis, scores sensibly
    bt = rolling_origin(world.spec, world.panel, origins=(8,), horizon=2, draws=30, seed=0)
    assert isinstance(bt, Backtest)
    assert all(s.mae < 5 * (world.noise_sd or 1.0) for s in bt.scores)


def test_rolling_origin_validates_origins(carry: SurfaceWorld) -> None:
    with pytest.raises(ValueError, match="outside"):
        rolling_origin(carry.spec, carry.panel, origins=(N_PERIODS,), horizon=1)
    with pytest.raises(ValueError, match="distinct"):
        rolling_origin(carry.spec, carry.panel, origins=(5, 5), horizon=1)
    with pytest.raises(ValueError, match="at least one origin"):
        rolling_origin(carry.spec, carry.panel, origins=(), horizon=1)


def test_rolling_origin_reports_failed_refits(carry: SurfaceWorld) -> None:
    out = rolling_origin(
        carry.spec, carry.panel, origins=(8,), horizon=2, backend="no-such-backend"
    )
    assert isinstance(out, Unsupported)
    assert out.detail["n_failed"] == "1"


# -- freeze and replay --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fitted(carry: SurfaceWorld) -> FitResult:
    return fit(carry.spec, carry.panel, draws=30, chains=1, seed=4)


def test_frozen_predictor_replays_prepare_plus_forward(
    carry: SurfaceWorld, fitted: FitResult
) -> None:
    fp = freeze(fitted)
    assert isinstance(fp, FrozenPredictor)
    assert fp.spec == carry.spec
    assert fp.provenance["spec_hash"] == carry.spec.content_hash()
    assert fp.provenance["n_draws"] == 30
    out = fp.predict(carry.panel)
    assert isinstance(out, PredictiveDraws)
    assert out.values.shape == (1, 30, N_UNITS, N_PERIODS)
    direct = predict(carry.surface, fitted.posterior, carry.data)  # type: ignore[arg-type]
    assert np.allclose(out.values, direct.values)
    assert list(out.coords["unit"]) == list(carry.spec.unit_labels)
    # a new panel: the same units, a different dose path
    new = surface_world(
        n_units=N_UNITS,
        n_periods=6,
        treatments=("a",),
        carryover={"a": GeometricCarryover(max_lag=4)},
        intercept="shared",
        doses=DosePlan(scale=50.0),
        noise_sd=0.1,
        seed=99,
    )
    replay = fp.predict(new.panel)
    assert isinstance(replay, PredictiveDraws)
    assert replay.values.shape == (1, 30, N_UNITS, 6)
    via_forward = predict(carry.surface, fitted.posterior, prepare(carry.spec, new.panel))  # type: ignore[arg-type]
    assert np.allclose(replay.values, via_forward.values)


def test_frozen_predictor_hash_and_unit_mismatch(carry: SurfaceWorld, fitted: FitResult) -> None:
    fp = freeze(fitted)
    assert isinstance(fp, FrozenPredictor)
    h = fp.content_hash
    assert len(h) == 64 and fp.content_hash == h
    other = freeze(fit(carry.spec, carry.panel, draws=30, chains=1, seed=5))
    assert isinstance(other, FrozenPredictor) and other.content_hash != h
    wrong = surface_world(n_units=N_UNITS + 1, n_periods=4, treatments=("a",), seed=1)
    out = fp.predict(wrong.panel)
    assert isinstance(out, Unsupported)
    assert "units" in out.missing


def test_freeze_without_a_posterior_is_unsupported(carry: SurfaceWorld) -> None:
    missing = fit(carry.spec, carry.panel, backend="no-such-backend")
    out = freeze(missing)
    assert isinstance(out, Unsupported) and "posterior" in out.missing
