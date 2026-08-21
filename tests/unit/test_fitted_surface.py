"""``FitResult`` as a ``SupportsEstimands`` producer: delegation, capabilities, counterfactuals."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from axiom.core import (
    Capability,
    Intervention,
    Posterior,
    PredictiveDraws,
    SupportsEstimands,
    SupportsIntervention,
    SupportsPosterior,
    TimeWindow,
    Unsupported,
)
from axiom.sim import DosePlan, arms_world, surface_world
from axiom.sim.surface_world import SurfaceWorld
from axiom.surface import FitResult, fit
from axiom.surface.carryover import GeometricCarryover
from axiom.surface.forward import (
    counterfactual_doses,
    marginal,
    marginal_shift,
    marginal_total,
    predict,
    predict_marginal,
    support_indicator,
)
from axiom.surface.kernels import PowerKernel


@pytest.fixture(scope="module")
def world() -> SurfaceWorld:
    return surface_world(
        n_units=4,
        n_periods=16,
        treatments=("a", "b"),
        carryover={"a": GeometricCarryover(max_lag=3)},
        doses=DosePlan(scale=50.0, zero_fraction=0.1),
        noise_sd=0.3,
        seed=3,
    )


@pytest.fixture(scope="module")
def res(world: SurfaceWorld) -> FitResult:
    out = fit(world.spec, world.panel, backend="laplace", draws=60, chains=2, seed=5)
    assert isinstance(out.posterior, Posterior)
    return out


def _theta(res: FitResult, d: int) -> dict[str, np.ndarray]:
    post = res.posterior
    assert isinstance(post, Posterior)
    return {name: post.flat(name)[d] for name in post.names()}


# -- protocol conformance ------------------------------------------------------------------


def test_fit_result_satisfies_supports_estimands(world: SurfaceWorld, res: FitResult) -> None:
    assert isinstance(res, SupportsPosterior)
    assert isinstance(res, SupportsIntervention)
    assert isinstance(res, SupportsEstimands)
    post = res.posterior
    assert isinstance(post, Posterior)
    assert res.names() == post.names()
    assert res.n_draws() == post.n_draws() == 120
    assert res.coords() == post.coords()
    assert np.array_equal(res.draws("beta_a"), post.draws("beta_a"))
    assert res.treatments == world.spec.treatments
    assert res.outcome == world.spec.outcome
    assert res.outcome_unit is None and res.dose_unit("a") == "USD"
    with pytest.raises(KeyError):
        res.dose_unit("zzz")
    assert res.shape == (4, 16) and res.n_units == 4 and res.n_periods == 16
    assert res.unit_labels == world.spec.unit_labels == ("u0", "u1", "u2", "u3")
    assert res.periods == tuple(float(t) for t in range(16))
    assert res.provenance["unit_labels"] == list(world.panel.units)


def test_declared_estimands_are_appended_and_deduplicated(res: FitResult) -> None:
    assert res.declared_estimands == ()
    with_two = res.with_estimands("a" * 64, "b" * 64)
    assert with_two.declared_estimands == ("a" * 64, "b" * 64)
    assert with_two.with_estimands("a" * 64, "c" * 64).declared_estimands == (
        "a" * 64,
        "b" * 64,
        "c" * 64,
    )
    assert res.declared_estimands == ()  # the original is untouched
    with pytest.raises(ValueError, match="non-empty"):
        res.with_estimands("")


def test_delegation_raises_clearly_without_a_posterior(res: FitResult) -> None:
    failed = replace(res, posterior=Unsupported(reason="backend missing"))
    for call in (failed.names, failed.n_draws, failed.coords):
        with pytest.raises(ValueError, match="no posterior.*Unsupported.*backend missing"):
            call()
    with pytest.raises(ValueError, match="no posterior"):
        failed.draws("beta_a")
    assert failed.capabilities() == frozenset()  # so its methods degrade, typed
    out = failed.predict_under(Intervention(doses={"a": 1.0}))
    assert isinstance(out, Unsupported) and out.missing == ("counterfactual",)
    assert "no posterior" in out.reason and "backend missing" in out.reason


def test_unit_labels_fall_back_to_provenance_then_index(res: FitResult) -> None:
    spec = res.surface.spec.model_copy(update={"intercept": "shared", "unit_labels": ()})
    from axiom.surface.model import Surface

    anonymous = replace(res, surface=Surface(spec))
    assert anonymous.unit_labels == ("u0", "u1", "u2", "u3")  # recorded by fit
    bare = replace(anonymous, provenance={})
    assert bare.unit_labels == ("0", "1", "2", "3")


# -- capabilities --------------------------------------------------------------------------


def test_capabilities_and_restrict(res: FitResult) -> None:
    assert res.capabilities() == {
        Capability.COUNTERFACTUAL,
        Capability.MARGINAL,
        Capability.TIME_WINDOW,
        Capability.PER_UNIT,
        Capability.PREDICTIVE,
    }
    crippled = res.restrict([Capability.MARGINAL, "time_window"])
    assert Capability.MARGINAL not in crippled.capabilities()
    assert Capability.TIME_WINDOW not in crippled.capabilities()
    assert Capability.COUNTERFACTUAL in crippled.capabilities()
    assert res.capabilities() >= crippled.capabilities()  # the original keeps everything
    assert crippled.restrict([Capability.COUNTERFACTUAL]).capabilities() == {
        Capability.PER_UNIT,
        Capability.PREDICTIVE,
    }
    with pytest.raises(ValueError):
        res.restrict(["not_a_capability"])
    iv = Intervention(doses={"a": 10.0})
    out = crippled.marginal_under(iv, "a")
    assert isinstance(out, Unsupported) and out.missing == ("marginal",)
    assert "dropped" in out.reason
    out = crippled.predict_under(iv, window=TimeWindow(start=0, stop=4))
    assert isinstance(out, Unsupported) and out.missing == ("time_window",)
    assert isinstance(crippled.predict_under(iv), PredictiveDraws)  # no window: still fine
    out = crippled.restrict([Capability.COUNTERFACTUAL]).predict_under(iv)
    assert isinstance(out, Unsupported) and out.missing == ("counterfactual",)


def test_per_unit_capability_follows_intercept_or_unit_count() -> None:
    one = arms_world(n_units=1, treatments=("dose",), seed=0)
    single = fit(one.spec, one.panel, backend="laplace", draws=20, chains=1, seed=0)
    assert Capability.PER_UNIT not in single.capabilities()
    many = arms_world(n_units=6, treatments=("dose",), seed=0)
    shared = fit(many.spec, many.panel, backend="laplace", draws=20, chains=1, seed=0)
    assert Capability.PER_UNIT in shared.capabilities()


# -- counterfactual doses ------------------------------------------------------------------


def test_counterfactual_doses_modes_and_support(world: SurfaceWorld, res: FitResult) -> None:
    a = world.panel.array("a")
    out = counterfactual_doses(res.surface, res.data, Intervention(doses={"a": 7.0}))
    assert np.array_equal(out["a"], np.full(a.shape, 7.0))
    assert np.array_equal(out["b"], res.data["b"]) and out["unit"] is not res.data["a"]
    assert np.array_equal(res.data["a"], a)  # the fitted data is not mutated
    sup = TimeWindow(start=4, stop=8)
    out = counterfactual_doses(
        res.surface, res.data, Intervention(doses={"a": 1.5}, mode="scale", window=sup)
    )
    expect = a.copy()
    expect[:, 4:8] *= 1.5
    assert np.allclose(out["a"], expect)
    out = counterfactual_doses(
        res.surface, res.data, Intervention(doses={"a": 2.0, "b": 3.0}, mode="shift")
    )
    assert np.allclose(out["a"], a + 2.0) and np.allclose(out["b"], world.panel.array("b") + 3.0)
    with pytest.raises(ValueError, match="does not have.*zzz"):
        counterfactual_doses(res.surface, res.data, Intervention(doses={"zzz": 1.0}))
    with pytest.raises(ValueError, match="runs past"):
        counterfactual_doses(
            res.surface,
            res.data,
            Intervention(doses={"a": 1.0}, window=TimeWindow(start=10, stop=20)),
        )


def test_counterfactual_doses_refuse_negative_realized_doses(
    world: SurfaceWorld, res: FitResult
) -> None:
    """A shift below the observed dose, a negative scale, or a negative level is a ValueError
    naming the treatment and the mode — never a finite number from a kernel below zero."""
    a = world.panel.array("a")
    assert np.any(a < 2.0)  # the shift below realizes a negative dose somewhere
    with pytest.raises(ValueError, match="mode 'shift'.*treatment 'a'.*negative dose"):
        counterfactual_doses(res.surface, res.data, Intervention(doses={"a": -2.0}, mode="shift"))
    with pytest.raises(ValueError, match="mode 'scale'.*treatment 'b'"):
        counterfactual_doses(res.surface, res.data, Intervention(doses={"b": -1.0}, mode="scale"))
    with pytest.raises(ValueError, match="mode 'set' at level -5.0 on treatment 'a'"):
        counterfactual_doses(res.surface, res.data, Intervention(doses={"a": -5.0}))
    # a shift that stays non-negative on its support is fine even if it would not be elsewhere
    sup = TimeWindow(start=0, stop=16)
    floor = float(a.min())
    out = counterfactual_doses(
        res.surface, res.data, Intervention(doses={"a": -floor}, mode="shift", window=sup)
    )
    assert np.all(out["a"] >= 0.0)
    # and the producer's entry points refuse too, rather than returning a number
    with pytest.raises(ValueError, match="negative dose"):
        res.predict_under(Intervention(doses={"a": -2.0}, mode="shift"))
    with pytest.raises(ValueError, match="negative dose"):
        res.marginal_under(Intervention(doses={"a": -1.0}), "a")
    assert np.array_equal(res.data["a"], a)  # nothing was mutated along the way


# -- predict_under -------------------------------------------------------------------------


def test_predict_under_is_forward_per_draw(world: SurfaceWorld, res: FitResult) -> None:
    iv = Intervention(doses={"a": 10.0}, version="v2")
    out = res.predict_under(iv, seed=1)
    assert isinstance(out, PredictiveDraws)
    assert out.values.shape == (2, 60, 4, 16)
    assert out.intervention == iv and out.window is None and out.seed == 1
    assert list(out.coords["unit"]) == list(world.spec.unit_labels)
    assert list(out.coords["period"]) == [float(t) for t in range(16)]
    assert np.allclose(np.asarray(out.coords["a"]).reshape(4, 16), 10.0)
    assert np.allclose(np.asarray(out.coords["b"]).reshape(4, 16), world.panel.array("b"))
    dose = dict(res.data)
    dose["a"] = np.full((4, 16), 10.0)
    flat = out.values.reshape(120, 4, 16)
    for d in (0, 37, 119):
        assert np.allclose(flat[d], res.surface.forward(dose, _theta(res, d)), atol=1e-12)
    # and it is exactly what forward.predict says on the same grid
    post = res.posterior
    assert isinstance(post, Posterior)
    assert np.array_equal(out.values, predict(res.surface, post, dose).values)


def test_predict_under_window_selects_after_the_full_evaluation(res: FitResult) -> None:
    iv = Intervention(doses={"a": 1.2}, mode="scale")
    full = res.predict_under(iv)
    part = res.predict_under(iv, window=TimeWindow(start=2, stop=6))
    assert isinstance(full, PredictiveDraws) and isinstance(part, PredictiveDraws)
    assert part.values.shape == (2, 60, 4, 4)
    assert np.array_equal(part.values, full.values[..., 2:6])  # carryover saw every period
    assert list(part.coords["period"]) == [2.0, 3.0, 4.0, 5.0]
    assert part.window == TimeWindow(start=2, stop=6)
    realized = np.asarray(part.coords["a"]).reshape(4, 4)
    assert np.allclose(realized, 1.2 * res.data["a"][:, 2:6])
    with pytest.raises(ValueError, match="runs past"):
        res.predict_under(iv, window=TimeWindow(start=0, stop=17))
    with pytest.raises(ValueError, match="does not have"):
        res.predict_under(Intervention(doses={"c": 1.0}))


# -- marginal_under ------------------------------------------------------------------------


def test_marginal_under_is_the_shift_marginal_per_draw(res: FitResult) -> None:
    iv = Intervention(doses={"a": 30.0})
    out = res.marginal_under(iv, "a", seed=2)
    assert isinstance(out, PredictiveDraws)
    assert out.values.shape == (2, 60, 4, 16) and out.intervention == iv and out.seed == 2
    dose = dict(res.data)
    dose["a"] = np.full((4, 16), 30.0)
    flat = out.values.reshape(120, 4, 16)
    for d in (0, 58, 119):
        theta = _theta(res, d)
        assert np.allclose(flat[d], marginal_shift(res.surface, theta, dose, "a"), atol=1e-12)
        # carryover on "a": neither the same-period nor the total-attribution marginal is it
        assert not np.allclose(flat[d], marginal(res.surface, theta, dose, "a"))
        assert not np.allclose(flat[d], marginal_total(res.surface, theta, dose, "a"))
        # but once every lag is inside the horizon all three agree (weights sum to one)
        assert np.allclose(flat[d][:, 3:-3], marginal_total(res.surface, theta, dose, "a")[:, 3:-3])
    # "b" has no carryover: shift == period == total
    out_b = res.marginal_under(iv, "b", window=TimeWindow(start=5, stop=9))
    assert isinstance(out_b, PredictiveDraws) and out_b.values.shape == (2, 60, 4, 4)
    theta = _theta(res, 3)
    assert np.allclose(
        out_b.values.reshape(120, 4, 4)[3], marginal(res.surface, theta, dose, "b")[:, 5:9]
    )
    with pytest.raises(ValueError, match="no treatment"):
        res.marginal_under(iv, "zzz")


def test_marginal_under_sums_to_the_derivative_of_the_windowed_outcome(res: FitResult) -> None:
    """``Σ_{t∈W} marginal_under`` is ``d/dδ Σ_{t∈W} predict_under`` under a common shift of
    the support's doses, on every window, including ones the support only spills into."""
    eps = 1e-5
    sup = TimeWindow(start=4, stop=8)
    base = Intervention(doses={"a": 1.0}, mode="shift", window=sup)
    up = Intervention(doses={"a": 1.0 + eps}, mode="shift", window=sup)
    m = res.marginal_under(base, "a")
    y0 = res.predict_under(base)
    y1 = res.predict_under(up)
    assert isinstance(m, PredictiveDraws)
    assert isinstance(y0, PredictiveDraws) and isinstance(y1, PredictiveDraws)
    # the support indicator is what the derivative is taken against
    assert np.array_equal(support_indicator(16, sup), np.r_[np.zeros(4), np.ones(4), np.zeros(8)])
    assert np.allclose(m.values[..., :4], 0.0)  # nothing reaches the periods before the support
    assert np.any(m.values[..., 8:11] != 0.0)  # carryover spills past it
    for start, stop in ((0, 16), (0, 4), (4, 8), (8, 12), (2, 10)):
        fd = (y1.values - y0.values)[..., start:stop].sum(axis=-1) / eps
        assert np.allclose(m.values[..., start:stop].sum(axis=-1), fd, rtol=1e-4, atol=1e-8), (
            start,
            stop,
        )
    # the reporting window selects after the whole horizon is evaluated
    part = res.marginal_under(base, "a", window=TimeWindow(start=6, stop=10))
    assert isinstance(part, PredictiveDraws)
    assert np.array_equal(part.values, m.values[..., 6:10])


def test_marginal_under_is_unsupported_where_the_slope_is_infinite() -> None:
    w = arms_world(n_units=4, doses={"dose": [0.0, 1.0, 2.0, 3.0]}, kernels=PowerKernel(), seed=0)
    res = fit(w.spec, w.panel, backend="laplace", draws=10, chains=1, seed=0)
    assert isinstance(res.posterior, Posterior)
    out = res.marginal_under(Intervention(doses={"dose": 0.0}), "dose")
    assert isinstance(out, Unsupported) and "not finite" in out.reason
    ok = res.marginal_under(Intervention(doses={"dose": 2.0}), "dose")
    assert isinstance(ok, PredictiveDraws) and np.all(np.isfinite(ok.values))


def test_predict_marginal_helper_horizons(res: FitResult) -> None:
    post = res.posterior
    assert isinstance(post, Posterior)
    period = predict_marginal(res.surface, post, res.data, "a", horizon="period")
    total = predict_marginal(res.surface, post, res.data, "a", horizon="total")
    assert period.values.shape == total.values.shape == (2, 60, 4, 16)
    theta = _theta(res, 7)
    assert np.allclose(
        period.values.reshape(120, 4, 16)[7], marginal(res.surface, theta, res.data, "a")
    )
    assert np.allclose(
        total.values.reshape(120, 4, 16)[7], marginal_total(res.surface, theta, res.data, "a")
    )
    assert np.allclose(total.values[..., -1], period.values[..., -1])  # no future at the end
    shift = predict_marginal(res.surface, post, res.data, "a", horizon="shift")
    assert np.allclose(
        shift.values.reshape(120, 4, 16)[7], marginal_shift(res.surface, theta, res.data, "a")
    )
    assert np.allclose(shift.values[..., 0], period.values[..., 0])  # only w_0 reaches t = 0
    sup = TimeWindow(start=2, stop=5)
    reached = predict_marginal(res.surface, post, res.data, "a", horizon="shift", support=sup)
    assert np.allclose(
        reached.values.reshape(120, 4, 16)[7],
        marginal_shift(res.surface, theta, res.data, "a", sup),
    )
    assert np.allclose(reached.values[..., :2], 0.0)
    with pytest.raises(ValueError, match="runs past"):
        support_indicator(16, TimeWindow(start=10, stop=20))
    with pytest.raises(KeyError):
        predict_marginal(res.surface, post, res.data, "zzz")
