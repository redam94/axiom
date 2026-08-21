from __future__ import annotations

import numpy as np
import pytest

from axiom.core import (
    Convolve,
    D,
    Data,
    Opaque,
    Spec,
    dimension,
    dimensionless,
    load_spec,
    params,
    value,
    walk,
)
from axiom.surface.carryover import (
    CARRYOVERS,
    CarryoverKernel,
    DelayedCarryover,
    GeometricCarryover,
    NoCarryover,
    WeibullCarryover,
    carryover_from_name,
)

DOSE = Data(name="x", dimension=D.currency)
T = "tv"
L = 6

THETA: dict[str, dict[str, float]] = {
    "geometric": {f"lam_{T}": 0.6},
    "delayed": {f"lam_{T}": 0.5, f"theta_{T}": 2.0},
    "weibull": {f"lam_{T}": 2.5, f"kappa_{T}": 1.7},
    "none": {},
}


def _kernel(name: str) -> CarryoverKernel:
    return carryover_from_name(name) if name == "none" else carryover_from_name(name, max_lag=L)


def _raw_closed_form(name: str, lags: np.ndarray) -> np.ndarray:
    v = THETA[name]
    match name:
        case "geometric":
            return np.asarray(v[f"lam_{T}"] ** lags)
        case "delayed":
            return np.asarray(v[f"lam_{T}"] ** ((lags - v[f"theta_{T}"]) ** 2))
        case "weibull":
            return np.asarray(np.exp(-((lags / v[f"lam_{T}"]) ** v[f"kappa_{T}"])))
        case "none":
            return np.asarray((lags == 0).astype(float))
    raise AssertionError(name)


@pytest.fixture(params=sorted(CARRYOVERS), ids=sorted(CARRYOVERS))
def name(request: pytest.FixtureRequest) -> str:
    return str(request.param)


def test_registry_is_complete_and_satisfies_the_protocol(name: str) -> None:
    kernel = _kernel(name)
    assert isinstance(kernel, Spec)
    assert isinstance(kernel, CarryoverKernel)
    assert kernel.name == name
    assert type(kernel).__mro__[1] is Spec
    assert set(CARRYOVERS.values()) == {
        GeometricCarryover,
        DelayedCarryover,
        WeibullCarryover,
        NoCarryover,
    }
    with pytest.raises(ValueError, match="unknown carryover family"):
        carryover_from_name("geo_metric")


def test_parameters_are_dimensionless_shapes_with_priors(name: str) -> None:
    kernel = _kernel(name)
    ps = kernel.parameters(T)
    assert [p.name for p in ps] == [f"{role}_{T}" for role in kernel.roles]
    assert all(kernel.roles[r] == "shape" for r in kernel.roles)
    for p in ps:
        assert p.is_shape
        assert p.prior is not None
    found = {p.name for p in params(kernel.weights(T))}
    assert found == {p.name for p in ps}


def test_weights_sum_to_one_and_match_closed_form(name: str) -> None:
    kernel = _kernel(name)
    w = value(kernel.weights(T), params=THETA[name])
    assert w.shape == (kernel.max_lag,)
    assert w.sum() == pytest.approx(1.0, abs=1e-12)
    raw = _raw_closed_form(name, np.arange(kernel.max_lag, dtype=float))
    np.testing.assert_allclose(w, raw / raw.sum(), rtol=1e-12)


def test_weights_use_no_opaque_node(name: str) -> None:
    assert not any(isinstance(n, Opaque) for _, n in walk(_kernel(name).weights(T)))


def test_apply_keeps_the_dose_dimension(name: str) -> None:
    kernel = _kernel(name)
    applied = kernel.apply(DOSE, T)
    assert dimension(applied) == D.currency
    assert dimension(kernel.weights(T)) == dimensionless()
    if name == "none":
        assert applied == DOSE
    else:
        assert isinstance(applied, Convolve)


def test_impulse_response_reproduces_the_weights(name: str) -> None:
    kernel = _kernel(name)
    n = 10
    impulse = np.zeros(n)
    impulse[0] = 1.0
    out = value(kernel.apply(DOSE, T), data={"x": impulse}, params=THETA[name])
    w = value(kernel.weights(T), params=THETA[name])
    expected = np.zeros(n)
    expected[: kernel.max_lag] = w
    np.testing.assert_allclose(out, expected, rtol=1e-12, atol=1e-15)


def test_apply_is_causal_and_linear(name: str) -> None:
    kernel = _kernel(name)
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 10, size=12)
    # shifting the input by one period shifts the output by one period
    out = value(kernel.apply(DOSE, T), data={"x": x}, params=THETA[name])
    shifted = value(
        kernel.apply(DOSE, T), data={"x": np.concatenate([[0.0], x[:-1]])}, params=THETA[name]
    )
    np.testing.assert_allclose(shifted[1:], out[:-1], rtol=1e-12)
    # linear: doubling the dose doubles the carried dose
    twice = value(kernel.apply(DOSE, T), data={"x": 2 * x}, params=THETA[name])
    np.testing.assert_allclose(twice, 2 * out, rtol=1e-12)


def test_half_life_halves_the_unnormalized_weight(name: str) -> None:
    kernel = _kernel(name)
    hl = kernel.half_life(THETA[name], T)
    assert hl.shape == ()
    if name == "none":
        assert hl == 0.0
        return
    peak = 0.0 if name != "delayed" else THETA["delayed"][f"theta_{T}"]
    at = _raw_closed_form(name, np.array([peak, peak + float(hl)]))
    assert at[1] / at[0] == pytest.approx(0.5, rel=1e-12)


def test_half_life_vectorizes_over_draws() -> None:
    lam = np.array([0.25, 0.5, 0.8])
    hl = GeometricCarryover(max_lag=4).half_life({f"lam_{T}": lam}, T)
    np.testing.assert_allclose(hl, np.log(0.5) / np.log(lam))
    assert hl[1] == pytest.approx(1.0)
    with pytest.raises(KeyError, match="missing"):
        GeometricCarryover(max_lag=4).half_life({}, T)


def test_half_life_is_infinite_at_no_decay_without_warnings() -> None:
    """λ = 1 is no decay: the half-life is +inf, not -inf, nan, or a warning."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        geo = GeometricCarryover(max_lag=3).half_life({f"lam_{T}": 1.0}, T)
        assert geo == np.inf
        delayed = DelayedCarryover(max_lag=3).half_life({f"lam_{T}": 1.0}, T)
        assert delayed == np.inf
        mixed = GeometricCarryover(max_lag=3).half_life({f"lam_{T}": np.array([0.5, 1.0])}, T)
        np.testing.assert_array_equal(mixed, [1.0, np.inf])
        mixed_d = DelayedCarryover(max_lag=3).half_life({f"lam_{T}": np.array([[0.25], [1.0]])}, T)
        np.testing.assert_allclose(mixed_d, [[np.sqrt(0.5)], [np.inf]])


@pytest.mark.parametrize("lam", [1.5, 0.0, -0.2, np.nan, np.inf])
def test_half_life_rejects_decay_outside_the_unit_interval(lam: float) -> None:
    for cls in (GeometricCarryover, DelayedCarryover):
        with pytest.raises(ValueError, match="lam_tv"):
            cls(max_lag=3).half_life({f"lam_{T}": lam}, T)
    with pytest.raises(ValueError, match="lam_tv"):
        GeometricCarryover(max_lag=3).half_life({f"lam_{T}": np.array([0.5, lam])}, T)


def test_weibull_half_life_rejects_nonpositive_scale_or_shape() -> None:
    import warnings

    wb = WeibullCarryover(max_lag=3)
    with pytest.raises(ValueError, match="kappa_tv"):
        wb.half_life({f"lam_{T}": 1.0, f"kappa_{T}": 0.0}, T)
    with pytest.raises(ValueError, match="kappa_tv"):
        wb.half_life({f"lam_{T}": 1.0, f"kappa_{T}": -1.0}, T)
    with pytest.raises(ValueError, match="lam_tv"):
        wb.half_life({f"lam_{T}": 0.0, f"kappa_{T}": 1.0}, T)
    with pytest.raises(ValueError, match="lam_tv"):
        wb.half_life({f"lam_{T}": np.nan, f"kappa_{T}": 1.0}, T)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert wb.half_life({f"lam_{T}": 2.0, f"kappa_{T}": 1.0}, T) == pytest.approx(
            2.0 * np.log(2.0)
        )


def test_delayed_needs_room_for_a_delay() -> None:
    with pytest.raises(ValueError):
        DelayedCarryover(max_lag=1)
    lam, theta = DelayedCarryover(max_lag=5).parameters(T)
    assert theta.prior is not None
    assert theta.prior.hyper == {"low": 0.0, "high": 4.0}
    w = value(DelayedCarryover(max_lag=5).weights(T), params={f"lam_{T}": 0.5, f"theta_{T}": 2.0})
    assert int(np.argmax(w)) == 2


def test_geometric_with_one_lag_is_the_identity() -> None:
    w = value(GeometricCarryover(max_lag=1).weights(T), params={f"lam_{T}": 0.3})
    np.testing.assert_allclose(w, [1.0])
    with pytest.raises(ValueError):
        GeometricCarryover(max_lag=0)


def test_weibull_kappa_one_is_geometric() -> None:
    lam = 2.0
    w = value(WeibullCarryover(max_lag=L).weights(T), params={f"lam_{T}": lam, f"kappa_{T}": 1.0})
    g = value(GeometricCarryover(max_lag=L).weights(T), params={f"lam_{T}": np.exp(-1 / lam)})
    np.testing.assert_allclose(w, g, rtol=1e-12)


def test_json_round_trip(name: str) -> None:
    kernel = _kernel(name)
    assert isinstance(kernel, Spec)
    back = load_spec(kernel.to_json())
    assert back == kernel
    assert type(back) is type(kernel)
    assert back.content_hash() == kernel.content_hash()
    applied = kernel.apply(DOSE, T)
    assert load_spec(applied.to_json()) == applied


def test_jax_agrees_with_numpy(name: str) -> None:
    pytest.importorskip("jax")
    from axiom.core import compile_jax

    kernel = _kernel(name)
    expected = value(kernel.weights(T), params=THETA[name])
    got = np.asarray(compile_jax(kernel.weights(T))({}, THETA[name]), dtype=float)
    np.testing.assert_allclose(got, expected, rtol=1e-5)
    x = np.linspace(0.0, 3.0, 9)
    expected_applied = value(kernel.apply(DOSE, T), data={"x": x}, params=THETA[name])
    got_applied = np.asarray(compile_jax(kernel.apply(DOSE, T))({"x": x}, THETA[name]), dtype=float)
    np.testing.assert_allclose(got_applied, expected_applied, rtol=1e-5, atol=1e-6)


# -- draw-shaped parameters (D1) -----------------------------------------------------------


def test_geometric_weights_normalize_per_draw() -> None:
    """Finding 1: (n, 1) decay rates give (n, L) weights, each row summing to one."""
    w = value(
        GeometricCarryover(max_lag=3).weights(T),
        params={f"lam_{T}": np.array([[0.2], [0.5], [0.9]])},
    )
    np.testing.assert_allclose(
        w,
        [[0.806, 0.161, 0.032], [0.571, 0.286, 0.143], [0.369, 0.332, 0.299]],
        atol=5e-4,
    )
    np.testing.assert_allclose(w.sum(axis=-1), 1.0, atol=1e-12)


def _batched_theta(name: str, n: int) -> dict[str, np.ndarray]:
    factors = np.linspace(0.7, 1.0, n)[:, None]  # keeps λ inside (0, 1) for the decays
    return {k: v * factors for k, v in THETA[name].items()}


def test_draw_shaped_weights_and_apply_match_per_draw_evaluation(name: str) -> None:
    kernel = _kernel(name)
    n = 4
    batched = _batched_theta(name, n)
    rng = np.random.default_rng(3)
    x = rng.uniform(0.0, 5.0, size=11)
    w = value(kernel.weights(T), params=batched)
    out = value(kernel.apply(DOSE, T), data={"x": x}, params=batched)
    if name == "none":
        assert w.shape == (1,)
        np.testing.assert_allclose(out, x)
        return
    assert w.shape == (n, kernel.max_lag)
    assert out.shape == (n, x.size)
    np.testing.assert_allclose(w.sum(axis=-1), 1.0, atol=1e-12)
    for i in range(n):
        row = {k: float(v[i, 0]) for k, v in batched.items()}
        np.testing.assert_allclose(w[i], value(kernel.weights(T), params=row), rtol=1e-12)
        np.testing.assert_allclose(
            out[i], value(kernel.apply(DOSE, T), data={"x": x}, params=row), rtol=1e-12
        )


def test_draw_shaped_apply_over_a_unit_by_time_signal() -> None:
    """A (draws, 1, L) kernel against a (units, T) signal gives (draws, units, T)."""
    kernel = GeometricCarryover(max_lag=3)
    lam = np.array([0.3, 0.6, 0.9])[:, None, None]
    x = np.arange(10.0).reshape(2, 5)
    out = value(kernel.apply(DOSE, T), data={"x": x}, params={f"lam_{T}": lam})
    assert out.shape == (3, 2, 5)
    for d in range(3):
        for u in range(2):
            single = value(
                kernel.apply(DOSE, T), data={"x": x[u]}, params={f"lam_{T}": float(lam[d, 0, 0])}
            )
            np.testing.assert_allclose(out[d, u], single, rtol=1e-12)


def test_jax_agrees_with_numpy_for_draw_shaped_apply(name: str) -> None:
    pytest.importorskip("jax")
    import jax

    from axiom.core import compile_jax

    jax.config.update("jax_enable_x64", True)
    kernel = _kernel(name)
    batched = _batched_theta(name, 3)
    x = np.linspace(0.0, 3.0, 9)
    expected = value(kernel.apply(DOSE, T), data={"x": x}, params=batched)
    got = np.asarray(compile_jax(kernel.apply(DOSE, T))({"x": x}, batched), dtype=float)
    assert got.shape == expected.shape
    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-12)


# -- Weibull lag-zero gradients (finding 4) ------------------------------------------------


@pytest.mark.parametrize(
    ("lam", "kappa"), [(0.5, 0.5), (2.0, 0.7), (1.0, 1.0), (3.0, 2.5), (0.1, 0.3)]
)
def test_weibull_gradients_are_finite_at_lag_zero(lam: float, kappa: float) -> None:
    """``l^κ / λ^κ`` keeps d/dλ and d/dκ of the lag-zero weight finite for every κ > 0."""
    pytest.importorskip("jax")
    import jax

    from axiom.core import compile_jax

    jax.config.update("jax_enable_x64", True)
    f = compile_jax(WeibullCarryover(max_lag=4).weights(T))
    jac = jax.jacfwd(lambda th: f({}, th))({f"lam_{T}": lam, f"kappa_{T}": kappa})
    for pname, g in jac.items():
        g = np.asarray(g)
        assert g.shape == (4,)
        assert np.all(np.isfinite(g)), (pname, g)
    # and the weights themselves are finite and normalized
    w = np.asarray(f({}, {f"lam_{T}": lam, f"kappa_{T}": kappa}))
    assert np.all(np.isfinite(w))
    assert w.sum() == pytest.approx(1.0)
