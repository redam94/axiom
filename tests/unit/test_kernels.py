from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from axiom.core import (
    Add,
    D,
    Data,
    Likelihood,
    ModelSpec,
    Param,
    Prior,
    Spec,
    dimension,
    dimensionless,
    jax_available,
    load_spec,
    log_density,
    params,
    unconstrain,
    value,
)
from axiom.surface.kernels import (
    KERNELS,
    ExponentialKernel,
    HillKernel,
    LinearKernel,
    LogisticKernel,
    PowerKernel,
    ResponseKernel,
    kernel_from_name,
)

DOSE = Data(name="x", dimension=D.currency)
T = "tv"
GRID = np.linspace(0.1, 5.0, 25)
jax_only = pytest.mark.skipif(not jax_available(), reason="jax not installed")

# parameter values per family, keyed by role (the treatment suffix is added below)
VALUES: dict[str, dict[str, float]] = {
    "hill": {"k": 2.0, "s": 1.5, "beta": 3.0},
    "logistic": {"k": 2.0, "beta": 3.0},
    "exponential": {"k": 2.0, "beta": 3.0},
    "power": {"k": 2.0, "s": 0.7, "beta": 3.0},
    "linear": {"beta_rate": 0.7},
}


def _theta(name: str, scale: float = 1.0) -> dict[str, float]:
    """Parameter values for ``name`` with every scale-role parameter multiplied by ``scale``."""
    kernel = kernel_from_name(name)
    out: dict[str, float] = {}
    for role, v in VALUES[name].items():
        if kernel.roles[role] == "scale":
            v = v * scale
        out[f"{role}_{T}"] = v
    return out


def _closed_form(name: str, x: np.ndarray) -> np.ndarray:
    v = VALUES[name]
    match name:
        case "hill":
            u = (x / v["k"]) ** v["s"]
            return np.asarray(v["beta"] * u / (1 + u))
        case "logistic":
            return np.asarray(v["beta"] * (2 / (1 + np.exp(-x / v["k"])) - 1))
        case "exponential":
            return np.asarray(v["beta"] * (1 - np.exp(-x / v["k"])))
        case "power":
            return np.asarray(v["beta"] * (x / v["k"]) ** v["s"])
        case "linear":
            return np.asarray(v["beta_rate"] * x)
    raise AssertionError(name)


@pytest.fixture(params=sorted(KERNELS), ids=sorted(KERNELS))
def name(request: pytest.FixtureRequest) -> str:
    return str(request.param)


def test_registry_is_complete_and_satisfies_the_protocol(name: str) -> None:
    cls = KERNELS[name]
    kernel = cls()
    assert isinstance(kernel, Spec)
    assert isinstance(kernel, ResponseKernel)
    assert kernel.name == name
    assert type(kernel).__mro__[1] is Spec  # flat: no base class between it and Spec
    built = kernel_from_name(name, reference_dose=3.0)
    assert isinstance(built, Spec)
    assert built == cls.model_validate({"reference_dose": 3.0})


def test_unknown_family_is_a_value_error() -> None:
    with pytest.raises(ValueError, match="unknown kernel family"):
        kernel_from_name("hil")


def test_every_family_is_in_the_registry() -> None:
    assert set(KERNELS.values()) == {
        HillKernel,
        LogisticKernel,
        ExponentialKernel,
        PowerKernel,
        LinearKernel,
    }


def test_parameter_names_dimensions_and_roles(name: str) -> None:
    kernel = kernel_from_name(name)
    ps = kernel.parameters(T, D.currency, D.outcome)
    assert [p.name for p in ps] == [f"{role}_{T}" for role in kernel.roles]
    for p in ps:
        role = kernel.roles[p.name.removesuffix(f"_{T}")]
        assert p.prior is not None
        if role == "scale":
            assert p.dimension == D.currency
            assert p.prior.family == "lognormal"
        elif role == "shape":
            assert p.dimension == dimensionless()
            assert p.is_shape
            # power is shipped for diminishing returns: its shape lives on (0, 1)
            assert p.prior.family == ("beta" if name == "power" else "gamma")
        else:
            assert p.prior.family == "halfnormal"
            assert p.dimension in (D.outcome, D.outcome / D.currency)


def test_reference_dose_centres_the_scale_prior() -> None:
    k, _, _ = HillKernel(reference_dose=50.0).parameters(T, D.currency, D.outcome)
    assert k.prior is not None
    assert k.prior.hyper == {"mu": pytest.approx(np.log(50.0)), "sigma": 1.0}
    beta = HillKernel(amplitude_scale=4.0).parameters(T, D.currency, D.outcome)[2]
    assert beta.prior is not None
    assert beta.prior.hyper == {"sigma": 4.0}
    with pytest.raises(ValueError):
        HillKernel(reference_dose=0.0)


def test_dimensions(name: str) -> None:
    kernel = kernel_from_name(name)
    assert dimension(kernel.response(DOSE, T)) == D.outcome
    assert dimension(kernel.saturation(DOSE, T)) == dimensionless()
    assert dimension(kernel.derivative(DOSE, T)) == D.outcome / D.currency
    # a different outcome dimension threads through
    assert dimension(kernel.response(DOSE, T, D.entity)) == D.entity
    assert dimension(kernel.derivative(DOSE, T, D.entity)) == D.entity / D.currency


def test_expression_params_match_declared_parameters(name: str) -> None:
    kernel = kernel_from_name(name)
    declared = kernel.parameters(T, D.currency, D.outcome)
    for expr in (kernel.response(DOSE, T), kernel.derivative(DOSE, T)):
        found = {p.name: p for p in params(expr)}
        assert set(found) == {p.name for p in declared}
        for p in declared:
            assert found[p.name] == p


def test_value_matches_closed_form(name: str) -> None:
    kernel = kernel_from_name(name)
    got = value(kernel.response(DOSE, T), data={"x": GRID}, params=_theta(name))
    np.testing.assert_allclose(got, _closed_form(name, GRID), rtol=1e-12, atol=1e-12)


def test_derivative_matches_central_differences(name: str) -> None:
    kernel = kernel_from_name(name)
    theta = _theta(name)
    h = 1e-5
    up = value(kernel.response(DOSE, T), data={"x": GRID + h}, params=theta)
    down = value(kernel.response(DOSE, T), data={"x": GRID - h}, params=theta)
    fd = (up - down) / (2 * h)
    got = value(kernel.derivative(DOSE, T), data={"x": GRID}, params=theta)
    np.testing.assert_allclose(np.broadcast_to(got, fd.shape), fd, atol=1e-6)


def test_zero_dose_gives_zero_response(name: str) -> None:
    kernel = kernel_from_name(name)
    got = value(kernel.response(DOSE, T), data={"x": np.array([0.0])}, params=_theta(name))
    assert got == pytest.approx(0.0)


def test_saturating_families_are_bounded_by_beta() -> None:
    for name in ("hill", "logistic", "exponential"):
        kernel = kernel_from_name(name)
        assert kernel.saturating
        far = value(kernel.saturation(DOSE, T), data={"x": np.array([1e9])}, params=_theta(name))
        assert far == pytest.approx(1.0, abs=1e-6)
    for name in ("power", "linear"):
        kernel = kernel_from_name(name)
        assert not kernel.saturating
        far = value(kernel.response(DOSE, T), data={"x": np.array([1e9])}, params=_theta(name))
        assert far > 1e3


def test_monotone_increasing(name: str) -> None:
    kernel = kernel_from_name(name)
    got = value(kernel.response(DOSE, T), data={"x": GRID}, params=_theta(name))
    assert np.all(np.diff(got) > 0)


def test_unit_invariance_of_shapes(name: str) -> None:
    """Exit criterion 6: rescale dose and scale together, shapes untouched, response unchanged."""
    kernel = kernel_from_name(name)
    factor = 1000.0  # e.g. currency in thousands
    expr = kernel.response(DOSE, T)
    base = value(expr, data={"x": GRID}, params=_theta(name))
    if name == "linear":
        theta = {f"beta_rate_{T}": VALUES["linear"]["beta_rate"] / factor}
    else:
        theta = _theta(name, scale=factor)
    rescaled = value(expr, data={"x": GRID * factor}, params=theta)
    np.testing.assert_allclose(rescaled, base, rtol=1e-10)


def test_derivative_scales_with_the_dose_unit() -> None:
    """d response / d dose in thousands is 1000 times the per-unit derivative."""
    kernel = HillKernel()
    factor = 1000.0
    d1 = value(kernel.derivative(DOSE, T), data={"x": GRID}, params=_theta("hill"))
    d2 = value(kernel.derivative(DOSE, T), data={"x": GRID * factor}, params=_theta("hill", factor))
    np.testing.assert_allclose(d2 * factor, d1, rtol=1e-10)


def test_json_round_trip(name: str) -> None:
    kernel = KERNELS[name].model_validate({"reference_dose": 12.5, "amplitude_scale": 0.3})
    back = load_spec(kernel.to_json())
    assert back == kernel
    assert type(back) is KERNELS[name]
    assert back.content_hash() == kernel.content_hash()
    assert KERNELS[name].from_json(kernel.to_json()) == kernel


def test_response_expression_round_trips(name: str) -> None:
    from axiom.core import Mul

    expr = kernel_from_name(name).response(DOSE, T)
    assert isinstance(expr, Mul)
    assert load_spec(expr.to_json()) == expr


def test_linear_kernel_rate_carries_outcome_per_dose() -> None:
    (rate,) = LinearKernel(reference_dose=10.0, amplitude_scale=5.0).parameters(
        T, D.currency, D.outcome
    )
    assert rate.name == f"beta_rate_{T}"
    assert rate.dimension == D.outcome / D.currency
    assert rate.prior is not None
    assert rate.prior.hyper == {"sigma": 0.5}
    sat = LinearKernel(reference_dose=10.0).saturation(DOSE, T)
    assert value(sat, data={"x": np.array([5.0])}) == pytest.approx(0.5)


# -- draw-shaped parameters ----------------------------------------------------------------


def _batched(theta: dict[str, float], n: int) -> dict[str, np.ndarray]:
    """Each parameter as an ``(n, 1)`` column of distinct draws (scaled 0.8 … 1.2 of ``theta``)."""
    factors = np.linspace(0.8, 1.2, n)[:, None]
    return {k: v * factors for k, v in theta.items()}


def test_draw_shaped_parameters_broadcast_to_draws_by_dose(name: str) -> None:
    """Parameters of shape (n, 1) against a dose grid give (n, T), row i being draw i."""
    kernel = kernel_from_name(name)
    n = 3
    batched = _batched(_theta(name), n)
    resp = value(kernel.response(DOSE, T), data={"x": GRID}, params=batched)
    deriv = value(kernel.derivative(DOSE, T), data={"x": GRID}, params=batched)
    assert resp.shape == (n, GRID.size)
    for i in range(n):
        row = {k: float(v[i, 0]) for k, v in batched.items()}
        np.testing.assert_allclose(
            resp[i], value(kernel.response(DOSE, T), data={"x": GRID}, params=row), rtol=1e-12
        )
        np.testing.assert_allclose(
            np.broadcast_to(deriv, (n, GRID.size))[i],
            np.broadcast_to(
                value(kernel.derivative(DOSE, T), data={"x": GRID}, params=row), GRID.shape
            ),
            rtol=1e-12,
        )


# -- jax gradients at zero dose (D2) -----------------------------------------------------


def _normal_model(kernel: ResponseKernel, name: str) -> ModelSpec:
    sigma = Param(
        name="sigma",
        dimension=D.outcome,
        prior=Prior(family="halfnormal", hyper={"sigma": 1.0}),
    )
    return ModelSpec(
        name=f"{name}-normal",
        mean=kernel.response(DOSE, T),
        outcome=Data(name="y", dimension=D.outcome),
        likelihood=Likelihood(family="normal", scale="sigma"),
        parameters=(*kernel.parameters(T, D.currency, D.outcome), sigma),
    )


def _central_differences(
    f: Callable[[dict[str, np.ndarray]], float], z: dict[str, np.ndarray], h: float = 1e-5
) -> dict[str, float]:
    out: dict[str, float] = {}
    for k in z:
        up = {kk: (vv + h if kk == k else vv) for kk, vv in z.items()}
        down = {kk: (vv - h if kk == k else vv) for kk, vv in z.items()}
        out[k] = (f(up) - f(down)) / (2 * h)
    return out


@jax_only
@pytest.mark.parametrize("name", ["hill", "power"])
def test_jax_gradient_is_finite_at_zero_dose_for_s_below_one(name: str) -> None:
    """D2: with s = 0.7 and zero-dose rows, every gradient is finite and matches finite differences.

    Written as ``(dose / k)^s`` the gradient in ``k`` is ``∞ · 0 = nan`` on the
    zero-dose rows; the scaled form ``x̃^s / (k̃^s + x̃^s)`` keeps it finite.
    """
    import jax

    from axiom.core import compile_log_density

    jax.config.update("jax_enable_x64", True)
    kernel = kernel_from_name(name)
    model = _normal_model(kernel, name)
    x = np.array([0.0, 0.0, 1.0, 2.0, 5.0])
    theta = {f"k_{T}": 1.5, f"s_{T}": 0.7, f"beta_{T}": 2.0, "sigma": 0.8}
    y = value(kernel.response(DOSE, T), data={"x": x}, params=theta) + np.array(
        [0.1, -0.2, 0.05, -0.1, 0.3]
    )
    data = {"x": x, "y": y}
    z = {k: np.asarray(v) for k, v in unconstrain(model, theta).items()}
    f = compile_log_density(model)
    assert float(f(data, z)) == pytest.approx(log_density(model, data, z), abs=1e-8)
    g = jax.grad(lambda zz: f(data, zz))(z)
    fd = _central_differences(lambda zz: log_density(model, data, zz), z)
    for k in z:
        assert np.isfinite(float(g[k])), k
        assert float(g[k]) == pytest.approx(fd[k], rel=1e-6, abs=1e-6), k


def test_power_default_shape_prior_lives_on_the_unit_interval() -> None:
    _, s, _ = PowerKernel().parameters(T, D.currency, D.outcome)
    assert s.prior is not None
    assert s.prior.family == "beta"
    assert s.prior.hyper == {"alpha": 2.0, "beta": 2.0}
    _, s_hill, _ = HillKernel().parameters(T, D.currency, D.outcome)
    assert s_hill.prior is not None
    assert s_hill.prior.family == "gamma"


def test_hill_and_power_values_are_unchanged_by_the_scaled_parameterization() -> None:
    """The D2 form is a rewrite, not a new family: same numbers at non-zero doses."""
    for name in ("hill", "power"):
        kernel = kernel_from_name(name)
        theta = {f"k_{T}": 1.5, f"s_{T}": 0.7, f"beta_{T}": 2.0}
        x = np.array([0.0, 0.3, 1.0, 2.0, 5.0])
        u = x / 1.5
        if name == "hill":
            expected = 2.0 * u**0.7 / (1 + u**0.7)
        else:
            expected = 2.0 * u**0.7
        got = value(kernel.response(DOSE, T), data={"x": x}, params=theta)
        np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-15)


# -- end to end: kernel + carryover + seasonality in one ModelSpec ----------------------------


def _end_to_end(s_value: float) -> tuple[ModelSpec, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Hill + geometric carryover + annual seasonality on a 16-row frame with zero-dose rows."""
    import pandas as pd

    from axiom.surface.carryover import GeometricCarryover
    from axiom.surface.nuisance import FourierSeasonality

    n = 16
    rng = np.random.default_rng(7)
    frame = pd.DataFrame({"t": np.arange(n, dtype=float), "x": rng.uniform(0.0, 4.0, n)})
    frame.loc[[0, 5], "x"] = 0.0  # zero-dose rows; row 0 is zero *after* carryover too
    season = FourierSeasonality(period=8.0, order=1)
    frame = season.augment(frame, "t", "s_")
    kernel = HillKernel(reference_dose=2.0)
    carry = GeometricCarryover(max_lag=3)
    mean = Add(terms=(kernel.response(carry.apply(DOSE, T), T), season.expr("s_")))
    sigma = Param(
        name="sigma", dimension=D.outcome, prior=Prior(family="halfnormal", hyper={"sigma": 1.0})
    )
    model = ModelSpec(
        name="e2e",
        mean=mean,
        outcome=Data(name="y", dimension=D.outcome),
        likelihood=Likelihood(family="normal", scale="sigma"),
        parameters=(
            *kernel.parameters(T, D.currency, D.outcome),
            *carry.parameters(T),
            *season.parameters("s_"),
            sigma,
        ),
    )
    theta = {
        f"k_{T}": 1.8,
        f"s_{T}": s_value,
        f"beta_{T}": 2.5,
        f"lam_{T}": 0.6,
        "s_coef_sin_1": 0.4,
        "s_coef_cos_1": -0.3,
        "sigma": 0.5,
    }
    data = {c: frame[c].to_numpy(dtype=float) for c in ("x", "s_sin_1", "s_cos_1")}
    data["y"] = value(mean, data=data, params=theta) + rng.normal(0.0, 0.5, n)
    z = {k: np.asarray(v) for k, v in unconstrain(model, theta).items()}
    return model, data, z


@jax_only
def test_end_to_end_model_spec_agrees_between_numpy_and_jax() -> None:
    import jax

    from axiom.core import compile_log_density

    jax.config.update("jax_enable_x64", True)
    model, data, z = _end_to_end(s_value=1.5)
    f = compile_log_density(model)
    assert float(f(data, z)) == pytest.approx(log_density(model, data, z), abs=1e-8)
    g = jax.grad(lambda zz: f(data, zz))(z)
    assert set(g) == set(z)
    for k, v in g.items():
        assert np.all(np.isfinite(np.asarray(v))), k
    fd = _central_differences(lambda zz: log_density(model, data, zz), z)
    for k in z:
        assert float(g[k]) == pytest.approx(fd[k], rel=1e-5, abs=1e-6), k


@jax_only
def test_carryover_gradient_with_s_below_one_and_a_zero_carried_dose() -> None:
    import jax

    from axiom.core import compile_log_density

    jax.config.update("jax_enable_x64", True)
    model, data, z = _end_to_end(s_value=0.7)
    f = compile_log_density(model)
    assert float(f(data, z)) == pytest.approx(log_density(model, data, z), abs=1e-8)
    g = jax.grad(lambda zz: f(data, zz))(z)
    assert np.isfinite(float(g[f"lam_{T}"]))
