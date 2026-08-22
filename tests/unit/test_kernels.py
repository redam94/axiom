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
    GaussianProcessKernel,
    HillKernel,
    LinearKernel,
    LogisticKernel,
    PiecewiseLinearKernel,
    PolynomialKernel,
    PowerKernel,
    ResponseKernel,
    SplineKernel,
    kernel_from_name,
)

#: The families whose response is a signed sum over a fixed basis rather than
#: ``amplitude · saturation``. They are not monotone, not bounded by one amplitude,
#: and linear in every parameter they declare.
BASIS = ("polynomial", "spline", "piecewise_linear")

#: Families that are not monotone in the dose. The basis families plus the GP, whose
#: sign comes from its coefficients rather than from its (positive) amplitude.
NON_MONOTONE = (*BASIS, "gaussian_process")

#: A fixed set of standard-normal GP coefficients, so the family's tests are deterministic.
_GP_Z = np.sin(np.arange(1, 64) * 2.399963) * 1.3

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
    # signed coefficients chosen so each basis family turns over inside GRID
    "polynomial": {"beta1": 2.0, "beta2": -0.9, "beta3": 0.1},
    "spline": {"beta1": 1.5, "beta2": -2.5},
    "piecewise_linear": {"beta1": 1.2, "beta2": -2.0},
    "gaussian_process": {
        "ell": 0.30,
        "beta": 2.0,
        **{f"z{j}": float(_GP_Z[j - 1]) for j in range(1, GaussianProcessKernel().n_basis + 1)},
    },
}


def _natural_spline_basis(kernel: SplineKernel, x: np.ndarray) -> np.ndarray:
    """An independent natural-cubic basis: ESL 5.2.1 with the constant dropped."""
    t = np.asarray(kernel.reduced_knots)
    last = t.size - 1
    u = x / kernel.reference_dose

    def d(k: int) -> np.ndarray:
        left = np.maximum(u - t[k], 0.0) ** 3
        right = np.maximum(u - t[last], 0.0) ** 3
        return np.asarray((left - right) / (t[last] - t[k]))

    return np.stack([u, *(d(k) - d(last - 1) for k in range(t.size - 2))], axis=-1)


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
        case "polynomial":
            u = x / PolynomialKernel().reference_dose
            return np.asarray(sum(v[f"beta{j}"] * u**j for j in (1, 2, 3)))
        case "spline":
            coefficients = np.asarray([v[f"beta{j}"] for j in (1, 2)])
            return np.asarray(_natural_spline_basis(SplineKernel(), x) @ coefficients)
        case "piecewise_linear":
            kernel = PiecewiseLinearKernel()
            u = x / kernel.reference_dose
            out = v["beta1"] * u
            for j, t in enumerate(kernel.reduced_knots, start=2):
                out = out + v[f"beta{j}"] * np.maximum(u - t, 0.0)
            return np.asarray(out)
        case "gaussian_process":
            kernel = GaussianProcessKernel()
            u = x / kernel.reference_dose - kernel.half_width
            total = np.zeros_like(u)
            for j, w in enumerate(kernel.frequencies, start=1):
                phi = np.sin(w * (u + kernel.boundary)) / np.sqrt(kernel.boundary)
                at_zero = np.sin(w * (kernel.boundary - kernel.half_width)) / np.sqrt(
                    kernel.boundary
                )
                sd = np.sqrt(np.sqrt(2 * np.pi) * v["ell"] * np.exp(-(v["ell"] ** 2) * w**2 / 2))
                total = total + sd * v[f"z{j}"] * (phi - at_zero)
            return np.asarray(v["beta"] * total)
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
        PolynomialKernel,
        SplineKernel,
        PiecewiseLinearKernel,
        GaussianProcessKernel,
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
            if name == "gaussian_process":
                # the GP's shapes are a lengthscale and its standard-normal coefficients
                expected = "lognormal" if p.name == f"ell_{T}" else "normal"
                assert p.prior.family == expected
                if expected == "normal":
                    assert p.prior.hyper == {"mu": 0.0, "sigma": 1.0}
            else:
                # power is shipped for diminishing returns: its shape lives on (0, 1)
                assert p.prior.family == ("beta" if name == "power" else "gamma")
        elif name in BASIS:
            # a basis coefficient is signed: the family exists to bend back down
            assert p.prior.family == "normal"
            assert p.prior.hyper == {"mu": 0.0, "sigma": 1.0}
            assert p.dimension == D.outcome
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
    """Every family whose amplitude is an asymptote is monotone. The basis families are not."""
    if name in NON_MONOTONE:
        pytest.skip("these families are non-monotone by construction")
    kernel = kernel_from_name(name)
    got = value(kernel.response(DOSE, T), data={"x": GRID}, params=_theta(name))
    assert np.all(np.diff(got) > 0)


def test_unit_invariance_of_shapes(name: str) -> None:
    """Exit criterion 6: rescale dose and scale together, shapes untouched, response unchanged."""
    kernel = kernel_from_name(name)
    factor = 1000.0  # e.g. currency in thousands
    base = value(kernel.response(DOSE, T), data={"x": GRID}, params=_theta(name))
    if name in NON_MONOTONE:
        # these families carry their scale in spec fields, not parameters: move the
        # reference dose and the knots with the unit and the coefficients are unchanged
        fields = {"reference_dose": kernel.reference_dose * factor}
        if hasattr(kernel, "knots"):
            fields["knots"] = tuple(t * factor for t in kernel.knots)
        rescaled_kernel = kernel.model_copy(update=fields)
        rescaled = value(
            rescaled_kernel.response(DOSE, T), data={"x": GRID * factor}, params=_theta(name)
        )
        np.testing.assert_allclose(rescaled, base, rtol=1e-10)
        return
    if name == "linear":
        theta = {f"beta_rate_{T}": VALUES["linear"]["beta_rate"] / factor}
    else:
        theta = _theta(name, scale=factor)
    rescaled = value(kernel.response(DOSE, T), data={"x": GRID * factor}, params=theta)
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
    # amplitude x saturation for the classic families; a sum over a basis for the rest
    assert isinstance(expr, Add if name in BASIS else Mul)
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


# -- basis families ------------------------------------------------------------------------


@pytest.mark.parametrize("name", BASIS)
def test_basis_families_turn_over(name: str) -> None:
    """The point of the whole family: a response that rises and then falls."""
    kernel = kernel_from_name(name)
    got = np.asarray(value(kernel.response(DOSE, T), data={"x": GRID}, params=_theta(name)))
    steps = np.diff(got)
    assert np.any(steps > 0) and np.any(steps < 0), got
    assert not kernel.saturating


@pytest.mark.parametrize("name", BASIS)
def test_basis_families_are_linear_in_every_parameter(name: str) -> None:
    """Every coefficient has the ``amplitude`` role, so ``linearize`` is exact, not local."""
    kernel = kernel_from_name(name)
    assert set(kernel.roles.values()) == {"amplitude"}
    theta = _theta(name)
    expr = kernel.response(DOSE, T)
    base = np.asarray(value(expr, data={"x": GRID}, params=theta))
    # doubling every coefficient doubles the response: it is a linear map of the vector
    doubled = np.asarray(value(expr, data={"x": GRID}, params={k: 2 * v for k, v in theta.items()}))
    np.testing.assert_allclose(doubled, 2 * base, rtol=1e-12)
    # and it is additive over the coefficient vector, one basis function at a time
    pieces = np.zeros_like(base)
    for stem in kernel.roles:
        only = {k: (v if k == f"{stem}_{T}" else 0.0) for k, v in theta.items()}
        pieces = pieces + np.asarray(value(expr, data={"x": GRID}, params=only))
    np.testing.assert_allclose(pieces, base, rtol=1e-12)


@pytest.mark.parametrize("name", BASIS)
def test_basis_coefficients_may_be_constrained_positive(name: str) -> None:
    """A signed prior is the default; a positive one is how you ask for monotone."""
    default = kernel_from_name(name)
    for p in default.parameters(T, D.currency, D.outcome):
        assert p.prior is not None and p.prior.family == "normal"
    positive = KERNELS[name].model_validate(
        {"amplitude_prior": Prior(family="halfnormal", hyper={"sigma": 2.0})}
    )
    for p in positive.parameters(T, D.currency, D.outcome):
        assert p.prior is not None and p.prior.family == "halfnormal"
    with pytest.raises(ValueError, match="amplitude_prior must be one of"):
        KERNELS[name].model_validate(
            {"amplitude_prior": Prior(family="uniform", hyper={"low": -1.0, "high": 1.0})}
        )


def test_polynomial_degree_sets_the_coefficient_count() -> None:
    for degree in (1, 2, 5, 8):
        kernel = PolynomialKernel(degree=degree)
        assert kernel.stems == tuple(f"beta{j}" for j in range(1, degree + 1))
        assert len(kernel.parameters(T, D.currency, D.outcome)) == degree
    with pytest.raises(ValueError):
        PolynomialKernel(degree=0)
    with pytest.raises(ValueError):
        PolynomialKernel(degree=9)


def test_polynomial_degree_one_is_the_linear_kernel() -> None:
    """A degree-1 polynomial and a linear kernel are one function, parameterized twice."""
    reference = 4.0
    poly = PolynomialKernel(reference_dose=reference, degree=1)
    linear = LinearKernel(reference_dose=reference)
    got = np.asarray(value(poly.response(DOSE, T), data={"x": GRID}, params={f"beta1_{T}": 3.0}))
    same = np.asarray(
        value(
            linear.response(DOSE, T), data={"x": GRID}, params={f"beta_rate_{T}": 3.0 / reference}
        )
    )
    np.testing.assert_allclose(got, same, rtol=1e-12)


def test_natural_spline_is_linear_outside_the_boundary_knots() -> None:
    """The 'natural' in natural cubic spline: no cubic tail to extrapolate off a cliff."""
    kernel = SplineKernel(reference_dose=10.0, knots=(2.0, 5.0, 8.0, 11.0))
    theta = {f"beta{j}_{T}": v for j, v in enumerate([4.0, -30.0, 22.0], start=1)}
    above = np.asarray(
        value(
            kernel.response(DOSE, T), data={"x": np.array([12.0, 14.0, 16.0, 18.0])}, params=theta
        )
    )
    below = np.asarray(
        value(kernel.response(DOSE, T), data={"x": np.array([0.0, 0.5, 1.0, 1.5])}, params=theta)
    )
    np.testing.assert_allclose(np.diff(above, 2), 0.0, atol=1e-9)
    np.testing.assert_allclose(np.diff(below, 2), 0.0, atol=1e-12)
    # a cubic polynomial over the same range does not have that property
    cubic = PolynomialKernel(reference_dose=10.0, degree=3)
    curved = np.asarray(
        value(
            cubic.response(DOSE, T),
            data={"x": np.array([12.0, 14.0, 16.0, 18.0])},
            params={f"beta{j}_{T}": v for j, v in enumerate([4.0, -3.0, 2.0], start=1)},
        )
    )
    assert np.max(np.abs(np.diff(curved, 2))) > 1e-3


def test_natural_spline_matches_an_independent_basis() -> None:
    kernel = SplineKernel(reference_dose=8.0, knots=(1.0, 3.0, 5.0, 7.0, 9.0))
    coefficients = np.array([1.2, -0.7, 0.4, 0.9])
    theta = {f"beta{j}_{T}": c for j, c in enumerate(coefficients, start=1)}
    got = np.asarray(value(kernel.response(DOSE, T), data={"x": GRID}, params=theta))
    want = _natural_spline_basis(kernel, GRID) @ coefficients
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)


def test_piecewise_linear_coefficients_are_slope_changes() -> None:
    """beta1 is the slope in the first segment; each later one is the change at its knot."""
    kernel = PiecewiseLinearKernel(reference_dose=10.0, knots=(3.0, 7.0))
    theta = {f"beta1_{T}": 5.0, f"beta2_{T}": -9.0, f"beta3_{T}": 6.0}
    # the derivative is in outcome per dose, so divide the coefficients by the reference
    slopes = np.asarray(
        value(kernel.derivative(DOSE, T), data={"x": np.array([1.0, 5.0, 9.0])}, params=theta)
    )
    np.testing.assert_allclose(
        slopes, np.array([5.0, 5.0 - 9.0, 5.0 - 9.0 + 6.0]) / 10.0, rtol=1e-12
    )
    # step(0) = 0, so a derivative read exactly at a knot is the slope arriving into it
    at_knot = float(
        np.asarray(value(kernel.derivative(DOSE, T), data={"x": np.array([3.0])}, params=theta))[0]
    )
    assert at_knot == pytest.approx(0.5)


@pytest.mark.parametrize("cls", [SplineKernel, PiecewiseLinearKernel])
def test_knots_must_be_increasing_and_positive(cls: type) -> None:
    with pytest.raises(ValueError, match="strictly increasing and strictly positive"):
        cls(knots=(0.5, 0.5, 0.9))
    with pytest.raises(ValueError, match="strictly increasing and strictly positive"):
        cls(knots=(0.0, 0.5, 0.9))
    with pytest.raises(ValueError, match="strictly increasing and strictly positive"):
        cls(knots=(0.9, 0.5, 0.2))
    with pytest.raises(ValueError, match="must be finite"):
        cls(knots=(0.2, 0.5, float("inf")))


def test_a_spline_needs_three_knots() -> None:
    with pytest.raises(ValueError, match="need at least 3 knots"):
        SplineKernel(knots=(0.3, 0.6))
    assert len(SplineKernel(knots=(0.3, 0.6, 0.9)).stems) == 2
    with pytest.raises(ValueError, match="need at least 1 knots"):
        PiecewiseLinearKernel(knots=())


@pytest.mark.parametrize("name", sorted(KERNELS))
def test_saturation_derivative_differentiates_the_saturation(name: str) -> None:
    """The protocol's new member, checked against central differences for every family."""
    kernel = kernel_from_name(name)
    theta = _theta(name)
    grid = GRID + 0.013  # off any default knot
    h = 1e-6
    up = np.asarray(value(kernel.saturation(DOSE, T), data={"x": grid + h}, params=theta))
    down = np.asarray(value(kernel.saturation(DOSE, T), data={"x": grid - h}, params=theta))
    got = np.asarray(value(kernel.saturation_derivative(DOSE, T), data={"x": grid}, params=theta))
    np.testing.assert_allclose(np.broadcast_to(got, up.shape), (up - down) / (2 * h), atol=1e-6)
    assert dimension(kernel.saturation_derivative(DOSE, T)) == dimensionless() / D.currency


@pytest.mark.parametrize("name", sorted(set(KERNELS) - set(BASIS) - {"linear"}))
def test_derivative_is_the_amplitude_times_the_saturation_derivative(name: str) -> None:
    """For a single-amplitude family the two are the same tree, so they cannot drift."""
    from axiom.core import Mul

    kernel = kernel_from_name(name)
    expr = kernel.derivative(DOSE, T)
    assert isinstance(expr, Mul)
    amplitude, rest = expr.factors
    assert kernel.roles[str(amplitude.name).removesuffix(f"_{T}")] == "amplitude"
    assert rest == kernel.saturation_derivative(DOSE, T)


@jax_only
@pytest.mark.parametrize("name", BASIS)
def test_basis_families_agree_between_numpy_and_jax(name: str) -> None:
    """Gate 9 in miniature: relu and step evaluate identically under both interpreters."""
    import jax

    from axiom.core import compile_log_density

    jax.config.update("jax_enable_x64", True)
    kernel = kernel_from_name(name)
    model = _normal_model(kernel, name)
    theta = {**_theta(name), "sigma": 0.8}
    x = np.array([0.0, 0.2, 0.5, 0.75, 1.0, 2.5, 5.0])
    y = np.asarray(value(kernel.response(DOSE, T), data={"x": x}, params=_theta(name))) + 0.1
    data = {"x": x, "y": y}
    z = {k: np.asarray(v) for k, v in unconstrain(model, theta).items()}
    f = compile_log_density(model)
    assert float(f(data, z)) == pytest.approx(log_density(model, data, z), abs=1e-10)
    g = jax.grad(lambda zz: f(data, zz))(z)
    fd = _central_differences(lambda zz: log_density(model, data, zz), z)
    for k in z:
        assert np.isfinite(float(g[k])), k
        assert float(g[k]) == pytest.approx(fd[k], rel=1e-6, abs=1e-6), k


# -- the Gaussian process (note 0006) ------------------------------------------------------


def test_gp_implied_covariance_matches_the_exact_one() -> None:
    """The claim that makes it a GP: the basis reproduces the covariance it approximates."""
    kernel = GaussianProcessKernel()
    for lengthscale in (0.15, 0.3, 0.5):
        u = np.linspace(0.0, 1.0, 41)
        implied = kernel.implied_covariance(lengthscale, u)
        exact = kernel.exact_covariance(lengthscale, u)
        assert np.max(np.abs(implied - exact)) < 0.02, lengthscale
        # symmetric, positive semi-definite, unit marginal variance
        np.testing.assert_allclose(implied, implied.T, atol=1e-12)
        assert np.min(np.linalg.eigvalsh(implied)) > -1e-9
        np.testing.assert_allclose(np.diag(implied), 1.0, atol=0.02)


@pytest.mark.parametrize("covariance", ["squared_exponential", "matern32", "matern52"])
def test_gp_covariance_families_are_each_approximated(covariance: str) -> None:
    kernel = GaussianProcessKernel(covariance=covariance, n_basis=48, boundary_factor=3.0)
    assert kernel.covariance_error(0.3) < 0.02, covariance
    # a Matern is rougher than a squared exponential, so its spectral tail is heavier
    tail = kernel.implied_covariance(0.3, np.array([0.0, 0.5]))[0, 1]
    assert 0.0 < tail < 1.0


def test_gp_reports_when_its_basis_is_too_small() -> None:
    """The approximation is allowed to be wrong; it is not allowed to be quiet about it."""
    kernel = GaussianProcessKernel()
    assert kernel.sufficient_for(0.3)
    assert kernel.sufficient_for(0.10)
    assert kernel.sufficient_for(0.70)
    # too few basis functions for a short lengthscale, too small a boundary for a long one
    assert not kernel.sufficient_for(0.03)
    assert not kernel.sufficient_for(2.0)
    assert kernel.covariance_error(0.03) > kernel.covariance_error(0.3)
    # more basis functions buy the short end
    assert GaussianProcessKernel(n_basis=96).sufficient_for(0.03)
    # a wider boundary buys the long end
    assert GaussianProcessKernel(n_basis=96, boundary_factor=8.0).sufficient_for(2.0)
    with pytest.raises(ValueError, match="lengthscale must be positive"):
        kernel.covariance_error(0.0)


def test_gp_boundary_and_frequencies() -> None:
    kernel = GaussianProcessKernel(n_basis=4, boundary_factor=2.0)
    assert kernel.half_width == 0.5
    assert kernel.boundary == 1.0
    np.testing.assert_allclose(
        kernel.frequencies, [np.pi * j / 2.0 for j in (1, 2, 3, 4)], rtol=1e-12
    )
    assert kernel.stems == ("ell", "z1", "z2", "z3", "z4", "beta")


def test_gp_has_one_amplitude_so_the_classic_invariant_holds() -> None:
    """Unlike the basis families: the sign lives in the coefficients, so beta stays positive."""
    from axiom.core import Mul

    kernel = GaussianProcessKernel()
    amplitudes = [s for s, r in kernel.roles.items() if r == "amplitude"]
    assert amplitudes == ["beta"]
    response = kernel.response(DOSE, T)
    assert isinstance(response, Mul)
    beta, saturation = response.factors
    assert beta.name == f"beta_{T}"
    assert saturation == kernel.saturation(DOSE, T)
    derivative = kernel.derivative(DOSE, T)
    assert isinstance(derivative, Mul)
    assert derivative.factors[1] == kernel.saturation_derivative(DOSE, T)
    with pytest.raises(ValueError, match="positive support"):
        GaussianProcessKernel(
            amplitude_prior=Prior(family="normal", hyper={"mu": 0.0, "sigma": 1.0})
        )


def test_gp_roughness_tracks_its_lengthscale() -> None:
    """The GP property: a shorter lengthscale wiggles more, relative to its own scale.

    Measured as total variation over the function's own standard deviation. The raw
    total variation is not the right statistic — the spectral density carries the
    amplitude too, so a short lengthscale gives a *smaller* function as well as a
    rougher one.
    """
    grid = np.linspace(0.0, 1.0, 101)
    kernel = GaussianProcessKernel(n_basis=48)
    roughness = []
    for lengthscale in (0.08, 0.15, 0.30, 0.50):
        theta = {f"ell_{T}": lengthscale, f"beta_{T}": 1.0}
        theta |= {f"z{j}_{T}": float(_GP_Z[j - 1]) for j in range(1, kernel.n_basis + 1)}
        got = np.ravel(np.asarray(value(kernel.response(DOSE, T), data={"x": grid}, params=theta)))
        roughness.append(float(np.sum(np.abs(np.diff(got))) / np.std(got)))
    assert all(a > b for a, b in zip(roughness[:-1], roughness[1:], strict=True)), roughness
    assert roughness[0] > 5 * roughness[-1], roughness


def test_gp_response_is_zero_at_zero_dose_for_any_coefficients() -> None:
    """Centring the basis is what keeps the surface intercept meaning what it says."""
    rng = np.random.default_rng(0)
    kernel = GaussianProcessKernel(reference_dose=40.0)
    for _ in range(5):
        theta = {f"ell_{T}": float(rng.uniform(0.1, 0.6)), f"beta_{T}": float(rng.uniform(0.5, 3))}
        theta |= {f"z{j}_{T}": float(rng.normal()) for j in range(1, kernel.n_basis + 1)}
        at_zero = value(kernel.response(DOSE, T), data={"x": np.array([0.0])}, params=theta)
        assert float(np.ravel(np.asarray(at_zero))[0]) == pytest.approx(0.0, abs=1e-12)


@jax_only
def test_gp_agrees_between_numpy_and_jax() -> None:
    """sin and cos through the whole log density, values and gradients."""
    import jax

    from axiom.core import compile_log_density

    jax.config.update("jax_enable_x64", True)
    kernel = GaussianProcessKernel(n_basis=6)
    model = _normal_model(kernel, "gaussian_process")
    theta = {f"ell_{T}": 0.3, f"beta_{T}": 2.0, "sigma": 0.5}
    theta |= {f"z{j}_{T}": float(_GP_Z[j - 1]) for j in range(1, 7)}
    x = np.linspace(0.0, 1.0, 9)
    y = np.ravel(np.asarray(value(kernel.response(DOSE, T), data={"x": x}, params=theta))) + 0.05
    data = {"x": x, "y": y}
    z = {k: np.asarray(v) for k, v in unconstrain(model, theta).items()}
    f = compile_log_density(model)
    assert float(f(data, z)) == pytest.approx(log_density(model, data, z), abs=1e-9)
    g = jax.grad(lambda zz: f(data, zz))(z)
    fd = _central_differences(lambda zz: log_density(model, data, zz), z)
    for k in z:
        assert np.isfinite(float(g[k])), k
        assert float(g[k]) == pytest.approx(fd[k], rel=1e-5, abs=1e-6), k
