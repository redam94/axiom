from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from axiom.core import (
    Add,
    Const,
    Convolve,
    D,
    Data,
    DimensionError,
    Div,
    Gather,
    Likelihood,
    ModelSpec,
    Mul,
    Param,
    Pow,
    Prior,
    Reduce,
    Spec,
    constrain,
    dimension,
    dimensionless,
    free_parameters,
    jax_available,
    latex,
    log_density,
    log_prior,
    unconstrain,
    value,
)

jax_only = pytest.mark.skipif(not jax_available(), reason="jax not installed")


def _panel_model() -> tuple[ModelSpec, dict[str, np.ndarray], dict[str, np.ndarray]]:
    unit = Data(name="unit", dimension=dimensionless())
    dose = Data(name="dose", dimension=D.currency)
    y = Data(name="y", dimension=D.outcome)
    k = Param(
        name="k",
        dimension=D.currency,
        prior=Prior(family="lognormal", hyper={"mu": float(np.log(50)), "sigma": 0.5}),
    )
    s = Param(
        name="s",
        dimension=dimensionless(),
        prior=Prior(family="gamma", hyper={"alpha": 4.0, "beta": 2.0}),
    )
    beta = Param(
        name="beta", dimension=D.outcome, prior=Prior(family="halfnormal", hyper={"sigma": 20.0})
    )
    lam = Param(
        name="lam",
        dimension=dimensionless(),
        prior=Prior(family="beta", hyper={"alpha": 2.0, "beta": 2.0}),
    )
    a_mean = Param(
        name="a_mean",
        dimension=D.outcome,
        prior=Prior(family="normal", hyper={"mu": 0.0, "sigma": 10.0}),
    )
    a_sd = Param(
        name="a_sd", dimension=D.outcome, prior=Prior(family="halfnormal", hyper={"sigma": 5.0})
    )
    alpha = Param(
        name="alpha",
        dimension=D.outcome,
        shape=(3,),
        prior=Prior(family="normal", hyper={"mu": "a_mean", "sigma": "a_sd"}),
    )
    sigma = Param(
        name="sigma", dimension=D.outcome, prior=Prior(family="halfnormal", hyper={"sigma": 5.0})
    )
    theta_ = Param(
        name="theta",
        dimension=dimensionless(),
        prior=Prior(family="uniform", hyper={"low": 0.0, "high": 3.0}),
    )
    lags = Const(value=(0.0, 1.0, 2.0, 3.0), dimension=dimensionless())
    raw = Pow(
        base=lam,
        exponent=Mul(
            factors=(
                Add(
                    terms=(
                        lags,
                        Mul(factors=(Const(value=-1.0, dimension=dimensionless()), theta_)),
                    )
                ),
            )
            * 2
        ),
    )
    w = Div(numerator=raw, denominator=Reduce(op="sum", arg=raw))
    u = Div(numerator=Convolve(signal=dose, kernel=w), denominator=k)
    hill = Mul(
        factors=(
            beta,
            Div(
                numerator=Pow(base=u, exponent=s),
                denominator=Add(
                    terms=(Const(value=1.0, dimension=dimensionless()), Pow(base=u, exponent=s))
                ),
            ),
        )
    )
    mean = Add(terms=(Gather(source=alpha, index=unit), hill))
    model = ModelSpec(
        name="panel",
        mean=mean,
        outcome=y,
        likelihood=Likelihood(family="normal", scale="sigma"),
        parameters=(k, s, beta, lam, theta_, a_mean, a_sd, alpha, sigma),
    )
    rng = np.random.default_rng(0)
    n = 20
    data = {"unit": rng.integers(0, 3, n), "dose": rng.uniform(0, 150, n)}
    theta = {
        "k": 50.0,
        "s": 2.0,
        "beta": 10.0,
        "lam": 0.5,
        "theta": 1.0,
        "a_mean": 1.0,
        "a_sd": 0.5,
        "alpha": np.array([0.5, 1.0, 1.5]),
        "sigma": 1.0,
    }
    data["y"] = value(mean, data=data, params=theta) + rng.normal(0, 1, n)
    return model, data, theta


def test_new_nodes_dimension_value_latex() -> None:
    lam = Param(name="lam", dimension=dimensionless())
    lags = Const(value=(0.0, 1.0, 2.0), dimension=dimensionless())
    raw = Pow(base=lam, exponent=lags)
    w = Div(numerator=raw, denominator=Reduce(op="sum", arg=raw))
    assert (
        dimension(w).is_dimensionless
        and lags.is_vector
        and not Const(value=1.0, dimension=D.time).is_vector
    )
    np.testing.assert_allclose(value(w, params={"lam": 0.5}), np.array([1, 0.5, 0.25]) / 1.75)
    assert "\\sum" in latex(w)
    alpha = Param(name="alpha", dimension=D.outcome, shape=(2,))
    g = Gather(source=alpha, index=Data(name="u", dimension=dimensionless()))
    assert dimension(g) == D.outcome
    np.testing.assert_allclose(
        value(g, data={"u": [1, 0, 1]}, params={"alpha": [3.0, 4.0]}), [4, 3, 4]
    )
    assert "_{[u]}" in latex(g)
    with pytest.raises(KeyError, match="index column"):
        value(g, params={"alpha": [1.0, 2.0]})
    assert Spec.from_json(g.to_json()) == g
    with pytest.raises(ValueError):
        Const(value=(), dimension=D.time)
    with pytest.raises(ValueError):
        Const(value=float("inf"), dimension=D.time)


def test_prior_validation_and_hierarchy() -> None:
    with pytest.raises(ValueError, match="takes"):
        Prior(family="normal", hyper={"mu": 0.0})
    with pytest.raises(ValueError, match="unexpected"):
        Prior(family="halfnormal", hyper={"sigma": 1.0, "mu": 0.0})
    p = Prior(family="normal", hyper={"mu": "m", "sigma": 1.0})
    assert p.parents == ("m",)
    with pytest.raises(ValueError):
        Param(name="a", dimension=D.outcome, shape=(0,))


def test_modelspec_closure_checks() -> None:
    y = Data(name="y", dimension=D.outcome)
    a = Param(
        name="a", dimension=D.outcome, prior=Prior(family="normal", hyper={"mu": 0.0, "sigma": 1.0})
    )
    sig = Param(
        name="sigma", dimension=D.outcome, prior=Prior(family="halfnormal", hyper={"sigma": 1.0})
    )
    lik = Likelihood(family="normal", scale="sigma")
    ModelSpec(name="ok", mean=a, outcome=y, likelihood=lik, parameters=(a, sig))
    with pytest.raises(ValueError, match="not declared"):
        ModelSpec(name="x", mean=a, outcome=y, likelihood=lik, parameters=(sig,))
    with pytest.raises(ValueError, match="scale parameter"):
        ModelSpec(name="x", mean=a, outcome=y, likelihood=lik, parameters=(a,))
    with pytest.raises(ValueError, match="no prior"):
        ModelSpec(
            name="x",
            mean=a,
            outcome=y,
            likelihood=lik,
            parameters=(a, Param(name="sigma", dimension=D.outcome)),
        )
    with pytest.raises(ValueError, match="cycle"):
        m = Param(
            name="m",
            dimension=D.outcome,
            prior=Prior(family="normal", hyper={"mu": "a2", "sigma": 1.0}),
        )
        a2 = Param(
            name="a2",
            dimension=D.outcome,
            prior=Prior(family="normal", hyper={"mu": "m", "sigma": 1.0}),
        )
        ModelSpec(name="x", mean=a2, outcome=y, likelihood=lik, parameters=(m, a2, sig))
    with pytest.raises(DimensionError, match="mean has dimension"):
        ModelSpec(
            name="x",
            mean=Param(name="t", dimension=D.time, prior=a.prior),
            outcome=y,
            likelihood=lik,
            parameters=(Param(name="t", dimension=D.time, prior=a.prior), sig),
        )
    with pytest.raises(DimensionError, match="scale parameter"):
        ModelSpec(
            name="x",
            mean=a,
            outcome=y,
            likelihood=Likelihood(family="lognormal", scale="sigma"),
            parameters=(a, sig),
        )
    with pytest.raises(ValueError):
        Likelihood(family="poisson", scale="s")
    with pytest.raises(ValueError):
        Likelihood(family="student_t", scale="s")
    fixed = Param(name="f", dimension=D.outcome, prior=Prior(family="fixed", hyper={"value": 2.0}))
    m2 = ModelSpec(
        name="fx", mean=Add(terms=(a, fixed)), outcome=y, likelihood=lik, parameters=(a, fixed, sig)
    )
    assert [p.name for p in free_parameters(m2)] == ["a", "sigma"]
    theta, _ = constrain(m2, {"a": 0.3, "sigma": 0.0})
    assert theta["f"] == 2.0 and theta["sigma"] == 1.0


def test_transforms_round_trip_and_jacobian() -> None:
    model, data, theta = _panel_model()
    z = unconstrain(model, theta)
    back, log_jac = constrain(model, z)
    for k_, v in theta.items():
        np.testing.assert_allclose(back[k_], v)
    # log k + log s + log beta + log(lam(1-lam)) + log(u(1-u)*3) for theta + log a_sd + log sigma
    expected = (
        np.log(50)
        + np.log(2.0)
        + np.log(10)
        + np.log(0.5 * 0.5)
        + np.log((1 / 3) * (2 / 3) * 3)
        + np.log(0.5)
        + np.log(1.0)
    )
    assert log_jac == pytest.approx(expected)


def test_log_prior_matches_scipy_and_density_is_finite() -> None:
    model, data, theta = _panel_model()
    lp = log_prior(model, theta)
    manual = (
        stats.lognorm.logpdf(50, s=0.5, scale=50)
        + stats.gamma.logpdf(2.0, a=4, scale=0.5)
        + stats.halfnorm.logpdf(10, scale=20)
        + stats.beta.logpdf(0.5, 2, 2)
        + stats.uniform.logpdf(1.0, 0, 3)
        + stats.norm.logpdf(1.0, 0, 10)
        + stats.halfnorm.logpdf(0.5, scale=5)
        + stats.norm.logpdf([0.5, 1.0, 1.5], 1.0, 0.5).sum()
        + stats.halfnorm.logpdf(1.0, scale=5)
    )
    assert lp == pytest.approx(manual)
    assert np.isfinite(log_density(model, data, unconstrain(model, theta)))


@jax_only
def test_jax_agrees_with_numpy_on_panel_model() -> None:
    import jax

    from axiom.core import compile_jax, compile_log_density

    jax.config.update("jax_enable_x64", True)
    model, data, theta = _panel_model()
    z = unconstrain(model, theta)
    np.testing.assert_allclose(
        np.asarray(compile_jax(model.mean)(data, theta)),
        value(model.mean, data=data, params=theta),
        rtol=1e-12,
    )
    assert float(compile_log_density(model)(data, z)) == pytest.approx(
        log_density(model, data, z), abs=1e-8
    )
    g = jax.grad(lambda zz: compile_log_density(model)(data, zz))(z)
    assert all(np.all(np.isfinite(np.asarray(v))) for v in g.values())
