"""Laplace approximation: closed forms, support handling, PD reporting, the parent's funnel.

The review cases are here by name: a non-unit-scale model (absolute ``gtol``
and an absolute PD floor both broke it), an exact saddle at the default
start, a hierarchical funnel with no mode, and an interval prior whose bound
is a fixed parameter.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from axiom.core import (
    Add,
    Apply,
    Const,
    D,
    Data,
    Gather,
    Likelihood,
    ModelSpec,
    Mul,
    Param,
    Posterior,
    Prior,
    Unsupported,
    Unverified,
    dimensionless,
    jax_available,
    log_density,
    unconstrain,
)
from axiom.infer.backend import Backend, PointEstimate, SampleSettings, get_backend
from axiom.infer.laplace import (
    LaplaceBackend,
    constrain_draws,
    find_mode,
    flat_layout,
    hessian_at,
    laplace,
)

OUT = D.outcome
Y = Data(name="y", dimension=OUT)

DERIVATIVES = ["finite_difference"] + (["jax"] if jax_available() else [])


def normal_normal(prior_sd: float = 10.0) -> ModelSpec:
    """y ~ N(mu, 1), mu ~ N(0, prior_sd); sigma fixed at 1."""
    mu = Param(
        name="mu", dimension=OUT, prior=Prior(family="normal", hyper={"mu": 0.0, "sigma": prior_sd})
    )
    sigma = Param(name="sigma", dimension=OUT, prior=Prior(family="fixed", hyper={"value": 1.0}))
    return ModelSpec(
        name="normal_normal",
        mean=mu,
        outcome=Y,
        likelihood=Likelihood(family="normal", scale="sigma"),
        parameters=(mu, sigma),
    )


def big_scale() -> ModelSpec:
    """y ~ N(mu, 1e5), mu ~ N(0, 1e8): the posterior sd is ~4.5e4 and the Hessian ~5e-10."""
    mu = Param(
        name="mu", dimension=OUT, prior=Prior(family="normal", hyper={"mu": 0.0, "sigma": 1e8})
    )
    sigma = Param(name="sigma", dimension=OUT, prior=Prior(family="fixed", hyper={"value": 1e5}))
    return ModelSpec(
        name="big_scale",
        mean=mu,
        outcome=Y,
        likelihood=Likelihood(family="normal", scale="sigma"),
        parameters=(mu, sigma),
    )


def normal_halfnormal() -> ModelSpec:
    """y ~ N(mu, sigma), mu ~ N(0, 10), sigma ~ HalfNormal(2)."""
    mu = Param(
        name="mu", dimension=OUT, prior=Prior(family="normal", hyper={"mu": 0.0, "sigma": 10.0})
    )
    sigma = Param(
        name="sigma", dimension=OUT, prior=Prior(family="halfnormal", hyper={"sigma": 2.0})
    )
    return ModelSpec(
        name="normal_halfnormal",
        mean=mu,
        outcome=Y,
        likelihood=Likelihood(family="normal", scale="sigma"),
        parameters=(mu, sigma),
    )


def collinear() -> ModelSpec:
    """y ~ N(a + b, 1) with effectively flat priors: the (1, -1) direction is unidentified."""
    flat = Prior(family="normal", hyper={"mu": 0.0, "sigma": 1e8})
    a = Param(name="a", dimension=OUT, prior=flat)
    b = Param(name="b", dimension=OUT, prior=flat)
    sigma = Param(name="sigma", dimension=OUT, prior=Prior(family="fixed", hyper={"value": 1.0}))
    return ModelSpec(
        name="collinear",
        mean=Add(terms=(a, b)),
        outcome=Y,
        likelihood=Likelihood(family="normal", scale="sigma"),
        parameters=(a, b, sigma),
    )


def saddle() -> ModelSpec:
    """y ~ N(a · b, 1), a, b ~ N(0, 10): ``z = 0`` (the default start) is an exact saddle."""
    a = Param(
        name="a", dimension=OUT, prior=Prior(family="normal", hyper={"mu": 0.0, "sigma": 10.0})
    )
    b = Param(
        name="b",
        dimension=dimensionless(),
        prior=Prior(family="normal", hyper={"mu": 0.0, "sigma": 10.0}),
    )
    sigma = Param(name="sigma", dimension=OUT, prior=Prior(family="fixed", hyper={"value": 1.0}))
    return ModelSpec(
        name="saddle",
        mean=Mul(factors=(a, b)),
        outcome=Y,
        likelihood=Likelihood(family="normal", scale="sigma"),
        parameters=(a, b, sigma),
    )


def tanh_wide_prior() -> ModelSpec:
    """y ~ N(tanh(b), 1), b ~ N(0, 1e8): a prior-scaled finite-difference step saturates tanh.

    The Hessian in ``b`` cannot be resolved by finite differences at that step
    (the truncation check catches it); jax resolves it exactly.
    """
    b = Param(
        name="b",
        dimension=dimensionless(),
        prior=Prior(family="normal", hyper={"mu": 0.0, "sigma": 1e8}),
    )
    sigma = Param(name="sigma", dimension=OUT, prior=Prior(family="fixed", hyper={"value": 1.0}))
    return ModelSpec(
        name="tanh_wide_prior",
        mean=Mul(factors=(Const(value=1.0, dimension=OUT), Apply(fn="tanh", arg=b))),
        outcome=Y,
        likelihood=Likelihood(family="normal", scale="sigma"),
        parameters=(b, sigma),
    )


def uniform_fixed_bound() -> ModelSpec:
    """mu ~ Uniform(lo, 5) where ``lo`` is a *fixed* parameter: ``unconstrain`` must see it."""
    lo = Param(name="lo", dimension=OUT, prior=Prior(family="fixed", hyper={"value": -5.0}))
    mu = Param(
        name="mu", dimension=OUT, prior=Prior(family="uniform", hyper={"low": "lo", "high": 5.0})
    )
    sigma = Param(name="sigma", dimension=OUT, prior=Prior(family="fixed", hyper={"value": 1.0}))
    return ModelSpec(
        name="uniform_fixed_bound",
        mean=mu,
        outcome=Y,
        likelihood=Likelihood(family="normal", scale="sigma"),
        parameters=(lo, mu, sigma),
    )


def hierarchical() -> ModelSpec:
    """Three units with their own intercepts drawn from a common normal with inferred spread.

    This is the shape that broke the parent's BFGS-``hess_inv`` Laplace path.
    """
    a_mean = Param(
        name="a_mean", dimension=OUT, prior=Prior(family="normal", hyper={"mu": 0.0, "sigma": 5.0})
    )
    a_sd = Param(name="a_sd", dimension=OUT, prior=Prior(family="halfnormal", hyper={"sigma": 2.0}))
    alpha = Param(
        name="alpha",
        dimension=OUT,
        shape=(3,),
        prior=Prior(family="normal", hyper={"mu": "a_mean", "sigma": "a_sd"}),
    )
    sigma = Param(
        name="sigma", dimension=OUT, prior=Prior(family="halfnormal", hyper={"sigma": 2.0})
    )
    idx = Data(name="unit_index", dimension=dimensionless())
    return ModelSpec(
        name="hierarchical",
        mean=Gather(source=alpha, index=idx),
        outcome=Y,
        likelihood=Likelihood(family="normal", scale="sigma"),
        parameters=(a_mean, a_sd, alpha, sigma),
    )


@pytest.fixture
def y50() -> np.ndarray:
    rng = np.random.default_rng(1)
    return 3.0 + rng.normal(0.0, 1.0, 50)


def closed_form(
    y: np.ndarray, prior_sd: float = 10.0, noise_sd: float = 1.0
) -> tuple[float, float]:
    prec = 1.0 / prior_sd**2 + y.size / noise_sd**2
    return float(y.sum() / noise_sd**2 / prec), float(1.0 / np.sqrt(prec))


def as_posterior(result: Posterior | Unverified) -> Posterior:
    assert isinstance(result, Posterior), getattr(result, "reason", result)
    return result


def as_estimate(result: PointEstimate | Unverified) -> PointEstimate:
    assert isinstance(result, PointEstimate), getattr(result, "reason", result)
    return result


# -- conjugate normal-normal ----------------------------------------------------------


def test_normal_normal_matches_closed_form(y50: np.ndarray) -> None:
    post = as_posterior(laplace(normal_normal(), {"y": y50}, draws=8000, seed=0))
    mean, sd = closed_form(y50)
    s = post.summary("mu")
    assert abs(s.mean - mean) / abs(mean) < 0.02
    assert abs(s.sd - sd) / sd < 0.02
    assert post.draws("mu").shape == (1, 8000)
    prov = post.provenance
    assert prov["method"] == "laplace"
    assert prov["seed"] == 0
    assert prov["hessian_pd"] is True
    assert prov["converged"] is True
    assert prov["verified"] is True
    assert prov["nonfinite_draw_frac"] == 0.0
    assert prov["optimizer"] == "trust-ncg"
    assert prov["newton_decrement"] < 1e-8
    assert prov["jitter"] == 0.0
    assert prov["fixed"] == {"sigma": 1.0}
    assert "sigma" not in post.names()


def test_find_mode_is_the_closed_form_mean(y50: np.ndarray) -> None:
    est = find_mode(normal_normal(), {"y": y50})
    mean, _ = closed_form(y50)
    assert isinstance(est, PointEstimate)
    assert est.converged
    assert est.hessian_pd is True
    assert est.newton_decrement is not None and est.newton_decrement < 1e-8
    assert est.method == "trust-ncg"
    assert abs(float(est.theta["mu"]) - mean) < 1e-6
    assert PointEstimate.from_json(est.to_json()) == est


@pytest.mark.parametrize("model", [normal_normal, normal_halfnormal, uniform_fixed_bound])
def test_point_estimate_log_density_is_the_numpy_log_density(
    model: type[ModelSpec], y50: np.ndarray
) -> None:
    spec = model()
    est = find_mode(spec, {"y": y50})
    fixed = {
        p.name: p.prior.hyper["value"]
        for p in spec.parameters
        if p.prior is not None and p.prior.family == "fixed"
    }
    z = unconstrain(spec, {**fixed, **est.theta})
    assert est.log_density == pytest.approx(log_density(spec, {"y": y50}, z), rel=1e-10)


def test_init_in_constrained_space_is_accepted(y50: np.ndarray) -> None:
    est = find_mode(normal_halfnormal(), {"y": y50}, init={"mu": 1.0, "sigma": 0.5})
    assert est.converged
    assert abs(float(est.theta["mu"]) - y50.mean()) < 0.05
    with pytest.raises(KeyError):
        find_mode(normal_halfnormal(), {"y": y50}, init={"mu": 1.0})


def test_seed_none_is_drawn_and_recorded(y50: np.ndarray) -> None:
    post = as_posterior(laplace(normal_normal(), {"y": y50}, draws=50, seed=None))
    assert isinstance(post.provenance["seed"], int)
    again = as_posterior(
        laplace(normal_normal(), {"y": y50}, draws=50, seed=post.provenance["seed"])
    )
    assert np.array_equal(again.draws("mu"), post.draws("mu"))


# -- non-unit scale: the absolute-tolerance bugs ----------------------------------------------


@pytest.mark.parametrize("derivatives", DERIVATIVES)
def test_non_unit_scale_mode_and_sd_match_closed_form(derivatives: str) -> None:
    """mu ~ N(0, 1e8), y ~ N(mu, 1e5), n = 5: an absolute gtol stopped at z=105023 (true 304066)
    and an absolute PD floor called the 5e-10 Hessian non-PD (sd 9497 instead of 44721)."""
    rng = np.random.default_rng(0)
    y = 3e5 + rng.normal(0.0, 1e5, 5)
    mean, sd = closed_form(y, prior_sd=1e8, noise_sd=1e5)
    est = find_mode(big_scale(), {"y": y}, derivatives=derivatives)  # type: ignore[arg-type]
    assert est.converged
    assert est.hessian_pd is True
    assert abs(float(est.theta["mu"]) - mean) / abs(mean) < 1e-6
    assert est.newton_decrement is not None and est.newton_decrement < 1e-8
    post = as_posterior(
        laplace(
            big_scale(), {"y": y}, draws=4000, seed=0, derivatives=derivatives  # type: ignore[arg-type]
        )
    )
    prov = post.provenance
    assert prov["hessian_pd"] is True
    assert prov["converged"] is True
    assert prov["min_eigenvalue"] == pytest.approx(5e-10, rel=1e-3)
    s = post.summary("mu")
    assert abs(s.mean - mean) / sd < 0.05
    assert abs(s.sd - sd) / sd < 0.03


# -- positive support -----------------------------------------------------------------


def test_positive_parameter_has_no_nonfinite_draws(y50: np.ndarray) -> None:
    post = as_posterior(laplace(normal_halfnormal(), {"y": y50}, draws=2000, seed=3))
    sigma = post.draws("sigma")
    assert np.all(np.isfinite(sigma))
    assert np.all(sigma > 0)
    assert post.provenance["nonfinite_draw_frac"] == 0.0
    assert post.provenance["hessian_pd"] is True
    # the Laplace sd of sigma is sensible (the posterior sd of sigma is ~ sigma / sqrt(2n))
    s = post.summary("sigma")
    assert 0.5 < s.mean < 1.5
    assert 0.05 < s.sd < 0.25


def test_uniform_prior_with_fixed_bound(y50: np.ndarray) -> None:
    """``unconstrain`` needs the fixed bound; before the fix this was ``KeyError: 'lo'``."""
    est = find_mode(uniform_fixed_bound(), {"y": y50}, init={"mu": 1.0})
    assert est.converged
    assert abs(float(est.theta["mu"]) - y50.mean()) < 0.05
    post = as_posterior(
        laplace(uniform_fixed_bound(), {"y": y50}, draws=500, seed=0, init={"mu": 1.0})
    )
    mu = post.draws("mu")
    assert np.all((mu > -5.0) & (mu < 5.0))
    assert abs(mu.mean() - y50.mean()) < 0.1
    assert post.provenance["fixed"] == {"lo": -5.0, "sigma": 1.0}


# -- not positive definite ---------------------------------------------------------------


def test_collinear_model_is_unverified_unless_allowed(
    y50: np.ndarray, caplog: pytest.LogCaptureFixture
) -> None:
    got = laplace(collinear(), {"y": y50}, draws=200, seed=0)
    assert isinstance(got, Unverified)
    assert "not numerically positive definite" in got.reason
    assert got.detail["hessian_pd"] == "False"
    assert got.detail["converged"] == "True"  # the identified direction did converge
    assert got.detail["optimizer"] == "trust-ncg"
    assert float(got.detail["min_eigenvalue"]) < 1e-6
    with caplog.at_level(logging.WARNING, logger="axiom.infer.laplace"):
        post = as_posterior(
            laplace(collinear(), {"y": y50}, draws=200, seed=0, allow_unverified=True)
        )
    prov = post.provenance
    assert prov["hessian_pd"] is False
    assert prov["verified"] is False
    assert prov["converged"] is True
    assert prov["min_eigenvalue"] < 1e-6
    assert prov["jitter"] > 0.0
    assert prov["nonfinite_draw_frac"] == 0.0
    assert np.all(np.isfinite(post.draws("a")))
    records = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("not numerically positive definite" in r.getMessage() for r in records)
    # the identified direction a + b is still right
    total = post.flat("a") + post.flat("b")
    assert abs(total.mean() - y50.mean()) < 0.1


def test_saddle_at_the_start_is_escaped_by_a_seeded_restart() -> None:
    """``z = 0`` has zero gradient and an indefinite Hessian; the old code called it converged."""
    y = np.array([3.0, 2.9, 3.1])
    est = find_mode(saddle(), {"y": y})
    assert est.converged
    assert est.hessian_pd is True
    assert est.min_eigenvalue is not None and est.min_eigenvalue > 0.0
    ab = float(est.theta["a"]) * float(est.theta["b"])
    assert abs(ab - 3.0) < 0.1
    post = as_posterior(laplace(saddle(), {"y": y}, draws=200, seed=1))
    assert post.provenance["n_restarts"] >= 1
    assert post.provenance["converged"] is True
    assert post.provenance["hessian_pd"] is True


# -- the parent's breaking case ---------------------------------------------------------


@pytest.fixture
def hier_data() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(7)
    unit = np.repeat(np.arange(3), 12)
    y = np.array([1.0, 2.5, 4.0])[unit] + rng.normal(0.0, 0.5, unit.size)
    return {"y": y, "unit_index": unit}


def test_hierarchical_model_has_finite_draws(hier_data: dict[str, np.ndarray]) -> None:
    post = as_posterior(laplace(hierarchical(), hier_data, draws=1000, seed=11))
    prov = post.provenance
    assert prov["nonfinite_draw_frac"] == 0.0
    assert prov["hessian_pd"] is True
    assert post.draws("alpha").shape == (1, 1000, 3)
    assert post.coords() == {"alpha_dim0": [0, 1, 2]}
    for name in post.names():
        assert np.all(np.isfinite(post.draws(name))), name
    assert np.all(post.draws("a_sd") > 0)
    assert np.all(post.draws("sigma") > 0)
    means = post.flat("alpha").mean(axis=0)
    assert np.allclose(means, [1.0, 2.5, 4.0], atol=0.4)


def test_funnel_without_a_mode_is_unverified_not_nan() -> None:
    """Identical units drive ``a_sd`` to zero: no mode exists; the old code returned NaN draws."""
    rng = np.random.default_rng(3)
    unit = np.repeat(np.arange(3), 12)
    data = {"y": 2.0 + rng.normal(0.0, 0.5, unit.size), "unit_index": unit}
    got = laplace(hierarchical(), data, draws=200, seed=0)
    assert isinstance(got, Unverified)
    # hessian_pd, not converged: the funnel is degenerate enough that whether the
    # optimizer stops just short of the neck or lands in it is a floating-point
    # detail of the platform, and `converged` records which happened. What is
    # true on every platform -- and what makes the mode unusable -- is that the
    # curvature there is not positive definite.
    assert got.detail["hessian_pd"] == "False"
    assert "min_eigenvalue" in got.detail and "optimizer" in got.detail
    est = find_mode(hierarchical(), data)
    # Two routes to the same verdict, and which one you get is a property of the
    # install rather than of the model: with jax, find_mode returns a mode it
    # knows did not converge; without it the finite-difference curvature check
    # cannot resolve a_sd, so find_mode declines to vouch for a mode at all.
    # "There is no mode here" is the invariant under test either way.
    assert isinstance(est, Unverified) or not est.converged
    forced = as_posterior(laplace(hierarchical(), data, draws=200, seed=0, allow_unverified=True))
    assert forced.provenance["verified"] is False
    assert forced.provenance["hessian_pd"] is False
    assert forced.provenance["nonfinite_draw_frac"] == 0.0
    for name in forced.names():
        assert np.all(np.isfinite(forced.draws(name))), name


def test_vector_parameters_flatten_and_restore(hier_data: dict[str, np.ndarray]) -> None:
    model = hierarchical()
    assert flat_layout(model) == (("a_mean", ()), ("a_sd", ()), ("alpha", (3,)), ("sigma", ()))
    est = find_mode(model, hier_data)
    assert est.converged
    assert isinstance(est.theta["alpha"], tuple)
    assert len(est.theta["alpha"]) == 3
    assert est.theta["a_sd"] > 0


# -- derivatives ---------------------------------------------------------------------


@pytest.mark.skipif(not jax_available(), reason="jax is not installed")
def test_jax_and_finite_difference_hessians_agree(hier_data: dict[str, np.ndarray]) -> None:
    model = hierarchical()
    theta = {"a_mean": 2.0, "a_sd": 1.2, "alpha": np.array([1.1, 2.4, 3.9]), "sigma": 0.6}
    z = unconstrain(model, theta)
    H_jax = hessian_at(model, hier_data, z, derivatives="jax")
    H_fd = hessian_at(model, hier_data, z, derivatives="finite_difference")
    assert not isinstance(H_jax, Unverified)
    assert not isinstance(H_fd, Unverified), H_fd.reason
    assert H_jax.shape == (6, 6)
    assert np.allclose(H_jax, H_jax.T)
    assert np.allclose(H_jax, H_fd, atol=1e-4, rtol=1e-4)


def test_finite_difference_path_runs_without_jax(y50: np.ndarray) -> None:
    post = as_posterior(
        laplace(normal_normal(), {"y": y50}, draws=4000, seed=0, derivatives="finite_difference")
    )
    mean, sd = closed_form(y50)
    assert post.provenance["derivatives"] == "finite_difference"
    s = post.summary("mu")
    assert abs(s.mean - mean) / abs(mean) < 0.02
    assert abs(s.sd - sd) / sd < 0.03


def test_finite_difference_big_scale_at_the_zero_mode_is_the_closed_form_or_unverified() -> None:
    """mu ~ N(0, 1e8), y ~ N(mu, 1e5), sum(y) = 0: the mode is exactly z = 0.

    With a unit step floor the second difference at z = 0 was round-off
    (``-1.9e-6`` for a true ``5e-10``): the polish called it a saddle,
    the restarts landed on noise of the other sign, and ``laplace`` returned
    a *verified* Gaussian with sd 1022 against a closed form of 44721. The
    step is now scaled by the prior and checked against its round-off floor:
    the result is either the closed form or ``Unverified``, never a verified
    wrong number.
    """
    y = np.array([-1e5, -5e4, 0.0, 5e4, 1e5])
    mean, sd = closed_form(y, prior_sd=1e8, noise_sd=1e5)
    assert mean == 0.0
    H = hessian_at(big_scale(), {"y": y}, {"mu": 0.0}, derivatives="finite_difference")
    if not isinstance(H, Unverified):
        assert H[0, 0] == pytest.approx(1.0 / sd**2, rel=1e-3)
    est = find_mode(big_scale(), {"y": y}, derivatives="finite_difference")
    if not isinstance(est, Unverified):
        assert est.converged and est.hessian_pd is True
        assert abs(float(est.theta["mu"])) < 1e-3 * sd
    got = laplace(big_scale(), {"y": y}, draws=4000, seed=0, derivatives="finite_difference")
    if isinstance(got, Unverified):
        assert "finite-difference curvature unresolved" in got.reason
        assert got.detail["unresolved"] == "mu"
        return
    prov = got.provenance
    assert prov["verified"] is True and prov["converged"] is True and prov["hessian_pd"] is True
    assert prov["unresolved_curvature"] == []
    assert prov["min_eigenvalue"] == pytest.approx(1.0 / sd**2, rel=1e-3)
    s = got.summary("mu")
    assert abs(s.sd - sd) / sd < 0.05
    assert abs(s.mean - mean) / sd < 0.05


def test_finite_difference_no_spurious_saddle_restarts(caplog: pytest.LogCaptureFixture) -> None:
    """n = 1, y = 3e5 on the 1e8-scale prior: the unit-floor step read the 1e-10 curvature at
    ``z = 0`` as a saddle and burned every restart of every optimizer (six WARNINGs) before
    BFGS happened to land near the answer."""
    y = np.array([3e5])
    mean, sd = closed_form(y, prior_sd=1e8, noise_sd=1e5)
    with caplog.at_level(logging.WARNING, logger="axiom.infer.laplace"):
        est = as_estimate(find_mode(big_scale(), {"y": y}, derivatives="finite_difference"))
    saddles = [r for r in caplog.records if "saddle" in r.getMessage()]
    assert saddles == []
    assert est.converged and est.hessian_pd is True
    assert est.method == "trust-ncg"
    assert abs(float(est.theta["mu"]) - mean) / sd < 1e-6
    assert est.min_eigenvalue == pytest.approx(1.0 / sd**2, rel=1e-3)


def test_unresolved_finite_difference_curvature_is_unverified_not_a_number() -> None:
    """A step of ``1.2e-4 · 1e8`` saturates ``tanh``: the step-doubling check fails, and every
    entry point says so instead of returning the noise."""
    model = tanh_wide_prior()
    y = np.array([0.5, 0.4, 0.6])
    got = find_mode(model, {"y": y}, derivatives="finite_difference")
    assert isinstance(got, Unverified)
    assert "finite-difference curvature unresolved for b" in got.reason
    assert "install jax or rescale" in got.reason
    assert got.detail["unresolved"] == "b"
    assert got.detail["min_eigenvalue"] == "unresolved"
    assert got.detail["derivatives"] == "finite_difference"
    forced = as_estimate(
        find_mode(model, {"y": y}, derivatives="finite_difference", allow_unverified=True)
    )
    assert forced.converged is False
    assert forced.hessian_pd is False
    assert forced.min_eigenvalue is None and forced.newton_decrement is None
    H = hessian_at(model, {"y": y}, {"b": 0.5}, derivatives="finite_difference")
    assert isinstance(H, Unverified)
    assert H.detail["unresolved"] == "b"
    post = laplace(model, {"y": y}, draws=50, seed=0, derivatives="finite_difference")
    assert isinstance(post, Unverified)
    assert "finite-difference curvature unresolved for b" in post.reason
    forced_post = as_posterior(
        laplace(
            model,
            {"y": y},
            draws=50,
            seed=0,
            derivatives="finite_difference",
            allow_unverified=True,
        )
    )
    prov = forced_post.provenance
    assert prov["verified"] is False and prov["hessian_pd"] is False and prov["converged"] is False
    assert prov["unresolved_curvature"] == ["b"]
    assert prov["min_eigenvalue"] is None and prov["newton_decrement"] is None
    if jax_available():
        est = as_estimate(find_mode(model, {"y": y}, derivatives="jax"))
        assert est.converged and est.hessian_pd is True
        assert abs(float(est.theta["b"]) - np.arctanh(0.5)) < 0.02


def test_finite_difference_scales_come_from_the_priors() -> None:
    """One scale per flat coordinate, in unconstrained units, read off each prior."""
    from axiom.infer.laplace import _scales, flat_layout

    model = hierarchical()
    # a_mean ~ N(0, 5): 5; a_sd ~ HalfNormal: 1 (log scale); alpha ~ N(a_mean, a_sd): the
    # parent's typical size, 2; sigma ~ HalfNormal: 1
    assert _scales(model, flat_layout(model)).tolist() == [5.0, 1.0, 2.0, 2.0, 2.0, 1.0]

    rate = Param(
        name="rate", dimension=OUT, prior=Prior(family="gamma", hyper={"alpha": 4.0, "beta": 2.0})
    )
    frac = Param(
        name="frac",
        dimension=dimensionless(),
        prior=Prior(family="beta", hyper={"alpha": 2.0, "beta": 2.0}),
    )
    lk = Param(
        name="lk", dimension=OUT, prior=Prior(family="lognormal", hyper={"mu": 0.0, "sigma": 0.3})
    )
    lk_child = Param(
        name="lk_child",
        dimension=OUT,
        prior=Prior(family="lognormal", hyper={"mu": 0.0, "sigma": "rate"}),
    )
    box = Param(
        name="box", dimension=OUT, prior=Prior(family="uniform", hyper={"low": -5.0, "high": 5.0})
    )
    sigma = Param(name="sigma", dimension=OUT, prior=Prior(family="fixed", hyper={"value": 1.0}))
    model = ModelSpec(
        name="families",
        mean=rate,
        outcome=Y,
        likelihood=Likelihood(family="normal", scale="sigma"),
        parameters=(rate, frac, lk, lk_child, box, sigma),
    )
    got = _scales(model, flat_layout(model))
    # gamma: 1/sqrt(alpha); beta: 1; lognormal: sigma; sigma naming a gamma parent: its mean
    # alpha/beta; uniform: 1
    assert got.tolist() == pytest.approx([0.5, 1.0, 0.3, 2.0, 1.0])


# -- helpers ---------------------------------------------------------------------------


def test_constrain_draws_applies_the_core_transform() -> None:
    model = normal_halfnormal()
    z = {"mu": np.array([0.0, 1.0, -2.0]), "sigma": np.array([0.0, np.log(2.0), np.log(0.5)])}
    theta = constrain_draws(model, z)
    assert np.allclose(theta["mu"], [0.0, 1.0, -2.0])
    assert np.allclose(theta["sigma"], [1.0, 2.0, 0.5])


def test_missing_data_column_is_a_key_error(y50: np.ndarray) -> None:
    with pytest.raises(KeyError, match="unit_index"):
        laplace(hierarchical(), {"y": y50}, draws=10, seed=0)


# -- backend ---------------------------------------------------------------------------


def test_laplace_backend_satisfies_protocol(y50: np.ndarray) -> None:
    backend = get_backend("laplace")
    assert isinstance(backend, LaplaceBackend)
    assert isinstance(backend, Backend)
    assert backend.name == "laplace"
    post = as_posterior(
        backend.sample(normal_normal(), {"y": y50}, draws=300, tune=50, chains=2, seed=5)
    )
    assert post.draws("mu").shape == (2, 300)
    prov = post.provenance
    assert prov["note"].startswith("laplace draws, not MCMC")
    assert (prov["draws"], prov["tune"], prov["chains"], prov["seed"]) == (300, 50, 2, 5)
    est = backend.optimize(normal_normal(), {"y": y50}, seed=None)
    assert est.converged
    post2 = as_posterior(backend.laplace(normal_normal(), {"y": y50}, draws=100, seed=1))
    assert post2.draws("mu").shape == (1, 100)


def test_laplace_backend_passes_unverified_through(y50: np.ndarray) -> None:
    backend = LaplaceBackend()
    assert isinstance(
        backend.sample(collinear(), {"y": y50}, draws=20, tune=0, chains=1, seed=0), Unverified
    )
    assert isinstance(backend.laplace(collinear(), {"y": y50}, draws=20, seed=0), Unverified)
    forced = backend.laplace(collinear(), {"y": y50}, draws=20, seed=0, allow_unverified=True)
    assert isinstance(forced, Posterior)
    assert forced.provenance["hessian_pd"] is False


def test_get_backend_unknown_is_unsupported() -> None:
    got = get_backend("stan")
    assert isinstance(got, Unsupported)
    assert "stan" in got.reason


def test_sample_settings_defaults_and_validation() -> None:
    s = SampleSettings()
    assert (s.draws, s.tune, s.chains, s.target_accept) == (1000, 1000, 4, 0.9)
    with pytest.raises(ValueError):
        SampleSettings(chains=0)
    with pytest.raises(ValueError):
        SampleSettings(target_accept=1.0)
    assert SampleSettings.from_json(s.to_json()) == s
