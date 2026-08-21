"""Posterior-geometry weak identification: ridge pairs, condition number, sd ratios, bounds."""

from __future__ import annotations

import numpy as np
import pytest

from axiom.core import (
    Add,
    Const,
    Data,
    Likelihood,
    ModelSpec,
    Mul,
    Param,
    Posterior,
    Prior,
    Unsupported,
    dimensionless,
)
from axiom.diagnose.weak_id import WeakIdReport, prior_moments, weak_identification
from axiom.sim import DosePlan, surface_world
from axiom.surface import FitResult, fit

NONE = dimensionless()


def _toy_model() -> ModelSpec:
    x = Data(name="x", dimension=NONE)
    a = Param(name="a", dimension=NONE, prior=Prior(family="normal", hyper={"mu": 0, "sigma": 2}))
    b = Param(name="b", dimension=NONE, prior=Prior(family="normal", hyper={"mu": 0, "sigma": 2}))
    r = Param(name="r", dimension=NONE, prior=Prior(family="beta", hyper={"alpha": 2, "beta": 2}))
    s = Param(name="sigma", dimension=NONE, prior=Prior(family="halfnormal", hyper={"sigma": 1}))
    mean = Add(terms=(a, Mul(factors=(b, x)), Mul(factors=(r, Const(value=0.0, dimension=NONE)))))
    return ModelSpec(
        name="toy",
        mean=mean,
        outcome=Data(name="y", dimension=NONE),
        likelihood=Likelihood(family="normal", scale="sigma"),
        parameters=(a, b, r, s),
    )


def test_prior_moments_closed_forms() -> None:
    assert prior_moments(Prior(family="normal", hyper={"mu": 1, "sigma": 3})) == (1.0, 3.0)
    m, s = prior_moments(Prior(family="lognormal", hyper={"mu": 0, "sigma": 0.5}))
    assert m == pytest.approx(np.exp(0.125)) and s == pytest.approx(
        np.sqrt((np.exp(0.25) - 1) * np.exp(0.25))
    )
    assert prior_moments(Prior(family="uniform", hyper={"low": 0, "high": 1}))[1] == pytest.approx(
        1 / np.sqrt(12)
    )
    assert prior_moments(Prior(family="gamma", hyper={"alpha": 4, "beta": 2})) == (2.0, 1.0)
    assert prior_moments(Prior(family="normal", hyper={"mu": "m", "sigma": 1}), {"m": 3.0}) == (
        3.0,
        1.0,
    )


def test_ridge_pair_condition_number_and_unlearned() -> None:
    rng = np.random.default_rng(0)
    n = 4000
    a = rng.normal(0.0, 0.2, n)
    b = a * 0.99 + rng.normal(0.0, 0.02, n)  # the ridge
    r = rng.beta(2, 2, n)  # posterior equals prior: unlearned
    post = Posterior({"a": a[None], "b": b[None], "r": r[None], "sigma": np.abs(a)[None] + 0.1})
    rep = weak_identification(post, _toy_model(), rho_threshold=0.9)
    assert isinstance(rep, WeakIdReport)
    assert rep.parameters == ("a", "b", "r", "sigma")
    assert [p[:2] for p in rep.high_pairs] == [("a", "b")]
    assert rep.high_pairs[0][2] > 0.98
    assert rep.condition_number > 50
    assert rep.sd_ratio["a"] == pytest.approx(0.1, rel=0.1)
    assert rep.sd_ratio["r"] == pytest.approx(1.0, rel=0.1)
    assert "r" in rep.unlearned and "a" not in rep.unlearned
    assert rep.saturated == ()
    assert not rep.passed
    assert WeakIdReport.from_json(rep.to_json()) == rep


def test_saturated_bounded_parameter_is_flagged() -> None:
    rng = np.random.default_rng(1)
    n = 1000
    r = np.where(rng.uniform(size=n) < 0.5, 1.0 - 1e-6, rng.beta(2, 2, n))
    post = Posterior(
        {
            "a": rng.normal(size=(1, n)),
            "b": rng.normal(size=(1, n)),
            "r": r[None],
            "sigma": rng.gamma(2, 1, size=(1, n)),
        }
    )
    rep = weak_identification(post, _toy_model(), parameters=("r",))
    assert isinstance(rep, WeakIdReport)
    assert rep.saturated == ("r",)
    assert not rep.passed


def test_validation_and_failures() -> None:
    post = Posterior({"a": np.zeros((1, 10)), "b": np.arange(10.0)[None]})
    with pytest.raises(ValueError):
        weak_identification(post)
    out = weak_identification(post, _toy_model(), parameters=("a", "b"))
    assert isinstance(out, Unsupported) and "constant" in out.reason
    with pytest.raises(ValueError):
        weak_identification(post, _toy_model(), parameters=("zzz",))


@pytest.mark.slow
def test_fit_result_uses_structural_parameters() -> None:
    world = surface_world(
        n_units=3,
        n_periods=12,
        treatments=("a",),
        doses=DosePlan(scale=50.0, zero_fraction=0.1),
        noise_sd=0.3,
        seed=3,
    )
    res: FitResult = fit(world.spec, world.panel, backend="laplace", draws=200, chains=1, seed=5)
    rep = weak_identification(res)
    assert isinstance(rep, WeakIdReport)
    structural = set(res.surface.linear) | set(res.surface.nonlinear)
    declared = {p.split("[")[0] for p in rep.parameters}
    assert declared <= structural
    assert "sigma" not in declared
    assert rep.n_draws == res.n_draws()
    assert all(0.0 < v for v in rep.sd_ratio.values())
