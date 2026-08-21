"""Prior→posterior learning and the surface prior predictive.

Both modules draw from the prior through ``axiom.diagnose.sbc.draw_prior``.
When that module is not yet importable (it is written concurrently) the
``prior_sampler`` fixture installs a stand-in built on
``axiom.sim.true_parameters(mode="prior")`` with the same signature.
"""

from __future__ import annotations

import importlib
import sys
import types
from collections.abc import Mapping

import numpy as np
import pytest

from axiom.core import (
    Add,
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
from axiom.diagnose.learning import LearningReport, bhattacharyya, learning
from axiom.diagnose.surface_prior import PriorPredictive, prior_predictive
from axiom.sim import DosePlan, surface_world, true_parameters
from axiom.surface import fit

NONE = dimensionless()
FALLBACK = "fallback:true_parameters"


def _fallback_draw_prior(
    model: ModelSpec, rng: np.random.Generator, *, n: int
) -> Mapping[str, np.ndarray]:
    seeds = rng.integers(0, 2**31 - 1, size=n)
    rows = [true_parameters(model, mode="prior", seed=int(s)) for s in seeds]
    return {k: np.stack([r[k] for r in rows]) for k in rows[0]}


@pytest.fixture(scope="module")
def prior_sampler() -> str:
    try:
        mod = importlib.import_module("axiom.diagnose.sbc")
        if hasattr(mod, "draw_prior"):
            return "axiom.diagnose.sbc.draw_prior"
    except ImportError:
        pass
    shim = types.ModuleType("axiom.diagnose.sbc")
    shim.draw_prior = _fallback_draw_prior  # type: ignore[attr-defined]
    sys.modules["axiom.diagnose.sbc"] = shim
    return FALLBACK


def _model() -> ModelSpec:
    x = Data(name="x", dimension=NONE)
    a = Param(name="a", dimension=NONE, prior=Prior(family="normal", hyper={"mu": 0, "sigma": 2}))
    b = Param(name="b", dimension=NONE, prior=Prior(family="normal", hyper={"mu": 0, "sigma": 2}))
    s = Param(name="sigma", dimension=NONE, prior=Prior(family="halfnormal", hyper={"sigma": 1}))
    return ModelSpec(
        name="toy",
        mean=Add(terms=(a, Mul(factors=(b, x)))),
        outcome=Data(name="y", dimension=NONE),
        likelihood=Likelihood(family="normal", scale="sigma"),
        parameters=(a, b, s),
    )


def test_bhattacharyya_closed_form() -> None:
    assert bhattacharyya(0, 1, 0, 1) == 1.0
    assert bhattacharyya(0, 1, 10, 1) == pytest.approx(np.exp(-100 / 8))
    assert bhattacharyya(0, 1, 0, 2) == pytest.approx(np.sqrt(4 / 5))
    with pytest.raises(ValueError):
        bhattacharyya(0, 0, 0, 1)


def test_learning_contraction_overlap_shift(prior_sampler: str) -> None:
    rng = np.random.default_rng(0)
    n = 4000
    post = Posterior(
        {
            "a": rng.normal(1.0, 0.2, (1, n)),  # learned: sd 0.2 vs prior 2
            "b": rng.normal(0.0, 2.0, (1, n)),  # untouched: prior-dominated
            "sigma": np.abs(rng.normal(0, 1, (1, n))),
        }
    )
    rep = learning(_model(), post, n_prior=20_000, seed=1, threshold=0.1)
    assert isinstance(rep, LearningReport)
    a, b = rep.get("a"), rep.get("b")
    assert a.contraction == pytest.approx(1 - 0.2**2 / 4, abs=0.02)
    assert a.shift == pytest.approx(0.5, abs=0.05)
    assert a.overlap < 0.5
    assert b.contraction == pytest.approx(0.0, abs=0.06)
    assert b.overlap > 0.97
    assert b.prior_dominated and not a.prior_dominated
    assert rep.prior_dominated == ("b", "sigma")
    assert not rep.passed
    assert rep.n_prior == 20_000 and rep.n_posterior == n and rep.seed == 1
    assert LearningReport.from_json(rep.to_json()) == rep


def test_learning_subset_and_errors(prior_sampler: str) -> None:
    post = Posterior({"a": np.random.default_rng(2).normal(size=(1, 50))})
    rep = learning(_model(), post, n_prior=500)
    assert isinstance(rep, LearningReport) and [p.name for p in rep.parameters] == ["a"]
    with pytest.raises(KeyError):
        learning(_model(), post, parameters=("b",), n_prior=500)
    with pytest.raises(ValueError):
        learning(_model(), post, n_prior=1)
    const = Posterior({"a": np.zeros((1, 50))})
    assert isinstance(learning(_model(), const, n_prior=500), Unsupported)


@pytest.fixture(scope="module")
def world():  # type: ignore[no-untyped-def]
    return surface_world(
        n_units=3,
        n_periods=12,
        treatments=("a", "b"),
        doses=DosePlan(scale=50.0, zero_fraction=0.1),
        noise_sd=0.3,
        seed=3,
    )


@pytest.mark.slow
def test_learning_on_a_fit(prior_sampler: str, world) -> None:  # type: ignore[no-untyped-def]
    res = fit(world.spec, world.panel, backend="laplace", draws=200, chains=1, seed=5)
    rep = learning(res.surface.model, res, n_prior=2000, seed=0)
    assert isinstance(rep, LearningReport)
    names = {p.name.split("[")[0] for p in rep.parameters}
    assert names <= {p.name for p in res.surface.model.free}
    assert all(-5 < p.contraction <= 1 for p in rep.parameters)


def test_prior_predictive_goes_through_forward(prior_sampler: str, world) -> None:  # type: ignore[no-untyped-def]
    pp = prior_predictive(world.spec, world.panel, n=60, seed=4, magnitude_factor=10.0)
    assert isinstance(pp, PriorPredictive)
    assert pp.treatments == ("a", "b")
    assert pp.n == 60 and pp.n_units == 3 and pp.n_periods == 12
    assert pp.spec_hash == world.spec.content_hash()
    assert set(pp.contribution) == {"a", "b"}
    assert pp.response_range.definition == "eti" and pp.response_range.mass == 0.9
    assert 0.0 <= pp.share_flagged <= 1.0
    assert pp.passed == (pp.share_flagged <= pp.tolerance)
    assert PriorPredictive.from_json(pp.to_json()) == pp
    # a Hill surface with a half-normal amplitude never contributes negatively
    assert pp.share_wrong_sign["a"] == 0.0
    strict = prior_predictive(world.spec, world.panel, n=60, seed=4, magnitude_factor=1e-9)
    assert isinstance(strict, PriorPredictive)
    assert strict.share_implausible_magnitude["a"] == 1.0 and not strict.passed
    with pytest.raises(ValueError):
        prior_predictive(world.spec, world.panel, n=1)


def test_which_prior_sampler_was_used(prior_sampler: str) -> None:
    # Records which sampler the run used; the report to the integrator states it.
    assert prior_sampler in ("axiom.diagnose.sbc.draw_prior", FALLBACK)
