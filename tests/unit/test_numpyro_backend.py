"""NumPyro backend (NUTS over the jax log density) and the arviz round trip.

Every test that touches numpyro or arviz skips when the extra is missing.
The sampling tests are tiny (1-2 chains x 100-200 draws, about a second each
including jit compilation).
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from axiom.core import (
    D,
    Data,
    Gather,
    Likelihood,
    ModelSpec,
    Param,
    Posterior,
    Prior,
    Unsupported,
    Unverified,
    dimensionless,
)
from axiom.infer import _arviz
from axiom.infer import numpyro_backend as nb
from axiom.infer.backend import Backend, PointEstimate, get_backend
from axiom.infer.diagnostics import diagnose, mcse_mean

OUT = D.outcome
Y = Data(name="y", dimension=OUT)


def normal_normal(prior_sd: float = 10.0) -> ModelSpec:
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


def normal_halfnormal() -> ModelSpec:
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


def uniform_fixed_bound() -> ModelSpec:
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
    return 3.0 + np.random.default_rng(1).normal(0.0, 1.0, 50)


@pytest.fixture
def hier_data() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(7)
    unit = np.repeat(np.arange(3), 12)
    y = np.array([1.0, 2.5, 4.0])[unit] + rng.normal(0.0, 0.5, unit.size)
    return {"y": y, "unit_index": unit}


def test_available_and_resolution() -> None:
    got = get_backend("numpyro")
    if nb.available():
        assert isinstance(got, nb.NumpyroBackend)
        assert isinstance(got, Backend)
        assert got.name == "numpyro"
        assert nb.missing() == ()
    else:
        assert isinstance(got, Unsupported)
        assert got.missing


needs_numpyro = pytest.mark.skipif(not nb.available(), reason="numpyro is not installed")


@needs_numpyro
def test_normal_normal_posterior_mean(y50: np.ndarray) -> None:
    post = nb.sample(normal_normal(), {"y": y50}, draws=200, tune=200, chains=2, seed=42)
    assert isinstance(post, Posterior)
    assert post.draws("mu").shape == (2, 200)
    prec = 1.0 / 100.0 + y50.size
    closed = y50.sum() / prec
    draws = post.draws("mu")
    assert abs(draws.mean() - closed) < 3 * mcse_mean(draws)
    prov = post.provenance
    assert prov["seed"] == 42
    assert prov["divergences"] == 0
    assert prov["backend"] == "numpyro"
    assert prov["method"] == "nuts"
    assert (prov["draws"], prov["tune"], prov["chains"]) == (200, 200, 2)
    assert 0.5 < prov["accept_prob"] <= 1.0
    assert prov["init"] == "laplace_mode"
    assert "init_note" not in prov
    assert prov["max_tree_depth_hits"] == 0
    assert prov["mean_num_steps"] > 0
    assert prov["nonfinite_draw_frac"] == 0.0
    assert "sigma" not in post.names()
    assert prov["fixed"] == {"sigma": 1.0}
    report = diagnose(post, ess_min=50)
    assert report.converged, report.failing


@needs_numpyro
def test_positive_parameter_is_constrained(y50: np.ndarray) -> None:
    backend = nb.NumpyroBackend()
    post = backend.sample(normal_halfnormal(), {"y": y50}, draws=200, tune=200, chains=2, seed=0)
    sigma = post.draws("sigma")
    assert np.all(sigma > 0)
    assert 0.6 < sigma.mean() < 1.4
    assert post.provenance["divergences"] == 0


@needs_numpyro
def test_vector_parameter_with_one_chain(hier_data: dict[str, np.ndarray]) -> None:
    post = nb.sample(hierarchical(), hier_data, draws=150, tune=150, chains=1, seed=5)
    assert post.draws("alpha").shape == (1, 150, 3)
    assert post.draws("a_sd").shape == (1, 150)
    assert post.coords() == {"alpha_dim0": [0, 1, 2]}
    assert post.provenance["init"] == "laplace_mode"
    assert np.all(post.draws("a_sd") > 0)
    means = post.flat("alpha").mean(axis=0)
    assert np.allclose(means, [1.0, 2.5, 4.0], atol=0.6)
    assert post.provenance["nonfinite_draw_frac"] == 0.0


@needs_numpyro
def test_seed_none_is_drawn_and_recorded(y50: np.ndarray) -> None:
    post = nb.sample(normal_normal(), {"y": y50}, draws=50, tune=50, chains=1, seed=None)
    seed = post.provenance["seed"]
    assert isinstance(seed, int)
    again = nb.sample(normal_normal(), {"y": y50}, draws=50, tune=50, chains=1, seed=seed)
    assert np.array_equal(again.draws("mu"), post.draws("mu"))


@needs_numpyro
def test_tree_depth_hits_are_recorded(y50: np.ndarray, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="axiom.infer.numpyro_backend"):
        post = nb.sample(
            normal_normal(), {"y": y50}, draws=100, tune=100, chains=1, seed=0, max_tree_depth=2
        )
    prov = post.provenance
    assert prov["max_tree_depth"] == 2
    assert prov["max_tree_depth_hits"] > 0
    assert prov["mean_num_steps"] <= 3.0  # a depth-2 tree holds at most 2^2 - 1 leapfrog steps
    assert any("max_tree_depth" in r.getMessage() for r in caplog.records)


@needs_numpyro
def test_init_falls_back_to_zeros_when_the_mode_is_unverified() -> None:
    """Identical units (a funnel): the Laplace mode search does not converge; the chains must
    start from zeros, not from wherever the optimizer gave up in the neck."""
    rng = np.random.default_rng(3)
    unit = np.repeat(np.arange(3), 12)
    data = {"y": 2.0 + rng.normal(0.0, 0.5, unit.size), "unit_index": unit}
    z, source, note = nb._init_z(hierarchical(), data, None, from_mode=True, seed=5)
    assert source == "zeros"
    assert note is not None and "mode search" in note
    assert set(z) == {"a_mean", "a_sd", "alpha", "sigma"}
    for name, v in z.items():
        assert np.all(v == 0.0), name
    assert z["alpha"].shape == (3,)
    post = nb.sample(hierarchical(), data, draws=30, tune=30, chains=1, seed=5)
    assert post.provenance["init"] == "zeros"
    assert "mode search" in post.provenance["init_note"]


@needs_numpyro
def test_init_ignores_a_converged_mode_whose_hessian_is_not_pd(
    y50: np.ndarray, caplog: pytest.LogCaptureFixture
) -> None:
    """Collinear a + b: the mode search converges (the identified direction does) but the
    Hessian is not PD. ``converged`` alone used to start the chains there; only a *verified*
    mode (converged and PD) is a start."""
    from axiom.core import Add

    flat = Prior(family="normal", hyper={"mu": 0.0, "sigma": 1e8})
    a = Param(name="a", dimension=OUT, prior=flat)
    b = Param(name="b", dimension=OUT, prior=flat)
    sigma = Param(name="sigma", dimension=OUT, prior=Prior(family="fixed", hyper={"value": 1.0}))
    model = ModelSpec(
        name="collinear",
        mean=Add(terms=(a, b)),
        outcome=Y,
        likelihood=Likelihood(family="normal", scale="sigma"),
        parameters=(a, b, sigma),
    )
    est = nb.NumpyroBackend().optimize(model, {"y": y50}, seed=0)
    assert est.converged and est.hessian_pd is False  # the case the gate must catch
    with caplog.at_level(logging.WARNING, logger="axiom.infer.numpyro_backend"):
        z, source, note = nb._init_z(model, {"y": y50}, None, from_mode=True, seed=0)
    assert source == "zeros"
    assert note is not None and "not positive definite" in note
    assert z["a"] == 0.0 and z["b"] == 0.0
    assert any("initializing chains at zero" in r.getMessage() for r in caplog.records)


@needs_numpyro
def test_init_z_handles_a_fixed_interval_bound(y50: np.ndarray) -> None:
    z, source, note = nb._init_z(
        uniform_fixed_bound(), {"y": y50}, {"mu": 1.0}, from_mode=False, seed=0
    )
    assert source == "init" and note is None
    assert z["mu"] == pytest.approx(np.log(0.6 / 0.4))
    z, source, note = nb._init_z(uniform_fixed_bound(), {"y": y50}, None, from_mode=True, seed=0)
    assert source == "laplace_mode" and note is None
    mu = 5.0 * np.tanh(float(z["mu"]) / 2.0)  # inverse of the logit on (-5, 5)
    assert abs(mu - y50.mean()) < 0.05


@needs_numpyro
def test_optimize_delegates_to_laplace(y50: np.ndarray) -> None:
    est = nb.NumpyroBackend().optimize(normal_normal(), {"y": y50}, seed=None)
    assert isinstance(est, PointEstimate)
    assert est.converged
    prec = 1.0 / 100.0 + y50.size
    assert abs(float(est.theta["mu"]) - y50.sum() / prec) < 1e-6


@needs_numpyro
def test_laplace_delegates(y50: np.ndarray) -> None:
    post = nb.NumpyroBackend().laplace(normal_normal(), {"y": y50}, draws=100, seed=1)
    assert isinstance(post, Posterior)
    assert post.provenance["method"] == "laplace"
    assert post.provenance["derivatives"] == "jax"


@needs_numpyro
def test_laplace_delegates_unverified(y50: np.ndarray) -> None:
    from axiom.core import Add

    flat = Prior(family="normal", hyper={"mu": 0.0, "sigma": 1e8})
    a = Param(name="a", dimension=OUT, prior=flat)
    b = Param(name="b", dimension=OUT, prior=flat)
    sigma = Param(name="sigma", dimension=OUT, prior=Prior(family="fixed", hyper={"value": 1.0}))
    model = ModelSpec(
        name="collinear",
        mean=Add(terms=(a, b)),
        outcome=Y,
        likelihood=Likelihood(family="normal", scale="sigma"),
        parameters=(a, b, sigma),
    )
    backend = nb.NumpyroBackend()
    assert isinstance(backend.laplace(model, {"y": y50}, draws=20, seed=1), Unverified)
    forced = backend.laplace(model, {"y": y50}, draws=20, seed=1, allow_unverified=True)
    assert isinstance(forced, Posterior)
    assert forced.provenance["hessian_pd"] is False


@needs_numpyro
def test_bad_sizes_and_settings_rejected(y50: np.ndarray) -> None:
    with pytest.raises(ValueError):
        nb.sample(normal_normal(), {"y": y50}, draws=0, tune=10, chains=1, seed=0)
    with pytest.raises(ValueError, match="target_accept"):
        nb.sample(
            normal_normal(), {"y": y50}, draws=10, tune=10, chains=1, seed=0, target_accept=1.0
        )
    with pytest.raises(ValueError, match="max_tree_depth"):
        nb.sample(
            normal_normal(), {"y": y50}, draws=10, tune=10, chains=1, seed=0, max_tree_depth=0
        )
    with pytest.raises(ValueError, match="target_accept"):
        nb.NumpyroBackend(target_accept=0.0)
    with pytest.raises(ValueError, match="max_tree_depth"):
        nb.NumpyroBackend(max_tree_depth=0)


# -- arviz ------------------------------------------------------------------------------


def _arviz_available() -> bool:
    for mod in ("arviz_base", "arviz"):
        try:
            __import__(mod)
        except ImportError:
            continue
        return True
    return False


@pytest.fixture
def hand_built() -> Posterior:
    rng = np.random.default_rng(9)
    return Posterior(
        {"mu": rng.standard_normal((2, 50)), "alpha": rng.standard_normal((2, 50, 3))},
        coords={"alpha_dim0": [0, 1, 2]},
        provenance={
            "seed": np.int64(9),
            "n_iter": np.int32(3),
            "backend": "hand",
            "hessian_pd": True,
            "converged": False,
            "nonfinite_draw_frac": 0.0,
            "fixed": {"sigma": 1.0},
            "note": None,
            "message": "null",
        },
    )


def test_to_inference_data_without_arviz_is_unsupported(hand_built: Posterior) -> None:
    if _arviz_available():
        pytest.skip("arviz is installed")
    got = _arviz.to_inference_data(hand_built)
    assert isinstance(got, Unsupported)


@pytest.mark.skipif(not _arviz_available(), reason="arviz is not installed")
def test_arviz_round_trip(hand_built: Posterior) -> None:
    idata = _arviz.to_inference_data(hand_built)
    assert not isinstance(idata, Unsupported)
    back = _arviz.from_inference_data(idata)
    assert back == hand_built
    prov = back.provenance
    assert prov["hessian_pd"] is True
    assert prov["converged"] is False
    assert prov["note"] is None
    assert prov["message"] == "null"  # a string that happens to say "null" stays a string
    assert prov["seed"] == 9 and type(prov["seed"]) is int
    assert prov["n_iter"] == 3 and type(prov["n_iter"]) is int
    assert prov["fixed"] == {"sigma": 1.0}
    assert back.coords() == {"alpha_dim0": [0, 1, 2]}
    attrs = _arviz._posterior_dataset(idata).attrs
    assert "provenance_json" in attrs
    assert attrs["hessian_pd"] == 1  # the flattened, human-readable copy is still there


@pytest.mark.skipif(not _arviz_available(), reason="arviz is not installed")
def test_from_inference_data_decodes_flattened_attrs(hand_built: Posterior) -> None:
    """Objects written by other tools have no ``provenance_json``; fall back to the flat attrs."""
    idata = _arviz.to_inference_data(hand_built)
    ds = _arviz._posterior_dataset(idata)
    del ds.attrs["provenance_json"]
    back = _arviz.from_inference_data(ds)
    prov = back.provenance
    assert prov["hessian_pd"] is True
    assert prov["note"] is None
    assert prov["fixed"] == {"sigma": 1.0}
