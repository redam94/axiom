"""Unit tests for ``axiom.diagnose.sbc``: prior draws, rank statistics, the loop, and the
negative control."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from axiom.core import (
    Add,
    Data,
    Likelihood,
    ModelSpec,
    Mul,
    Param,
    Prior,
    Unsupported,
    dimensionless,
)
from axiom.diagnose.sbc import (
    ParameterRanks,
    SBCResult,
    SBCSpec,
    default_bins,
    draw_prior,
    rank_uniformity,
    sbc,
    sbc_pool,
    sbc_surface,
    simulate_outcome,
)
from axiom.meta import PoolPriors, PoolSpec
from axiom.sim import DosePlan, surface_world
from axiom.surface import LinearKernel

DL = dimensionless()
AMP = Prior(family="lognormal", hyper={"mu": 0.0, "sigma": 0.5})


def _linear_model(
    *, sigma_fixed: float | None = 0.5, beta_mu: float = 0.0, beta_sd: float = 1.0
) -> ModelSpec:
    """``y = alpha + beta x`` with a known noise sd: the Laplace posterior is exact here."""
    sigma_prior = (
        Prior(family="fixed", hyper={"value": sigma_fixed})
        if sigma_fixed is not None
        else Prior(family="halfnormal", hyper={"sigma": 1.0})
    )
    return ModelSpec(
        name="line",
        mean=Add(
            terms=(
                Param(
                    name="alpha",
                    dimension=DL,
                    prior=Prior(family="normal", hyper={"mu": 0.0, "sigma": 1.0}),
                ),
                Mul(
                    factors=(
                        Param(
                            name="beta",
                            dimension=DL,
                            prior=Prior(family="normal", hyper={"mu": beta_mu, "sigma": beta_sd}),
                        ),
                        Data(name="x", dimension=DL),
                    )
                ),
            )
        ),
        outcome=Data(name="y", dimension=DL),
        likelihood=Likelihood(family="normal", scale="sigma"),
        parameters=(
            Param(
                name="alpha",
                dimension=DL,
                prior=Prior(family="normal", hyper={"mu": 0.0, "sigma": 1.0}),
            ),
            Param(
                name="beta",
                dimension=DL,
                prior=Prior(family="normal", hyper={"mu": beta_mu, "sigma": beta_sd}),
            ),
            Param(name="sigma", dimension=DL, prior=sigma_prior),
        ),
    )


def _data(n: int = 40) -> dict[str, np.ndarray]:
    x = np.linspace(-1.0, 1.0, n)
    return {"x": x, "y": np.zeros(n)}


# -- draw_prior ------------------------------------------------------------------------------


def test_draw_prior_shapes_and_hierarchy() -> None:
    world = surface_world(
        n_units=5, n_periods=4, treatments=("a",), intercept="hierarchical", seed=0
    )
    draws = draw_prior(world.model, np.random.default_rng(0), n=400)
    for p in world.model.parameters:
        assert draws[p.name].shape == (400, *p.shape), p.name
    # a hierarchical unit intercept is drawn around its drawn parent, resolved first
    unit = next(p for p in world.model.parameters if p.shape)
    assert unit.prior is not None and unit.prior.parents
    parent = unit.prior.parents[0]
    centred = draws[unit.name] - draws[parent][:, None]
    assert abs(float(centred.mean())) < 0.15


def test_draw_prior_matches_prior_moments_and_is_seeded() -> None:
    model = _linear_model(sigma_fixed=None)
    a = draw_prior(model, np.random.default_rng(1), n=4000)
    b = draw_prior(model, np.random.default_rng(1), n=4000)
    assert all(np.array_equal(a[k], b[k]) for k in a)
    assert abs(a["alpha"].mean()) < 0.06 and abs(a["alpha"].std() - 1.0) < 0.06
    assert np.all(a["sigma"] > 0) and abs(a["sigma"].mean() - np.sqrt(2 / np.pi)) < 0.05
    fixed = draw_prior(_linear_model(sigma_fixed=0.3), np.random.default_rng(0), n=3)
    assert np.array_equal(fixed["sigma"], [0.3, 0.3, 0.3])
    with pytest.raises(ValueError, match="n must be"):
        draw_prior(model, np.random.default_rng(0), n=0)


def test_simulate_outcome_uses_the_likelihood_family() -> None:
    model = _linear_model(sigma_fixed=0.5)
    data = _data(2000)
    theta = {"alpha": np.asarray(1.0), "beta": np.asarray(2.0), "sigma": np.asarray(0.5)}
    y = simulate_outcome(model, data, theta, np.random.default_rng(0))
    resid = y - (1.0 + 2.0 * data["x"])
    assert abs(resid.mean()) < 0.05 and abs(resid.std() - 0.5) < 0.03


# -- rank statistics -------------------------------------------------------------------------


def test_default_bins_divides_and_respects_expected_count() -> None:
    assert default_bins(200, 19) == 20
    assert default_bins(60, 19) == 5
    assert default_bins(12, 19) == 2
    assert default_bins(3, 19) == 2


def test_rank_uniformity_accepts_uniform_and_rejects_skewed_ranks() -> None:
    rng = np.random.default_rng(0)
    ok = rank_uniformity("u", rng.integers(0, 20, size=400), n_ranks=19, alpha=0.05)
    assert ok.passed and ok.bins == 20 and sum(ok.histogram) == 400
    assert ok.ecdf_statistic <= ok.ecdf_band
    skewed = rank_uniformity(
        "s", np.minimum(rng.integers(0, 8, size=400), 19), n_ranks=19, alpha=0.05
    )
    assert not skewed.passed and skewed.chi2_p_value < 1e-6
    with pytest.raises(ValueError, match="no ranks"):
        rank_uniformity("e", [], n_ranks=19, alpha=0.05)
    with pytest.raises(ValueError, match="must lie on"):
        rank_uniformity("e", [25], n_ranks=19, alpha=0.05)
    with pytest.raises(ValueError, match="divide"):
        rank_uniformity("e", [1, 2, 3], n_ranks=19, alpha=0.05, bins=3)


def test_parameter_ranks_validates_its_bookkeeping() -> None:
    with pytest.raises(ValidationError):
        ParameterRanks(
            name="x",
            n=3,
            n_ranks=19,
            bins=2,
            ranks=(1, 2),
            histogram=(2, 1),
            chi2_statistic=0.0,
            chi2_p_value=1.0,
            ecdf_statistic=0.0,
            ecdf_band=0.5,
            alpha=0.05,
            passed=True,
        )


def test_sbc_spec_validation() -> None:
    SBCSpec(n_simulations=10, draws=50, rank_draws=19, bins=4)
    with pytest.raises(ValidationError, match="exceeds draws"):
        SBCSpec(n_simulations=10, draws=10, rank_draws=20)
    with pytest.raises(ValidationError, match="divide"):
        SBCSpec(n_simulations=10, rank_draws=19, bins=3)
    with pytest.raises(ValidationError):
        SBCSpec(n_simulations=1)
    with pytest.raises(ValidationError, match="distinct"):
        SBCSpec(n_simulations=5, parameters=("a", "a"))


# -- the loop --------------------------------------------------------------------------------


def test_sbc_on_an_exact_model_is_uniform_and_bookkept() -> None:
    model = _linear_model(sigma_fixed=0.5)
    spec = SBCSpec(n_simulations=40, draws=100, seed=3, alpha=0.05)
    out = sbc(model, _data(), spec=spec)
    assert isinstance(out, SBCResult)
    assert out.n_simulations == 40 and out.n_fitted + out.n_failed_fits == 40
    assert out.n_failed_fits == 0
    assert {p.name for p in out.parameters} == {"alpha", "beta"}  # sigma is fixed: not ranked
    assert out.alpha_per_parameter == pytest.approx(0.025)
    assert out.passed, [(p.name, p.chi2_p_value) for p in out.parameters]
    assert out.model_hash == model.content_hash() == out.refit_model_hash
    round_trip = SBCResult.from_json(out.to_json())
    assert round_trip == out


def test_sbc_negative_control_fails_with_a_shifted_prior() -> None:
    model = _linear_model(sigma_fixed=0.5)
    wrong = _linear_model(sigma_fixed=0.5, beta_mu=3.0, beta_sd=0.1)
    spec = SBCSpec(n_simulations=40, draws=100, seed=3, alpha=0.05, parameters=("beta",))
    out = sbc(model, _data(), spec=spec, refit=wrong)
    assert isinstance(out, SBCResult)
    assert out.refit_model_hash == wrong.content_hash() != out.model_hash
    assert not out.passed and out.failed_parameters == ("beta",)


def test_sbc_refuses_mismatched_refit_and_unknown_parameters() -> None:
    model = _linear_model(sigma_fixed=0.5)
    with pytest.raises(ValueError, match="same free parameters"):
        sbc(model, _data(), spec=SBCSpec(n_simulations=2), refit=_linear_model(sigma_fixed=None))
    with pytest.raises(ValueError, match="not free parameters"):
        sbc(model, _data(), spec=SBCSpec(n_simulations=2, parameters=("sigma",)))


def test_sbc_unknown_backend_is_unsupported() -> None:
    out = sbc(_linear_model(), _data(), spec=SBCSpec(n_simulations=2, backend="no-such-backend"))
    assert isinstance(out, Unsupported)


def test_sbc_custom_simulator_and_vector_parameters() -> None:
    world = surface_world(
        n_units=3,
        n_periods=6,
        treatments=("a",),
        kernels=LinearKernel(reference_dose=1.0, amplitude_prior=AMP),
        intercept="hierarchical",
        noise_sd=0.3,
        seed=0,
    )
    calls: list[int] = []

    def simulate(theta: dict[str, np.ndarray], rng: np.random.Generator) -> np.ndarray:
        calls.append(1)
        return simulate_outcome(world.model, world.data, theta, rng)

    out = sbc(world.model, world.data, spec=SBCSpec(n_simulations=4, draws=40), simulate=simulate)
    assert isinstance(out, SBCResult)
    assert len(calls) == 4
    names = {p.name for p in out.parameters}
    unit = next(p for p in world.model.parameters if p.shape)
    assert f"{unit.name}[0]" in names and f"{unit.name}[2]" in names


def test_sbc_surface_and_pool_smoke() -> None:
    world = surface_world(
        n_units=4,
        n_periods=8,
        treatments=("a",),
        kernels=LinearKernel(reference_dose=1.0, amplitude_prior=AMP),
        intercept="shared",
        noise_sd=0.3,
        seed=1,
        doses=DosePlan(zero_fraction=0.2),
    )
    out = sbc_surface(world.spec, world.panel, sbc_spec=SBCSpec(n_simulations=4, draws=40))
    assert isinstance(out, SBCResult) and out.n_fitted + out.n_failed_fits == 4
    assert {p.name for p in out.parameters} <= {p.name for p in world.model.free}

    from tests.unit.test_meta_pool import _corpus

    pooled = sbc_pool(
        PoolSpec(family="f", priors=PoolPriors(tau_fixed=0.3)),
        _corpus(0, k=8),
        sbc_spec=SBCSpec(n_simulations=4, draws=40),
    )
    assert isinstance(pooled, SBCResult) and {p.name for p in pooled.parameters} == {"mu_f"}
    missing = sbc_pool(PoolSpec(family="other"), _corpus(0, k=8), sbc_spec=SBCSpec(n_simulations=4))
    assert isinstance(missing, Unsupported)
