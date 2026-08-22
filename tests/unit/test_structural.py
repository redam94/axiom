"""design.structural on a real ``axiom.surface`` Surface: one Hill treatment with carryover."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from axiom.core import Unsupported, jax_available
from axiom.design.schedule import alternating, constant, contrast_score, pulse
from axiom.design.structural import (
    FisherInformation,
    IdentifiabilityRidge,
    IdentifyingDesign,
    design_to_identify,
    expected_posterior_sd,
    fisher_information,
    identifiability_ridge,
    ridge_of,
)
from axiom.sim import DosePlan, surface_world
from axiom.surface import Design, GeometricCarryover, HillKernel

N_UNITS = 2
N_PERIODS = 12
NOISE_SD = 0.5
TRUTH = {"beta_a": 10.0, "alpha": 5.0, "k_a": 50.0, "s_a": 2.0, "lam_a": 0.5}
PRIORS = {"alpha": 5.0, "beta_a": 5.0, "k_a": 20.0, "s_a": 1.0, "lam_a": 0.3}


@pytest.fixture(scope="module")
def world() -> Any:
    return surface_world(
        n_units=N_UNITS,
        n_periods=N_PERIODS,
        treatments=("a",),
        kernels=HillKernel(reference_dose=50.0, amplitude_scale=10.0),
        carryover=GeometricCarryover(max_lag=4),
        doses=DosePlan(scale=50.0),
        intercept="shared",
        truth=TRUTH,
        noise_sd=NOISE_SD,
        seed=0,
    )


def _data_with(world: Any, doses: np.ndarray) -> dict[str, np.ndarray]:
    data = dict(world.data)
    data["a"] = doses
    return data


def _fi(world: Any, doses: np.ndarray, **kw: Any) -> FisherInformation:
    fi = fisher_information(world.surface, _data_with(world, doses), world.theta, NOISE_SD, **kw)
    assert isinstance(fi, FisherInformation), fi
    return fi


def test_information_on_carryover_increases_with_contrast(world: Any) -> None:
    # The same pulse pattern at growing amplitude around the same mean dose: the
    # contrast score rises and so does the information on the carryover decay
    # and the total information (the determinant), starting from a singular flat design.
    ladder = (
        constant(N_PERIODS, 50.0),
        pulse(N_PERIODS, 60.0, 40.0, on=2, off=2),
        pulse(N_PERIODS, 80.0, 20.0, on=2, off=2),
        pulse(N_PERIODS, 100.0, 0.0, on=2, off=2),
    )
    scores = [contrast_score(s) for s in ladder]
    assert scores[0] == 0.0 and all(a < b for a, b in zip(scores, scores[1:], strict=False))
    lam_info: list[float] = []
    dets: list[float] = []
    for s in ladder:
        fi = _fi(world, s.as_grid(N_UNITS), method="finite")
        assert fi.parameters == ("alpha", "beta_a", "lam_a", "s_a", "k_a")
        assert fi.n_observations == N_UNITS * N_PERIODS
        i = fi.index("lam_a")
        lam_info.append(float(fi.as_array()[i, i]))
        dets.append(fi.det)
    assert all(a < b for a, b in zip(lam_info, lam_info[1:], strict=False))
    assert dets[0] == pytest.approx(0.0, abs=1e-6)
    assert all(a < b for a, b in zip(dets[1:], dets[2:], strict=False))
    # The maximal-contrast alternation is also identifying.
    swing = _fi(world, alternating(N_PERIODS, 80.0, 20.0).as_grid(N_UNITS), method="finite")
    assert (
        not swing.singular
        and swing.as_array()[swing.index("lam_a")][swing.index("lam_a")] > lam_info[0]
    )


def test_constant_schedule_is_singular_and_refuses_flat_prior_inversion(world: Any) -> None:
    fi = _fi(world, constant(N_PERIODS, 50.0).as_grid(N_UNITS), method="finite")
    assert fi.singular
    assert isinstance(fi.covariance(), Unsupported)
    flat = expected_posterior_sd(None, fi)
    assert isinstance(flat, Unsupported)
    assert "singular" in flat.reason or "no information" in flat.reason
    # Pairwise correlations need the inverse, so the ridge with pairs is refused too.
    assert isinstance(ridge_of(fi, pairs=(("beta_a", "k_a"),)), Unsupported)
    # The ridge itself (no pairs) still names the flat direction.
    ridge = ridge_of(fi)
    assert isinstance(ridge, IdentifiabilityRidge)
    assert ridge.condition_number > 1e10
    # A pulse makes the same surface identifiable.
    pulsed = _fi(world, pulse(N_PERIODS, 80.0, 20.0, on=2, off=2).as_grid(N_UNITS), method="finite")
    assert not pulsed.singular
    assert not isinstance(pulsed.covariance(), Unsupported)


@pytest.mark.skipif(not jax_available(), reason="jax not installed")
def test_finite_and_jax_derivatives_agree(world: Any) -> None:
    import jax

    # Same convention as tests/unit/test_ascent.py: x64 on for the derivative, then restored.
    was = bool(getattr(jax.config, "jax_enable_x64", False))
    jax.config.update("jax_enable_x64", True)
    try:
        doses = pulse(N_PERIODS, 80.0, 20.0, on=3, off=2).as_grid(N_UNITS)
        fd = _fi(world, doses, method="finite")
        jx = _fi(world, doses, method="jax")
    finally:
        jax.config.update("jax_enable_x64", was)
    assert fd.method == "finite" and jx.method == "jax"
    assert fd.parameters == jx.parameters
    np.testing.assert_allclose(fd.as_array(), jx.as_array(), rtol=1e-6, atol=1e-9)


def test_expected_posterior_sd_decreases_with_n(world: Any) -> None:
    doses = pulse(N_PERIODS, 80.0, 20.0, on=2, off=2).as_grid(N_UNITS)
    fi1 = _fi(world, doses, method="finite")
    sd1 = expected_posterior_sd(PRIORS, fi1)
    assert isinstance(sd1, dict)
    # Doubling the observations through the additive law halves the information's scale.
    fi2 = fi1 + fi1
    assert fi2.n_observations == 2 * fi1.n_observations
    sd2 = expected_posterior_sd(PRIORS, fi2)
    assert isinstance(sd2, dict)
    for name in fi1.parameters:
        assert sd2[name] < sd1[name] < PRIORS[name]
    # And the sd is never larger than the prior: information only shrinks it.
    sd_prior_only = expected_posterior_sd(
        PRIORS, fi1.model_copy(update={"matrix": tuple((0.0,) * fi1.p for _ in range(fi1.p))})
    )
    assert isinstance(sd_prior_only, dict)
    for name in fi1.parameters:
        assert sd_prior_only[name] == pytest.approx(PRIORS[name])


def test_information_scales_with_noise(world: Any) -> None:
    doses = pulse(N_PERIODS, 80.0, 20.0, on=2, off=2).as_grid(N_UNITS)
    data = _data_with(world, doses)
    a = fisher_information(world.surface, data, world.theta, 0.5, method="finite")
    b = fisher_information(world.surface, data, world.theta, 1.0, method="finite")
    assert isinstance(a, FisherInformation) and isinstance(b, FisherInformation)
    np.testing.assert_allclose(a.as_array(), 4.0 * b.as_array(), rtol=1e-12)


def test_parameters_subset_and_errors(world: Any) -> None:
    doses = pulse(N_PERIODS, 80.0, 20.0, on=2, off=2).as_grid(N_UNITS)
    fi = _fi(world, doses, method="finite", parameters=("beta_a", "k_a"))
    assert fi.parameters == ("beta_a", "k_a")
    with pytest.raises(ValueError, match="not coordinates"):
        _fi(world, doses, method="finite", parameters=("nope",))
    with pytest.raises(ValueError, match="noise_sd"):
        fisher_information(world.surface, _data_with(world, doses), world.theta, 0.0)
    with pytest.raises(KeyError):
        fi.index("lam_a")


def test_identifiability_ridge_reports_beta_k_coupling(world: Any) -> None:
    doses = pulse(N_PERIODS, 80.0, 20.0, on=2, off=2).as_grid(N_UNITS)
    ridge = identifiability_ridge(
        world.surface,
        _data_with(world, doses),
        world.theta,
        NOISE_SD,
        pairs=(("beta_a", "k_a"),),
        method="finite",
    )
    assert isinstance(ridge, IdentifiabilityRidge)
    assert set(ridge.direction) == set(ridge.parameters)
    assert np.isclose(sum(v * v for v in ridge.direction.values()), 1.0)
    assert ridge.min_eigenvalue >= 0.0 and ridge.condition_number >= 1.0
    (rho,) = ridge.correlations
    # beta and k of a saturating kernel trade off: strongly positively correlated.
    assert rho > 0.7
    assert len(ridge.ridge_parameters) >= 1


def test_design_to_identify_basics(world: Any) -> None:
    steady = world.surface.steady_state()
    theta = {k: v for k, v in world.theta.items() if k != "lam_a"}
    candidates = Design(
        treatments=("a",), points=((0.0,), (20.0,), (50.0,), (100.0,), (200.0,)), kind="grid"
    )
    out = design_to_identify(
        steady,
        candidates,
        theta,
        NOISE_SD,
        target="k_a",
        n=8,
        prior_sds={"alpha": 5.0, "beta_a": 5.0, "k_a": 20.0, "s_a": 1.0},
        seed=0,
        method="finite",
    )
    assert isinstance(out, IdentifyingDesign), out
    assert out.target == "k_a" and out.design.n == 8 and len(out.indices) == 8
    assert out.expected_sd == pytest.approx(out.expected_sds["k_a"])
    assert out.expected_sd < 20.0
    # Reproducible with the same seed.
    again = design_to_identify(
        steady,
        candidates,
        theta,
        NOISE_SD,
        target="k_a",
        n=8,
        prior_sds={"alpha": 5.0, "beta_a": 5.0, "k_a": 20.0, "s_a": 1.0},
        seed=0,
        method="finite",
    )
    assert isinstance(again, IdentifyingDesign)
    assert again.indices == out.indices
    with pytest.raises(ValueError, match="target"):
        design_to_identify(steady, candidates, theta, NOISE_SD, target="zzz", n=3, method="finite")
    # One row under flat priors cannot identify four parameters.
    flat = design_to_identify(
        steady, candidates, theta, NOISE_SD, target="k_a", n=1, seed=0, method="finite"
    )
    assert isinstance(flat, Unsupported)
