"""Phase 3 recovery: the surface model recovers its world's truth; unit invariance holds.

Fast tier uses Laplace. The NUTS coverage tier is ``slow`` and states its N
and acceptance region (review B5).
"""

from __future__ import annotations

import numpy as np
import pytest

from axiom.core import D, Posterior, Treatment, clopper_pearson, is_failure
from axiom.data import Panel
from axiom.sim import DosePlan, arms_world, surface_world
from axiom.surface import (
    GeometricCarryover,
    HillKernel,
    PiecewiseLinearKernel,
    PolynomialKernel,
    SplineKernel,
    fit,
)

pytestmark = pytest.mark.recovery


def _world(seed: int = 0, noise_sd: float = 0.5):  # type: ignore[no-untyped-def]
    return surface_world(
        n_units=4,
        n_periods=40,
        treatments=("a",),
        kernels=HillKernel(reference_dose=50.0, amplitude_scale=10.0),
        carryover=GeometricCarryover(max_lag=4),
        doses=DosePlan(scale=50.0, spread=0.8, zero_fraction=0.05),
        intercept="shared",
        truth={"beta_a": 10.0, "alpha": 5.0, "k_a": 50.0, "s_a": 2.0, "lam_a": 0.5},
        noise_sd=noise_sd,
        seed=seed,
    )


def test_laplace_recovers_structural_parameters() -> None:
    world = _world()
    res = fit(world.spec, world.panel, backend="laplace", draws=2000, seed=0)
    post = res.posterior
    assert isinstance(post, Posterior), post
    assert res.converged
    for name in ("beta_a", "k_a", "s_a", "lam_a", "sigma"):
        s = post.summary(name, definition="hdi", mass=0.95)
        truth = float(np.asarray(world.theta[name]))
        assert s.interval.contains(truth) or abs(s.mean - truth) < 3 * s.sd, (name, truth, s)


def test_negative_control_wrong_kernel_family_misses() -> None:
    """A linear response fitted to a saturating world: the misfit shows up in the noise scale."""
    from axiom.surface import LinearKernel

    world = _world(seed=1)
    wrong = world.spec.model_copy(
        update={"kernels": {"a": LinearKernel(reference_dose=50.0, amplitude_scale=10.0)}}
    )
    res = fit(wrong, world.panel, backend="laplace", draws=500, seed=0)
    post = res.posterior
    if is_failure(post):
        return  # an honest refusal is an acceptable outcome for a misspecified model
    right = fit(world.spec, world.panel, backend="laplace", draws=500, seed=0).posterior
    assert isinstance(right, Posterior)
    assert post.summary("sigma").mean > 1.5 * right.summary("sigma").mean


def test_unit_invariance_of_fit() -> None:
    world = _world(seed=2)
    frame = world.panel.frame
    roles = world.panel.roles
    cents_roles = roles.model_copy(
        update={"treatments": {"a": Treatment(name="a", dimension=D.currency, unit="cents")}}
    )
    cents_panel = Panel(frame.assign(a=frame["a"] * 100.0), cents_roles)
    cents_spec = world.spec.model_copy(
        update={
            "treatments": (Treatment(name="a", dimension=D.currency, unit="cents"),),
            "kernels": {"a": HillKernel(reference_dose=5000.0, amplitude_scale=10.0)},
        }
    )
    usd = fit(world.spec, world.panel, backend="laplace", draws=2000, seed=0).posterior
    cents = fit(cents_spec, cents_panel, backend="laplace", draws=2000, seed=0).posterior
    assert isinstance(usd, Posterior) and isinstance(cents, Posterior)
    for shape in ("s_a", "lam_a"):
        a, b = usd.summary(shape), cents.summary(shape)
        assert abs(a.mean - b.mean) < 3 * max(a.sd, b.sd) + 0.05, shape
    ka, kc = usd.summary("k_a"), cents.summary("k_a")
    assert kc.mean / ka.mean == pytest.approx(100.0, rel=0.1)


@pytest.mark.slow
def test_nuts_interval_coverage() -> None:
    """90% HDIs cover truth at the nominal rate over N=60 worlds (exact binomial region)."""
    pytest.importorskip("numpyro")
    n = 60
    hits = 0
    for seed in range(n):
        world = _world(seed=seed)
        res = fit(
            world.spec, world.panel, backend="numpyro", draws=400, tune=400, chains=2, seed=seed
        )
        post = res.posterior
        assert isinstance(post, Posterior), post
        s = post.summary("beta_a", definition="hdi", mass=0.9)
        hits += int(s.interval.contains(float(np.asarray(world.theta["beta_a"]))))
    region = clopper_pearson(n, 0.9, alpha=0.001)
    assert region.accepts(hits), (hits, region)


# -- basis families (note 0005) ------------------------------------------------------------

BASIS_WORLDS = [
    pytest.param(
        PolynomialKernel(reference_dose=100.0, amplitude_scale=5.0, degree=3),
        {"alpha": 1.0, "beta1_a": 9.0, "beta2_a": -16.0, "beta3_a": 8.0},
        id="polynomial",
    ),
    pytest.param(
        SplineKernel(reference_dose=100.0, amplitude_scale=5.0, knots=(25.0, 50.0, 75.0)),
        {"alpha": 1.0, "beta1_a": 6.0, "beta2_a": -9.0},
        id="spline",
    ),
    pytest.param(
        PiecewiseLinearKernel(reference_dose=100.0, amplitude_scale=5.0, knots=(40.0,)),
        {"alpha": 1.0, "beta1_a": 5.0, "beta2_a": -11.0},
        id="piecewise_linear",
    ),
]


@pytest.mark.parametrize(("kernel", "truth"), BASIS_WORLDS)
def test_basis_families_recover_a_response_that_turns_over(
    kernel: object, truth: dict[str, float]
) -> None:
    """The families exist for non-monotone worlds, so the recovery world is one.

    ``truth`` is stated rather than drawn from the prior centre: the centre of a signed
    coefficient prior is zero, so ``truth_mode="centre"`` would build a world in which
    the treatment does nothing and the recovery would be vacuous.
    """
    world = arms_world(
        n_units=400,
        treatments=("a",),
        kernels=kernel,  # type: ignore[arg-type]
        doses={"a": np.linspace(0.0, 100.0, 400)},
        truth=truth,
        noise_sd=0.5,
        seed=3,
    )
    response = np.ravel(np.asarray(world.mean))
    steps = np.diff(response)
    assert np.any(steps > 0) and np.any(steps < 0), "the recovery world must turn over"

    res = fit(world.spec, world.panel, backend="laplace", draws=800, chains=1, seed=3)
    post = res.posterior
    assert isinstance(post, Posterior), post
    assert res.converged
    for name in (*truth, "sigma"):
        s = post.summary(name, definition="hdi", mass=0.9)
        assert s.interval.contains(float(np.asarray(world.theta[name]))), (name, s)


def test_a_saturating_family_cannot_hold_a_reversal() -> None:
    """The negative control for note 0005: the monotone family misses, and says where.

    The noise is small here on purpose. The point is not that a Hill curve fits a
    reversing world badly at any noise level — at the noise the other tests use, the
    misfit is inside the residual and neither family can tell. It is that the misfit goes
    into ``sigma`` and *stays* there as the data get better, which is what a shape
    assumption the data reject looks like.
    """
    kernel, truth = BASIS_WORLDS[0].values  # type: ignore[misc]
    world = arms_world(
        n_units=400,
        treatments=("a",),
        kernels=kernel,
        doses={"a": np.linspace(0.0, 100.0, 400)},
        truth=truth,
        noise_sd=0.05,
        seed=3,
    )
    basis = fit(world.spec, world.panel, backend="laplace", draws=400, chains=1, seed=3)
    wrong = world.spec.model_copy(
        update={"kernels": {"a": HillKernel(reference_dose=50.0, amplitude_scale=10.0)}}
    )
    saturating = fit(wrong, world.panel, backend="laplace", draws=400, chains=1, seed=3)
    assert isinstance(basis.posterior, Posterior)
    assert isinstance(saturating.posterior, Posterior)
    basis_sigma = basis.posterior.summary("sigma").mean
    wrong_sigma = saturating.posterior.summary("sigma").mean
    # the basis family recovers the noise it was given; the monotone one cannot
    assert basis_sigma == pytest.approx(0.05, rel=0.15), basis_sigma
    assert wrong_sigma > 3 * basis_sigma, (wrong_sigma, basis_sigma)
