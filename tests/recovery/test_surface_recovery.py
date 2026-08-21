"""Phase 3 recovery: the surface model recovers its world's truth; unit invariance holds.

Fast tier uses Laplace. The NUTS coverage tier is ``slow`` and states its N
and acceptance region (review B5).
"""

from __future__ import annotations

import numpy as np
import pytest

from axiom.core import D, Posterior, Treatment, clopper_pearson, is_failure
from axiom.data import Panel
from axiom.sim import DosePlan, surface_world
from axiom.surface import GeometricCarryover, HillKernel, fit

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
