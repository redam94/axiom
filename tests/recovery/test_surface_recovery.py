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
    """The recovery world. Its dose plan is a design decision, not a default.

    A Hill surface's intercept, amplitude and shape are confounded when the doses
    do not span the rising part of the curve: ``design.identifiability_ridge`` on
    the narrow plan this world used to carry (``spread=0.8``, ``zero_fraction=0.05``)
    reports a condition number of 847 with ``corr(beta_a, s_a) = -0.97``, and the
    fixed-truth coverage of ``beta_a`` there is 45 %, not 90 % — under Laplace as
    well as NUTS, so it was never a sampler problem. Widening the dose distribution
    and putting a fifth of the rows at zero dose takes the condition number to 119
    and the coverage back to nominal. See ``docs/notes/0006``.
    """
    return surface_world(
        n_units=4,
        n_periods=40,
        treatments=("a",),
        kernels=HillKernel(reference_dose=50.0, amplitude_scale=10.0),
        carryover=GeometricCarryover(max_lag=4),
        doses=DosePlan(scale=50.0, spread=1.3, zero_fraction=0.2),
        intercept="shared",
        truth={"beta_a": 10.0, "alpha": 5.0, "k_a": 50.0, "s_a": 2.0, "lam_a": 0.5},
        noise_sd=noise_sd,
        seed=seed,
    )


def test_the_recovery_world_is_identified() -> None:
    """Guard the design decision above: if the dose plan narrows, this fails first."""
    from axiom.design import identifiability_ridge

    world = _world()
    theta = {k: v for k, v in world.theta.items() if k != "sigma"}
    ridge = identifiability_ridge(
        world.surface,
        {"a": world.data["a"]},
        theta,
        0.5,
        pairs=(("beta_a", "s_a"), ("alpha", "beta_a")),
        method="finite",
    )
    assert not is_failure(ridge)
    assert ridge.condition_number < 200.0, ridge.condition_number
    assert all(abs(c) < 0.95 for c in ridge.correlations), ridge.correlations


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


def test_a_gaussian_process_learns_a_curve_from_no_shipped_family() -> None:
    """The GP's claim: it fits a dose-response that no parametric family here can express.

    The truth is a rise, a turn and a slow drift — outside Hill, outside a
    natural cubic on any small knot set, and outside a low-degree polynomial.
    A correctly-specified GP should recover the noise scale it was given; a
    saturating family has to bury the misfit in ``sigma`` instead.
    """
    import pandas as pd

    from axiom.core import Outcome, Treatment
    from axiom.data import Panel, RoleMap
    from axiom.surface import GaussianProcessKernel, SurfaceSpec, forward

    rng = np.random.default_rng(4)
    n, top, noise = 300, 40.0, 0.8
    dose = rng.uniform(0.0, top, n)

    def curve(x: np.ndarray) -> np.ndarray:
        u = x / 12.0
        return np.asarray(9.0 * u / (1.0 + u**2) * 2.2 + 0.02 * x)

    outcome = Outcome(name="y", dimension=D.outcome, unit="u", aggregation="mean")
    drug = Treatment(name="dose", dimension=D.currency, unit="mg")
    panel = Panel(
        pd.DataFrame(
            {
                "unit": [f"u{i:03d}" for i in range(n)],
                "t": 0,
                "y": curve(dose) + rng.normal(0.0, noise, n),
                "dose": dose,
            }
        ),
        RoleMap(unit="unit", time="t", outcome=("y", outcome), treatments={"dose": drug}),
    )

    kernel = GaussianProcessKernel(reference_dose=top, amplitude_scale=10.0)
    spec = SurfaceSpec(
        name="gp",
        treatments=(drug,),
        outcome=outcome,
        kernels={"dose": kernel},
        intercept="shared",
        noise_scale=1.0,
    )
    gp = fit(spec, panel, backend="laplace", draws=600, chains=1, seed=1)
    assert gp.converged
    assert isinstance(gp.posterior, Posterior)

    # the fitted lengthscale is inside the band the basis can actually represent
    lengthscale = gp.posterior.summary("ell_dose").mean
    assert kernel.sufficient_for(lengthscale), lengthscale

    # it recovers the noise it was given, so the curve is not being absorbed into sigma
    assert gp.posterior.summary("sigma").mean == pytest.approx(noise, rel=0.2)

    grid = np.linspace(0.0, top, 81)
    theta = {
        p.name: float(gp.posterior.summary(p.name).mean)
        for p in gp.surface.model.parameters
        if p.name != "sigma"
    }
    predicted = np.ravel(np.asarray(forward(gp.surface, {"dose": grid}, theta)))
    target = curve(grid) - curve(np.zeros(1))[0]
    gp_rmse = float(np.sqrt(np.mean((predicted - target) ** 2)))

    saturating = fit(
        spec.model_copy(
            update={"kernels": {"dose": HillKernel(reference_dose=12.0, amplitude_scale=10.0)}}
        ),
        panel,
        backend="laplace",
        draws=600,
        chains=1,
        seed=1,
    )
    assert isinstance(saturating.posterior, Posterior)
    hill_theta = {
        p.name: float(saturating.posterior.summary(p.name).mean)
        for p in saturating.surface.model.parameters
        if p.name != "sigma"
    }
    hill_rmse = float(
        np.sqrt(
            np.mean(
                (
                    np.ravel(np.asarray(forward(saturating.surface, {"dose": grid}, hill_theta)))
                    - target
                )
                ** 2
            )
        )
    )
    # the misfit a saturating family cannot hold goes into its residual scale
    assert saturating.posterior.summary("sigma").mean > 1.5 * gp.posterior.summary("sigma").mean
    # the GP tracks the curve to a few per cent of its span, and better than the wrong family
    span = float(target.max() - target.min())
    assert gp_rmse < 0.1 * span, (gp_rmse, span)
    assert gp_rmse < 0.75 * hill_rmse, (gp_rmse, hill_rmse)
