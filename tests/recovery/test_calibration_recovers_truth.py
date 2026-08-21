"""Phase 6 exit criterion 1 (D6.7): randomized evidence corrects a confounded surface fit.

The setting
-----------
A ``surface_world`` with one Hill treatment ``a`` (truth ``beta_a = 10``,
``k_a = 50``, ``s_a = 2``, shared intercept ``alpha = 5``, no carryover)
generates an honest panel. The *observational* panel the analyst sees is
that panel plus ``gamma · u``, where ``u`` is an **unobserved confounder
correlated with the dose** (the standardized dose plus independent noise):
units were dosed more when something else was also lifting the outcome.
The world's own spec omits ``u``, so fitting it to the observational panel
inflates the amplitude — verified first: the biased posterior mean of
``beta_a`` sits more than 2 posterior sd above the truth.

A randomized experiment on the *true* world measures the cumulative
contrast ``set a = 100`` versus ``set a = 0`` over the full horizon, per
unit (``Level(unit="individual")``): the truth is computed from
``world.forward`` and the ``Measurement`` is ``truth + se · ε`` with a
fixed seed and ``se = 1 %`` of the truth.

What is asserted
----------------
* **Both routes move the amplitude toward the truth.** The prior route
  (``derive_prior`` from the biased fit's beta / realized-contrast draws,
  refit on the observational panel) and the likelihood route
  (``fit_calibrated``) both leave ``beta_a``'s posterior mean strictly
  closer to 10 than the biased fit, by a stated number of biased sd.
* **The mechanism that separates the routes, exactly.** The prior route
  changes *only* the amplitude prior: every other parameter's prior is
  unchanged and the change in ``log_prior`` it introduces has zero
  gradient in ``k_a`` and ``s_a`` (finite differences are exactly zero).
  The likelihood route's constraint expression depends on ``k_a`` and
  ``s_a`` (finite differences are non-zero), so its soft constraint pulls
  the curve shape as well as the amplitude.
* **The likelihood route moves the shape posterior** by more than one
  biased posterior sd in ``k_a`` or ``s_a``, and its realized contrast
  ``agrees`` with the randomized measurement (``check.agreement``).
* **Negative controls.** A measurement placed 5 se *above* the biased
  fit's own realized contrast moves ``beta_a`` *away* from the truth under
  both routes, and a measurement 5 se off the truth is not ``agrees`` on
  the calibrated fits (its ``|z|`` exceeds the true measurement's), so the
  test cannot pass vacuously.

Modelling limitation, stated rather than loosened
-------------------------------------------------
The roadmap phrases the distinction as "the likelihood route moves the
curve shape, the prior route does not". That is true of what each route
*adds to the density* (asserted exactly above), and it is **not** true of
the posterior on a confounded panel: the prior route's tight amplitude
prior forces the likelihood to re-explain the confounded outcome through
``k_a`` / ``s_a``, so the shape *posterior* moves under the prior route
too — in this scenario by 1.5 sd (``k_a``) and 2.6 sd (``s_a``), i.e. no
less than under the likelihood route. For the same reason the prior route
does not reach ``agrees``: its design factor is ``mean(contribution /
beta)`` under the *biased* shape (``k ≈ 110`` rather than 50), so the
implied amplitude prior (mean ≈ 16.5) is only part of the way to 10 and
the realized contrast at dose 100 stays above the measurement. The test
therefore asserts for the prior route that the realized contrast moves
toward the measurement and that ``|z|`` shrinks, not that it agrees.
Every number quoted here is reproduced by the Laplace fits below at the
stated seeds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from axiom.calibrate import (
    Agreement,
    CalibratedSpec,
    Measurement,
    agreement,
    attach,
    derive_prior,
    fit_calibrated,
)
from axiom.core import (
    Intervention,
    ModelSpec,
    Population,
    Posterior,
    TimeWindow,
    log_prior,
    value,
)
from axiom.data import Panel
from axiom.estimands import Estimand, Level, Quantity, RealizedDraws, realize
from axiom.sim import DosePlan, SurfaceWorld, surface_world
from axiom.surface import FitResult, HillKernel, Surface, build, fit

pytestmark = pytest.mark.recovery

TRUTH = {"beta_a": 10.0, "alpha": 5.0, "k_a": 50.0, "s_a": 2.0}
HI, LO = 100.0, 0.0
SHAPE = ("k_a", "s_a")

WORLD_SEED = 0
CONFOUNDER_SEED = 123
MEASUREMENT_SEED = 7
FIT_SEED = 1


# -- scenario ------------------------------------------------------------------------------


@dataclass(frozen=True)
class Scenario:
    world: SurfaceWorld
    observed: Panel
    estimand: Estimand
    truth: float
    measurement: Measurement


def _world(n_units: int, n_periods: int, noise_sd: float) -> SurfaceWorld:
    return surface_world(
        n_units=n_units,
        n_periods=n_periods,
        treatments=("a",),
        kernels=HillKernel(reference_dose=50.0, amplitude_scale=10.0),
        doses=DosePlan(scale=50.0, spread=0.8, zero_fraction=0.05),
        intercept="shared",
        truth=TRUTH,
        noise_sd=noise_sd,
        seed=WORLD_SEED,
    )


def _confound(panel: Panel, gamma: float, seed: int) -> Panel:
    """The panel with an unobserved confounder ``u`` (correlated with the dose) added to ``y``.

    ``u = standardized(dose) + 0.3 · ε``: its correlation with the dose is
    what biases the amplitude; the noise keeps it from being a deterministic
    function of the dose.
    """
    rng = np.random.default_rng(seed)
    frame = panel.frame
    a = frame["a"].to_numpy(dtype=np.float64)
    u = (a - a.mean()) / a.std() + 0.3 * rng.standard_normal(a.size)
    return Panel(frame.assign(y=frame["y"] + gamma * u), panel.roles)


def _estimand(world: SurfaceWorld) -> Estimand:
    spec = world.spec
    return Estimand(
        name="cumulative_lift_100_vs_0",
        quantity=Quantity(kind="contrast"),
        treatment=spec.treatment("a"),
        intervention=Intervention(doses={"a": HI}),
        reference=Intervention(doses={"a": LO}),
        outcome=spec.outcome,
        population=Population(name="panel_units"),
        window=TimeWindow(start=0, stop=world.n_periods, basis="cumulative"),
        level=Level(unit="individual"),
        dimension=spec.outcome_dimension,
    )


def _truth_contrast(world: SurfaceWorld) -> float:
    """Per-unit cumulative contrast at the truth, through the world's one ``forward``."""
    hi = world.forward({"a": HI})
    lo = world.forward({"a": LO})
    return float(np.mean(np.sum(hi - lo, axis=1)))


def _scenario(
    *,
    n_units: int,
    n_periods: int,
    noise_sd: float = 2.0,
    gamma: float = 1.0,
    se_frac: float = 0.01,
) -> Scenario:
    world = _world(n_units, n_periods, noise_sd)
    estimand = _estimand(world)
    truth = _truth_contrast(world)
    se = se_frac * truth
    noise = float(np.random.default_rng(MEASUREMENT_SEED).standard_normal())
    measurement = Measurement(
        estimand=estimand,
        estimate=truth + se * noise,
        se=se,
        method="randomized_contrast",
        n_units=n_units,
        n_periods=n_periods,
        source="rct_true_world",
    )
    return Scenario(
        world, _confound(world.panel, gamma, CONFOUNDER_SEED), estimand, truth, measurement
    )


# -- fitting helpers -----------------------------------------------------------------------


def _posterior(result: FitResult) -> Posterior:
    assert isinstance(result.posterior, Posterior), result.posterior
    assert result.converged
    return result.posterior


def _moments(result: FitResult) -> dict[str, tuple[float, float]]:
    post = _posterior(result)
    return {
        name: (float(post.flat(name).mean()), float(post.flat(name).std(ddof=1)))
        for name in ("beta_a", "k_a", "s_a", "alpha", "sigma")
    }


def _prior_route(
    sc: Scenario, biased: FitResult, measurement: Measurement, draws: int
) -> tuple[CalibratedSpec, FitResult]:
    """``derive_prior`` from the biased fit's paired draws, then refit on the observed panel."""
    realized = realize(sc.estimand, biased, assume_identified=True, keep_draws=True)
    assert isinstance(realized, RealizedDraws), realized
    contribution = np.asarray(realized.draws, dtype=np.float64).reshape(-1)
    beta = _posterior(biased).flat("beta_a").reshape(-1)
    assert contribution.shape == beta.shape  # the same draws, in the same order
    calibrated = derive_prior(
        [measurement], sc.world.spec, "a", beta_draws=beta, contribution_draws=contribution
    )
    assert isinstance(calibrated, CalibratedSpec), calibrated
    return calibrated, fit(
        calibrated.spec, sc.observed, backend="laplace", draws=draws, seed=FIT_SEED
    )


def _likelihood_route(sc: Scenario, measurement: Measurement, draws: int) -> FitResult:
    result = fit_calibrated(
        sc.world.spec, sc.observed, [measurement], backend="laplace", draws=draws, seed=FIT_SEED
    )
    assert isinstance(result, FitResult), result
    return result


def _agreement(result: FitResult, measurement: Measurement) -> Agreement:
    out = agreement(result, measurement, seed=0)
    assert isinstance(out, Agreement), out
    return out


def _mode(result: FitResult) -> dict[str, np.ndarray]:
    post = _posterior(result)
    return {name: np.mean(post.flat(name), axis=0) for name in post.names()}


def _shifted(theta: dict[str, np.ndarray], name: str, factor: float) -> dict[str, np.ndarray]:
    return {**theta, name: np.asarray(theta[name], dtype=np.float64) * factor}


def _off_measurement(sc: Scenario, biased: FitResult, k: float = 5.0) -> Measurement:
    """A measurement ``k`` se *above* the biased fit's own realized contrast (negative control)."""
    anchor = _agreement(biased, sc.measurement).posterior_mean
    se = sc.measurement.se
    return sc.measurement.model_copy(update={"estimate": anchor + k * se, "source": "rct_off"})


# -- the mechanism, asserted exactly ------------------------------------------------------


def _assert_prior_route_touches_only_the_amplitude_prior(
    sc: Scenario,
    calibrated: CalibratedSpec,
    observed_data: dict[str, np.ndarray],
    theta: dict[str, np.ndarray],
) -> None:
    before, after = build(sc.world.spec), build(calibrated.spec)
    assert calibrated.parameter == "beta_a"
    assert [p.name for p in before.parameters] == [p.name for p in after.parameters]
    for p, q in zip(before.parameters, after.parameters, strict=True):
        if p.name == "beta_a":
            assert q.prior == calibrated.prior and p.prior != q.prior
        else:
            assert p.prior == q.prior, p.name
    # the mean tree is unchanged apart from the prior the amplitude Param carries: it
    # evaluates identically
    np.testing.assert_array_equal(
        np.asarray(value(before.mean, data=observed_data, params=theta)),
        np.asarray(value(after.mean, data=observed_data, params=theta)),
    )
    # the route's contribution to the density, log_prior(after) - log_prior(before), does not
    # depend on the shape or scale: its finite differences in k_a and s_a vanish (to rounding
    # of the summed log prior, 1e-12 relative) ...
    delta = log_prior(after, theta) - log_prior(before, theta)
    assert delta != 0.0
    for name in SHAPE:
        moved = _shifted(theta, name, 1.2)
        assert log_prior(after, moved) - log_prior(before, moved) == pytest.approx(
            delta, rel=1e-12
        ), name
    # ... while it does depend on the amplitude
    moved = _shifted(theta, "beta_a", 1.2)
    assert log_prior(after, moved) - log_prior(before, moved) != pytest.approx(delta, rel=1e-6)


def _assert_constraint_depends_on_the_shape(
    sc: Scenario, observed_data: dict[str, np.ndarray], theta: dict[str, np.ndarray]
) -> None:
    model = attach(sc.measurement, Surface(sc.world.spec), observed_data)
    assert isinstance(model, ModelSpec)
    (constraint,) = model.constraints
    assert constraint.family == "normal" and constraint.scale == sc.measurement.se
    at = float(value(constraint.expr, data=observed_data, params=theta))
    for name in ("beta_a", *SHAPE):
        moved = float(value(constraint.expr, data=observed_data, params=_shifted(theta, name, 1.2)))
        assert moved != pytest.approx(at, rel=1e-6), name
    # and at the truth it realizes the truth (the contrast is built from the one forward)
    assert float(
        value(constraint.expr, data=observed_data, params=sc.world.theta)
    ) == pytest.approx(sc.truth, rel=1e-9)


# -- the runs -------------------------------------------------------------------------------


@dataclass(frozen=True)
class Runs:
    sc: Scenario
    biased: FitResult
    calibrated: CalibratedSpec
    prior: FitResult
    likelihood: FitResult


def _run(sc: Scenario, draws: int) -> Runs:
    biased = fit(sc.world.spec, sc.observed, backend="laplace", draws=draws, seed=FIT_SEED)
    calibrated, prior = _prior_route(sc, biased, sc.measurement, draws)
    return Runs(sc, biased, calibrated, prior, _likelihood_route(sc, sc.measurement, draws))


def _assert_bias_and_both_routes_move_toward_truth(
    runs: Runs, *, prior_min_sd: float, likelihood_min_sd: float
) -> tuple[dict[str, tuple[float, float]], ...]:
    truth = TRUTH["beta_a"]
    b, p, k = _moments(runs.biased), _moments(runs.prior), _moments(runs.likelihood)
    b_mean, b_sd = b["beta_a"]
    # 1. the confounded panel biases the amplitude upward, by more than 2 posterior sd
    assert (b_mean - truth) / b_sd > 2.0, (b_mean, b_sd)
    # 2./3. both routes move the amplitude toward the truth, each by a stated number of sd
    gap = b_mean - truth
    for label, (mean, _), minimum in (
        ("prior", p["beta_a"], prior_min_sd),
        ("likelihood", k["beta_a"], likelihood_min_sd),
    ):
        assert abs(mean - truth) < abs(gap), (label, mean, b_mean)
        assert (b_mean - mean) / b_sd > minimum, (label, mean, b_mean, b_sd)
    return b, p, k


# -- tests ----------------------------------------------------------------------------------


@pytest.mark.slow
def test_both_routes_recover_the_amplitude_and_the_likelihood_route_moves_the_shape() -> None:
    sc = _scenario(n_units=6, n_periods=40)
    assert sc.truth == pytest.approx(40 * TRUTH["beta_a"] * (HI / TRUTH["k_a"]) ** 2 / (1 + 4.0))
    runs = _run(sc, draws=1000)
    b, p, k = _assert_bias_and_both_routes_move_toward_truth(
        runs, prior_min_sd=1.0, likelihood_min_sd=0.5
    )

    # the mechanism (exact): the prior route changes only the amplitude prior; the likelihood
    # route's constraint depends on the shape
    mode = _mode(runs.biased)
    _assert_prior_route_touches_only_the_amplitude_prior(
        sc, runs.calibrated, dict(runs.biased.data), mode
    )
    _assert_constraint_depends_on_the_shape(sc, dict(runs.biased.data), mode)

    # 4. the likelihood route moves the shape posterior by more than one biased sd
    shape_move = {name: abs(k[name][0] - b[name][0]) / b[name][1] for name in SHAPE}
    assert max(shape_move.values()) > 1.0, shape_move
    # The prior route's shape posterior moves as well — through the likelihood, not the
    # prior (module docstring). Recorded, not asserted either way; the exact statement about
    # the prior route is the mechanism assertion above.
    prior_shape_move = {name: abs(p[name][0] - b[name][0]) / b[name][1] for name in SHAPE}
    assert all(np.isfinite(v) for v in prior_shape_move.values())

    # 6. agreement: the likelihood route agrees with the randomized measurement; the prior route
    # moves its realized contrast toward it and shrinks |z| (module docstring on why not more)
    ag_b = _agreement(runs.biased, sc.measurement)
    ag_p = _agreement(runs.prior, sc.measurement)
    ag_k = _agreement(runs.likelihood, sc.measurement)
    assert ag_b.verdict == "disagrees" and abs(ag_b.z) > 2.0
    assert ag_k.verdict == "agrees" and abs(ag_k.z) < 1.0, (ag_k.z, ag_k.posterior_mean)
    assert abs(ag_p.posterior_mean - sc.truth) < abs(ag_b.posterior_mean - sc.truth)
    assert abs(ag_p.z) < abs(ag_b.z)

    # 5. negative controls
    off = _off_measurement(sc, runs.biased)
    off_likelihood = _likelihood_route(sc, off, draws=1000)
    _, off_prior = _prior_route(sc, runs.biased, off, draws=1000)
    for label, result in (("likelihood", off_likelihood), ("prior", off_prior)):
        mean = _moments(result)["beta_a"][0]
        assert mean > b["beta_a"][0], (label, mean, b["beta_a"][0])
    five_off = sc.measurement.model_copy(
        update={"estimate": sc.truth + 5 * sc.measurement.se, "source": "rct_5se_off"}
    )
    for label, result in (("likelihood", runs.likelihood), ("prior", runs.prior)):
        wrong = _agreement(result, five_off)
        assert wrong.verdict != "agrees", (label, wrong.z)
    # on the route that agrees with the truth, the off measurement sits strictly further out
    # (the prior route's realized contrast is still above the truth, so an upward offset is
    # nearer to it — the |z| ordering is only a statement about the agreeing route)
    assert abs(_agreement(runs.likelihood, five_off).z) > abs(ag_k.z)


def test_smoke_both_routes_move_the_amplitude_toward_truth() -> None:
    """The same scenario at smoke size (3 × 20, 300 draws): bias, both routes, mechanism."""
    sc = _scenario(n_units=3, n_periods=20)
    runs = _run(sc, draws=300)
    b, _, k = _assert_bias_and_both_routes_move_toward_truth(
        runs, prior_min_sd=1.0, likelihood_min_sd=0.5
    )
    mode = _mode(runs.biased)
    _assert_prior_route_touches_only_the_amplitude_prior(
        sc, runs.calibrated, dict(runs.biased.data), mode
    )
    _assert_constraint_depends_on_the_shape(sc, dict(runs.biased.data), mode)
    assert max(abs(k[n][0] - b[n][0]) / b[n][1] for n in SHAPE) > 1.0
    assert _agreement(runs.likelihood, sc.measurement).verdict == "agrees"
    off_mean = _moments(_likelihood_route(sc, _off_measurement(sc, runs.biased), 300))["beta_a"][0]
    assert off_mean > b["beta_a"][0]
