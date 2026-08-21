"""The likelihood route: constraints built from measurements, attached, and fit."""

from __future__ import annotations

import numpy as np
import pytest

from axiom.calibrate.evidence import Measurement
from axiom.calibrate.likelihood import (
    attach,
    constraint_for,
    fit_calibrated,
    lognormal_mu_from_moments,
    lognormal_sigma_from_moments,
)
from axiom.core import (
    Constraint,
    D,
    Intervention,
    ModelSpec,
    Population,
    Posterior,
    TimeWindow,
    Treatment,
    Unsupported,
    dimensionless,
    jax_available,
    log_density,
    unconstrain,
    value,
)
from axiom.estimands import Estimand, Level, Quantity
from axiom.sim import SurfaceWorld, surface_world
from axiom.surface import HillKernel, fit
from axiom.surface.carryover import GeometricCarryover

TRUTH = {"beta_a": 1.5, "k_a": 1.0, "s_a": 2.0}


def _world(noise_sd: float = 0.05, carryover: GeometricCarryover | None = None) -> SurfaceWorld:
    return surface_world(
        n_units=4,
        n_periods=12,
        treatments=("a",),
        kernels=HillKernel(reference_dose=1.0),
        carryover=carryover,
        intercept="shared",
        truth=TRUTH,
        noise_sd=noise_sd,
        seed=7,
    )


def _estimand(
    world: SurfaceWorld,
    *,
    kind: str = "contrast",
    hi: float = 2.0,
    lo: float = 0.0,
    mode: str = "set",
    window: TimeWindow | None = None,
    level: str = "individual",
    scale: str = "natural",
    treatment: Treatment | None = None,
) -> Estimand:
    spec = world.spec
    t = treatment or spec.treatment("a")
    out_dim = spec.outcome_dimension
    dim = {
        "contrast": out_dim,
        "ratio": out_dim / D.currency,
        "marginal": out_dim / D.currency,
        "area": out_dim * D.currency,
        "elasticity": dimensionless(),
    }[kind]
    return Estimand(
        name=f"lift_{kind}",
        quantity=Quantity(kind=kind, scale=scale),  # type: ignore[arg-type]
        treatment=t,
        intervention=Intervention(doses={"a": hi}, mode=mode),  # type: ignore[arg-type]
        reference=(
            Intervention(doses={"a": lo}, mode=mode)  # type: ignore[arg-type]
            if kind in ("contrast", "ratio", "area")
            else None
        ),
        outcome=spec.outcome,
        population=Population(name="panel_units"),
        window=window or TimeWindow(start=0, stop=world.n_periods, basis="cumulative"),
        level=Level(unit=level),  # type: ignore[arg-type]
        dimension=dim,
    )


def _measurement(estimand: Estimand, estimate: float, se: float) -> Measurement:
    return Measurement(
        estimand=estimand, estimate=estimate, se=se, source="experiment_1", method="difference"
    )


def _hill(d: float) -> float:
    u = (d / TRUTH["k_a"]) ** TRUTH["s_a"]
    return TRUTH["beta_a"] * u / (1.0 + u)


def test_lognormal_moments_match_golden() -> None:
    assert lognormal_sigma_from_moments(1.0, 0.5) == pytest.approx(0.47238072707743883, rel=1e-12)
    assert lognormal_sigma_from_moments(2.5, 0.4) == pytest.approx(0.1589899593819325, rel=1e-12)
    sigma = lognormal_sigma_from_moments(2.0, 0.3)
    assert lognormal_mu_from_moments(2.0, 0.3) == pytest.approx(np.log(2.0) - sigma**2 / 2)
    # the moments round-trip: a lognormal(mu, sigma) has mean 2.0 and sd 0.3
    mu = lognormal_mu_from_moments(2.0, 0.3)
    assert np.exp(mu + sigma**2 / 2) == pytest.approx(2.0)
    assert np.sqrt((np.exp(sigma**2) - 1) * np.exp(2 * mu + sigma**2)) == pytest.approx(0.3)
    with pytest.raises(ValueError):
        lognormal_sigma_from_moments(0.0, 0.5)
    with pytest.raises(ValueError):
        lognormal_sigma_from_moments(1.0, -0.1)


def test_contrast_constraint_evaluates_to_the_aggregate_functional() -> None:
    world = _world()
    surface, data, theta = world.surface, world.data, world.theta
    t = world.n_periods
    per_cell = _hill(2.0) - _hill(0.0)
    cases = {
        ("cumulative", "individual"): per_cell * t,
        ("per_period", "individual"): per_cell,
        ("cumulative", "cluster"): per_cell * t * world.n_units,
        ("per_period", "aggregate"): per_cell * world.n_units,
    }
    for (basis, level), expected in cases.items():
        est = _estimand(
            world,
            window=TimeWindow(start=0, stop=t, basis=basis),  # type: ignore[arg-type]
            level=level,
        )
        c = constraint_for(_measurement(est, expected, 0.1), surface, data)
        assert isinstance(c, Constraint)
        assert c.family == "normal" and c.scale == 0.1 and c.observed == expected
        got = value(c.expr, data=data, params=theta)
        assert np.shape(got) == () and float(got) == pytest.approx(expected)
        assert c.detail["estimand_hash"] == est.content_hash()
        assert c.detail["measurement"] == "experiment_1"
    # a sub-window of the horizon
    est = _estimand(world, window=TimeWindow(start=3, stop=8, basis="cumulative"))
    c = constraint_for(_measurement(est, 1.0, 0.1), surface, data)
    assert isinstance(c, Constraint)
    assert float(value(c.expr, data=data, params=theta)) == pytest.approx(per_cell * 5)
    assert c.detail["window"] == "[3,8):cumulative"
    with pytest.raises(ValueError, match="runs past"):
        constraint_for(
            _measurement(_estimand(world, window=TimeWindow(start=0, stop=t + 1)), 1.0, 0.1),
            surface,
            data,
        )


def test_ratio_and_marginal_are_cell_means() -> None:
    world = _world()
    surface, data, theta = world.surface, world.data, world.theta
    ratio = _estimand(world, kind="ratio", hi=2.0, lo=0.5, level="cluster")
    c = constraint_for(_measurement(ratio, 1.0, 0.1), surface, data)
    assert isinstance(c, Constraint)
    assert float(value(c.expr, data=data, params=theta)) == pytest.approx(
        (_hill(2.0) - _hill(0.5)) / 1.5
    )
    marginal = _estimand(world, kind="marginal", hi=1.0)
    c = constraint_for(_measurement(marginal, 1.0, 0.1), surface, data)
    assert isinstance(c, Constraint)
    # d/dd beta u^s/(1+u^s) at d = k: beta s / (4 k)
    assert float(value(c.expr, data=data, params=theta)) == pytest.approx(
        TRUTH["beta_a"] * TRUTH["s_a"] / 4.0
    )


def test_supplied_doses_replace_the_panel_doses() -> None:
    world = _world()
    surface, data, theta = world.surface, world.data, world.theta
    est = _estimand(world, kind="contrast", hi=2.0, lo=1.0, mode="scale")
    c = constraint_for(_measurement(est, 1.0, 0.1), surface, data, doses={"a": 0.5})
    assert isinstance(c, Constraint)
    assert c.detail["doses"] == "supplied:a"
    assert float(value(c.expr, data=data, params=theta)) == pytest.approx(
        (_hill(1.0) - _hill(0.5)) * world.n_periods
    )
    # a path: the same for every unit, as a grid or a vector
    path = np.linspace(0.2, 1.0, world.n_periods)
    expected = float(np.sum([_hill(2 * d) - _hill(d) for d in path]))
    for doses in (path, np.tile(path, (world.n_units, 1))):
        c = constraint_for(_measurement(est, 1.0, 0.1), surface, data, doses={"a": doses})
        assert isinstance(c, Constraint)
        assert float(value(c.expr, data=data, params=theta)) == pytest.approx(expected)
    # without doses the panel's observed doses are the base
    c = constraint_for(_measurement(est, 1.0, 0.1), surface, data)
    assert isinstance(c, Constraint) and c.detail["doses"] == "observed"
    d = data["a"]
    assert float(value(c.expr, data=data, params=theta)) == pytest.approx(
        float(np.mean(np.sum(_hill_array(2 * d) - _hill_array(d), axis=1)))
    )


def _hill_array(d: np.ndarray) -> np.ndarray:
    u = (d / TRUTH["k_a"]) ** TRUTH["s_a"]
    return np.asarray(TRUTH["beta_a"] * u / (1.0 + u))


def test_set_intervention_on_a_carryover_surface_uses_constant_paths() -> None:
    world = _world(carryover=GeometricCarryover(max_lag=4))
    surface, data, theta = world.surface, world.data, world.theta
    est = _estimand(world, window=TimeWindow(start=2, stop=10, basis="cumulative"))
    c = constraint_for(_measurement(est, 1.0, 0.1), surface, data)
    assert isinstance(c, Constraint)
    hi = world.forward({"a": 2.0})
    lo = world.forward({"a": 0.0})
    expected = float(np.mean(np.sum((hi - lo)[:, 2:10], axis=1)))
    assert float(value(c.expr, data=data, params=theta)) == pytest.approx(expected)
    # a marginal through carryover is not a per-row expression
    out = constraint_for(_measurement(_estimand(world, kind="marginal"), 1.0, 0.1), surface, data)
    assert isinstance(out, Unsupported) and "carryover" in out.reason


def test_lognormal_path_for_log_scale_estimands() -> None:
    world = _world()
    surface, data, theta = world.surface, world.data, world.theta
    est = _estimand(world, kind="ratio", hi=2.0, lo=0.0, scale="log")
    m = _measurement(est, 0.6, 0.12)
    c = constraint_for(m, surface, data)
    assert isinstance(c, Constraint)
    assert c.family == "lognormal"
    assert c.observed == 0.6
    assert c.scale == pytest.approx(lognormal_sigma_from_moments(0.6, 0.12))
    assert float(value(c.expr, data=data, params=theta)) == pytest.approx(_hill(2.0) / 2.0)
    assert isinstance(constraint_for(_measurement(est, -0.1, 0.12), surface, data), Unsupported)


def test_unsupported_paths_are_typed() -> None:
    world = _world()
    surface, data = world.surface, world.data
    for kind in ("area", "elasticity"):
        out = constraint_for(_measurement(_estimand(world, kind=kind), 1.0, 0.1), surface, data)
        assert isinstance(out, Unsupported) and kind in out.reason
    grid = np.linspace(0.1, 1.0, world.n_units * world.n_periods).reshape(
        world.n_units, world.n_periods
    )
    est = _estimand(world, hi=2.0, lo=1.0, mode="scale")
    out = constraint_for(_measurement(est, 1.0, 0.1), surface, data, doses={"a": grid})
    assert isinstance(out, Unsupported) and "per_unit_dose_grid" in out.missing
    other_unit = world.spec.treatment("a").model_copy(update={"unit": "EUR"})
    out = constraint_for(
        _measurement(_estimand(world, treatment=other_unit), 1.0, 0.1), surface, data
    )
    assert isinstance(out, Unsupported) and "unit_conversion" in out.missing
    strata = _estimand(world).model_copy(
        update={"population": Population(name="p", strata={"size": {"big": 0.5, "small": 0.5}})}
    )
    assert isinstance(constraint_for(_measurement(strata, 1.0, 0.1), surface, data), Unsupported)
    assert isinstance(
        attach(_measurement(_estimand(world, kind="area"), 1.0, 0.1), surface, data), Unsupported
    )
    assert isinstance(
        fit_calibrated(
            world.spec, world.panel, [_measurement(_estimand(world, kind="area"), 1.0, 0.1)]
        ),
        Unsupported,
    )
    with pytest.raises(KeyError):
        constraint_for(_measurement(_estimand(world), 1.0, 0.1), surface, data, doses={"b": 1.0})


def test_attach_appends_a_validated_constraint() -> None:
    world = _world()
    est = _estimand(world)
    model = attach(_measurement(est, 10.0, 0.5), world.surface, world.data)
    assert isinstance(model, ModelSpec)
    assert model.mean == world.model.mean and len(model.constraints) == 1
    assert model.constraints[0].name == "lift_contrast@experiment_1"
    z = unconstrain(model, world.theta)
    assert np.isfinite(log_density(model, world.data, z))
    assert log_density(model, world.data, z) != log_density(world.model, world.data, z)
    if jax_available():
        import jax

        from axiom.core import compile_log_density

        jax.config.update("jax_enable_x64", True)
        assert float(compile_log_density(model)(world.data, z)) == pytest.approx(
            log_density(model, world.data, z), abs=1e-10
        )


@pytest.mark.slow
def test_tight_constraint_at_truth_pulls_the_amplitude_mode() -> None:
    """A noisy panel leaves beta loose; a tight measurement of the contrast pins it."""
    world = _world(noise_sd=3.0)
    est = _estimand(world)
    truth = float(value(world.model.mean, data=world.data, params=world.theta).mean())
    del truth
    per_cell = _hill(2.0) - _hill(0.0)
    target = per_cell * world.n_periods
    m = _measurement(est, target, 0.01 * target)
    plain = fit(world.spec, world.panel, backend="laplace", draws=300, seed=1)
    calibrated = fit_calibrated(world.spec, world.panel, [m], backend="laplace", draws=300, seed=1)
    assert not isinstance(calibrated, Unsupported)
    assert isinstance(plain.posterior, Posterior) and isinstance(calibrated.posterior, Posterior)
    beta_plain = plain.posterior.flat("beta_a")
    beta_cal = calibrated.posterior.flat("beta_a")
    # the constrained posterior is nearer the truth and tighter in the amplitude
    assert abs(beta_cal.mean() - TRUTH["beta_a"]) < abs(beta_plain.mean() - TRUTH["beta_a"])
    assert beta_cal.std() < beta_plain.std()
    # and its realized contrast sits on the measurement within its se
    c = calibrated.provenance["constraints"][0]
    assert c["observed"] == target and c["family"] == "normal"
    assert calibrated.provenance["measurement_sources"] == ["experiment_1"]
    assert calibrated.provenance["constrained_model_hash"] != calibrated.provenance["model_hash"]
    assert calibrated.provenance["route"] == "likelihood"
    expr = attach(m, world.surface, world.data)
    assert isinstance(expr, ModelSpec)
    mode = {
        k: np.median(calibrated.posterior.flat(k), axis=0) for k in calibrated.posterior.names()
    }
    realized = float(value(expr.constraints[0].expr, data=world.data, params=mode))
    assert realized == pytest.approx(target, rel=0.05)
