"""Posterior-versus-measurement agreement on a simulated world."""

from __future__ import annotations

import numpy as np
import pytest

from axiom.calibrate.check import Agreement, agreement
from axiom.calibrate.evidence import Measurement
from axiom.core import Capability, Intervention, Population, Spec, TimeWindow, Unsupported
from axiom.estimands import Estimand, Level, Quantity
from axiom.sim import SurfaceWorld, surface_world
from axiom.surface import FitResult, HillKernel, fit

TRUTH = {"beta_a": 1.5, "k_a": 1.0, "s_a": 2.0}


@pytest.fixture(scope="module")
def fitted() -> tuple[SurfaceWorld, FitResult]:
    world = surface_world(
        n_units=4,
        n_periods=10,
        treatments=("a",),
        kernels=HillKernel(reference_dose=1.0),
        intercept="shared",
        truth=TRUTH,
        noise_sd=0.05,
        seed=11,
    )
    return world, fit(world.spec, world.panel, backend="laplace", draws=400, seed=3)


def _estimand(world: SurfaceWorld) -> Estimand:
    spec = world.spec
    return Estimand(
        name="lift",
        quantity=Quantity(kind="contrast"),
        treatment=spec.treatment("a"),
        intervention=Intervention(doses={"a": 2.0}),
        reference=Intervention(doses={"a": 0.0}),
        outcome=spec.outcome,
        population=Population(name="panel_units"),
        window=TimeWindow(start=0, stop=world.n_periods, basis="cumulative"),
        level=Level(unit="individual"),
        dimension=spec.outcome_dimension,
    )


def _truth_contrast(world: SurfaceWorld) -> float:
    hi = world.forward({"a": 2.0})
    lo = world.forward({"a": 0.0})
    return float(np.mean(np.sum(hi - lo, axis=1)))


def test_agreement_on_a_simulated_world(fitted: tuple[SurfaceWorld, FitResult]) -> None:
    world, result = fitted
    est = _estimand(world)
    target = _truth_contrast(world)
    se = 0.1 * target
    ok = agreement(result, Measurement(estimand=est, estimate=target, se=se, source="s1"))
    assert isinstance(ok, Agreement)
    assert ok.verdict == "agrees" and abs(ok.z) < 1.0 and ok.inside and ok.p > 0.3
    assert ok.posterior_mean == pytest.approx(target, rel=0.1)
    assert ok.interval.definition == "hdi" and ok.interval.mass == 0.9
    assert ok.realized.status == "downgraded"  # identification was assumed, and says so
    assert ok.estimand_hash == est.content_hash()
    assert Spec.from_json(ok.to_json()) == ok
    off = agreement(result, Measurement(estimand=est, estimate=target + 5 * se, se=se, source="s2"))
    assert isinstance(off, Agreement)
    assert off.verdict == "disagrees" and off.z > 2.0 and not off.inside and off.p < 0.05
    mid = agreement(
        result,
        Measurement(estimand=est, estimate=target + 1.5 * se, se=se, source="s3"),
        tension_at=0.5,
        disagrees_at=3.0,
    )
    assert isinstance(mid, Agreement) and mid.verdict == "tension"
    eti = agreement(
        result,
        Measurement(estimand=est, estimate=target, se=se, source="s1"),
        definition="eti",
        mass=0.5,
    )
    assert (
        isinstance(eti, Agreement) and eti.interval.definition == "eti" and eti.interval.mass == 0.5
    )


def test_agreement_typed_failures_and_arguments(fitted: tuple[SurfaceWorld, FitResult]) -> None:
    world, result = fitted
    m = Measurement(estimand=_estimand(world), estimate=1.0, se=0.1, source="s1")
    out = agreement(result.restrict({Capability.COUNTERFACTUAL}), m)
    assert isinstance(out, Unsupported) and "counterfactual" in out.missing
    with pytest.raises(ValueError, match="tension_at"):
        agreement(result, m, tension_at=2.0, disagrees_at=1.0)
