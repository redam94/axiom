"""Unit tests for ``axiom.diagnose.coverage``."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from axiom.core import Population, Posterior, Prior, TimeWindow, Unverified, clopper_pearson
from axiom.diagnose.coverage import (
    CoverageResult,
    EstimandCoverageResult,
    ParameterCoverage,
    SupportsWorld,
    WorldView,
    coverage,
    estimand_coverage,
    misspecify,
    truth_producer,
)
from axiom.estimands import Level, realize, standard_estimands
from axiom.sim import DosePlan, surface_world
from axiom.surface import FitResult, GeometricCarryover, HillKernel, LinearKernel, NoCarryover, fit

AMP = Prior(family="lognormal", hyper={"mu": 0.0, "sigma": 0.5})


def _world(seed: int, *, hill: bool = False, **kw: object) -> SupportsWorld:
    kernel = (
        HillKernel(reference_dose=1.0, amplitude_prior=AMP)
        if hill
        else LinearKernel(reference_dose=1.0, amplitude_prior=AMP)
    )
    return surface_world(
        n_units=4,
        n_periods=8,
        treatments=("a",),
        kernels=kernel,
        intercept="shared",
        noise_sd=0.3,
        seed=seed,
        doses=DosePlan(zero_fraction=0.2),
        **kw,  # type: ignore[arg-type]
    )


def test_world_view_and_misspecify() -> None:
    w = _world(0, carryover=GeometricCarryover(max_lag=4))
    assert isinstance(w, SupportsWorld)
    view = misspecify(w, w.spec.model_copy(update={"carryover": {"a": NoCarryover()}}))
    assert isinstance(view, WorldView) and isinstance(view, SupportsWorld)
    assert view.panel is w.panel and "lam_a" in view.theta


def test_coverage_counts_bookkeeping_and_region() -> None:
    out = coverage(_world, n=6, mass=0.9, alpha=0.01, draws=60, seed=5)
    assert isinstance(out, CoverageResult)
    assert out.n == 6 and out.n_fitted + out.n_failed_fits == 6
    region = clopper_pearson(out.n_fitted, 0.9, 0.01)
    for p in out.parameters:
        assert p.n == out.n_fitted and p.region == region and 0 <= p.covered <= p.n
        assert p.passed == region.accepts(p.covered)
        assert p.definition == "eti" and p.mass == 0.9
    assert {p.name for p in out.parameters} == {"alpha", "beta_rate_a", "sigma"}
    assert CoverageResult.from_json(out.to_json()) == out
    narrowed = coverage(_world, n=3, parameters=("sigma",), draws=40, seed=5)
    assert [p.name for p in narrowed.parameters] == ["sigma"]


def test_coverage_records_failed_fits_and_validates_arguments() -> None:
    def failing(world: SupportsWorld, seed: int) -> FitResult:
        res = fit(world.spec, world.panel, backend="laplace", draws=30, chains=1, seed=seed)
        if seed % 2 == 0:
            return FitResult(
                res.surface,
                Unverified(reason="declined on purpose"),
                res.data,
                None,
                res.provenance,
            )
        return res

    out = coverage(_world, n=4, fit=failing, seed=0)
    assert out.n_failed_fits == 2 and out.n_fitted == 2
    assert all("declined on purpose" in r for r in out.failure_reasons)
    assert all(p.n == 2 for p in out.parameters)
    with pytest.raises(ValueError, match="n must be"):
        coverage(_world, n=0)
    with pytest.raises(ValueError, match="not free"):
        coverage(_world, n=1, parameters=("nope",), draws=20)
    with pytest.raises(ValueError, match="eti or hdi"):
        coverage(_world, n=1, definition="wald")


def test_parameter_coverage_validates_consistency() -> None:
    region = clopper_pearson(10, 0.9, 0.01)
    with pytest.raises(ValidationError, match="passed must equal"):
        ParameterCoverage(
            name="x", n=10, covered=2, mass=0.9, definition="eti", region=region, passed=True
        )
    with pytest.raises(ValidationError, match="acceptance region"):
        ParameterCoverage(
            name="x", n=12, covered=2, mass=0.9, definition="eti", region=region, passed=False
        )


def test_truth_producer_realizes_the_true_estimand() -> None:
    w = surface_world(
        n_units=3,
        n_periods=5,
        treatments=("a",),
        kernels=HillKernel(reference_dose=1.0, amplitude_prior=AMP),
        intercept="shared",
        seed=2,
    )
    producer = truth_producer(w)
    assert isinstance(producer.posterior, Posterior) and producer.n_draws() == 2
    est = standard_estimands(
        w.spec.treatments[0],
        w.spec.outcome,
        Population(name="all"),
        TimeWindow(start=0, stop=5),
        Level(unit="aggregate"),
        dose=1.0,
    ).get("contrast_at_dose")
    got = realize(est, producer, assume_identified=True, definition="eti", mass=0.9)
    assert hasattr(got, "summary")
    # at the prior centre beta = 1 and k = 1, s = 1: the Hill response at dose 1 is 0.5 per cell
    expected = 0.5 * w.n_units * w.n_periods
    assert got.summary.mean == pytest.approx(expected, rel=1e-9)  # type: ignore[union-attr]
    assert got.summary.sd == 0.0  # type: ignore[union-attr]


def _hill_world(seed: int) -> SupportsWorld:
    return _world(seed, hill=True)


def test_estimand_coverage_bookkeeping() -> None:
    w = _hill_world(0)
    est = standard_estimands(
        w.spec.treatments[0],
        w.spec.outcome,
        Population(name="all"),
        TimeWindow(start=0, stop=8),
        Level(unit="aggregate"),
        dose=1.0,
    )
    chosen = [est.get("contrast_at_dose"), est.get("marginal_at_dose")]
    out = estimand_coverage(_hill_world, estimands=chosen, n=4, draws=60, seed=1)
    assert isinstance(out, EstimandCoverageResult)
    assert out.n_fitted + out.n_failed_fits == 4
    assert [e.name for e in out.estimands] == ["contrast_at_dose", "marginal_at_dose"]
    for e in out.estimands:
        assert e.n == out.n_fitted and len(e.true_values) == e.n
        assert e.region == clopper_pearson(e.n, 0.9, 0.01)
        assert e.passed == e.region.accepts(e.covered)
    assert EstimandCoverageResult.from_json(out.to_json()) == out
    with pytest.raises(ValueError, match="distinct"):
        estimand_coverage(_hill_world, estimands=[chosen[0], chosen[0]], n=1)


def test_estimand_coverage_uses_truth_world_for_a_misspecified_view() -> None:
    gen = _world(0, hill=True, carryover=GeometricCarryover(max_lag=4))
    est = standard_estimands(
        gen.spec.treatments[0],
        gen.spec.outcome,
        Population(name="all"),
        TimeWindow(start=0, stop=8),
        Level(unit="aggregate"),
        dose=1.0,
    ).get("contrast_at_dose")
    naive_spec = gen.spec.model_copy(update={"carryover": {"a": NoCarryover()}})
    originals: dict[str, SupportsWorld] = {}

    def factory(seed: int) -> SupportsWorld:
        w = _world(seed, hill=True, carryover=GeometricCarryover(max_lag=4))
        originals[w.panel.content_hash()] = w
        return misspecify(w, naive_spec)

    out = estimand_coverage(
        factory,
        estimands=[est],
        n=2,
        draws=40,
        seed=3,
        truth_world=lambda view: originals[view.panel.content_hash()],
    )
    assert out.n_fitted == 2 and out.estimands[0].n == 2
    assert all(np.isfinite(out.estimands[0].true_values))
