"""Gate (Phase 8, exit criterion 2): nominal 90 % intervals cover the truth at the nominal
rate on a ``sim`` world, and a deliberately mis-specified world fails the same check.

The criterion is the exact Clopper–Pearson acceptance region
``core.clopper_pearson(n, 0.9, alpha=0.01)`` on the count of covering
intervals, stated per tier:

* **fast** (default): ``n = 30`` replications of a Hill surface world
  (``8 × 16`` panel, Laplace, 100 draws), region ``[22, 30]`` of 30
  (73.3 %–100 %). Small, but the mis-specified world — truth generated
  with geometric carryover (``lambda = 0.7``, six lags), fitted without —
  covers the intercept about once in 30 and fails by a mile.
* **slow** (``-m slow``): ``n = 200``, region ``[168, 190]`` of 200
  (84 %–95 %) — the roadmap's "88–92 %" cannot be an *exact* test at
  ``n = 200`` (the binomial sd of the rate is 2.1 points), so the exact
  region is what is asserted and the observed rate is reported alongside.
  The same tier scores two realized estimands (``contrast_at_dose``,
  ``marginal_at_dose``) through ``estimand_coverage`` at ``n = 100``.
"""

from __future__ import annotations

import pytest

from axiom.core import Population, Prior, TimeWindow, clopper_pearson
from axiom.diagnose.coverage import (
    CoverageResult,
    EstimandCoverageResult,
    SupportsWorld,
    coverage,
    estimand_coverage,
    misspecify,
)
from axiom.estimands import Level, standard_estimands
from axiom.sim import DosePlan, surface_world
from axiom.surface import GeometricCarryover, HillKernel, NoCarryover

AMPLITUDE = Prior(family="lognormal", hyper={"mu": 0.0, "sigma": 0.5})
MASS = 0.9
ALPHA = 0.01
FAST_N = 30
SLOW_N = 200
N_UNITS, N_PERIODS = 8, 16


def _world(seed: int, *, carryover: bool = False) -> SupportsWorld:
    return surface_world(
        n_units=N_UNITS,
        n_periods=N_PERIODS,
        treatments=("a",),
        kernels=HillKernel(reference_dose=1.0, amplitude_prior=AMPLITUDE),
        carryover=GeometricCarryover(max_lag=6) if carryover else None,
        truth={"lam_a": 0.7} if carryover else None,
        intercept="shared",
        noise_sd=0.3,
        seed=seed,
        doses=DosePlan(zero_fraction=0.2),
    )


def _misspecified(seed: int) -> SupportsWorld:
    """Truth with carryover; fitted as if there were none."""
    w = _world(seed, carryover=True)
    return misspecify(w, w.spec.model_copy(update={"carryover": {"a": NoCarryover()}}))


def _report(label: str, out: CoverageResult) -> str:
    lines = [f"{label}: n={out.n} fitted={out.n_fitted} failed_fits={out.n_failed_fits}"]
    for p in out.parameters:
        lines.append(
            f"  {p.name}: {p.covered}/{p.n} = {p.rate:.3f} "
            f"region [{p.region.lower}, {p.region.upper}] passed={p.passed}"
        )
    return "\n".join(lines)


def _estimands(world: SupportsWorld) -> list:  # type: ignore[type-arg]
    registry = standard_estimands(
        world.spec.treatments[0],
        world.spec.outcome,
        Population(name="all"),
        TimeWindow(start=0, stop=N_PERIODS),
        Level(unit="aggregate"),
        dose=1.0,
    )
    return [registry.get("contrast_at_dose"), registry.get("marginal_at_dose")]


# -- fast tier -------------------------------------------------------------------------------


def test_region_is_stated_for_the_fast_n() -> None:
    region = clopper_pearson(FAST_N, MASS, ALPHA)
    assert (region.lower, region.upper) == (22, 30)


def test_well_specified_world_covers_and_misspecified_world_fails_fast() -> None:
    good = coverage(_world, n=FAST_N, mass=MASS, alpha=ALPHA, draws=100, seed=100)
    assert good.n_failed_fits == 0, good.failure_reasons
    assert good.nominal_region == clopper_pearson(FAST_N, MASS, ALPHA)
    assert good.passed, _report("well-specified", good)
    bad = coverage(_misspecified, n=FAST_N, mass=MASS, alpha=ALPHA, draws=100, seed=100)
    assert not bad.passed, _report("mis-specified", bad)
    assert "alpha" in bad.failed_parameters, _report("mis-specified", bad)


# -- slow tier -------------------------------------------------------------------------------


@pytest.mark.slow
def test_region_is_stated_for_the_slow_n() -> None:
    region = clopper_pearson(SLOW_N, MASS, ALPHA)
    assert (region.lower, region.upper) == (168, 190)


@pytest.mark.slow
def test_well_specified_world_covers_and_misspecified_world_fails_slow() -> None:
    good = coverage(_world, n=SLOW_N, mass=MASS, alpha=ALPHA, draws=200, seed=1000)
    assert good.n_failed_fits == 0, good.failure_reasons
    assert good.passed, _report("well-specified", good)
    bad = coverage(_misspecified, n=SLOW_N, mass=MASS, alpha=ALPHA, draws=200, seed=1000)
    assert not bad.passed and "alpha" in bad.failed_parameters, _report("mis-specified", bad)


@pytest.mark.slow
def test_estimand_coverage_slow() -> None:
    out = estimand_coverage(
        _world, estimands=_estimands(_world(0)), n=100, mass=MASS, alpha=ALPHA, draws=200, seed=2000
    )
    assert isinstance(out, EstimandCoverageResult)
    assert out.n_failed_fits == 0, out.failure_reasons
    assert out.passed, [(e.name, e.covered, e.n, e.region) for e in out.estimands]
