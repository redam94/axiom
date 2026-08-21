"""Refutation checks: placebo, permutation, random subset, added noise — rules and provenance."""

from __future__ import annotations

import numpy as np
import pytest

from axiom.core import Unsupported
from axiom.diagnose.refute import (
    Refutation,
    added_noise,
    permutation,
    placebo_treatment,
    random_subset,
)
from axiom.diagnose.spec_curve import default_estimand, realized_point
from axiom.sim import DosePlan, surface_world
from axiom.sim.surface_world import SurfaceWorld
from axiom.surface import FitResult, fit

DRAWS = 40


@pytest.fixture(scope="module")
def world() -> SurfaceWorld:
    # A real, well-identified effect: many zero-dose cells pin the intercept, low noise.
    return surface_world(
        n_units=6,
        n_periods=10,
        treatments=("a",),
        intercept="shared",
        doses=DosePlan(scale=50.0, zero_fraction=0.3),
        noise_sd=0.1,
        seed=21,
    )


@pytest.fixture(scope="module")
def res(world: SurfaceWorld) -> FitResult:
    out = fit(world.spec, world.panel, backend="laplace", draws=DRAWS, chains=1, seed=2)
    assert out.converged
    return out


def _original(world: SurfaceWorld, res: FitResult) -> float:
    got = realized_point(default_estimand(world.spec, world.panel), res, definition="eti", mass=0.9)
    assert not isinstance(got, Unsupported)
    return got.result.summary.mean


# -- placebo ---------------------------------------------------------------------------------


def test_placebo_effect_vanishes_and_passes(world: SurfaceWorld, res: FitResult) -> None:
    r = placebo_treatment(res, seed=0, draws=DRAWS, panel=world.panel)
    assert isinstance(r, Refutation)
    assert r.kind == "placebo_treatment" and r.treatment == "a"
    assert r.original == pytest.approx(_original(world, res))
    assert r.original > 0 and r.original_interval.lower > 0  # the real effect is there
    # the placebo estimate is near zero relative to the real one and does not reach it
    assert abs(r.refuted_mean) < 0.25 * r.original
    assert float(r.detail["placebo_ratio"]) == pytest.approx(r.refuted_mean / r.original)
    assert r.refuted_interval.upper < r.original
    assert r.p_value < r.alpha and r.passed
    assert r.n == DRAWS and r.n_refits == 1 and r.n_failed == 0
    assert "N = 40" in r.rule and "reassigned across units" in r.rule
    assert r.alpha == pytest.approx(0.1)
    assert r.refuted_interval.definition == "eti" and r.refuted_interval.mass == 0.9
    assert r.estimand_hash == default_estimand(world.spec, world.panel).content_hash()
    back = Refutation.from_json(r.to_json())
    assert back == r


def test_placebo_pools_draws_over_refits_and_rebuilds_the_panel(
    world: SurfaceWorld, res: FitResult
) -> None:
    # without `panel=` the check rebuilds the fitted panel from the fit's arrays
    r = placebo_treatment(res, seed=5, n=2, draws=DRAWS)
    assert isinstance(r, Refutation)
    assert r.n == 2 * DRAWS and r.n_refits == 2 and len(r.refuted_estimates) == 2
    assert r.passed


def test_placebo_is_deterministic_by_seed(world: SurfaceWorld, res: FitResult) -> None:
    a = placebo_treatment(res, seed=7, draws=DRAWS, panel=world.panel)
    b = placebo_treatment(res, seed=7, draws=DRAWS, panel=world.panel)
    assert a == b


# -- permutation -----------------------------------------------------------------------------


def test_permutation_p_value_states_n_and_rejects_the_null(
    world: SurfaceWorld, res: FitResult
) -> None:
    r = permutation(res, n=5, seed=0, alpha=0.2, draws=DRAWS, panel=world.panel)
    assert isinstance(r, Refutation)
    assert r.kind == "permutation"
    assert r.n == 5 and r.n_refits == 5 and r.n_failed == 0
    assert len(r.refuted_estimates) == 5
    # the null is centred near zero and far below the real effect
    assert abs(r.refuted_mean) < 0.3 * r.original
    assert r.p_value == pytest.approx(1.0 / 6.0)  # no permutation beats the original
    assert r.detail["n_extreme"] == 0.0
    assert r.passed and "N = 5" in r.rule
    with pytest.raises(ValueError, match="cannot reach"):
        permutation(res, n=3, alpha=0.05, panel=world.panel)


# -- random subset and added noise ---------------------------------------------------------------


def test_random_subset_is_stable_on_a_clean_world(world: SurfaceWorld, res: FitResult) -> None:
    r = random_subset(res, fraction=0.5, n=4, seed=0, alpha=0.1, draws=DRAWS, panel=world.panel)
    assert isinstance(r, Refutation)
    assert r.kind == "random_subset"
    assert r.detail["units_kept"] == 3.0 and r.detail["n_units"] == 6.0
    assert r.n == 4 and r.n_failed == 0
    pts = np.asarray(r.refuted_estimates)
    # subset estimates scatter around the original, and the original is not in the tails
    assert np.all(np.abs(pts - r.original) < 0.5 * abs(r.original))
    assert r.p_value >= r.alpha and r.passed
    assert "random subsets of 3 of 6 units" in r.rule
    with pytest.raises(ValueError, match="keeps every"):
        random_subset(res, fraction=0.99, n=2, panel=world.panel)
    with pytest.raises(ValueError, match="fraction"):
        random_subset(res, fraction=1.5, n=2, panel=world.panel)


def test_added_noise_is_stable_and_reports_sigma(world: SurfaceWorld, res: FitResult) -> None:
    r = added_noise(res, sd_fraction=0.5, n=4, seed=0, alpha=0.1, draws=DRAWS, panel=world.panel)
    assert isinstance(r, Refutation)
    assert r.kind == "added_noise"
    assert r.detail["sigma_source"] == "posterior mean of 'sigma'"
    assert float(r.detail["noise_sd"]) == pytest.approx(0.5 * float(r.detail["sigma_hat"]))
    assert float(r.detail["sigma_hat"]) == pytest.approx(world.noise_sd or 0.0, rel=0.5)
    pts = np.asarray(r.refuted_estimates)
    assert np.all(np.abs(pts - r.original) < 0.5 * abs(r.original))
    assert r.passed
    with pytest.raises(ValueError, match="sd_fraction"):
        added_noise(res, sd_fraction=0.0, panel=world.panel)


# -- failures are typed -------------------------------------------------------------------------


def test_checks_refuse_a_fit_without_a_posterior(world: SurfaceWorld) -> None:
    missing = fit(world.spec, world.panel, backend="no-such-backend")
    with pytest.raises(ValueError, match="no posterior"):
        placebo_treatment(missing, panel=world.panel)
    with pytest.raises(ValueError, match="no posterior"):
        permutation(missing, n=20, panel=world.panel)


def test_every_failed_refit_is_counted_not_dropped(world: SurfaceWorld, res: FitResult) -> None:
    # refits through a missing backend all fail: the check returns Unsupported, with counts
    out = permutation(res, n=20, seed=0, backend="no-such-backend", panel=world.panel)
    assert isinstance(out, Unsupported)
    assert out.detail["n_failed"] == "20"
    assert "every one of 20" in out.reason


def test_refutation_invariants() -> None:
    base = dict(
        kind="permutation",
        estimand_name="e",
        estimand_hash="h",
        treatment="a",
        original=1.0,
        original_interval={"lower": 0.5, "upper": 1.5, "definition": "eti", "mass": 0.9},
        refuted_mean=0.0,
        refuted_sd=0.1,
        refuted_interval={"lower": -0.2, "upper": 0.2, "definition": "eti", "mass": 0.9},
        p_value=0.05,
        n=10,
        alpha=0.1,
        rule="p below alpha",
        passed=True,
    )
    with pytest.raises(ValueError, match="at least one successful"):
        Refutation(**base, n_refits=10, n_failed=10)
    with pytest.raises(ValueError, match="exceed"):
        Refutation(**base, n_refits=5, n_failed=6)
    assert Refutation(**base, n_refits=10, n_failed=1).passed
