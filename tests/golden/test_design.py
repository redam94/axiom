"""Golden: every ``planning.*`` case in the parent fixture reproduced by ``axiom.design``.

Parent vocabulary maps onto axiom's: ``sigma_k → prior_sd``, ``sigma_exp →
experiment_se``, ``sigma_post → posterior_sd``, ``tau → prior_sd``, ``weeks
→ periods``, ``geo_holdout → "cluster_holdout"``, ``ghost_ads → "ghost"``,
``roi_median → reference_value``, ``cpa_* → precision.cost_per_outcome_*``
with ``lift → effect``. An unmapped ``planning`` function fails the test
rather than being skipped: the fixture is the contract.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import pytest

from axiom.design.eig import decayed_sd, eig_gaussian, experiment_se_for_design
from axiom.design.evoi import preposterior_sd_ratio
from axiom.design.precision import (
    cost_per_outcome_interval,
    cost_per_outcome_power,
    max_detectable_cost_per_outcome,
)
from tests.golden.test_manifest import load

pytestmark = pytest.mark.golden

_DESIGN_KIND = {"geo_holdout": "cluster_holdout", "ghost_ads": "ghost"}


def _cpa_interval(kw: Mapping[str, Any]) -> dict[str, Any]:
    r = cost_per_outcome_interval(kw["cost"], kw["lift"], kw["se_lift"])
    return {
        "alpha": r.alpha,
        "cpa": r.estimate,
        "hi": r.upper,
        "lo": r.lower,
        "lift_interval": [r.effect_interval.lower, r.effect_interval.upper],
        "naive_hi": r.naive_interval.upper,
        "naive_lo": r.naive_interval.lower,
        "naive_se": r.naive_se,
        "status": r.status,
    }


_PORTS: dict[str, Callable[[Mapping[str, Any]], Any]] = {
    "planning.cpa::cpa_interval": _cpa_interval,
    "planning.cpa::cpa_power": lambda kw: cost_per_outcome_power(
        kw["cost"], kw["true_lift"], kw["se_lift"]
    ).power,
    "planning.cpa::max_detectable_cpa": lambda kw: max_detectable_cost_per_outcome(
        kw["cost"], kw["lift_mde"]
    ),
    "planning.eig::decayed_sigma": lambda kw: decayed_sd(
        kw["sigma_post"], kw["weeks_elapsed"], kw["half_life_weeks"]
    ),
    "planning.eig::eig_gaussian": lambda kw: eig_gaussian(kw["sigma_k"], kw["sigma_exp"]),
    "planning.eig::sigma_exp_for_design": lambda kw: experiment_se_for_design(
        _DESIGN_KIND[kw["design_type"]], kw["roi_median"]
    ),
    "planning.evoi::preposterior_sd_ratio": lambda kw: preposterior_sd_ratio(
        kw["tau"], kw["sigma_exp"]
    ),
}

CASES = sorted(k for k in load() if k.startswith("planning."))


def _compare(got: Any, want: Any, rtol: float, path: str) -> None:
    if isinstance(want, dict):
        assert isinstance(got, dict), f"{path}: expected a mapping, got {type(got).__name__}"
        assert set(got) == set(want), f"{path}: keys {set(got) ^ set(want)} differ"
        for k in want:
            _compare(got[k], want[k], rtol, f"{path}.{k}")
    elif isinstance(want, list):
        assert len(got) == len(want), f"{path}: length differs"
        for i, (g, w) in enumerate(zip(got, want, strict=True)):
            _compare(g, w, rtol, f"{path}[{i}]")
    elif isinstance(want, str):
        assert got == want, f"{path}: {got!r} != {want!r}"
    else:
        assert got == pytest.approx(want, rel=rtol), f"{path}: {got!r} != {want!r}"


def test_every_planning_case_is_mapped() -> None:
    assert CASES, "the fixture has no planning.* cases"
    unmapped = sorted({c.rsplit("::", 1)[0] for c in CASES} - set(_PORTS))
    assert not unmapped, f"planning functions without an axiom port: {unmapped}"


@pytest.mark.parametrize("key", CASES)
def test_planning_golden(key: str) -> None:
    case = load()[key]
    fn = _PORTS[key.rsplit("::", 1)[0]]
    got = fn(case["kwargs"])
    _compare(got, case["value"], float(case["rtol"]), key)
