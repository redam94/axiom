"""Golden: every ``calibration.*`` fixture key reproduces the parent's value at rtol 1e-12.

``lognormal_sigma_from_moments`` lives in ``axiom.calibrate.likelihood``; the
module is skipped (not silently passed) while that module is absent.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from axiom.calibrate.evidence import combine_inverse_variance
from axiom.calibrate.prior import design_factor, mean_sd_to_gamma

FIXTURE = Path(__file__).parent / "parent_values.json"


def load() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text())


def _lognormal_sigma_from_moments() -> Callable[..., float]:
    try:
        mod = importlib.import_module("axiom.calibrate.likelihood")
    except ImportError as exc:  # pragma: no cover - only before the module lands
        pytest.skip(f"axiom.calibrate.likelihood not available yet: {exc}")
    fn: Callable[..., float] = mod.lognormal_sigma_from_moments
    return fn


def _cases(prefix: str) -> list[tuple[str, dict[str, Any]]]:
    data = load()
    return sorted((k, v) for k, v in data.items() if k.startswith(prefix))


FUNCTIONS: dict[str, Callable[..., Any]] = {
    "calibration.experiment::combine_inverse_variance": combine_inverse_variance,
    "calibration.experiment::design_factor": design_factor,
    "calibration.experiment::mean_sd_to_gamma": mean_sd_to_gamma,
}


def test_every_calibration_key_is_covered() -> None:
    keys = {k.rsplit("::", 1)[0] for k, _ in _cases("calibration.")}
    covered = set(FUNCTIONS) | {"calibration.likelihood::lognormal_sigma_from_moments"}
    assert keys <= covered, f"uncovered golden keys: {sorted(keys - covered)}"


@pytest.mark.parametrize(("key", "case"), _cases("calibration.experiment::"))
def test_experiment_values(key: str, case: dict[str, Any]) -> None:
    fn = FUNCTIONS[key.rsplit("::", 1)[0]]
    got = fn(**case["kwargs"])
    np.testing.assert_allclose(np.asarray(got), np.asarray(case["value"]), rtol=case["rtol"])


@pytest.mark.parametrize(("key", "case"), _cases("calibration.likelihood::"))
def test_likelihood_values(key: str, case: dict[str, Any]) -> None:
    fn = _lognormal_sigma_from_moments()
    got = fn(**case["kwargs"])
    np.testing.assert_allclose(got, case["value"], rtol=case["rtol"])


def test_design_factor_is_mean_of_ratios() -> None:
    """Pin the formula choice: the other candidates miss the fixture by >1e-4."""
    case = load()["calibration.experiment::design_factor::seeded_lognormal"]
    b = np.asarray(case["kwargs"]["beta_samples"])
    c = np.asarray(case["kwargs"]["contribution_samples"])
    target = case["value"]
    assert design_factor(b, c) == pytest.approx(float(np.mean(c / b)), rel=1e-15)
    for other in (c.mean() / b.mean(), np.median(c / b), (b * c).sum() / (b * b).sum()):
        assert abs(other - target) / target > 1e-4
