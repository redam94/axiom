from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from axiom.core import Unsupported
from axiom.design import (
    STAGEWISE_ORDERING,
    LookSchedule,
    StoppedEstimate,
    StoppingRule,
    monitor,
    obrien_fleming,
    pocock,
    stagewise_tail,
    stopped_estimate,
)

_FOUR = LookSchedule(labels=("L1", "L2", "L3", "L4"), information=(0.25, 0.5, 0.75, 1.0))


def _pocock() -> StoppingRule:
    return StoppingRule(name="Pocock-4", looks=_FOUR, boundaries=(pocock(0.05, _FOUR),))


def _obf() -> StoppingRule:
    return StoppingRule(name="OBF-4", looks=_FOUR, boundaries=(obrien_fleming(0.05, _FOUR),))


def _one_look() -> StoppingRule:
    looks = LookSchedule(labels=("final",), information=(1.0,))
    return StoppingRule(name="one-look", looks=looks, boundaries=(pocock(0.05, looks),))


def _stop_at(rule: StoppingRule, look: int, z: float, se: float | None = None) -> StoppedEstimate:
    zs = [0.5] * look + [z]
    ses = None if se is None else [1.0] * look + [se]
    result = stopped_estimate(monitor(rule, zs, ses=ses))
    assert isinstance(result, StoppedEstimate)
    return result


# -- the tail probability -------------------------------------------------------------------


def test_the_tail_is_strictly_increasing_in_drift() -> None:
    rule = _pocock()
    tails = [stagewise_tail(rule, 2, 2.5, d) for d in (-4.0, -1.0, 0.0, 1.0, 3.0, 6.0)]
    assert all(a < b for a, b in zip(tails, tails[1:], strict=False))
    assert 0.0 <= tails[0] and tails[-1] <= 1.0


def test_at_the_first_look_the_tail_is_the_fixed_sample_tail() -> None:
    """Nothing precedes look 1, so the stage-wise set is just ``Z_1 >= z``."""
    rule = _pocock()
    t, z, drift = 0.25, 2.5, 1.3
    b_mean, b_sd = drift * t, math.sqrt(t)
    expected = float(stats.norm.sf((z * math.sqrt(t) - b_mean) / b_sd))
    assert stagewise_tail(rule, 0, z, drift) == pytest.approx(expected, abs=1e-9)


def test_the_tail_refuses_a_look_outside_the_schedule() -> None:
    rule = _pocock()
    for bad in (-1, 4):
        with pytest.raises(ValueError, match="outside the schedule"):
            stagewise_tail(rule, bad, 2.0, 0.0)
    with pytest.raises(ValueError, match="n_grid"):
        stagewise_tail(rule, 1, 2.0, 0.0, n_grid=10)
    with pytest.raises(ValueError, match="finite"):
        stagewise_tail(rule, 1, float("nan"), 0.0)


# -- the fixed-sample limit -----------------------------------------------------------------


def test_one_look_reproduces_the_fixed_sample_answer_exactly() -> None:
    estimate = _stop_at(_one_look(), 0, 2.5)
    half = float(stats.norm.ppf(0.975))
    assert estimate.drift == pytest.approx(2.5, abs=1e-6)
    assert estimate.naive_drift == pytest.approx(2.5, abs=1e-12)
    assert estimate.drift_interval.lower == pytest.approx(2.5 - half, abs=1e-6)
    assert estimate.drift_interval.upper == pytest.approx(2.5 + half, abs=1e-6)
    assert estimate.bias == pytest.approx(0.0, abs=1e-6)


def test_a_first_look_stop_is_not_corrected_and_the_docstring_says_so() -> None:
    """A property of the ordering, not an omission — see the module docstring."""
    estimate = _stop_at(_pocock(), 0, 2.5)
    assert estimate.bias == pytest.approx(0.0, abs=1e-6)
    assert estimate.drift == pytest.approx(estimate.naive_drift, abs=1e-6)


# -- the correction itself ------------------------------------------------------------------


def test_the_correction_grows_with_the_look_index() -> None:
    rule = _pocock()
    biases = [_stop_at(rule, k, 2.5).bias for k in range(4)]
    assert biases[0] == pytest.approx(0.0, abs=1e-6)
    assert all(a < b for a, b in zip(biases, biases[1:], strict=False))
    assert biases[-1] == pytest.approx(0.219, abs=0.01)


def test_a_rule_that_rarely_stops_early_has_little_to_correct() -> None:
    """O'Brien-Fleming's early boundaries are far out, so conditioning on them buys less."""
    for look in (2, 3):
        pocock_bias = _stop_at(_pocock(), look, 2.5).bias
        obf_bias = _stop_at(_obf(), look, 2.5).bias
        assert 0.0 < obf_bias < pocock_bias
    assert _stop_at(_obf(), 2, 2.5).bias == pytest.approx(0.027, abs=0.005)
    # z = 2.5 does not even reach O'Brien-Fleming's first two boundaries.
    assert _obf().boundaries[0].z[:2] == pytest.approx((4.049, 2.863), abs=0.005)


def test_the_correction_is_towards_the_null_in_both_directions() -> None:
    high = _stop_at(_pocock(), 3, 2.5)
    low = _stop_at(_pocock(), 3, -2.5)
    assert high.bias > 0 and abs(high.drift) < abs(high.naive_drift)
    assert low.bias < 0 and abs(low.drift) < abs(low.naive_drift)
    assert low.drift == pytest.approx(-high.drift, abs=1e-3)


def test_the_interval_brackets_the_estimate_and_carries_its_definition() -> None:
    estimate = _stop_at(_pocock(), 2, 2.5)
    interval = estimate.drift_interval
    assert interval.lower < estimate.drift < interval.upper
    assert interval.definition == "stagewise" and interval.mass == 0.95
    assert "STAGEWISE" in interval.text()

    narrow = _stop_at(_pocock(), 2, 2.5)
    wide = stopped_estimate(monitor(_pocock(), [0.5, 0.5, 2.5]), mass=0.99)
    assert isinstance(wide, StoppedEstimate)
    assert wide.drift_interval.width > narrow.drift_interval.width


# -- scales ---------------------------------------------------------------------------------


def test_the_effect_scale_appears_only_when_a_standard_error_does() -> None:
    without = _stop_at(_pocock(), 2, 2.5)
    assert without.effect is None and without.effect_interval is None
    assert without.se_full is None and without.effect_bias is None
    assert "drift" in without.text()

    with_se = _stop_at(_pocock(), 2, 2.5, se=0.4)
    assert with_se.se_full == pytest.approx(0.4 * math.sqrt(0.75))
    assert with_se.naive_effect == pytest.approx(2.5 * 0.4)  # z * se at the stopping look
    assert with_se.effect == pytest.approx(with_se.drift * with_se.se_full)
    assert with_se.effect_bias == pytest.approx(with_se.bias * with_se.se_full)
    assert with_se.effect_interval is not None
    assert with_se.effect_interval.definition == "stagewise"
    assert "naive" in with_se.text()


def test_the_naive_drift_is_the_maximum_likelihood_one() -> None:
    estimate = _stop_at(_pocock(), 2, 2.5)
    assert estimate.naive_drift == pytest.approx(2.5 / math.sqrt(0.75))


# -- paths that cannot be corrected ---------------------------------------------------------


def test_a_path_that_ran_out_of_statistics_is_unsupported() -> None:
    result = stopped_estimate(monitor(_pocock(), [0.5, 0.7]))
    assert isinstance(result, Unsupported)
    assert "no final analysis" in result.reason
    assert not result


def test_a_completed_study_is_corrected_too() -> None:
    """It is biased by the looks it could have stopped at, not only by the one it took."""
    result = stopped_estimate(monitor(_pocock(), [0.5, 0.7, 1.0, 1.5]))
    assert isinstance(result, StoppedEstimate)
    assert result.decision == "completed"
    assert not result.stopped_early
    assert result.bias > 0.0


def test_mass_outside_the_unit_interval_is_refused() -> None:
    path = monitor(_pocock(), [0.5, 0.5, 2.5])
    for bad in (0.0, 1.0, 1.5):
        with pytest.raises(ValueError, match="mass must be in"):
            stopped_estimate(path, mass=bad)


# -- the ledger -----------------------------------------------------------------------------


def test_the_ledger_line_names_what_it_corrects() -> None:
    estimate = _stop_at(_pocock(), 3, 2.5)
    line = estimate.ledger_line()
    assert line.kind == "stopped_estimate"
    assert line.assumption == STAGEWISE_ORDERING
    assert line.assumption.state == "asserted"
    assert line.detail["corrects"] == "stopped_estimate_bias"
    assert line.detail["canonical"] == "canonical_joint_distribution"
    assert f"{estimate.drift:.4g}" in line.statement


def test_the_estimate_round_trips() -> None:
    estimate = _stop_at(_pocock(), 2, 2.5, se=0.4)
    assert StoppedEstimate.from_json(estimate.to_json()) == estimate
    assert str(estimate) == estimate.text()


# -- the claims the module docstring makes --------------------------------------------------


@pytest.mark.slow
def test_the_estimate_is_median_unbiased_and_the_interval_covers() -> None:
    """The two numbers the module docstring quotes, simulated against a known drift."""
    rule = _pocock()
    theta, n = 2.0, 3000
    rng = np.random.default_rng(0)
    t = np.asarray(_FOUR.information)
    dt = np.diff(np.concatenate([[0.0], t]))
    b = np.cumsum(rng.normal(theta * dt, np.sqrt(dt), size=(n, 4)), axis=1)
    z = b / np.sqrt(t)

    naive, corrected, covered = [], [], 0
    for row in z:
        estimate = stopped_estimate(monitor(rule, list(row)), n_grid=201)
        assert isinstance(estimate, StoppedEstimate)
        naive.append(estimate.naive_drift)
        corrected.append(estimate.drift)
        covered += estimate.drift_interval.contains(theta)

    # The Monte Carlo standard error of a median at n = 3000 is about 0.023.
    assert float(np.median(naive)) == pytest.approx(2.12, abs=0.03)
    assert float(np.median(corrected)) == pytest.approx(theta, abs=0.06)
    assert abs(float(np.median(corrected)) - theta) < abs(float(np.median(naive)) - theta)
    # Coverage: binomial se at 0.95 and n = 3000 is 0.004.
    assert covered / n == pytest.approx(0.95, abs=0.015)
    # Median-unbiased is not mean-unbiased, and the module docstring says so.
    assert float(np.mean(corrected)) > theta
