from __future__ import annotations

import numpy as np
import pytest
from scipy import special, stats

from axiom.design import (
    INDEPENDENT_READOUTS,
    OnlineProgram,
    Readout,
    gamma_sequence,
    online_decisions,
)

_ALPHA = 0.05


def _stream(truth: np.ndarray, *, seed: int = 0) -> list[Readout]:
    rng = np.random.default_rng(seed)
    p = 2 * stats.norm.sf(np.abs(rng.normal(truth, 1.0)))
    return [Readout(experiment=f"E{i:03d}", p_value=float(x)) for i, x in enumerate(p)]


def _early_then_late(n: int = 200) -> np.ndarray:
    truth = np.zeros(n)
    truth[:20] = 4.0  # the first twenty work
    truth[120:125] = 4.0  # then a long null run, then five more
    return truth


# -- the sequence ---------------------------------------------------------------------------


def test_the_gamma_sequence_is_a_normalized_power_law() -> None:
    gamma = gamma_sequence(10)
    assert gamma[0] == pytest.approx(1.0 / float(special.zeta(1.6, 1)))
    assert gamma[3] == pytest.approx(4.0**-1.6 / float(special.zeta(1.6, 1)))
    assert all(b < a for a, b in zip(gamma, gamma[1:], strict=False))
    assert sum(gamma) < 1.0  # a truncation of a sequence that sums to exactly one
    assert sum(gamma_sequence(50_000)) == pytest.approx(1.0, abs=0.01)


def test_a_faster_decay_spends_more_early() -> None:
    fast, slow = gamma_sequence(5, decay=2.5), gamma_sequence(5, decay=1.2)
    assert fast[0] > slow[0]
    assert fast[-1] < slow[-1]


def test_the_sequence_refuses_a_decay_that_does_not_converge() -> None:
    for bad in (1.0, 0.5, -1.0):
        with pytest.raises(ValueError, match="greater than 1"):
            gamma_sequence(5, decay=bad)
    with pytest.raises(ValueError, match="at least 1"):
        gamma_sequence(0)


# -- the behaviour that matters -------------------------------------------------------------


def test_the_level_shrinks_through_a_null_run_and_recovers_after_a_rejection() -> None:
    """The property to understand before using it: a programme that discovers can keep testing."""
    program = online_decisions(_stream(_early_then_late()), alpha=_ALPHA)
    levels = program.levels
    assert levels[19] > levels[0]  # early rejections pay the budget back
    assert levels[99] < levels[19] / 100  # a long null run spends it down
    assert levels[125] > levels[99] * 100  # and five more rejections restore it


def test_it_finds_the_real_effects_without_the_uncorrected_false_ones() -> None:
    truth = _early_then_late()
    program = online_decisions(_stream(truth), alpha=_ALPHA)
    rejected = [d.index - 1 for d in program.decisions if d.reject]
    assert sum(1 for i in rejected if truth[i] > 0) >= 23
    assert sum(1 for i in rejected if truth[i] == 0) == 0
    assert len(program.rejected_uncorrected) > len(program.rejected)
    assert program.expected_false_uncorrected == pytest.approx(10.0)


def test_arrival_order_is_part_of_the_procedure() -> None:
    truth = _early_then_late()
    forward = online_decisions(_stream(truth), alpha=_ALPHA)
    reversed_stream = list(reversed(_stream(truth)))
    backward = online_decisions(reversed_stream, alpha=_ALPHA)
    assert set(forward.rejected) != set(backward.rejected)
    assert forward.levels[0] == backward.levels[0]  # the first level cannot know the order


def test_every_decision_is_its_own_level() -> None:
    program = online_decisions(_stream(_early_then_late()), alpha=_ALPHA)
    for decision in program.decisions:
        assert decision.reject == (decision.p_value <= decision.level)
        assert 0.0 <= decision.level <= 1.0


# -- the guarantee --------------------------------------------------------------------------


@pytest.mark.slow
def test_the_false_discovery_rate_is_controlled_over_the_stream() -> None:
    proportions = []
    for seed in range(200):
        rng = np.random.default_rng(seed)
        truth = np.zeros(300)
        truth[:30] = 3.5
        rng.shuffle(truth)
        program = online_decisions(_stream(truth, seed=seed + 1000), alpha=_ALPHA)
        rejected = [d.index - 1 for d in program.decisions if d.reject]
        if rejected:
            proportions.append(sum(1 for i in rejected if truth[i] == 0) / len(rejected))
        else:
            proportions.append(0.0)
    assert float(np.mean(proportions)) <= _ALPHA


# -- what it refuses ------------------------------------------------------------------------


def test_it_needs_p_values_and_a_budget_that_fits() -> None:
    stream = _stream(np.zeros(5))
    with pytest.raises(ValueError, match="at least one readout"):
        online_decisions([])
    with pytest.raises(ValueError, match="alpha must be in"):
        online_decisions(stream, alpha=0.0)
    with pytest.raises(ValueError, match=r"w0 must be in \(0, alpha\]"):
        online_decisions(stream, alpha=0.05, w0=0.1)
    evalues = [Readout(experiment="E1", evalue=30.0), Readout(experiment="E2", evalue=1.0)]
    with pytest.raises(ValueError, match="needs a p-value on every readout"):
        online_decisions(evalues)


def test_repeated_keys_are_refused() -> None:
    twice = [Readout(experiment="E1", p_value=0.01), Readout(experiment="E1", p_value=0.4)]
    with pytest.raises(ValueError, match="keys must be distinct"):
        online_decisions(twice)


def test_it_names_the_condition_the_batch_route_does_not_need() -> None:
    program = online_decisions(_stream(_early_then_late()), alpha=_ALPHA)
    line = program.ledger_line()
    assert line.kind == "online_error_control"
    assert line.assumption == INDEPENDENT_READOUTS
    assert "arbitrary dependence" in INDEPENDENT_READOUTS.challenged_by
    assert "design.program" in INDEPENDENT_READOUTS.challenged_by
    assert "however long the stream runs" in program.bound


def test_the_programme_round_trips_and_summarizes() -> None:
    program = online_decisions(_stream(_early_then_late()), alpha=_ALPHA, stream="2026")
    assert OnlineProgram.from_json(program.to_json()) == program
    assert "decided on arrival" in program.summary()
    assert "at the first readout" in program.summary()
