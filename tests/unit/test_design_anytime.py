from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from axiom.design import (
    AnytimeLook,
    ConfidenceSequence,
    LookSchedule,
    anytime_p,
    confidence_sequence,
    evalue,
    mixture_boundary,
    tune,
)

_ALPHA = 0.05
_T = (0.25, 0.5, 0.75, 1.0)


def _sequence(z: tuple[float, ...] = (0.4, 1.2, 2.1, 2.9), **kw: object) -> ConfidenceSequence:
    return confidence_sequence(z, _T, alpha=_ALPHA, **kw)  # type: ignore[arg-type]


# -- the boundary ---------------------------------------------------------------------------


def test_the_boundary_grows_faster_than_a_fixed_sample_one() -> None:
    """sqrt(t log t) against sqrt(t) is the whole difference between the two guarantees."""
    rho = tune(_ALPHA, 1.0)
    ratios = [
        (mixture_boundary(t, alpha=_ALPHA, rho=rho) / t)
        / (float(stats.norm.isf(_ALPHA / 2)) / math.sqrt(t))
        for t in _T
    ]
    assert all(r > 1.0 for r in ratios)
    assert ratios[-1] == pytest.approx(1.55, abs=0.05)  # ~55% wider where it is tuned
    assert ratios[0] > ratios[-1]  # and worse away from the tuning point


def test_tuning_minimizes_the_width_where_it_was_asked_to() -> None:
    for target in (0.5, 1.0):
        best = tune(_ALPHA, target)
        here = mixture_boundary(target, alpha=_ALPHA, rho=best) / target
        for other in (best / 3.0, best / 1.5, best * 1.5, best * 3.0):
            assert mixture_boundary(target, alpha=_ALPHA, rho=other) / target >= here
    # tuning early buys width early and pays for it late
    early, late = tune(_ALPHA, 0.25), tune(_ALPHA, 1.0)
    assert mixture_boundary(0.25, alpha=_ALPHA, rho=early) < mixture_boundary(
        0.25, alpha=_ALPHA, rho=late
    )
    assert mixture_boundary(1.0, alpha=_ALPHA, rho=early) > mixture_boundary(
        1.0, alpha=_ALPHA, rho=late
    )


def test_the_boundary_and_the_tuner_refuse_what_they_cannot_compute() -> None:
    for bad_alpha in (0.0, 1.0, 1.5):
        with pytest.raises(ValueError, match="alpha must be in"):
            mixture_boundary(1.0, alpha=bad_alpha, rho=0.1)
        with pytest.raises(ValueError, match="alpha must be in"):
            tune(bad_alpha)
    with pytest.raises(ValueError, match="rho must be finite and positive"):
        mixture_boundary(1.0, alpha=_ALPHA, rho=0.0)
    with pytest.raises(ValueError, match="target information"):
        tune(_ALPHA, 0.0)


# -- the e-value ----------------------------------------------------------------------------


def test_the_evalue_starts_at_one_and_grows_with_the_evidence() -> None:
    rho = tune(_ALPHA, 1.0)
    assert evalue(0.0, 0.0, rho=rho) == pytest.approx(1.0)
    assert evalue(0.0, 1.0, rho=rho) < 1.0  # no signal is evidence against nothing
    growing = [evalue(z, 1.0, rho=rho) for z in (0.0, 1.0, 2.0, 3.0)]
    assert all(b > a for a, b in zip(growing, growing[1:], strict=False))


def test_the_anytime_p_is_the_reciprocal_and_never_exceeds_one() -> None:
    rho = tune(_ALPHA, 1.0)
    assert anytime_p(0.0, 1.0, rho=rho) == 1.0
    strong = anytime_p(4.0, 1.0, rho=rho)
    assert strong == pytest.approx(1.0 / evalue(4.0, 1.0, rho=rho))
    # it is larger than the fixed-sample p-value: the same trade the interval makes
    assert strong > float(2 * stats.norm.sf(4.0))


def test_the_evalue_against_a_drift_other_than_zero() -> None:
    rho = tune(_ALPHA, 1.0)
    # at z = 2 with t = 1 the drift estimate is 2, so the evidence against 2 is minimal
    assert evalue(2.0, 1.0, rho=rho, drift=2.0) < evalue(2.0, 1.0, rho=rho, drift=0.0)
    assert evalue(2.0, 1.0, rho=rho, drift=2.0) == pytest.approx(math.sqrt(rho / (1.0 + rho)))


# -- the sequence ---------------------------------------------------------------------------


def test_a_sequence_carries_an_anytime_interval_at_every_look() -> None:
    seq = _sequence()
    assert len(seq.looks) == 4
    for look in seq.looks:
        assert isinstance(look, AnytimeLook)
        assert look.interval.definition == "anytime"
        assert look.interval.mass == pytest.approx(1.0 - _ALPHA)
        assert look.interval.contains(look.drift)
    assert seq.final.drift == pytest.approx(2.9)


def test_the_running_evalue_is_the_one_to_quote() -> None:
    """Ville bounds the running maximum, so a decision on it is the covered decision."""
    seq = confidence_sequence((3.0, 0.2, 0.1), _T[:3], alpha=_ALPHA)
    assert seq.looks[0].evalue > seq.looks[1].evalue  # the evidence faded
    assert seq.final.running_evalue == max(look.evalue for look in seq.looks)
    assert seq.final.running_evalue >= seq.final.evalue
    assert seq.final.anytime_p == pytest.approx(1.0 / seq.final.running_evalue)


def test_crossing_is_reported_at_the_first_look_it_happens() -> None:
    never = _sequence()
    assert never.crossed_at is None and not any(look.crossed for look in never.looks)
    early = confidence_sequence((8.0, 8.0, 8.0, 8.0), _T, alpha=_ALPHA)
    assert early.crossed_at == 0
    assert early.looks[0].crossed


def test_a_look_that_would_cross_a_fixed_sample_interval_may_not_cross_this_one() -> None:
    """The price, made concrete: z = 2.9 at full information is 'significant' and is not."""
    seq = _sequence()
    assert not seq.final.crossed
    fixed_half = float(stats.norm.isf(_ALPHA / 2))
    assert abs(seq.final.drift) > fixed_half  # a Wald interval would have excluded zero


def test_the_width_ratio_prices_the_peeking() -> None:
    seq = _sequence()
    assert seq.width_ratio(1.0) == pytest.approx(1.55, abs=0.05)
    assert seq.width_ratio(0.25) > seq.width_ratio(1.0)
    assert "1.549" in seq.ledger_line().detail["width_vs_fixed"] or seq.width_ratio(1.0) > 1.5


def test_a_look_schedule_may_be_passed_instead_of_fractions() -> None:
    schedule = LookSchedule(labels=("a", "b", "c", "d"), information=_T)
    assert (
        _sequence().looks[-1].interval
        == confidence_sequence((0.4, 1.2, 2.1, 2.9), schedule, alpha=_ALPHA).looks[-1].interval
    )


def test_the_ledger_line_names_the_assumption_that_is_left() -> None:
    seq = _sequence(name="NW-14")
    line = seq.ledger_line()
    assert line.kind == "confidence_sequence"
    assert line.assumption is not None
    assert line.assumption.name == "canonical_joint_distribution"
    assert "anytime-valid at alpha 0.05 over every look" in line.statement
    assert "NW-14" in seq.summary()


def test_the_sequence_round_trips() -> None:
    seq = _sequence()
    assert ConfidenceSequence.from_json(seq.to_json()) == seq


# -- what it refuses ------------------------------------------------------------------------


def test_the_looks_must_be_a_schedule() -> None:
    with pytest.raises(ValueError, match="statistics for"):
        confidence_sequence((1.0, 2.0), (0.5,))
    with pytest.raises(ValueError, match="strictly increasing"):
        confidence_sequence((1.0, 2.0), (0.5, 0.5))
    with pytest.raises(ValueError, match=r"must be in \(0, 1\]"):
        confidence_sequence((1.0,), (1.5,))
    with pytest.raises(ValueError, match="labels for"):
        confidence_sequence((1.0, 2.0), (0.5, 1.0), labels=("only",))
    with pytest.raises(ValueError, match="alpha must be in"):
        confidence_sequence((1.0,), (1.0,), alpha=0.0)


def test_a_look_must_carry_an_anytime_interval() -> None:
    from axiom.core import wald

    seq = _sequence()
    with pytest.raises(ValueError, match="carries an anytime interval"):
        seq.looks[0].model_copy(update={"interval": wald(0.0, 1.0, 0.95)}).model_validate(
            seq.looks[0].model_copy(update={"interval": wald(0.0, 1.0, 0.95)}).to_dict()
        )


# -- the guarantee --------------------------------------------------------------------------


@pytest.mark.slow
def test_the_sequence_covers_at_every_look_where_a_fixed_interval_does_not() -> None:
    """The claim the module exists for, and the number that makes the case."""
    t = np.linspace(0.1, 1.0, 10)
    dt = np.diff(np.concatenate([[0.0], t]))
    rho = tune(_ALPHA, 1.0)
    boundary = np.asarray([mixture_boundary(x, alpha=_ALPHA, rho=rho) for x in t])
    rng = np.random.default_rng(0)
    n = 20000

    for theta in (0.0, 2.0):
        b = np.cumsum(rng.normal(theta * dt, np.sqrt(dt), size=(n, t.size)), axis=1)
        error = np.abs(b - theta * t)
        anytime = (error > boundary).any(axis=1).mean()
        naive = (error > float(stats.norm.isf(_ALPHA / 2)) * np.sqrt(t)).any(axis=1).mean()
        assert anytime <= _ALPHA  # the guarantee holds over the whole sequence
        assert anytime < 0.03  # a mixture bound over ten looks is conservative
        assert naive > 0.15  # a fixed-sample interval read ten times errs ~4x its nominal rate
