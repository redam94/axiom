"""meta.privacy and meta.publish: k-anonymity, dominance, the epsilon ledger, DP releases."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.special import ndtr

from axiom.core import Blocked, Verdict
from axiom.meta.privacy import (
    Cell,
    EpsilonCharge,
    EpsilonLedger,
    PrivacyPolicy,
    cell_from_records,
    charge,
    check_cell,
    contributor_totals,
)
from axiom.meta.publish import (
    EpsilonSplit,
    Release,
    carry_forward,
    gaussian_sigma,
    gaussian_sigma_classical,
    jaccard_distance,
    laplace_scale,
    orthogonal_split,
    release,
)

POLICY = PrivacyPolicy(
    k=3, dominance_p=0.5, dominance_top_n=2, dominance_top_p=0.8, epsilon_total=1.0
)
CELL = Cell(records=(("a", 1.0), ("b", 1.2), ("c", 0.9), ("d", 1.1)), name="family/x")


# -- cells and policy --------------------------------------------------------------------------


def test_cell_properties_and_from_records() -> None:
    assert CELL.n == 4 and CELL.contributors == frozenset("abcd")
    assert contributor_totals(Cell(records=(("a", 1.0), ("a", -2.0), ("b", 1.0)))) == {
        "a": 3.0,
        "b": 1.0,
    }

    class R:
        def __init__(self, c: str, e: float) -> None:
            self.contributor, self.estimate = c, e

    c = cell_from_records([R("x", 0.1), R("y", 0.2)], name="n")
    assert c.records == (("x", 0.1), ("y", 0.2)) and c.name == "n"
    with pytest.raises(ValueError):
        Cell(records=())
    with pytest.raises(ValueError):
        Cell(records=(("a", math.nan),))
    with pytest.raises(ValueError):
        PrivacyPolicy(k=1)


def test_cell_below_k_is_blocked_not_clipped() -> None:
    two = Cell(records=(("a", 1.0), ("b", 1.0)))  # k - 1 contributors
    v = check_cell(two, POLICY)
    assert isinstance(v, Verdict) and v.status == "blocked" and not v.licensed
    assert "2 contributor" in v.reason and v.route == "k_anonymity"
    ledger = EpsilonLedger(budget=1.0)
    out = release(two, POLICY, ledger, release_id="r", epsilon=0.1, clip=(0.0, 2.0), seed=1)
    assert isinstance(out, Blocked)
    assert not hasattr(out, "value")
    # three records from only two distinct contributors is still below k
    dup = Cell(records=(("a", 1.0), ("a", 1.0), ("b", 1.0)))
    assert check_cell(dup, POLICY).status == "blocked"


def test_dominance_refusals() -> None:
    single = Cell(records=(("a", 10.0), ("b", 1.0), ("c", 1.0), ("d", 1.0)))
    v = check_cell(single, POLICY)
    assert v.status == "blocked" and v.route == "dominance_p"
    top2 = Cell(records=(("a", 4.5), ("b", 4.5), ("c", 1.0), ("d", 1.0)))
    v = check_cell(top2, POLICY)
    assert v.status == "blocked" and v.route == "dominance_top_n"
    assert check_cell(CELL, POLICY).status == "identified"
    # top-n rule is vacuous when n covers the whole cell
    assert check_cell(Cell(records=(("a", 1.0), ("b", 1.0), ("c", 1.0))), POLICY).licensed
    assert isinstance(
        release(
            top2,
            POLICY,
            EpsilonLedger(budget=1.0),
            release_id="r",
            epsilon=0.1,
            clip=(0, 5),
            seed=0,
        ),
        Blocked,
    )


# -- ledger ------------------------------------------------------------------------------------


def test_ledger_conserves_budget_and_blocks_overspend() -> None:
    ledger = EpsilonLedger(budget=1.0)
    eps = [0.1, 0.2, 0.3, 0.15]
    for i, e in enumerate(eps):
        out = charge(ledger, release_id=f"r{i}", epsilon=e, mechanism="laplace")
        assert isinstance(out, EpsilonLedger)
        ledger = out
    assert ledger.used + ledger.remaining == pytest.approx(1.0, abs=1e-12)
    assert ledger.used == pytest.approx(sum(eps), abs=1e-12)
    assert len(ledger.charges) == 4 and isinstance(ledger.charges[0], EpsilonCharge)
    before = ledger.content_hash()
    over = charge(ledger, release_id="big", epsilon=0.5, mechanism="laplace")
    assert isinstance(over, Blocked)
    assert ledger.content_hash() == before and ledger.remaining == pytest.approx(0.25)
    # exact remainder is allowed; a duplicate id is refused
    full = charge(
        ledger, release_id="last", epsilon=ledger.remaining, mechanism="gaussian", delta=1e-6
    )
    assert isinstance(full, EpsilonLedger) and full.remaining == pytest.approx(0.0, abs=1e-12)
    assert isinstance(charge(ledger, release_id="r0", epsilon=0.01, mechanism="laplace"), Blocked)
    with pytest.raises(ValueError):
        charge(ledger, release_id="neg", epsilon=-0.1, mechanism="laplace")
    with pytest.raises(ValueError):
        EpsilonLedger(
            budget=0.1, charges=(EpsilonCharge(release_id="x", epsilon=0.2, mechanism="laplace"),)
        )


# -- release -----------------------------------------------------------------------------------


def test_release_laplace_scale_clip_and_reproducibility() -> None:
    ledger = EpsilonLedger(budget=1.0)
    cell = Cell(records=(("a", 1.0), ("b", 5.0), ("c", -3.0), ("d", 1.0)))
    out = release(cell, POLICY, ledger, release_id="r1", epsilon=0.5, clip=(0.0, 2.0), seed=42)
    assert not isinstance(out, Blocked)
    rel, new_ledger = out
    assert isinstance(rel, Release)
    assert rel.sensitivity == pytest.approx(2.0 / 4)
    assert rel.noise_scale == pytest.approx(rel.sensitivity / 0.5)
    assert rel.noise_scale == laplace_scale(rel.sensitivity, 0.5)
    assert rel.n_clipped == 2 and (rel.clip_lower, rel.clip_upper) == (0.0, 2.0)
    expected_noise = float(np.random.default_rng(42).laplace(0.0, rel.noise_scale))
    clipped_mean = (1.0 + 2.0 + 0.0 + 1.0) / 4
    assert rel.value == pytest.approx(clipped_mean + expected_noise, rel=1e-12)
    assert rel.seed == 42 and rel.mechanism == "laplace" and rel.delta == 0.0
    assert rel.contributors == ("a", "b", "c", "d")
    assert rel.interval.definition == "wald" and rel.interval.contains(rel.value)
    assert rel.interval.upper - rel.value == pytest.approx(-rel.noise_scale * math.log(0.05))
    assert new_ledger.used == pytest.approx(0.5) and ledger.used == 0.0
    # same seed -> same value; different seed -> different value
    again = release(cell, POLICY, ledger, release_id="r1", epsilon=0.5, clip=(0.0, 2.0), seed=42)
    assert not isinstance(again, Blocked) and again[0].value == rel.value
    other = release(cell, POLICY, ledger, release_id="r1", epsilon=0.5, clip=(0.0, 2.0), seed=43)
    assert not isinstance(other, Blocked) and other[0].value != rel.value


def test_release_gaussian_analytic_sigma() -> None:
    # sigma solves Phi(D/(2s) - eps*s/D) - exp(eps) * Phi(-D/(2s) - eps*s/D) = delta
    d, eps, delta = 0.5, 0.8, 1e-5
    s = gaussian_sigma(d, eps, delta)
    lhs = ndtr(d / (2 * s) - eps * s / d) - math.exp(eps) * ndtr(-d / (2 * s) - eps * s / d)
    assert lhs == pytest.approx(delta, rel=1e-8)
    # tighter than the classical calibration, and monotone in epsilon
    assert s < gaussian_sigma_classical(d, eps, delta)
    assert gaussian_sigma(d, 2.0, delta) < s
    assert gaussian_sigma(0.0, eps, delta) == 0.0
    out = release(
        CELL,
        POLICY,
        EpsilonLedger(budget=1.0),
        release_id="g",
        epsilon=eps,
        clip=(0.0, 2.0),
        seed=3,
        mechanism="gaussian",
        delta=delta,
    )
    assert not isinstance(out, Blocked)
    rel, led = out
    assert rel.noise_scale == pytest.approx(gaussian_sigma(2.0 / 4, eps, delta))
    assert rel.delta == delta and led.delta_used == delta
    assert rel.value == pytest.approx(
        float(np.mean([1.0, 1.2, 0.9, 1.1]))
        + float(np.random.default_rng(3).normal(0.0, rel.noise_scale))
    )


def test_release_overspend_is_blocked_with_ledger_unchanged() -> None:
    ledger = EpsilonLedger(budget=0.3)
    out = release(CELL, POLICY, ledger, release_id="r", epsilon=0.5, clip=(0.0, 2.0), seed=0)
    assert isinstance(out, Blocked) and ledger.used == 0.0
    with pytest.raises(ValueError):
        release(CELL, POLICY, ledger, release_id="r", epsilon=0.1, clip=(2.0, 0.0), seed=0)


# -- split and carry-forward -----------------------------------------------------------------


def test_orthogonal_split_sums_to_epsilon() -> None:
    s = orthogonal_split(0.7, 3)
    assert isinstance(s, EpsilonSplit) and s.parts == 3
    assert math.fsum(s.epsilons) == pytest.approx(0.7, abs=1e-15)
    assert all(e == pytest.approx(0.7 / 3) for e in s.epsilons)
    assert "sequential" in s.composition
    ledger = EpsilonLedger(budget=0.7)
    for i, e in enumerate(s.epsilons):
        out = charge(ledger, release_id=f"p{i}", epsilon=e, mechanism="laplace")
        assert isinstance(out, EpsilonLedger)
        ledger = out
    assert ledger.remaining == pytest.approx(0.0, abs=1e-12)
    with pytest.raises(ValueError):
        orthogonal_split(0.7, 0)


def test_carry_forward_below_threshold_free_above_blocked() -> None:
    out = release(
        CELL,
        POLICY,
        EpsilonLedger(budget=1.0),
        release_id="r1",
        epsilon=0.2,
        clip=(0.0, 2.0),
        seed=1,
    )
    assert not isinstance(out, Blocked)
    prev, ledger = out
    # one contributor swapped out of four: Jaccard distance 1 - 3/5 = 0.4
    churned = Cell(records=(("a", 1.0), ("b", 1.2), ("c", 0.9), ("e", 1.0)))
    assert jaccard_distance(frozenset("abcd"), frozenset("abce")) == pytest.approx(0.4)
    kept = carry_forward(prev, churned, POLICY, churn_threshold=0.5)
    assert isinstance(kept, Release)
    assert kept.value == prev.value and kept.carried_from == "r1" and kept.epsilon == prev.epsilon
    assert ledger.used == pytest.approx(0.2)  # nothing new charged
    blocked = carry_forward(prev, churned, POLICY, churn_threshold=0.3)
    assert isinstance(blocked, Blocked) and "0.400" in blocked.reason
    # a cell that no longer clears the policy cannot be carried forward either
    small = Cell(records=(("a", 1.0), ("b", 1.0)))
    assert isinstance(carry_forward(prev, small, POLICY, churn_threshold=1.0), Blocked)
    with pytest.raises(ValueError):
        carry_forward(prev, churned, POLICY, churn_threshold=1.5)
