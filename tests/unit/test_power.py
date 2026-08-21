"""Power / MDE / sample size consistency and the cluster-design arithmetic."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from axiom.core import D, Unit, Unsupported
from axiom.design.cluster import (
    Assignment,
    ClusterDesign,
    HoldoutTradeoff,
    cluster_mde,
    cluster_power,
    clusters_needed,
    design_effect,
    effective_sample_size,
    holdout_tradeoff,
    match_clusters,
)
from axiom.design.power import (
    MDE,
    PowerCurve,
    PowerResult,
    SampleSize,
    coefficient_mde,
    coefficient_power,
    coefficient_sample_size,
    difference_se,
    mde,
    power,
    power_curve,
    power_from_se,
    sample_size,
)


def textbook_power(effect: float, se: float, alpha: float) -> float:
    z = stats.norm.ppf(1 - alpha / 2)
    r = abs(effect) / se
    return float(stats.norm.cdf(r - z) + stats.norm.cdf(-r - z))


# -- power -----------------------------------------------------------------------------


def test_power_matches_closed_form() -> None:
    res = power(100, 0.5, 1.0)
    assert isinstance(res, PowerResult)
    se = 1.0 / math.sqrt(100 * 0.25)
    assert res.se == pytest.approx(se)
    assert res.power == pytest.approx(textbook_power(0.5, se, 0.05), rel=1e-12)
    one = power(100, 0.5, 1.0, two_sided=False)
    assert one.power == pytest.approx(stats.norm.cdf(0.5 / se - stats.norm.ppf(0.95)), rel=1e-12)
    assert one.power > res.power
    assert difference_se(100, 1.0, 0.5) == pytest.approx(se)
    assert difference_se(100, 1.0, 0.2) > se


def test_power_at_zero_effect_is_alpha_and_monotone() -> None:
    assert power(50, 0.0, 1.0, alpha=0.05).power == pytest.approx(0.05, rel=1e-9)
    powers = [power(n, 0.3, 1.0).power for n in (20, 50, 100, 400)]
    assert powers == sorted(powers)
    by_effect = [power(100, e, 1.0).power for e in (0.1, 0.2, 0.5, 1.0)]
    assert by_effect == sorted(by_effect)
    assert power(100, 0.3, 2.0).power < power(100, 0.3, 1.0).power
    assert power(100, -0.5, 1.0).power == pytest.approx(power(100, 0.5, 1.0).power)


def test_power_from_se_and_coefficient_power_agree() -> None:
    a = power_from_se(0.4, 0.1)
    b = coefficient_power(0.4, 0.1)
    assert a.design == "coefficient" and a.n is None
    assert a.power == pytest.approx(b.power) == pytest.approx(textbook_power(0.4, 0.1, 0.05))
    assert power_from_se(0.4, 0.0).power == 1.0
    assert power_from_se(0.0, 0.0).power == pytest.approx(0.05)


# -- mde and sample size: inverses of power --------------------------------------------


@pytest.mark.parametrize("two_sided", [True, False])
@pytest.mark.parametrize("target", [0.5, 0.8, 0.95])
def test_mde_is_exact_inverse_of_power(two_sided: bool, target: float) -> None:
    m = mde(80, 2.0, power=target, two_sided=two_sided)
    assert isinstance(m, MDE)
    achieved = power(80, m.effect, 2.0, two_sided=two_sided).power
    assert achieved == pytest.approx(target, rel=1e-9)
    textbook = (stats.norm.ppf(1 - (0.025 if two_sided else 0.05)) + stats.norm.ppf(target)) * m.se
    assert m.effect <= textbook * (1 + 1e-12)
    if not two_sided:
        assert m.effect == pytest.approx(textbook)


def test_coefficient_mde_matches_difference_mde_at_same_se() -> None:
    se = difference_se(80, 2.0)
    assert coefficient_mde(se).effect == pytest.approx(mde(80, 2.0).effect)
    assert coefficient_mde(0.0).effect == 0.0


def test_sample_size_reaches_target_and_is_minimal() -> None:
    out = sample_size(0.5, 1.0, power=0.8)
    assert isinstance(out, SampleSize)
    assert out.power >= 0.8
    assert out.n_treated is not None and out.n_control is not None
    assert out.n_treated + out.n_control == out.n
    assert power(out.n, 0.5, 1.0).power >= 0.8 - 1e-12
    smaller = power(out.n - 1, 0.5, 1.0).power
    assert smaller < 0.8
    # Continuous textbook n for these numbers is ~125.6; the integer answer is a neighbour.
    textbook_n = (1.0 * (stats.norm.ppf(0.975) + stats.norm.ppf(0.8)) / 0.5) ** 2 / 0.25
    assert abs(out.n - textbook_n) < 3


def test_sample_size_monotone_and_unsupported() -> None:
    ns = []
    for e in (1.0, 0.5, 0.25):
        out = sample_size(e, 1.0)
        assert isinstance(out, SampleSize)
        ns.append(out.n)
    assert ns == sorted(ns)
    assert isinstance(sample_size(0.0, 1.0), Unsupported)
    assert isinstance(sample_size(1e-9, 1.0), Unsupported)
    unbalanced = sample_size(0.5, 1.0, allocation=0.2)
    assert isinstance(unbalanced, SampleSize) and unbalanced.n > ns[1]


def test_coefficient_sample_size_scales_as_inverse_square_root() -> None:
    out = coefficient_sample_size(0.3, 0.2, 100)
    assert isinstance(out, SampleSize)
    assert out.power >= 0.8 and out.design == "coefficient"
    assert out.se == pytest.approx(0.2 * math.sqrt(100 / out.n))
    assert isinstance(coefficient_sample_size(0.0, 0.2, 100), Unsupported)


def test_power_curve_interpolates_power() -> None:
    curve = power_curve(100, 1.0, [0.0, 0.25, 0.5, 1.0])
    assert isinstance(curve, PowerCurve)
    assert curve.powers[0] == pytest.approx(0.05, rel=1e-9)
    assert list(curve.powers) == sorted(curve.powers)
    assert curve.power_at(0.5) == pytest.approx(power(100, 0.5, 1.0).power)
    assert curve.power_at(5.0) == curve.powers[-1]


def test_power_validation_errors() -> None:
    with pytest.raises(ValueError):
        power(1, 0.5, 1.0)
    with pytest.raises(ValueError):
        power(100, 0.5, 0.0)
    with pytest.raises(ValueError):
        power(100, 0.5, 1.0, alpha=1.5)
    with pytest.raises(ValueError):
        power(100, 0.5, 1.0, allocation=1.0)
    with pytest.raises(ValueError):
        mde(100, 1.0, power=0.01, alpha=0.05)
    with pytest.raises(ValueError):
        power_curve(100, 1.0, [])
    with pytest.raises(ValueError):
        power_from_se(0.5, -1.0)
    with pytest.raises(ValueError):
        coefficient_sample_size(0.3, 0.0, 100)


# -- clusters --------------------------------------------------------------------------


def cluster_unit() -> Unit:
    return Unit(name="region", dimension=D.entity, kind="cluster")


def test_design_effect_and_ess() -> None:
    assert design_effect(1, 0.5) == 1.0
    assert design_effect(10, 0.0) == 1.0
    assert design_effect(10, 0.1) == pytest.approx(1.9)
    assert effective_sample_size(20, 10, 0.1) == pytest.approx(200 / 1.9)
    assert effective_sample_size(20, 10, 0.0) == 200.0
    with pytest.raises(ValueError):
        design_effect(0, 0.1)
    with pytest.raises(ValueError):
        design_effect(10, 1.0)
    with pytest.raises(ValueError):
        effective_sample_size(0, 10, 0.1)


def test_cluster_power_reduces_to_individual_power() -> None:
    design = ClusterDesign(unit=cluster_unit(), n_clusters=20, cluster_size=10, icc=0.1)
    assert design.design_effect == pytest.approx(1.9)
    assert design.n_individuals == 200
    res = cluster_power(design, 0.3, 1.0)
    direct = power(200, 0.3, math.sqrt(1.9))
    assert res.power == pytest.approx(direct.power)
    no_icc = ClusterDesign(unit=cluster_unit(), n_clusters=20, cluster_size=10, icc=0.0)
    assert cluster_power(no_icc, 0.3, 1.0).power > res.power
    m = cluster_mde(design, 1.0, power=0.8)
    assert cluster_power(design, m.effect, 1.0).power == pytest.approx(0.8, rel=1e-9)
    with pytest.raises(ValueError):
        ClusterDesign(
            unit=Unit(name="person", dimension=D.entity), n_clusters=20, cluster_size=10, icc=0.1
        )
    with pytest.raises(ValueError):
        ClusterDesign(unit=cluster_unit(), n_clusters=1, cluster_size=10, icc=0.1)


def test_clusters_needed_counts_clusters() -> None:
    out = clusters_needed(0.3, 1.0, cluster_size=10, icc=0.1)
    assert isinstance(out, SampleSize)
    assert out.n_treated is not None and out.n_control is not None
    assert out.n_treated + out.n_control == out.n
    assert out.sd == pytest.approx(math.sqrt(1.9 / 10))
    more = clusters_needed(0.3, 1.0, cluster_size=10, icc=0.3)
    assert isinstance(more, SampleSize) and more.n > out.n
    assert isinstance(clusters_needed(0.0, 1.0, cluster_size=10, icc=0.1), Unsupported)


def test_match_clusters_pairs_and_balances() -> None:
    rng = np.random.default_rng(0)
    pre = rng.normal(size=(9, 5)) + np.arange(9)[:, None]
    a = match_clusters(pre, seed=1)
    assert isinstance(a, Assignment)
    assert a.n_treated == 4 and a.n_control == 5
    assert len(a.pairs) == 4 and len(a.unpaired) == 1
    assert sorted(a.treated + a.control) == list(range(9))
    assert a.unit.kind == "cluster" and a.metric == "level"
    # Neighbouring levels are paired: every pair differs by one rank.
    for i, j in a.pairs:
        assert abs(i - j) <= 2
    assert abs(a.pre_smd) < 1.5
    again = match_clusters(pre, seed=1)
    assert again.treated == a.treated
    traj = match_clusters(pre, metric="trajectory", seed=2, labels=[f"r{i}" for i in range(9)])
    assert traj.labels[0] == "r0" and len(traj.pairs) == 4
    with pytest.raises(ValueError):
        match_clusters(pre[:1])
    with pytest.raises(ValueError):
        match_clusters(pre, labels=["a", "b"])


def test_holdout_tradeoff_prefers_balance() -> None:
    design = ClusterDesign(unit=cluster_unit(), n_clusters=20, cluster_size=10, icc=0.1)
    t = holdout_tradeoff(design, 1.0, fractions=(0.1, 0.3, 0.5))
    assert isinstance(t, HoldoutTradeoff)
    assert t.best_fraction == 0.5
    assert min(t.relative_mde) == 1.0 and t.relative_mde[0] > 1.0
    assert t.n_holdout == (2, 6, 10)
    assert list(t.mdes) == sorted(t.mdes, reverse=True)
    with pytest.raises(ValueError):
        holdout_tradeoff(design, 1.0, fractions=(0.01,))
    with pytest.raises(ValueError):
        holdout_tradeoff(design, 1.0, fractions=())
