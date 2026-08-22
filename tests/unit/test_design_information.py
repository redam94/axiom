"""``design.eig``, ``design.evoi``, ``design.precision``, ``design.anchor``: closed forms."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from scipy.special import ndtr

from axiom.core import Unsupported
from axiom.design.anchor import AnchoredEffect, anchor_draws, anchor_effect
from axiom.design.eig import (
    DESIGN_RELATIVE_SE,
    EIGEstimate,
    ReExperimentTiming,
    decayed_sd,
    eig_gaussian,
    eig_monte_carlo,
    experiment_se_for_design,
    information_half_life,
    time_to_re_experiment,
)
from axiom.design.evoi import (
    DecisionSpec,
    EVOIResult,
    evoi_gaussian,
    evpi,
    evpi_gaussian,
    evsi,
    evsi_gaussian,
    preposterior_sd,
    preposterior_sd_ratio,
)
from axiom.design.precision import (
    CostPerOutcomeInterval,
    cost_per_outcome_interval,
    cost_per_outcome_power,
    max_detectable_cost_per_outcome,
)

# -- eig -------------------------------------------------------------------------------


def test_eig_gaussian_closed_form_and_entropy_drop() -> None:
    assert eig_gaussian(1.0, 1.0) == pytest.approx(0.5 * math.log(2.0))
    # equals ln(prior_sd / posterior_sd) with the Gaussian posterior sd
    s, e = 2.0, 0.5
    post = 1.0 / math.sqrt(1 / s**2 + 1 / e**2)
    assert eig_gaussian(s, e) == pytest.approx(math.log(s / post))
    assert eig_gaussian(1.0, 0.1) > eig_gaussian(1.0, 1.0) > eig_gaussian(1.0, 10.0)


@pytest.mark.parametrize("args", [(0.0, 1.0), (1.0, 0.0), (-1.0, 1.0), (math.inf, 1.0)])
def test_eig_gaussian_rejects_non_positive(args: tuple[float, float]) -> None:
    with pytest.raises(ValueError):
        eig_gaussian(*args)


def test_experiment_se_for_design_table_and_override() -> None:
    assert experiment_se_for_design("cluster_holdout", 2.0) == pytest.approx(0.2)
    assert experiment_se_for_design("ghost", -2.0) == pytest.approx(0.5)
    assert experiment_se_for_design("bespoke", 4.0, relative_se=0.05) == pytest.approx(0.2)
    out = experiment_se_for_design("bespoke", 4.0)
    assert isinstance(out, Unsupported)
    assert "bespoke" in out.missing
    assert "cluster_holdout" in out.detail["known_kinds"]
    assert set(DESIGN_RELATIVE_SE) == {"cluster_holdout", "ghost"}
    with pytest.raises(ValueError):
        experiment_se_for_design("ghost", 0.0)
    with pytest.raises(ValueError):
        experiment_se_for_design("ghost", 1.0, relative_se=-0.1)


def test_decayed_sd_doubles_variance_per_half_life() -> None:
    assert decayed_sd(0.4, 0.0, 52.0) == 0.4
    assert decayed_sd(1.0, 26.0, 26.0) == pytest.approx(math.sqrt(2.0))
    assert decayed_sd(1.0, 104.0, 26.0) == pytest.approx(4.0)
    with pytest.raises(ValueError):
        decayed_sd(1.0, -1.0, 26.0)
    with pytest.raises(ValueError):
        decayed_sd(1.0, 1.0, 0.0)


def test_information_half_life_roundtrip() -> None:
    prior, post, hl = 2.0, 0.5, 13.0
    t = information_half_life(prior, post, hl)
    gained = math.log(prior / post)
    remaining = math.log(prior / decayed_sd(post, t, hl))
    assert remaining == pytest.approx(gained / 2.0)
    assert information_half_life(1.0, 1.0, hl) == 0.0
    with pytest.raises(ValueError, match="cannot lose information"):
        information_half_life(1.0, 2.0, hl)


def test_time_to_re_experiment_threshold_logic() -> None:
    r = time_to_re_experiment(0.2, 10.0, 0.5, 0.1, design_kind="cluster_holdout")
    assert isinstance(r, ReExperimentTiming)
    # at the computed time the decayed sd yields exactly min_eig
    assert eig_gaussian(decayed_sd(0.2, r.periods, 10.0), 0.5) == pytest.approx(0.1)
    assert r.eig_now == pytest.approx(eig_gaussian(0.2, 0.5))
    assert r.design_kind == "cluster_holdout"
    # already worth it: zero wait
    now = time_to_re_experiment(5.0, 10.0, 0.5, 0.1)
    assert now.periods == 0.0 and now.eig_now > 0.1


def test_eig_monte_carlo_converges_to_gaussian() -> None:
    rng = np.random.default_rng(0)
    prior_sd, se = 1.0, 0.7
    draws = prior_sd * rng.standard_normal(20_000)
    est = eig_monte_carlo(draws, se, n_sims=6000, seed=1)
    assert isinstance(est, EIGEstimate)
    exact = eig_gaussian(prior_sd, se)
    assert abs(est.eig - exact) < 4.0 * est.se + 0.01
    assert est.method == "nested_monte_carlo" and est.n_prior_draws == 20_000
    assert est.seed == 1 and est.n_sims == 6000


def test_eig_monte_carlo_validation() -> None:
    with pytest.raises(ValueError):
        eig_monte_carlo([1.0], 1.0)
    with pytest.raises(ValueError):
        eig_monte_carlo([0.0, 1.0], 1.0, n_sims=1)
    with pytest.raises(ValueError):
        eig_monte_carlo([0.0, 1.0], 0.0)


# -- evoi ------------------------------------------------------------------------------


def test_preposterior_sd_ratio_and_limits() -> None:
    assert preposterior_sd_ratio(1.0, 0.5) == pytest.approx(math.sqrt(0.8))
    assert preposterior_sd(2.0, 1.0) == pytest.approx(2.0 * math.sqrt(0.8))
    assert preposterior_sd_ratio(1.0, 1e-6) == pytest.approx(1.0, abs=1e-9)
    assert preposterior_sd_ratio(1.0, 1e6) == pytest.approx(1e-6, rel=1e-6)
    with pytest.raises(ValueError):
        preposterior_sd_ratio(0.0, 1.0)


def test_evpi_gaussian_matches_partial_expectation() -> None:
    d = DecisionSpec(threshold=0.5, value_per_outcome_unit=3.0, numeraire="units")
    m, s = 0.2, 1.0
    gap = m - d.threshold
    z = gap / s
    expected = 3.0 * (s * math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi) + gap * ndtr(z))
    assert evpi_gaussian(d, m, s) == pytest.approx(expected)
    # symmetric in the sign of the gap (decision indifferent at the threshold is worth most)
    assert evpi_gaussian(d, 0.5, 1.0) >= evpi_gaussian(d, 0.2, 1.0)
    assert evpi_gaussian(d, 0.5, 1.0) >= evpi_gaussian(d, 0.8, 1.0)


def test_evsi_is_bounded_by_evpi_and_monotone_in_experiment_se() -> None:
    d = DecisionSpec(threshold=0.0)
    ses = [0.05, 0.2, 1.0, 5.0]
    values = [evsi_gaussian(d, 0.3, 1.0, se) for se in ses]
    assert all(v <= evpi_gaussian(d, 0.3, 1.0) + 1e-12 for v in values)
    assert values == sorted(values, reverse=True)
    assert values[0] == pytest.approx(evpi_gaussian(d, 0.3, 1.0), rel=1e-2)
    assert values[-1] < 0.2 * values[0]
    r = evoi_gaussian(d, 0.3, 1.0, 0.2)
    assert isinstance(r, EVOIResult)
    assert r.method == "gaussian" and r.evsi <= r.evpi
    assert r.preposterior_sd == pytest.approx(preposterior_sd(1.0, 0.2))


def test_monte_carlo_evpi_and_evsi_match_gaussian() -> None:
    d = DecisionSpec(threshold=0.2, value_per_outcome_unit=2.0)
    rng = np.random.default_rng(3)
    m, s = 0.0, 1.0
    draws = m + s * rng.standard_normal(20_000)
    mc = evpi(d, draws)
    assert mc == pytest.approx(evpi_gaussian(d, m, s), rel=0.05)
    r = evsi(d, draws, 0.5, n_sims=3000, seed=4)
    exact = evsi_gaussian(d, m, s, 0.5)
    assert abs(r.evsi - exact) < 4.0 * r.evsi_se + 0.05 * exact
    assert r.evsi <= r.evpi + 1e-12
    assert r.method == "monte_carlo" and r.n_sims == 3000 and r.seed == 4
    assert r.preposterior_sd == pytest.approx(preposterior_sd(s, 0.5), rel=0.1)


def test_evpi_with_custom_value_fn_and_validation() -> None:
    d = DecisionSpec()
    draws = np.linspace(-1, 1, 101)
    # a payoff that is always positive leaves no value in information
    assert evpi(d, draws, value_fn=lambda t: np.ones_like(t)) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        evpi(d, draws, value_fn=lambda t: t[:10])
    with pytest.raises(ValueError):
        DecisionSpec(value_per_outcome_unit=0.0)
    with pytest.raises(ValueError):
        DecisionSpec(threshold=math.nan)
    with pytest.raises(ValueError):
        evsi(d, draws, 0.0)


# -- precision -------------------------------------------------------------------------


def test_cost_per_outcome_interval_bounded_case() -> None:
    r = cost_per_outcome_interval(50_000.0, 1200.0, 300.0)
    assert isinstance(r, CostPerOutcomeInterval)
    assert r.status == "bounded" and r.upper is not None
    assert r.estimate == pytest.approx(50_000.0 / 1200.0)
    assert r.lower == pytest.approx(50_000.0 / r.effect_interval.upper)
    assert r.upper == pytest.approx(50_000.0 / r.effect_interval.lower)
    assert r.naive_interval.definition == "wald" and r.mass == pytest.approx(0.95)
    assert r.contains(r.estimate)
    # Fieller is asymmetric: wider on the right than the naive interval
    assert r.upper > r.naive_interval.upper


def test_cost_per_outcome_interval_unbounded_when_lift_not_significant() -> None:
    r = cost_per_outcome_interval(50_000.0, 300.0, 300.0)
    assert r.status == "unbounded" and r.upper is None
    assert r.effect_interval.lower <= 0.0
    assert r.lower > 0.0 and r.contains(1e12)
    with pytest.raises(ValueError):
        cost_per_outcome_interval(50_000.0, -1.0, 300.0)
    with pytest.raises(ValueError):
        cost_per_outcome_interval(50_000.0, 1.0, 300.0, alpha=1.0)


def test_cost_per_outcome_power_and_threshold() -> None:
    r = cost_per_outcome_power(50_000.0, 1200.0, 300.0)
    assert r.effect_required == 0.0 and r.threshold is None
    assert r.power == pytest.approx(float(ndtr(1200.0 / 300.0 - 1.959963984540054)))
    t = cost_per_outcome_power(50_000.0, 1200.0, 300.0, threshold=50.0)
    assert t.effect_required == pytest.approx(1000.0)
    assert t.power < r.power
    assert max_detectable_cost_per_outcome(50_000.0, 800.0) == 62.5
    with pytest.raises(ValueError):
        max_detectable_cost_per_outcome(50_000.0, 0.0)


# -- anchor ----------------------------------------------------------------------------


@dataclass
class _Posterior:
    table: Mapping[str, npt.NDArray[np.float64]]

    def draws(self, name: str) -> npt.NDArray[np.float64]:
        return self.table[name]

    def names(self) -> frozenset[str]:
        return frozenset(self.table)

    def coords(self) -> Mapping[str, Sequence[Any]]:
        return {}

    def n_draws(self) -> int:
        return int(next(iter(self.table.values())).shape[1])


def test_anchor_draws_quantile_logic() -> None:
    x = np.linspace(0.0, 1.0, 1001)
    r = anchor_draws(x, "theta", 0.3, credence=0.9)
    assert isinstance(r, AnchoredEffect)
    assert r.probability_exceeds_mde == pytest.approx(0.7, abs=2e-3)
    assert r.anchored_effect == pytest.approx(0.1, abs=2e-3)
    assert r.already_believed is False
    confident = anchor_draws(x, "theta", 0.05, credence=0.9)
    assert confident.already_believed is True
    assert confident.probability_exceeds_mde >= 0.9
    assert r.mde_ratio == pytest.approx(0.3 / r.posterior_mean)
    assert r.n_draws == 1001 and r.posterior_interval.mass == 0.9


def test_anchor_draws_gaussian_probability_agrees_for_normal_draws() -> None:
    rng = np.random.default_rng(5)
    x = 1.0 + 0.5 * rng.standard_normal(50_000)
    r = anchor_draws(x, "theta", 1.2)
    assert r.probability_exceeds_mde == pytest.approx(r.probability_exceeds_mde_gaussian, abs=0.01)


def test_anchor_draws_validation_and_unsupported() -> None:
    with pytest.raises(ValueError):
        anchor_draws([0.0, 1.0], "theta", 0.0)
    with pytest.raises(ValueError):
        anchor_draws([0.0, 1.0], "theta", 1.0, credence=1.0)
    assert isinstance(anchor_draws([1.0], "theta", 1.0), Unsupported)
    bad = anchor_draws([1.0, math.nan, 2.0], "theta", 1.0)
    assert isinstance(bad, Unsupported) and bad.detail["non_finite"] == "1"


def test_anchor_effect_selects_parameter_and_index() -> None:
    rng = np.random.default_rng(6)
    post = _Posterior(
        {
            "scalar": 0.5 + 0.1 * rng.standard_normal((2, 500, 1)),
            "vector": np.stack(
                [
                    0.2 + 0.1 * rng.standard_normal((2, 500)),
                    2.0 + 0.1 * rng.standard_normal((2, 500)),
                ],
                axis=-1,
            ),
        }
    )
    s = anchor_effect(post, "scalar", 0.3)
    assert isinstance(s, AnchoredEffect) and s.parameter == "scalar"
    assert s.already_believed is True
    v1 = anchor_effect(post, "vector", 0.3, index=1)
    assert isinstance(v1, AnchoredEffect) and v1.parameter == "vector[1]"
    assert v1.posterior_mean == pytest.approx(2.0, abs=0.05)
    v0 = anchor_effect(post, "vector", 0.3, index=0)
    assert isinstance(v0, AnchoredEffect) and v0.already_believed is False
    assert isinstance(anchor_effect(post, "vector", 0.3), Unsupported)
    assert isinstance(anchor_effect(post, "vector", 0.3, index=5), Unsupported)
    missing = anchor_effect(post, "absent", 0.3)
    assert isinstance(missing, Unsupported) and "absent" in missing.missing
