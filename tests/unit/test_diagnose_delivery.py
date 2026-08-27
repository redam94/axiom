from __future__ import annotations

import numpy as np
import pytest

from axiom.design import ArmAllocation, assign
from axiom.diagnose import (
    SRM_ALPHA,
    BalanceCheck,
    Delivery,
    DeliveryReport,
    SampleRatio,
    arm_counts,
    balance,
    check_delivery,
    delivery,
    sample_ratio,
)

_EQUAL = ArmAllocation.equal("control", "treated")
_TRI = ArmAllocation(arms=("control", "low", "high"), shares=(0.5, 0.25, 0.25))
_UNITS = tuple(f"u{i:05d}" for i in range(2000))


# -- the sample-ratio check -----------------------------------------------------------------


def test_a_clean_split_passes_and_leaves_the_assumption_unverified() -> None:
    """A passing check is absence of evidence, not evidence of a working randomizer."""
    result = sample_ratio({"control": 5012, "treated": 4988}, _EQUAL)
    assert not result.mismatch
    assert result.p_value > 0.5
    assert result.assumption().name == "random_assignment"
    assert result.assumption().state == "unverified"
    assert "no mismatch" in result.ledger_line().statement


def test_the_split_everyone_ships_by_accident_is_caught() -> None:
    """51/49 on 100k units: invisible by eye, p = 2.5e-10."""
    result = sample_ratio({"control": 51000, "treated": 49000}, _EQUAL)
    assert result.mismatch
    assert result.p_value < 1e-9
    assert result.chi_square == pytest.approx(40.0)
    assert result.assumption().state == "violated"
    assert "MISMATCH" in result.ledger_line().statement


def test_the_default_alpha_is_the_srm_convention_not_five_percent() -> None:
    """A check that runs weekly at 0.05 cries wolf weekly."""
    assert SRM_ALPHA == 0.001
    borderline = sample_ratio({"control": 5000, "low": 2500, "high": 2300}, _TRI)
    assert 0.001 < borderline.p_value < 0.05
    assert not borderline.mismatch
    assert sample_ratio({"control": 5000, "low": 2500, "high": 2300}, _TRI, alpha=0.05).mismatch


def test_it_names_the_arm_that_moved() -> None:
    result = sample_ratio({"control": 5000, "low": 2500, "high": 2300}, _TRI)
    assert result.worst_arm == "high"
    assert result.realized_share("high") == pytest.approx(2300 / 9800)
    assert result.arms[2].deviation == pytest.approx(2300 - 0.25 * 9800)
    with pytest.raises(KeyError, match="no arm"):
        result.realized_share("placebo")


def test_an_assignment_can_be_checked_against_its_own_allocation() -> None:
    assigned = assign(_UNITS, _EQUAL, salt="s")
    result = sample_ratio(assigned)
    assert result.n == 2000
    assert not result.mismatch  # a hash split of 2000 should not trip 0.001


def test_counts_that_do_not_match_the_allocation_are_refused() -> None:
    with pytest.raises(ValueError, match="need the allocation"):
        sample_ratio({"control": 10, "treated": 10})
    with pytest.raises(ValueError, match="no count for arm"):
        sample_ratio({"control": 10}, _EQUAL)
    with pytest.raises(ValueError, match="unknown arm"):
        sample_ratio({"control": 10, "treated": 10, "placebo": 3}, _EQUAL)
    with pytest.raises(ValueError, match="non-negative"):
        sample_ratio({"control": 10, "treated": -1}, _EQUAL)
    with pytest.raises(ValueError, match="at least one unit"):
        sample_ratio({"control": 0, "treated": 0}, _EQUAL)


# -- delivery -------------------------------------------------------------------------------


def test_arms_reached_at_different_rates_are_flagged() -> None:
    result = delivery({"control": 5000, "treated": 5000}, {"control": 4900, "treated": 3000})
    assert result.differential_delivery
    assert result.differential == pytest.approx(0.38)
    assert result.assumption().name == "exposure_known"
    assert result.assumption().state == "violated"
    assert "DIFFERENTIAL" in result.ledger_line().statement


def test_equal_counts_with_equal_rates_pass() -> None:
    result = delivery({"control": 5000, "treated": 5000}, {"control": 4500, "treated": 4480})
    assert not result.differential_delivery
    assert result.overall_rate == pytest.approx(0.898)
    assert result.assumption().state == "unverified"


def test_full_and_empty_exposure_are_degenerate_not_significant() -> None:
    everyone = delivery({"control": 100, "treated": 100}, {"control": 100, "treated": 100})
    nobody = delivery({"control": 100, "treated": 100}, {"control": 0, "treated": 0})
    for result in (everyone, nobody):
        assert not result.differential_delivery
        assert result.p_value == 1.0 and result.chi_square == 0.0


def test_delivery_refuses_impossible_and_mismatched_inputs() -> None:
    with pytest.raises(ValueError, match="cannot be exposed without being assigned"):
        delivery({"control": 10, "treated": 10}, {"control": 11, "treated": 10})
    with pytest.raises(ValueError, match="same arms"):
        delivery({"control": 10, "treated": 10}, {"control": 10, "other": 10})
    with pytest.raises(ValueError, match="at least two arms"):
        delivery({"control": 10}, {"control": 10})
    with pytest.raises(ValueError, match="two arms with units"):
        delivery({"control": 10, "treated": 0}, {"control": 5, "treated": 0})


def test_an_empty_arm_has_no_rate_rather_than_a_zero_one() -> None:
    result = delivery(
        {"control": 10, "treated": 0, "extra": 10}, {"control": 5, "treated": 0, "extra": 4}
    )
    empty = next(row for row in result.arms if row.arm == "treated")
    assert np.isnan(empty.rate)


# -- balance --------------------------------------------------------------------------------


def _skewed(seed: int = 0) -> tuple[object, dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    assigned = assign(_UNITS, _EQUAL, salt="s")
    return assigned, {
        "x": rng.normal(size=len(_UNITS)),
        "y": rng.normal(size=len(_UNITS)),
        "skew": rng.normal(size=len(_UNITS)) + 0.5 * assigned.arm_of,
    }


def test_a_planted_imbalance_is_found_and_the_others_are_not() -> None:
    assigned, covariates = _skewed()
    result = balance(covariates, assigned)
    assert result.imbalanced == ("skew",)
    by_name = {t.covariate: t for t in result.tests}
    assert by_name["skew"].smd > 0.4 and by_name["skew"].adjusted_p < 1e-20
    assert by_name["x"].smd < 0.1 and not by_name["x"].imbalanced
    assert result.assumption().state == "violated"


def test_the_correction_is_family_wise_by_default_and_can_be_changed() -> None:
    assigned, covariates = _skewed()
    holm = balance(covariates, assigned)
    raw = balance(covariates, assigned, correction="none")
    assert holm.correction == "holm"
    for a, b in zip(holm.tests, raw.tests, strict=True):
        assert a.adjusted_p >= b.adjusted_p
        assert b.adjusted_p == b.p_value


def test_a_constant_covariate_is_skipped_rather_than_passed() -> None:
    assigned, covariates = _skewed()
    result = balance({**covariates, "flat": np.ones(len(_UNITS))}, assigned)
    assert result.skipped == ("flat",)
    assert "flat" not in [t.covariate for t in result.tests]
    assert any(row.covariate == "flat" for row in result.rows)  # still in the smd table


def test_balance_needs_covariates_and_an_allocation_for_a_bare_index() -> None:
    assigned, covariates = _skewed()
    with pytest.raises(ValueError, match="at least one covariate"):
        balance({}, assigned)
    with pytest.raises(ValueError, match="needs the allocation"):
        balance(covariates, assigned.arm_of)
    assert balance(covariates, assigned.arm_of, _EQUAL).imbalanced == ("skew",)


# -- the report -----------------------------------------------------------------------------


def test_the_report_runs_what_it_can_and_names_what_it_ran() -> None:
    assigned, covariates = _skewed()
    only_ratio = check_delivery({"control": 5012, "treated": 4988}, _EQUAL)
    assert only_ratio.checks_run == ("sample_ratio",)
    assert only_ratio.verdict().status == "identified"
    assert only_ratio.verdict().route == "sample_ratio"

    everything = check_delivery(
        assigned,
        exposed={"control": 900, "treated": 800},
        covariates=covariates,
    )
    assert everything.checks_run == ("sample_ratio", "delivery", "balance")
    assert [line.kind for line in everything.ledger()] == ["sample_ratio", "delivery", "balance"]


def test_any_failed_check_blocks_and_every_failure_is_named() -> None:
    assigned, covariates = _skewed()
    report = check_delivery(
        assigned, exposed={"control": 900, "treated": 500}, covariates=covariates
    )
    verdict = report.verdict()
    assert verdict.status == "blocked"
    assert "delivery" in verdict.reason and "balance (skew)" in verdict.reason
    assert len(report.failures) == 2
    assert "blocked" in report.summary()


def test_a_delivery_failure_is_never_downgraded() -> None:
    """No assumption licenses you past a broken split; the vocabulary reflects that."""
    report = check_delivery({"control": 51000, "treated": 49000}, _EQUAL)
    assert report.verdict().status == "blocked"
    assert report.verdict().assumptions == ()


def test_covariates_without_an_arm_index_are_refused_not_skipped() -> None:
    _, covariates = _skewed()
    with pytest.raises(ValueError, match="need arm_of"):
        check_delivery({"control": 1000, "treated": 1000}, _EQUAL, covariates=covariates)


def test_arm_counts_refuses_a_label_the_allocation_does_not_name() -> None:
    assert arm_counts(["control", "treated", "control"], _EQUAL) == {"control": 2, "treated": 1}
    assert arm_counts([], _EQUAL) == {"control": 0, "treated": 0}
    with pytest.raises(ValueError, match="not in the allocation"):
        arm_counts(["control", "treatment"], _EQUAL)


def test_the_results_round_trip() -> None:
    assigned, covariates = _skewed()
    report = check_delivery(
        assigned, exposed={"control": 900, "treated": 800}, covariates=covariates
    )
    assert DeliveryReport.from_json(report.to_json()) == report
    assert SampleRatio.from_json(report.ratio.to_json()) == report.ratio
    assert report.exposure is not None and Delivery.from_json(report.exposure.to_json())
    assert report.covariates is not None and BalanceCheck.from_json(report.covariates.to_json())
