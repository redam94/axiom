from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from axiom.core import Unverified
from axiom.identify import (
    EXCLUSION,
    MONOTONICITY,
    ComplianceReport,
    ComplianceTable,
    compliance,
    compliance_table,
    complier_effect,
    intention_to_treat,
)

_EFFECT = 2.0


def _world(
    *,
    n: int = 4000,
    complier_share: float = 0.6,
    always_taker_share: float = 0.05,
    effect: float = _EFFECT,
    seed: int = 0,
) -> pd.DataFrame:
    """Compliers respond to assignment; always- and never-takers do not.

    The exclusion restriction holds by construction: the outcome depends on
    ``exposed`` and never on ``assigned``.
    """
    rng = np.random.default_rng(seed)
    u = rng.random(n)
    complier = u < complier_share
    always = (u >= complier_share) & (u < complier_share + always_taker_share)
    assigned = (rng.random(n) < 0.5).astype(float)
    exposed = np.where(always, 1.0, np.where(complier, assigned, 0.0))
    y = 10.0 + effect * exposed + rng.normal(0.0, 1.0, n)
    return pd.DataFrame({"y": y, "assigned": assigned, "exposed": exposed})


# -- the table ------------------------------------------------------------------------------


def test_the_table_sizes_the_three_types_without_naming_a_member() -> None:
    table = compliance_table(_world(), "assigned", "exposed")
    assert table.complier_share == pytest.approx(0.6, abs=0.05)
    assert table.always_taker_share == pytest.approx(0.05, abs=0.02)
    assert table.never_taker_share == pytest.approx(0.35, abs=0.05)
    total = table.complier_share + table.always_taker_share + table.never_taker_share
    assert total == pytest.approx(1.0)
    assert table.monotonic and not table.one_sided and not table.perfect


def test_a_one_sided_design_has_no_always_takers() -> None:
    table = compliance_table(_world(always_taker_share=0.0), "assigned", "exposed")
    assert table.one_sided and table.always_taker_share == 0.0
    assert not table.perfect  # never-takers remain


def test_perfect_compliance_is_recognized() -> None:
    table = compliance_table(
        _world(complier_share=1.0, always_taker_share=0.0), "assigned", "exposed"
    )
    assert table.perfect and table.complier_share == 1.0
    assert table.never_taker_share == 0.0 and table.always_taker_share == 0.0


def test_the_table_refuses_columns_it_cannot_read() -> None:
    frame = _world()
    with pytest.raises(ValueError, match="same column"):
        compliance_table(frame, "assigned", "assigned")
    with pytest.raises(KeyError, match="not in frame"):
        compliance_table(frame, "assigned", "nope")
    with pytest.raises(ValueError, match="must be 0/1"):
        compliance_table(frame.assign(exposed=frame["y"]), "assigned", "exposed")


def test_a_table_needs_units_in_both_arms() -> None:
    with pytest.raises(ValueError, match="units in both arms"):
        ComplianceTable(
            assignment="a",
            exposure="e",
            n=10,
            n_assigned=10,
            n_control=0,
            exposed_assigned=5,
            exposed_control=0,
        )
    with pytest.raises(ValueError, match="every unit"):
        ComplianceTable(
            assignment="a",
            exposure="e",
            n=10,
            n_assigned=6,
            n_control=6,
            exposed_assigned=1,
            exposed_control=1,
        )
    with pytest.raises(ValueError, match="cannot be exposed"):
        ComplianceTable(
            assignment="a",
            exposure="e",
            n=10,
            n_assigned=5,
            n_control=5,
            exposed_assigned=6,
            exposed_control=0,
        )


# -- the two estimands ----------------------------------------------------------------------


def test_they_are_two_quantities_and_the_ratio_between_them_is_the_share() -> None:
    """The arithmetic behind reading one as the other, which is why both are reported."""
    report = compliance(_world(), "y", "assigned", "exposed")
    assert report.complier is not None
    assert report.itt.estimate == pytest.approx(report.complier.estimate * report.share, rel=1e-6)
    assert report.itt.estimate < report.complier.estimate  # ITT is diluted by non-compliance
    assert report.complier.estimate == pytest.approx(_EFFECT, abs=0.15)


def test_the_itt_needs_nothing_but_the_randomization() -> None:
    frame = _world()
    itt = intention_to_treat(frame, "y", "assigned")
    assert itt.method == "ols" and itt.treatment == "assigned"
    assert itt.estimate == pytest.approx(_EFFECT * 0.6, abs=0.15)


def test_the_complier_effect_is_two_stage_least_squares_not_a_second_wald_ratio() -> None:
    frame = _world()
    result = complier_effect(frame, "y", "exposed", "assigned")
    assert not isinstance(result, Unverified)
    assert result.method == "2sls" and result.treatment == "exposed"
    assert "first_stage_f" in result.detail


def test_covariates_reach_both_estimands() -> None:
    frame = _world()
    rng = np.random.default_rng(1)
    frame = frame.assign(age=rng.normal(size=len(frame)))
    report = compliance(frame, "y", "assigned", "exposed", covariates=("age",))
    assert report.itt.covariates == ("age",)
    assert report.complier is not None and report.complier.covariates == ("age",)


# -- what licenses the second one -----------------------------------------------------------


def test_the_complier_effect_is_downgraded_and_never_identified() -> None:
    report = compliance(_world(), "y", "assigned", "exposed")
    verdict = report.verdict()
    assert verdict.status == "downgraded"
    assert verdict.route == "2sls"
    assert [a.name for a in verdict.assumptions] == [
        "exclusion_restriction",
        "monotonicity",
        "instrument_strength",
    ]
    assert EXCLUSION.state == "unverified" and MONOTONICITY.state == "unverified"
    assert f"{report.share:.1%}" in verdict.reason


def test_perfect_compliance_is_identified_by_the_randomization_alone() -> None:
    report = compliance(
        _world(complier_share=1.0, always_taker_share=0.0), "y", "assigned", "exposed"
    )
    verdict = report.verdict()
    assert verdict.status == "identified" and verdict.route == "randomization"
    assert report.perfect_compliance
    assert report.complier is None  # there is no separate quantity to estimate
    assert report.unverified is not None and "no instrumental variation" in report.unverified.reason
    assert "the same quantity" in report.summary()


def test_defiers_block_the_complier_effect() -> None:
    frame = _world()
    flipped = frame.assign(exposed=1.0 - frame["exposed"])
    report = compliance(flipped, "y", "assigned", "exposed")
    assert not report.table.monotonic
    assert report.table.monotonicity().state == "violated"
    assert report.verdict().status == "blocked"
    assert "monotonicity" in report.verdict().reason


def test_an_assignment_that_moved_nothing_blocks() -> None:
    frame = _world()
    report = compliance(frame.assign(exposed=0.0), "y", "assigned", "exposed")
    assert report.complier is None
    assert report.verdict().status == "blocked"
    assert "no compliers" in report.verdict().reason
    assert report.itt is not None  # the ITT survives; it needed nothing from exposure


def test_a_weak_first_stage_is_reported_not_hidden() -> None:
    report = compliance(
        _world(n=600, complier_share=0.02, always_taker_share=0.0, seed=5),
        "y",
        "assigned",
        "exposed",
    )
    assert report.complier is not None
    assert report.instrument_strength is not None
    assert report.instrument_strength.name == "instrument_strength"
    # A first stage this thin makes the ratio enormously uncertain, whatever the F says.
    assert report.complier.se > 5.0 * report.itt.se


# -- the readout ----------------------------------------------------------------------------


def test_the_ledger_names_both_estimands_and_their_populations() -> None:
    report = compliance(_world(), "y", "assigned", "exposed")
    lines = report.ledger()
    assert [line.kind for line in lines] == ["compliance", "compliance_estimands"]
    assert lines[0].assumption is not None and lines[0].assumption.name == "monotonicity"
    assert lines[1].assumption == EXCLUSION
    assert "two estimands, not two estimates of one" in lines[1].statement
    assert "complier share" in lines[0].statement


def test_the_summary_says_which_population_each_number_is_over() -> None:
    report = compliance(_world(), "y", "assigned", "exposed")
    summary = report.summary()
    assert "intention to treat" in summary and "complier effect" in summary
    assert "all 4000 assigned units" in summary
    assert "whose exposure assignment moved" in summary


def test_a_report_carries_either_an_effect_or_the_reason_there_is_none() -> None:
    report = compliance(_world(), "y", "assigned", "exposed")
    with pytest.raises(ValueError, match="either a complier effect or the reason"):
        report.model_copy(update={"unverified": Unverified(reason="both")}).model_validate(
            report.model_copy(update={"unverified": Unverified(reason="both")}).to_dict()
        )


def test_the_report_round_trips() -> None:
    report = compliance(_world(), "y", "assigned", "exposed")
    assert ComplianceReport.from_json(report.to_json()) == report
    assert report.mass == 0.95
    with pytest.raises(ValueError, match="mass must be in"):
        compliance(_world(), "y", "assigned", "exposed", mass=1.5)


def test_an_unverified_reason_is_not_lost_to_its_falsiness() -> None:
    """``Unverified.__bool__`` is False; the report must test it against None."""
    report = compliance(_world().assign(exposed=0.0), "y", "assigned", "exposed")
    assert report.unverified is not None and not report.unverified
    assert "no compliers" in report.verdict().reason
    assert "no compliers" in report.summary()
    assert "no compliers" in report.ledger()[1].statement
