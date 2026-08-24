"""The provenance check — the thing that makes model-written prose publishable."""

from __future__ import annotations

import pytest
from axiom.core import Assumption, Interval

from axiom_dossier import EvidenceBuilder, licensed_numbers, literals, unverified


@pytest.fixture
def evidence():
    return (
        EvidenceBuilder("HYPER-3", "Does 40 mg lower pressure?")
        .finding(
            "contrast",
            Interval(lower=-16.7, upper=-8.1, definition="eti", mass=0.9),
            label="40 mg vs control",
            unit="mmHg",
            precision=1,
        )
        .diagnostic("coverage", 0.94, label="Interval coverage")
        .step("design", "Design", what="Two arms of 300 participants.")
        .assume(
            Assumption(
                name="no_unmeasured_confounding",
                facet="population",
                statement="age is the only common cause",
                state="unverified",
            )
        )
        .build()
    )


# -- extraction -------------------------------------------------------------------------


def test_literals_carry_the_precision_they_were_printed_at() -> None:
    found = literals("a 12.40 b 3 c -0.5 d 1,234 e 1.2e-3")
    assert [x.text for x in found] == ["12.40", "3", "-0.5", "1,234", "1.2e-3"]
    assert [x.decimals for x in found] == [2, 0, 1, 0, 1]
    assert found[3].value == 1234.0


def test_a_number_inside_a_word_is_not_a_literal() -> None:
    assert [x.text for x in literals("HYPER-3 v2 covid19 x1y")] == []


def test_a_percentage_is_recognised_as_one() -> None:
    found = literals("90 % of the mass, and 40 mg of dose")
    assert (found[0].text, found[0].percent) == ("90", True)
    assert (found[1].text, found[1].percent) == ("40", False)


# -- licensing --------------------------------------------------------------------------


def test_the_record_licenses_its_points_bounds_masses_and_own_prose(evidence) -> None:
    allowed = licensed_numbers(evidence)
    for n in (-12.4, -16.7, -8.1, 0.9, 0.94, 300.0):
        assert any(abs(a - n) < 1e-9 for a in allowed), f"{n} should be licensed"


def test_a_mass_licenses_its_percentage_form(evidence) -> None:
    assert unverified("the 90 % interval", evidence) == ()
    assert unverified("the 0.9 mass", evidence) == ()


def test_the_magnitude_of_a_negative_effect_is_licensed(evidence) -> None:
    """Prose carries the sign in the verb: an effect of -12.4 is "lowered by 12.4"."""
    assert unverified("pressure fell by 12.4 mmHg", evidence) == ()


def test_rounding_is_allowed_downwards_but_not_upwards(evidence) -> None:
    assert unverified("about 12 mmHg", evidence) == ()
    assert unverified("12.4 mmHg", evidence) == ()
    # claiming more precision than the analysis produced is not licensed
    assert [x.text for x in unverified("12.4372 mmHg", evidence)] == ["12.4372"]


# -- the failure it exists to catch ------------------------------------------------------


def test_an_invented_p_value_and_sample_size_are_caught(evidence) -> None:
    bad = unverified("the effect was 12.4 mmHg (p = 0.031, n = 812)", evidence)
    assert [x.text for x in bad] == ["0.031", "812"]


def test_prose_that_only_restates_the_record_passes(evidence) -> None:
    text = (
        "Across two arms of 300 participants, the 40 mg arm lowered systolic pressure "
        "by 12.4 mmHg (90 % ETI -16.7 to -8.1). Interval coverage was 0.94."
    )
    assert unverified(text, evidence) == ()


def test_an_explicit_allowance_can_license_a_number_the_record_lacks(evidence) -> None:
    assert [x.text for x in unverified("in 2026", evidence)] == ["2026"]
    assert unverified("in 2026", evidence, allow=[2026]) == ()
