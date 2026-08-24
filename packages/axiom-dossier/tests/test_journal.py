"""Journal shape, verbosity, interpretation, and the claims gate."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from axiom.core import Assumption, Interval, Unsupported, Verdict
from axiom.report import Heading, Table

from axiom_dossier import (
    JOURNAL_SECTIONS,
    JOURNAL_THEME,
    VERBOSITY,
    EvidenceBuilder,
    Narrator,
    Offline,
    build,
    conclusions_section,
    discussion_section,
    introduction_section,
    licensed_claims,
    methods_section,
    reading_of,
    unlicensed_claims,
)

STANDING = Assumption(
    name="no_unmeasured_confounding",
    facet="population",
    statement="age is the only common cause of dose and pressure",
    challenged_by="a sensitivity analysis",
    state="unverified",
)


def evidence_with(threshold: float | None, *, lower: float, upper: float, identified: bool = True):
    builder = EvidenceBuilder("HYPER-3", "Does 40 mg lower systolic pressure?")
    builder.verdict(
        Verdict(
            status="identified" if identified else "blocked",
            reason=(
                "age blocks the only back-door path" if identified else "a confounder is unmeasured"
            ),
            route="backdoor" if identified else "",
        )
    )
    builder.step("design", "Design", what="Two arms of 300 participants.", why="Powered at 80 %.")
    builder.finding(
        "contrast",
        Interval(lower=lower, upper=upper, definition="eti", mass=0.9),
        label="40 mg vs control",
        unit="mmHg",
        precision=1,
        threshold=threshold,
        beneficial="lower",
    )
    builder.diagnostic("coverage", 0.94, label="Interval coverage")
    builder.assume(STANDING)
    return builder.build()


# -- interpretation ----------------------------------------------------------------------


def test_an_interval_clear_of_the_threshold_settles_the_question() -> None:
    ev = evidence_with(0.0, lower=-16.7, upper=-8.1)
    text = reading_of(ev.findings[0], causal=True)
    assert "entirely below" in text
    assert "settled" in text
    assert "favourable" in text


def test_an_interval_spanning_the_threshold_is_unsettled_not_null() -> None:
    """The distinction applied work destroys most often."""
    ev = evidence_with(0.0, lower=-9.0, upper=3.0)
    text = reading_of(ev.findings[0], causal=True)
    assert "do not settle" in text
    assert "unsettled question rather than a finding of no effect" in text
    assert "no effect" in text  # said, but only to deny it


def test_with_no_threshold_it_reports_the_estimate_and_stops() -> None:
    ev = evidence_with(None, lower=-16.7, upper=-8.1)
    text = reading_of(ev.findings[0], causal=True)
    assert "No decision threshold was recorded" in text
    assert "stops short" in text


def test_an_unidentified_effect_is_read_as_a_difference_not_an_effect() -> None:
    ev = evidence_with(0.0, lower=-16.7, upper=-8.1, identified=False)
    section = discussion_section(ev)
    text = " ".join(getattr(b, "text", "") for b in section.blocks)
    assert "not identified" in text
    assert "requires an assumption this analysis does not supply" in text
    assert "the observed difference" in text


def test_the_conclusion_answers_the_question_and_names_its_conditions() -> None:
    ev = evidence_with(0.0, lower=-16.7, upper=-8.1)
    text = " ".join(getattr(b, "text", "") for b in conclusions_section(ev).blocks)
    assert text.count("Yes") == 1
    assert "causal reading" in text
    assert "no unmeasured confounding" in text


def test_the_conclusion_refuses_to_answer_when_the_interval_spans() -> None:
    ev = evidence_with(0.0, lower=-9.0, upper=3.0)
    text = " ".join(getattr(b, "text", "") for b in conclusions_section(ev).blocks)
    assert "Not settled by these data" in text


# -- verbosity ---------------------------------------------------------------------------


def test_verbosity_changes_what_the_draft_contains_not_just_its_length() -> None:
    ev = evidence_with(0.0, lower=-16.7, upper=-8.1)
    lengths = {}
    for level in ("brief", "standard", "full"):
        blocks = methods_section(ev, verbosity=level).blocks
        lengths[level] = len(blocks)
    assert lengths["brief"] < lengths["standard"] <= lengths["full"]

    # brief drops the assumption table entirely; standard carries it
    assert not any(isinstance(b, Table) for b in methods_section(ev, verbosity="brief").blocks)
    assert any(isinstance(b, Table) for b in methods_section(ev, verbosity="standard").blocks)


def test_full_verbosity_gives_each_finding_its_own_heading() -> None:
    ev = evidence_with(0.0, lower=-16.7, upper=-8.1)
    plain = discussion_section(ev, verbosity="standard").blocks
    full = discussion_section(ev, verbosity="full").blocks
    assert not any(isinstance(b, Heading) for b in plain)
    assert any(isinstance(b, Heading) for b in full)


def test_an_unknown_verbosity_is_refused() -> None:
    ev = evidence_with(0.0, lower=-16.7, upper=-8.1)
    with pytest.raises(ValueError, match="unknown verbosity"):
        methods_section(ev, verbosity="chatty")  # type: ignore[arg-type]


def test_every_level_declares_the_same_switches() -> None:
    keys = [set(v) for v in VERBOSITY.values()]
    assert all(k == keys[0] for k in keys), "a level is missing a switch"


# -- journal shape -----------------------------------------------------------------------


def test_the_journal_shape_is_a_paper_and_the_order_is_the_conventional_one() -> None:
    ev = evidence_with(0.0, lower=-16.7, upper=-8.1)
    built = build(ev, style="journal")
    titles = [s.title for s in built.report.sections]
    assert titles[0] == "Abstract", "the abstract is not numbered"
    assert titles[1:] == [
        "1. Introduction",
        "2. Methods",
        "3. Results",
        "4. Model checking",
        "5. Discussion",
        "6. Conclusions",
        "7. Limitations",
        "8. Provenance",
    ]
    assert JOURNAL_SECTIONS.index("discussion") > JOURNAL_SECTIONS.index("results")


def test_the_journal_theme_is_serif_in_every_format() -> None:
    """Times-Roman is a PostScript name; HTML needs to be told what it means."""
    assert JOURNAL_THEME.font == "Times-Roman"
    assert "serif" in JOURNAL_THEME.font_fallback


def test_tables_and_figures_are_numbered_for_reference() -> None:
    ev = evidence_with(0.0, lower=-16.7, upper=-8.1)
    built = build(ev, style="journal")
    captions = [b.caption for s in built.report.sections for b in s.blocks if isinstance(b, Table)]
    assert captions[0].startswith("Table 1.")
    assert captions[1].startswith("Table 2.")


def test_the_plain_style_is_unnumbered_and_keeps_the_readout_order() -> None:
    ev = evidence_with(0.0, lower=-16.7, upper=-8.1)
    titles = [s.title for s in build(ev, style="plain").report.sections]
    assert titles == ["Methods", "Results", "Model checking", "Limitations", "Provenance"]


def test_an_unknown_style_is_refused() -> None:
    ev = evidence_with(0.0, lower=-16.7, upper=-8.1)
    with pytest.raises(ValueError, match="unknown style"):
        build(ev, style="manuscript")  # type: ignore[arg-type]


def test_the_abstract_falls_back_to_a_generated_one_with_no_model() -> None:
    ev = evidence_with(0.0, lower=-16.7, upper=-8.1)
    built = build(ev, style="journal")
    abstract = built.report.sections[0]
    text = " ".join(getattr(b, "text", "") for b in abstract.blocks)
    for label in ("Objective", "Methods", "Results", "Conclusions"):
        assert label in text
    assert built.missing() == ()


def test_the_introduction_states_the_threshold_that_decides_the_question() -> None:
    ev = evidence_with(0.0, lower=-16.7, upper=-8.1)
    text = " ".join(getattr(b, "text", "") for b in introduction_section(ev).blocks)
    assert "the value it turns on is 0.0 mmHg" in text
    assert "spanning it does not" in text


def test_a_journal_report_renders(tmp_path) -> None:  # type: ignore[no-untyped-def]
    ev = evidence_with(0.0, lower=-16.7, upper=-8.1)
    out = build(ev, style="journal", verbosity="full").write(str(tmp_path / "paper.pdf"))
    assert not isinstance(out, Unsupported), out
    assert (tmp_path / "paper.pdf").read_bytes().startswith(b"%PDF")


# -- the claims gate ---------------------------------------------------------------------


def test_an_unlicensed_claim_is_caught_even_with_no_number_in_the_sentence() -> None:
    ev = evidence_with(0.0, lower=-16.7, upper=-8.1)
    found = unlicensed_claims("The effect is robust and statistically significant.", ev)
    kinds = {c.kind for c in found}
    assert kinds == {"robustness", "significance"}
    assert {c.text.lower() for c in found} == {"robust", "statistically significant"}


def test_an_overlapping_claim_is_reported_once() -> None:
    ev = evidence_with(0.0, lower=-16.7, upper=-8.1)
    found = unlicensed_claims("a statistically significant result", ev)
    assert [c.text.lower() for c in found] == ["statistically significant"]


def test_a_claim_the_record_itself_makes_is_licensed() -> None:
    builder = EvidenceBuilder("T", "q?")
    builder.finding("c", 1.0, label="C")
    builder.step("s", "Step", what="A robust variance estimator was used.")
    ev = builder.build()
    assert "robust" in licensed_claims(ev)
    assert unlicensed_claims("the estimator is robust", ev) == ()


def test_narration_is_rejected_for_an_unlicensed_claim(tmp_path) -> None:  # type: ignore[no-untyped-def]
    @dataclass(frozen=True)
    class Overclaimer:
        name: str = "overclaimer"

        def generate(self, prompt: str, *, system: str = "", temperature: float = 0.2) -> str:
            return "The result is definitive and proves the treatment works."

    ev = evidence_with(0.0, lower=-16.7, upper=-8.1)
    built = build(ev, style="journal", narrator=Narrator(prose=Overclaimer(), light=Offline()))
    bad = built.rejected()
    assert bad, "an overclaiming model was not caught"
    for n in bad:
        assert n.overclaimed
        assert "definitive" in n.overclaimed or "proves" in n.overclaimed
        assert "unlicensed claim" in n.summary()
