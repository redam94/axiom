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


def contested_evidence():
    """Four findings on the good side of the threshold and one on the bad side.

    The shape that matters: a record whose findings disagree, so that answering
    from ``findings[0]`` would quote a benefit while a harm sits two rows down.
    """
    builder = EvidenceBuilder("HYPER-3", "Which dose, and is any of them harming anyone?")
    builder.verdict(Verdict(status="identified", reason="randomised", route="backdoor"))
    for key, label, lower, upper in (
        ("d10", "10 mg vs control, pooled", -6.63, -3.91),
        ("d20", "20 mg vs control, pooled", -6.17, -3.44),
        ("d40", "40 mg vs control, pooled", -2.97, -0.10),
        ("d40_young", "40 mg vs control, age 25-35", -8.17, -2.36),
        ("d40_old", "40 mg vs control, age 51+", 3.98, 8.42),
    ):
        builder.finding(
            key,
            Interval(lower=lower, upper=upper, definition="wald", mass=0.9),
            label=label,
            unit="mmHg",
            precision=2,
            threshold=0.0,
            beneficial="lower",
        )
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


def test_an_unidentified_quantity_is_read_as_an_estimate_not_an_effect() -> None:
    ev = evidence_with(0.0, lower=-16.7, upper=-8.1, identified=False)
    section = discussion_section(ev)
    text = " ".join(getattr(b, "text", "") for b in section.blocks)
    assert "not identified" in text
    assert "requires an assumption this analysis does not supply" in text
    # The reading itself must not call the quantity an effect. The preamble may,
    # and has to: the sentence that refuses the causal reading names what is
    # being refused.
    assert "for the estimate" in text
    assert "for the effect" not in text


def test_an_identified_effect_is_read_as_an_effect() -> None:
    ev = evidence_with(0.0, lower=-16.7, upper=-8.1, identified=True)
    text = " ".join(getattr(b, "text", "") for b in discussion_section(ev).blocks)
    assert "for the effect" in text


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
    """Every part of the five-question abstract, as prose rather than as labels."""
    ev = evidence_with(0.0, lower=-16.7, upper=-8.1)
    built = build(ev, style="journal")
    abstract = built.report.sections[0]
    text = " ".join(getattr(b, "text", "") for b in abstract.blocks)
    assert "Does 40 mg lower systolic pressure?" in text  # what was asked
    assert "identified" in text  # how it was answered
    assert "-16.7" in text and "-8.1" in text  # what was found, with its interval
    assert "assumption" in text  # what that tells us, and under what
    assert "**" not in text, "the abstract is emphasised, which the guide forbids"
    assert built.missing() == ()


def test_the_abstract_states_one_quantity_and_names_the_rest() -> None:
    """An abstract that lists every finding has become the results table."""
    ev = contested_evidence()
    text = " ".join(
        getattr(b, "text", "") for b in build(ev, style="journal").report.sections[0].blocks
    )
    # The finding the answer turns on is the harmful one, and it is the only
    # one that arrives with an interval attached.
    assert "6.20" in text
    assert "-5.27" not in text and "-4.80" not in text
    assert "pooled" in text  # the others are named, not restated


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


# -- the discussion says agreement once --------------------------------------------------


def _prose(section) -> str:
    return " ".join(getattr(b, "text", "") for b in section.blocks)


def test_the_discussion_does_not_reprint_the_results_table() -> None:
    """The guide asks this section to repeat the results "without referring to stats"."""
    ev = contested_evidence()
    text = _prose(discussion_section(ev, verbosity="full"))
    for number in ("-6.63", "-6.17", "-2.97", "-8.17", "8.42"):
        assert number not in text, f"{number} is stated twice, once here and once in Results"
    # It still reads every finding: what each one settled is there in words.
    assert "favourable side" in text and "unfavourable side" in text


def test_findings_that_agree_are_read_once_and_the_odd_one_out_in_full() -> None:
    """Four identically shaped sentences is what made this section unreadable."""
    ev = contested_evidence()
    paragraphs = [
        b.text for b in discussion_section(ev, verbosity="full").blocks if hasattr(b, "text")
    ]
    grouped = [p for p in paragraphs if "every one of those questions is settled" in p]
    assert grouped, "the four agreeing findings were not collapsed into one reading"
    assert "10 mg vs control, pooled" in grouped[0] and "20 mg vs control, pooled" in grouped[0]
    # The finding that disagrees is read on its own, at length.
    alone = [p for p in paragraphs if p.startswith("40 mg vs control, age 51+:")]
    assert alone and "unfavourable side" in alone[0]


def test_the_conclusions_state_one_number_and_name_the_rest() -> None:
    """Answer the question, say what it rests on, stop — not list the findings again."""
    ev = contested_evidence()
    text = _prose(conclusions_section(ev))
    assert "6.20" in text, "the finding the answer turns on keeps its magnitude"
    for number in ("-5.27", "-4.80", "-1.53", "-5.26"):
        assert number not in text, f"{number} is restated in the conclusions"
    assert "10 mg vs control, pooled" in text, "the other findings are named"
    assert "**" not in text, "the conclusions are emphasised, which the guide forbids"


def test_no_generated_section_uses_emphasis_outside_a_heading() -> None:
    """ "Do not emphasize things with boldface (except headings)" — the guide's FAQ."""
    ev = contested_evidence()
    built = build(ev, style="journal", verbosity="full")
    for section in built.report.sections:
        for block in section.blocks:
            text = getattr(block, "text", "")
            if isinstance(block, Heading):
                continue
            assert "**" not in text, f"{section.title} sets prose in bold: {text[:80]}"


def test_no_generated_prose_contains_a_bracketed_plural() -> None:
    """A paper does not contain "assumption(s)"; that is a template showing through."""
    for ev in (contested_evidence(), evidence_with(0.0, lower=-16.7, upper=-8.1)):
        for verbosity in ("brief", "standard", "full"):
            built = build(ev, style="journal", verbosity=verbosity)
            for section in built.report.sections:
                for block in section.blocks:
                    text = getattr(block, "text", "")
                    assert "(s)" not in text, f"{section.title}: {text[:90]}"
