"""Narration: the plumbing, and the refusal that makes it safe."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from axiom.core import Assumption, Interval, Unsupported, Verdict
from axiom.report import Heading, Paragraph, Section

from axiom_dossier import (
    EvidenceBuilder,
    Gemini,
    LanguageModel,
    Narration,
    Narrator,
    Offline,
    build,
    established_note,
    evidence_brief,
    exhibit_note,
    methods_section,
    narrate_text,
)
from axiom_dossier.narrate import (
    ABSTRACT_LICENCE,
    LICENCE,
    SECTION_INSTRUCTION,
    VERBOSITY,
    _budget,
)


@pytest.fixture
def evidence():
    return (
        EvidenceBuilder("HYPER-3", "Does 40 mg lower pressure?")
        .step("design", "Design", what="Two arms of 300 participants.", why="Powered at 80 %.")
        .step("estimation", "Estimation", what="Ordinary least squares.", why="Linear response.")
        .finding(
            "contrast",
            Interval(lower=-16.7, upper=-8.1, definition="eti", mass=0.9),
            label="40 mg vs control",
            unit="mmHg",
            precision=1,
        )
        .assume(
            Assumption(
                name="no_unmeasured_confounding",
                facet="population",
                statement="age is the only common cause",
                challenged_by="a sensitivity analysis",
                state="unverified",
            )
        )
        .build()
    )


def _threshold_evidence():
    """One finding with a decision threshold, so it has a reading to be shown."""
    return (
        EvidenceBuilder("HYPER-3", "Does 40 mg lower pressure?")
        .verdict(Verdict(status="identified", reason="randomised", route="backdoor"))
        .finding(
            "contrast",
            Interval(lower=-16.7, upper=-8.1, definition="eti", mass=0.9),
            label="40 mg vs control",
            unit="mmHg",
            precision=1,
            threshold=0.0,
            beneficial="lower",
        )
        .build()
    )


def _contested_evidence():
    """Findings that disagree, so the one the answer turns on is not the first."""
    builder = EvidenceBuilder("HYPER-3", "Which dose, and is any of them harming anyone?")
    for key, label, lower, upper in (
        ("d10", "10 mg vs control, pooled", -6.63, -3.91),
        ("d20", "20 mg vs control, pooled", -6.17, -3.44),
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
    return builder.build()


@dataclass(frozen=True)
class Fabricator:
    """A model that answers fluently and makes a number up. The case that matters."""

    text: str = "The effect was 12.4 mmHg (p = 0.004, n = 1200), a robust finding."

    @property
    def name(self) -> str:
        return "fabricator"

    def generate(self, prompt: str, *, system: str = "", temperature: float = 0.2) -> str:
        return self.text


@dataclass(frozen=True)
class Absent:
    """A model that cannot be reached — the missing-extra / missing-key path."""

    @property
    def name(self) -> str:
        return "absent"

    def generate(self, prompt: str, *, system: str = "", temperature: float = 0.2) -> Unsupported:
        return Unsupported(reason="no key configured", missing=("GEMINI_API_KEY",))


def test_the_stand_ins_satisfy_the_protocol() -> None:
    for model in (Offline(), Fabricator(), Absent(), Gemini()):
        assert isinstance(model, LanguageModel)


def test_the_brief_contains_every_number_the_model_may_use(evidence) -> None:
    brief = evidence_brief(evidence)
    assert "-12.4" in brief and "-16.7" in brief and "-8.1" in brief
    # written as English rather than as an identifier: a model handed snake_case
    # prints snake_case into the abstract
    assert "no unmeasured confounding" in brief
    assert "no_unmeasured_confounding" not in brief
    assert "300 participants" in brief


# -- the refusal ------------------------------------------------------------------------


def test_a_fabricated_number_gets_the_narration_rejected_and_the_draft_kept(evidence) -> None:
    draft = "The generated draft, which is correct."
    out = narrate_text("results", draft, evidence, Fabricator())
    assert isinstance(out, Narration)
    assert out.narrated and not out.verified
    assert out.text == draft, "the correct draft must survive"
    assert set(out.rejected) == {"0.004", "1200"}
    assert "REJECTED" not in out.summary() and "rejected" in out.summary()


def test_a_faithful_rewrite_is_accepted(evidence) -> None:
    @dataclass(frozen=True)
    class Faithful:
        name: str = "faithful"

        def generate(self, prompt: str, *, system: str = "", temperature: float = 0.2) -> str:
            return "The 40 mg arm lowered pressure by 12.4 mmHg (90 % ETI -16.7 to -8.1)."

    out = narrate_text("results", "draft", evidence, Faithful())
    assert isinstance(out, Narration)
    assert out.verified and out.narrated
    assert "12.4" in out.text


def test_an_unreachable_model_is_unsupported_not_a_crash(evidence) -> None:
    out = narrate_text("results", "draft", evidence, Absent())
    assert isinstance(out, Unsupported)
    assert "GEMINI_API_KEY" in out.missing


def test_a_document_still_builds_when_the_model_is_unreachable(evidence) -> None:
    """The whole point of the split: no key, still a report."""
    built = build(evidence, narrator=Narrator(prose=Absent(), light=Absent()))
    assert built.missing() == ()
    assert len(built.report.sections) == 5
    assert all(not n.narrated for n in built.narrations)


# -- structure --------------------------------------------------------------------------


def test_narrating_a_section_does_not_orphan_its_headings(evidence) -> None:
    """A heading whose whole run was prose goes with the prose; one over a table stays."""
    section = methods_section(evidence)
    assert [b.text for b in section.blocks if isinstance(b, Heading)] == [
        "Design",
        "Estimation",
        "Assumptions this analysis rests on",
    ]

    narrated, record = Narrator(prose=Offline(), light=Offline()).section(section, evidence)
    assert record.verified
    headings = [b.text for b in narrated.blocks if isinstance(b, Heading)]
    assert headings == ["Assumptions this analysis rests on"], "prose headings are now empty"
    assert sum(isinstance(b, Paragraph) for b in narrated.blocks) == 1


def test_narration_never_touches_the_blocks_that_carry_numbers(evidence) -> None:
    from axiom_dossier import results_section

    section = results_section(evidence)
    metrics = [b for b in section.blocks if getattr(b, "block", "") == "metric"]
    narrated, _ = Narrator(prose=Offline(), light=Offline()).section(section, evidence)
    assert [b for b in narrated.blocks if getattr(b, "block", "") == "metric"] == metrics


def test_the_provenance_section_is_never_narrated(evidence) -> None:
    built = build(evidence, narrator=Narrator(prose=Offline(), light=Offline()))
    assert {n.key for n in built.narrations} == {
        "methods",
        "results",
        "diagnostics",
        "limitations",
    }


# -- the section contract ----------------------------------------------------------------
#
# What each section is *shown* is the whole of the anti-repetition mechanism, so
# it is tested the way the numeric check is: by what cannot get through. A
# prohibition in a prompt is a hope; a fact withheld from the brief is a
# guarantee.


@dataclass
class Recorder:
    """A model that keeps every prompt it was handed, and echoes the draft back.

    Echoing rather than inventing keeps the numeric check satisfied, so a whole
    document can be built and the prompts inspected afterwards.
    """

    prompts: list[str]

    @property
    def name(self) -> str:
        return "recorder"

    def generate(self, prompt: str, *, system: str = "", temperature: float = 0.2) -> str:
        self.prompts.append(prompt)
        return prompt.split("DRAFT:\n", 1)[-1].strip() or "."


def test_every_narrated_section_has_both_a_licence_and_an_instruction() -> None:
    """The two tables are read together; a key in one and not the other is a bug."""
    assert set(LICENCE) == set(SECTION_INSTRUCTION) | {"abstract"}
    for key in SECTION_INSTRUCTION:
        assert key in LICENCE, f"{key} would silently fall back to the full licence"


def test_the_results_section_is_the_only_one_shown_the_numbers(evidence) -> None:
    """The intervals live in one section. Every other refers to it."""
    results = evidence_brief(evidence, licence=LICENCE["results"])
    assert "-16.7" in results and "-8.1" in results
    for key in ("introduction", "methods", "diagnostics", "discussion", "limitations"):
        brief = evidence_brief(evidence, licence=LICENCE[key])
        assert "-16.7" not in brief, f"{key} can see the interval and so can repeat it"


def test_the_discussion_is_shown_the_reading_and_not_the_interval() -> None:
    """It has to say what the finding means, and must not restate the number."""
    ev = _threshold_evidence()
    brief = evidence_brief(ev, licence=LICENCE["discussion"])
    assert "40 mg vs control" in brief
    assert "favourable side" in brief
    assert "-16.7" not in brief and "-8.1" not in brief


def test_the_limitations_section_is_the_only_one_shown_what_an_assumption_claims(
    evidence,
) -> None:
    """Eight sections naming the same assumption is what made these reports unreadable."""
    claimed = "age is the only common cause"
    assert claimed in evidence_brief(evidence, licence=LICENCE["limitations"])
    for key in ("introduction", "methods", "results", "diagnostics", "discussion", "conclusions"):
        assert claimed not in evidence_brief(evidence, licence=LICENCE[key])
    # The methods and the discussion still learn that it is standing, by name,
    # so they can point at it without arguing it.
    methods = evidence_brief(evidence, licence=LICENCE["methods"])
    assert "no unmeasured confounding" in methods
    assert "limitations section" in methods


def test_the_conclusions_are_shown_one_number_and_no_others() -> None:
    """A conclusions section that lists the findings has become a second results section."""
    ev = _contested_evidence()
    brief = evidence_brief(ev, licence=LICENCE["conclusions"])
    assert "6.20" in brief, "the finding the answer turns on arrives with its magnitude"
    assert "-5.27" not in brief and "-4.80" not in brief
    assert "10 mg vs control, pooled" in brief, "the rest are named"


def test_the_established_note_is_empty_for_the_first_section_written() -> None:
    assert established_note(()) == ""
    note = established_note(("introduction", "methods"))
    assert "the question" in note and "how the effect is identified" in note
    assert "do not restate them" in note


def test_the_exhibit_note_asks_the_prose_to_name_its_figure() -> None:
    note = exhibit_note((("Figure 2", "Estimated quantities with their intervals."),))
    assert "Figure 2" in note
    assert "by number" in note


def test_a_section_is_told_what_the_ones_before_it_established(evidence) -> None:
    """Cohesion is threaded through the build, not hoped for in each prompt."""
    recorder = Recorder(prompts=[])
    build(evidence, style="journal", narrator=Narrator(prose=recorder, light=Offline()))
    joined = "\n\n@@\n\n".join(recorder.prompts)
    assert "already established" in joined
    # The first narrated section has nothing behind it and is told nothing.
    assert "already established" not in recorder.prompts[0]
    # A later one is told about the earlier ones by name.
    assert any("(methods)" in p and "(results)" in p for p in recorder.prompts)


def test_the_prose_is_asked_to_name_the_numbered_exhibits(evidence) -> None:
    """A report whose text never names its figures is one whose reader never opens them."""
    recorder = Recorder(prompts=[])
    build(evidence, style="journal", narrator=Narrator(prose=recorder, light=Offline()))
    owning = [p for p in recorder.prompts if "exhibits in this section" in p]
    assert owning, "no section was told which exhibits it owns"
    assert any("Figure 1" in p or "Table 1" in p for p in owning)


def test_the_abstract_is_shown_one_magnitude_and_not_the_results_table() -> None:
    """It has nothing to repeat, but it is still not the results section."""
    ev = _contested_evidence()
    brief = evidence_brief(ev, licence=ABSTRACT_LICENCE)
    assert "8.42" in brief, "the headline finding keeps its interval"
    assert "-6.63" not in brief and "-6.17" not in brief
    assert "10 mg vs control, pooled" in brief, "the rest are named"


def test_a_short_draft_is_not_padded_to_the_verbosity_target() -> None:
    """Verbosity buys content; a section with little content cannot spend it."""
    long_target = int(VERBOSITY["full"]["sentences"])
    assert _budget("One sentence.", long_target) == 3
    assert _budget(" ".join(f"Sentence {i}." for i in range(20)), long_target) == long_target
    # An explicit budget wins, which is how the abstract keeps its own count.
    recorder = Recorder(prompts=[])
    narrate_text("introduction", "One sentence.", _threshold_evidence(), recorder, budget=9)
    assert "about 9 sentences" in recorder.prompts[0]


def test_the_facts_block_carries_facts_and_not_instructions() -> None:
    """A directive inside the facts gets published as prose; the discussion did."""
    ev = _threshold_evidence()
    for key in ("methods", "discussion", "conclusions"):
        brief = evidence_brief(ev, licence=LICENCE[key])
        for directive in ("do not", "Do not", "Say how many", "point the reader"):
            assert directive not in brief, f"{key}'s facts block gives an instruction"


def test_the_prompt_names_every_phrase_the_claims_check_rejects() -> None:
    """A rule that costs the whole rewrite should not be a hidden one."""
    from axiom_dossier import CLAIM_WORDS
    from axiom_dossier.narrate import SYSTEM

    for phrases in CLAIM_WORDS.values():
        for phrase in phrases:
            assert phrase in SYSTEM, f"{phrase!r} is checked but never named in the prompt"


def test_every_field_the_draft_renders_is_licensed() -> None:
    """A numeral the report puts in front of the model must not then be refused.

    ``instead`` and ``readout`` are printed verbatim by ``methods_section``, and
    an exhibit's caption is printed under it. A narration that repeats one of
    those is faithful, and licensing only ``what``/``why``/``detail`` threw it
    away — which cost the section its prose and looked, in the provenance
    appendix, exactly like a model that had invented a number.
    """
    from axiom_dossier import Exhibit, licensed_numbers

    ev = (
        EvidenceBuilder("T", "q?")
        .step(
            "s",
            "Step",
            what="Nothing numeric here.",
            why="Because.",
            instead="a round hole would have to be 13 degrees wide",
            readout=("alpha    : 41 counts",),
            exhibits=(Exhibit(key="table_1", kind="table", caption="7 rows of it"),),
        )
        .build()
    )
    licensed = set(licensed_numbers(ev))
    for value in (13.0, 41.0, 7.0):
        assert value in licensed, f"{value} is printed in the draft and not licensed"


def test_a_report_may_count_what_it_holds(evidence) -> None:
    """Every generated section states a count; none of them is stored as a number."""
    from axiom_dossier import licensed_numbers

    licensed = set(licensed_numbers(evidence))
    assert float(len(evidence.steps)) in licensed
    assert float(len(evidence.unresolved())) in licensed
    assert float(len(evidence.findings)) in licensed


def test_a_section_may_cite_the_exhibit_number_it_was_given(evidence) -> None:
    """The check must not reject the sentence the instruction asked for."""

    @dataclass(frozen=True)
    class Citer:
        name: str = "citer"

        def generate(self, prompt: str, *, system: str = "", temperature: float = 0.2) -> str:
            return "The recorded diagnostics are shown in Table 4."

    out = narrate_text(
        "diagnostics",
        "draft",
        evidence,
        Citer(),
        exhibits=(("Table 4", "Recorded diagnostics."),),
    )
    assert isinstance(out, Narration)
    assert out.verified, f"citing the exhibit it was given was rejected: {out.rejected}"

    # And without being given that exhibit, the same numeral is still refused.
    bare = narrate_text("diagnostics", "draft", evidence, Citer())
    assert isinstance(bare, Narration)
    assert not bare.verified and "4" in bare.rejected


def test_narration_keeps_the_equations_and_the_readouts() -> None:
    """The estimating equation is the one thing a methods section exists to show.

    ``_with_prose`` replaced every ``Paragraph`` in a section, and an equation is
    a paragraph of backticked lines — so narrating a report deleted its
    mathematics and its printed output, in every narrated document the package
    had produced.
    """
    from axiom_dossier import Narrator, build

    ev = (
        EvidenceBuilder("E", "q?")
        .step(
            "fit",
            "Fit",
            what="Least squares on the change.",
            why="Randomization licenses it.",
            equations=("y = a + b * x + e",),
            readout=("coef   b : 1.4",),
        )
        .build()
    )
    built = build(ev, style="journal", narrator=Narrator(prose=Offline(), light=Offline()))
    methods = next(s for s in built.report.sections if "Methods" in s.title)
    text = " ".join(getattr(b, "text", "") for b in methods.blocks)
    assert "y = a + b * x + e" in text, "narration deleted the equation"
    assert "coef   b : 1.4" in text, "narration deleted the readout"


def test_a_verbatim_block_is_not_handed_to_the_model_as_prose() -> None:
    """It is the record showing itself; there is nothing to rewrite."""
    from axiom.report import Paragraph

    from axiom_dossier.narrate import _draft_of, _is_verbatim
    from axiom_dossier.sections import equation_text

    equations = Paragraph(text=equation_text(("y = a + b * x",)))
    assert _is_verbatim(equations)
    assert not _is_verbatim(Paragraph(text="Ordinary prose about the fit."))
    section = Section(title="M", blocks=(Paragraph(text="Prose."), equations))
    assert _draft_of(section) == "Prose."
