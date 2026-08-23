"""Narration: the plumbing, and the refusal that makes it safe."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from axiom.core import Assumption, Interval, Unsupported
from axiom.report import Heading, Paragraph

from axiom_dossier import (
    EvidenceBuilder,
    Gemini,
    LanguageModel,
    Narration,
    Narrator,
    Offline,
    build,
    evidence_brief,
    methods_section,
    narrate_text,
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
    assert "no_unmeasured_confounding" in brief
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
