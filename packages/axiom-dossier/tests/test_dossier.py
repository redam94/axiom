"""Sections and assembly: what the generated document is obliged to contain."""

from __future__ import annotations

import pytest
from axiom.core import Assumption, Interval, LedgerLine, Unsupported
from axiom.identify import CausalGraph, identify

from axiom_dossier import (
    DEFAULT_SECTIONS,
    EvidenceBuilder,
    build,
    context_for,
    limitations_section,
    methods_section,
    results_section,
    standing_assumptions,
)

STANDING = Assumption(
    name="no_unmeasured_confounding",
    facet="population",
    statement="age is the only common cause of dose and pressure",
    challenged_by="a sensitivity analysis at plausible confounder strength",
    state="unverified",
)


@pytest.fixture
def evidence():
    graph = CausalGraph.from_edges("age -> dose, age -> pressure, dose -> pressure")
    return (
        EvidenceBuilder("HYPER-3", "Does 40 mg lower systolic pressure?")
        .verdict(identify(graph, "dose", "pressure"))
        .step("design", "Design", what="Two arms of 300 participants.", why="Powered at 80 %.")
        .finding(
            "contrast",
            Interval(lower=-16.7, upper=-8.1, definition="eti", mass=0.9),
            label="40 mg vs control",
            unit="mmHg",
            source="estimands.realize",
            precision=1,
        )
        .diagnostic("coverage", 0.94, label="Interval coverage")
        .assume(STANDING)
        .ledger_lines(
            [
                LedgerLine(
                    kind="assumption", statement="Age adjusts the dose effect.", assumption=STANDING
                )
            ]
        )
        .provenance(seed="0")
        .build()
    )


def test_the_methods_section_names_the_route_and_the_adjustment_set(evidence) -> None:
    """It is generated from the verdict, so it cannot describe a route nobody took."""
    text = " ".join(getattr(b, "text", "") for b in methods_section(evidence).blocks)
    assert "backdoor" in text
    assert "adjusts for age" in text
    assert "300 participants" in text


def test_the_limitations_section_cannot_be_shorter_than_the_truth(evidence) -> None:
    text = " ".join(getattr(b, "text", "") for b in limitations_section(evidence).blocks)
    for name in evidence.unresolved():
        assert name.replace("_", " ") in text
    assert "sensitivity analysis" in text, "what would challenge it is stated too"


def test_a_finding_is_emitted_as_a_metric_so_its_interval_cannot_be_dropped(evidence) -> None:
    blocks = results_section(evidence).blocks
    metrics = [b for b in blocks if getattr(b, "block", "") == "metric"]
    assert [b.source for b in metrics] == ["contrast"]
    assert metrics[0].unit == "mmHg"


def test_standing_assumptions_are_exactly_the_unresolved_ones(evidence) -> None:
    assert tuple(a.name for a in standing_assumptions(evidence)) == evidence.unresolved()


def test_the_document_builds_with_no_missing_context_keys(evidence) -> None:
    built = build(evidence)
    assert built.missing() == ()
    assert [s.title for s in built.report.sections] == [
        "Methods",
        "Results",
        "Model checking",
        "Limitations",
        "Provenance",
    ]
    assert built.rejected() == ()


def test_the_context_carries_the_three_generated_tables(evidence) -> None:
    ctx = context_for(evidence)
    assert [r["Assumption"] for r in ctx["assumption_table"]] == ["no unmeasured confounding"]
    assert [r["Key"] for r in ctx["provenance_table"]] == ["contrast", "coverage"]
    fields = [r["Field"] for r in ctx["run_table"]]
    assert "seed" in fields and "evidence hash" in fields


def test_the_evidence_hash_is_recorded_in_the_document(evidence) -> None:
    """Same evidence, same claims — the hash is what makes that checkable."""
    ctx = context_for(evidence)
    row = next(r for r in ctx["run_table"] if r["Field"] == "evidence hash")
    assert row["Value"] == evidence.content_hash()


def test_sections_can_be_chosen_and_an_unknown_one_is_refused(evidence) -> None:
    built = build(evidence, sections=("results",))
    assert [s.title for s in built.report.sections] == ["Results"]
    with pytest.raises(ValueError, match="unknown section"):
        build(evidence, sections=("methods", "epilogue"))


def test_the_default_order_is_what_a_reader_expects() -> None:
    assert DEFAULT_SECTIONS == (
        "methods",
        "results",
        "remarks",
        "diagnostics",
        "limitations",
        "provenance",
    )


def test_a_record_with_no_remarks_gets_no_empty_remarks_section(evidence) -> None:
    """The section is dropped rather than rendered saying it has nothing to say."""
    assert not evidence.remarks
    assert "What the run showed" not in [s.title for s in build(evidence).report.sections]

    spoken = evidence.model_copy(update={"remarks": ("It did what it said.",)})
    titles = [s.title for s in build(spoken).report.sections]
    assert "What the run showed" in titles


def test_an_analysis_with_no_diagnostics_says_so_rather_than_omitting_the_section() -> None:
    bare = EvidenceBuilder("T").finding("c", 1.0, label="C").build()
    built = build(bare)
    text = " ".join(getattr(b, "text", "") for s in built.report.sections for b in s.blocks)
    assert "unexamined" in text


def test_it_renders_to_html(evidence, tmp_path) -> None:
    out = build(evidence).write(str(tmp_path / "r.html"))
    assert not isinstance(out, Unsupported), out
    body = (tmp_path / "r.html").read_text()
    assert "HYPER-3" in body
    assert "-16.7" in body, "the interval reaches the page"
