"""Turning an example's own record into a report, including the awkward text."""

from __future__ import annotations

import pytest
from axiom.core import Unsupported
from axiom.report import Paragraph

from axiom_dossier import build, evidence_from_record, literal, tables_from_record

RECORD = {
    "field": "economics",
    "title": "An effect you can only see sideways",
    "question": "Does a training programme raise earnings?",
    "steps": [
        {
            "n": 1,
            "title": "Write down the world",
            "why": "The obstacle is a structure, not a missing column.",
            "instead": "Starting from the estimator.",
            "blocks": [
                {"type": "say", "text": "Drive pushes both enrolment and earnings."},
                {"type": "out", "lines": ["naive OLS 2.347", "2SLS 1.999", "", "truth 2.000"]},
                {
                    "type": "table",
                    "columns": ["estimator", "estimate"],
                    "rows": [["OLS", "2.347"], ["2SLS", "1.999"]],
                    "caption": "Two estimators",
                },
            ],
        },
        {
            "n": 2,
            "title": "Check the instrument",
            "why": "A weak instrument is not a neutral loss of precision.",
            "instead": None,
            "blocks": [{"type": "say", "text": "The first stage is strong."}],
        },
    ],
    "findings": ["OLS overstates the programme by +0.35.", "Bias is not a small-sample problem."],
    "figures": {},
}


def test_the_record_becomes_an_evidence_that_keeps_the_reasoning() -> None:
    ev = evidence_from_record(RECORD)
    assert ev.title == "An effect you can only see sideways"
    assert len(ev.steps) == 2
    assert ev.steps[0].why.startswith("The obstacle")
    # the rejected alternative is the most useful thing in an example
    assert ev.steps[0].detail["considered instead"] == "Starting from the estimator."
    assert "naive OLS 2.347" in ev.steps[0].detail["readout"]
    assert ev.steps[1].detail.get("considered instead") is None


def test_the_analysts_own_words_are_carried_verbatim() -> None:
    ev = evidence_from_record(RECORD)
    assert ev.remarks == tuple(RECORD["findings"])


def test_remarks_are_never_narrated() -> None:
    """A model rewriting a person's conclusion, still attributed to them, is a lie."""
    from axiom_dossier import Narrator, Offline

    built = build(
        evidence_from_record(RECORD),
        style="journal",
        narrator=Narrator(prose=Offline(), light=Offline()),
    )
    assert "remarks" not in {n.key for n in built.narrations}
    section = next(s for s in built.report.sections if "What the run showed" in s.title)
    rendered = " ".join(b.text for b in section.blocks if isinstance(b, Paragraph))
    for remark in RECORD["findings"]:
        assert remark in rendered


def test_tables_come_back_keyed_in_document_order() -> None:
    tables = tables_from_record(RECORD)
    assert list(tables) == ["table_1"]
    assert tables["table_1"] == [
        {"estimator": "OLS", "estimate": "2.347"},
        {"estimator": "2SLS", "estimate": "1.999"},
    ]


def test_braces_in_prose_do_not_become_template_placeholders() -> None:
    """Three of the twelve examples write ``{}`` or ``{n}`` into their narration."""
    assert literal("a dict {} and a {n}") == "a dict {{}} and a {{n}}"

    awkward = {
        **RECORD,
        "findings": ["The empty set {} is not a covariate, and neither is {n}."],
        "question": "What does {alpha} do?",
    }
    built = build(evidence_from_record(awkward), style="journal")
    assert built.missing() == (), "a brace was read as a context key"


def test_a_record_with_no_quantities_still_renders(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The examples narrate results in prose, so there is nothing to put in a metric."""
    ev = evidence_from_record(RECORD)
    assert ev.findings == ()
    built = build(ev, style="journal", verbosity="full")
    assert built.missing() == ()
    out = built.write(str(tmp_path / "r.pdf"))
    assert not isinstance(out, Unsupported), out
    assert (tmp_path / "r.pdf").read_bytes().startswith(b"%PDF")


def test_the_abstract_does_not_print_empty_headings() -> None:
    """A structured abstract whose labels stand alone is worse than a short one."""
    built = build(evidence_from_record(RECORD), style="journal")
    text = " ".join(getattr(b, "text", "") for b in built.report.sections[0].blocks)
    assert "Methods. Conclusions." not in text
    assert "Results." in text and "OLS overstates" in text
    # no assumptions were recorded, which is not the same as none being needed
    assert "not the same as none being required" in text


def test_an_empty_record_is_refused_rather_than_producing_a_blank_report() -> None:
    with pytest.raises(ValueError, match="at least one section"):
        build(evidence_from_record({"title": "x"}), sections=())
