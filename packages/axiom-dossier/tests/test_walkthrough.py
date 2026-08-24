"""Turning an example's own record into a report, including the awkward text."""

from __future__ import annotations

import pytest
from axiom.core import Unsupported
from axiom.report import Paragraph

from axiom_dossier import (
    JOURNAL_THEME,
    build,
    evidence_from_record,
    exhibits_from_record,
    figures_from_record,
    literal,
    tables_from_record,
)

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
            "blocks": [
                {"type": "say", "text": "The first stage is strong."},
                {
                    "type": "figure",
                    "name": "first_stage",
                    "kind": "bars",
                    "opt": {"rows": "@rows", "xLabel": "F statistic"},
                    "title": "The first stage, by cohort",
                    "note": "Anything under ten is a weak instrument",
                },
            ],
        },
    ],
    "findings": ["OLS overstates the programme by +0.35.", "Bias is not a small-sample problem."],
    "figures": {"first_stage": {"rows": [{"label": "1998", "value": 41.2}]}},
}


def test_the_record_becomes_an_evidence_that_keeps_the_reasoning() -> None:
    ev = evidence_from_record(RECORD)
    assert ev.title == "An effect you can only see sideways"
    assert len(ev.steps) == 2
    assert ev.steps[0].why.startswith("The obstacle")
    # the rejected alternative is the most useful thing in an example
    assert ev.steps[0].instead == "Starting from the estimator."
    assert ev.steps[1].instead == ""


def test_printed_output_is_kept_as_lines_rather_than_run_together() -> None:
    """Column-aligned output is aligned in columns; joining it loses the columns."""
    ev = evidence_from_record(RECORD)
    assert ev.steps[0].readout == ("naive OLS 2.347", "2SLS 1.999", "truth 2.000")
    assert ev.steps[0].detail == {}, "printed output is not a design parameter"


def test_a_step_names_the_exhibits_it_produced() -> None:
    ev = evidence_from_record(RECORD)
    (table,) = ev.steps[0].exhibits
    assert (table.key, table.kind, table.caption) == ("table_1", "table", "Two estimators")
    (figure,) = ev.steps[1].exhibits
    assert figure.key == "figure_1" and figure.kind == "figure"
    assert figure.caption == (
        "The first stage, by cohort. Anything under ten is a weak instrument"
    ), "a title and its note are two sentences, not one run-on"


def test_the_recorded_charts_are_drawn_and_reach_the_document() -> None:
    figures = figures_from_record(RECORD, theme=JOURNAL_THEME)
    assert list(figures) == ["figure_1"]
    built = build(
        evidence_from_record(RECORD),
        style="journal",
        verbosity="full",
        extra_exhibits=exhibits_from_record(RECORD, theme=JOURNAL_THEME),
    )
    assert built.missing() == ()
    sources = {getattr(b, "source", "") for s in built.report.sections for b in s.blocks}
    assert {"figure_1", "table_1"} <= sources, "the run's own exhibits were dropped"


def test_an_exhibit_with_no_data_behind_it_is_dropped_rather_than_breaking_the_render() -> None:
    """A chart plotly cannot draw must not leave the report naming a key nobody filled."""
    built = build(evidence_from_record(RECORD), style="journal", verbosity="full")
    assert built.missing() == ()
    sources = {getattr(b, "source", "") for s in built.report.sections for b in s.blocks}
    assert "figure_1" not in sources and "table_1" not in sources


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


def test_the_evidence_holds_raw_text_and_the_escape_happens_at_the_paragraph() -> None:
    """Escaping on the way in put ``{{'alpha': 1.41}}`` in a design table cell."""
    braced = {
        **RECORD,
        "steps": [
            {
                "n": 1,
                "title": "Print a dict",
                "why": "Because {a} is what the run printed.",
                "blocks": [{"type": "out", "lines": ["detail : {'alpha': 1.41}"]}],
            }
        ],
    }
    ev = evidence_from_record(braced)
    assert ev.steps[0].readout == ("detail : {'alpha': 1.41}",), "the record is kept as it was"
    assert "{{" not in ev.steps[0].why
    built = build(ev, style="journal", verbosity="full")
    assert built.missing() == ()
    text = " ".join(getattr(b, "text", "") for s in built.report.sections for b in s.blocks)
    assert "{{'alpha': 1.41}}" in text, "the paragraph is where the brace is escaped"


def test_a_record_with_no_quantities_still_renders(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The examples narrate results in prose, so there is nothing to put in a metric."""
    ev = evidence_from_record(RECORD)
    assert ev.findings == ()
    built = build(ev, style="journal", verbosity="full")
    assert built.missing() == ()
    out = built.write(str(tmp_path / "r.pdf"))
    assert not isinstance(out, Unsupported), out
    assert (tmp_path / "r.pdf").read_bytes().startswith(b"%PDF")


def test_the_abstract_covers_every_part_it_has_material_for() -> None:
    """One flowing paragraph, and no part written as a bare label.

    The UCSD guide asks for "a one-paragraph summary of the whole study" and
    forbids emphasis outside a heading, so the abstract carries no ``Objective.``
    / ``Results.`` labels to leave stranded. What it must still not do is skip a
    part it has the material for: this record has no quantities, so its result
    comes from the analyst's own remark.
    """
    built = build(evidence_from_record(RECORD), style="journal")
    text = " ".join(getattr(b, "text", "") for b in built.report.sections[0].blocks)
    assert "**" not in text, "the abstract is emphasised, which the guide forbids"
    assert "raise earnings?" in text
    assert "OLS overstates" in text
    # no assumptions were recorded, which is not the same as none being needed
    assert "not the same as none being required" in text


def test_an_empty_record_is_refused_rather_than_producing_a_blank_report() -> None:
    with pytest.raises(ValueError, match="at least one section"):
        build(evidence_from_record({"title": "x"}), sections=())
