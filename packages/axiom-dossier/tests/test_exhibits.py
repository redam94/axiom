"""Figures, data tables, and the APA manuscript shape."""

from __future__ import annotations

import pytest
from axiom.core import Assumption, Interval, Unsupported, Verdict
from axiom.report import Figure, Table

from axiom_dossier import (
    APA_SECTIONS,
    APA_THEME,
    EvidenceBuilder,
    apa_caption,
    build,
    design_rows,
    diagnostics_plot,
    diagnostics_rows,
    figures_for,
    findings_plot,
    findings_rows,
    running_head,
    tables_for,
)
from axiom_dossier.figures import SPREAD_LIMIT, available

STANDING = Assumption(
    name="no_unmeasured_confounding",
    facet="population",
    statement="age is the only common cause",
    challenged_by="a sensitivity analysis",
    state="unverified",
)


def evidence(*, spans: bool = False):
    builder = EvidenceBuilder("HYPER-3", "Which dose goes forward?")
    builder.verdict(Verdict(status="identified", reason="age blocks it", route="backdoor"))
    builder.step(
        "protocol",
        "Protocol",
        what="Two arms.",
        why="Powered at 80 %.",
        detail={"arms": "2", "allocation": "2:1"},
    )
    builder.finding(
        "helps",
        Interval(lower=-16.7, upper=-8.1, definition="eti", mass=0.9),
        label="40 mg vs control",
        unit="mmHg",
        precision=1,
        threshold=0.0,
        beneficial="lower",
    )
    builder.finding(
        "harms" if not spans else "unsettled",
        (
            Interval(lower=3.98, upper=8.42, definition="eti", mass=0.9)
            if not spans
            else Interval(lower=-2.0, upper=4.0, definition="eti", mass=0.9)
        ),
        label="40 mg vs control, age 51+",
        unit="mmHg",
        precision=1,
        threshold=0.0,
        beneficial="lower",
    )
    builder.diagnostic("coverage", 0.94, label="Interval coverage", precision=2)
    builder.assume(STANDING)
    return builder.build()


# -- tables ------------------------------------------------------------------------------


def test_the_findings_table_reports_where_each_interval_landed() -> None:
    rows = findings_rows(evidence())
    assert [r["Relative to threshold"] for r in rows] == ["below threshold", "above threshold"]
    assert rows[0]["Interval"] == "-16.7 to -8.1 (90% ETI)"
    assert rows[0]["Unit"] == "mmHg"


def test_a_spanning_interval_is_never_called_a_null_in_the_table() -> None:
    rows = findings_rows(evidence(spans=True))
    assert rows[1]["Relative to threshold"] == "spans threshold"
    assert "no effect" not in " ".join(rows[1].values()).lower()


def test_the_design_table_is_the_steps_own_recorded_parameters() -> None:
    rows = design_rows(evidence())
    parameters = {r["Parameter"] for r in rows}
    assert {"arms", "allocation"} <= parameters
    # the identification step contributes its parameters too, which is the point:
    # the design table is every setting the record holds, not one step's
    assert {"route", "status"} <= parameters
    assert {r["Step"] for r in rows} == {"Protocol", "Identification"}


def test_diagnostics_rows_describe_the_fit() -> None:
    assert [r["Check"] for r in diagnostics_rows(evidence())] == ["Interval coverage"]


def test_only_non_empty_tables_reach_the_context() -> None:
    bare = EvidenceBuilder("T").finding("c", 1.0, label="C").build()
    keys = tables_for(bare)
    assert "findings_table" in keys
    assert "design_table" not in keys, "no steps means no design table to name"


# -- figures -----------------------------------------------------------------------------

needs_plotly = pytest.mark.skipif(not available(), reason="plotly is not installed")


@needs_plotly
def test_the_findings_plot_colours_each_row_by_the_side_it_settled_on() -> None:
    figure = findings_plot(evidence())
    assert not isinstance(figure, Unsupported), figure
    colours = list(figure.data[0].marker.color)
    assert colours[0] != colours[1], "a benefit and a harm must not be the same colour"
    assert len(set(colours)) == 2


@needs_plotly
def test_an_unsettled_row_is_drawn_as_neither() -> None:
    """Three states, three colours: favourable, unfavourable, and not settled."""
    figure = findings_plot(evidence(spans=True))
    colours = list(figure.data[0].marker.color)
    assert colours[1] == "#5b6472", "a spanning interval is grey, not green or red"


@needs_plotly
def test_the_figure_carries_no_title_because_the_caption_does() -> None:
    figure = findings_plot(evidence())
    assert figure.layout.title.text in (None, "")
    assert figure.layout.xaxis.title.text == "mmHg", "an axis without its unit is not a measurement"


@needs_plotly
def test_the_threshold_is_marked_when_the_findings_share_one() -> None:
    figure = findings_plot(evidence())
    assert figure.layout.shapes, "no rule was drawn at the threshold"
    assert any("threshold" in str(a.text) for a in figure.layout.annotations)


@needs_plotly
def test_incommensurable_diagnostics_are_refused_rather_than_drawn_misleadingly() -> None:
    """A share of 0.797 beside a count of 372 draws the share as nothing at all."""
    builder = EvidenceBuilder("T")
    builder.finding("c", 1.0, label="C")
    builder.diagnostic("share", 0.797, label="Retention")
    builder.diagnostic("count", 372.0, label="Units analysed")
    out = diagnostics_plot(builder.build())
    assert isinstance(out, Unsupported)
    assert "orders of magnitude" in out.reason
    assert float(out.detail["limit"]) == SPREAD_LIMIT


@needs_plotly
def test_comparable_diagnostics_do_get_drawn() -> None:
    builder = EvidenceBuilder("T")
    builder.finding("c", 1.0, label="C")
    builder.diagnostic("a", 0.94, label="Coverage")
    builder.diagnostic("b", 0.88, label="Power")
    assert not isinstance(diagnostics_plot(builder.build()), Unsupported)


@needs_plotly
def test_a_diagnostic_is_measured_against_the_axis_not_the_largest_bar() -> None:
    """A z of -2.26 beside 40 depots: 44x by ratio, 47x of the axis it is drawn on."""
    builder = EvidenceBuilder("T")
    builder.finding("c", 1.0, label="C")
    builder.diagnostic("z", -2.26, label="Agreement z")
    builder.diagnostic("power", 0.9, label="Power")
    builder.diagnostic("units", 40.0, label="Depots randomized")
    out = diagnostics_plot(builder.build())
    assert isinstance(out, Unsupported), "a bar of a fiftieth of the axis is not a bar"
    assert out.detail["span"] == "42.26"


@needs_plotly
def test_a_negative_diagnostic_is_drawn_against_a_baseline_with_its_label_clear() -> None:
    """Two z-scores are commensurable; what they need is a zero to be read against."""
    builder = EvidenceBuilder("T")
    builder.finding("c", 1.0, label="C")
    builder.diagnostic("before", -2.26, label="Agreement z, observational fit")
    builder.diagnostic("after", -1.01, label="Agreement z, calibrated fit")
    figure = diagnostics_plot(builder.build())
    assert not isinstance(figure, Unsupported), figure
    assert figure.layout.xaxis.zeroline, "a bar drawn leftwards from nothing starts nowhere"
    assert figure.data[0].textposition == "outside"
    assert figure.data[0].cliponaxis is False
    low, high = figure.layout.xaxis.range
    assert low < -2.26 and high > 0.0, "no room for the label of the longest bar"


@needs_plotly
def test_diagnostics_that_are_all_positive_keep_the_axis_at_zero() -> None:
    builder = EvidenceBuilder("T")
    builder.finding("c", 1.0, label="C")
    builder.diagnostic("a", 0.94, label="Coverage")
    builder.diagnostic("b", 0.88, label="Power")
    figure = diagnostics_plot(builder.build())
    assert not isinstance(figure, Unsupported), figure
    assert not figure.layout.xaxis.zeroline, "nothing crosses zero; the baseline is the axis"
    low, _ = figure.layout.xaxis.range
    assert low < 0.0, "the axis still starts at zero, with room for a label"


@needs_plotly
def test_only_drawable_figures_reach_the_context() -> None:
    keys = figures_for(evidence())
    assert "findings_figure" in keys
    bare = EvidenceBuilder("T").diagnostic("d", 1.0, label="D").build()
    assert "findings_figure" not in figures_for(bare), "no findings, no findings figure"


# -- APA ---------------------------------------------------------------------------------


def test_the_running_head_is_capped_on_a_word_boundary() -> None:
    long = "A sequential dose-finding trial in adults with elevated blood pressure"
    head = running_head(long)
    assert len(head) <= 50
    assert head == head.upper()
    assert not head.endswith(" ")
    assert running_head("Short") == "SHORT"


def test_the_apa_caption_italicises_the_label_and_not_the_description() -> None:
    assert apa_caption("Figure", 1, "Mean change.") == "*Figure 1*. Mean change."


def test_the_apa_theme_is_twelve_point_times_with_inch_margins() -> None:
    assert (APA_THEME.font, APA_THEME.base_size, APA_THEME.margin) == ("Times-Roman", 12.0, 72.0)
    assert APA_THEME.title_size == APA_THEME.base_size, "APA sets no display type"


def test_the_apa_manuscript_gathers_its_exhibits_after_the_text() -> None:
    built = build(evidence(), style="apa", authors=("A. Author",), affiliation="Somewhere")
    titles = [s.title for s in built.report.sections]
    assert titles[0] == "Author Note"
    assert titles[-2:] == ["Tables", "Figures"] if available() else titles[-1] == "Tables"
    assert built.missing() == ()
    assert built.report.subtitle == "A. Author · Somewhere"


def test_apa_exhibits_are_numbered_once() -> None:
    """Numbering a self-numbered section again produced 'Figure 1. Figure 1.'."""
    built = build(evidence(), style="apa")
    captions = [
        b.caption
        for s in built.report.sections
        for b in s.blocks
        if isinstance(b, (Figure, Table)) and b.caption
    ]
    for caption in captions:
        assert caption.count("Table") <= 1 and caption.count("Figure") <= 1


def test_embedded_exhibits_sit_with_the_prose_that_discusses_them() -> None:
    built = build(evidence(), style="journal", exhibits="embedded")
    results = next(s for s in built.report.sections if "Results" in s.title)
    assert any(isinstance(b, Figure) for b in results.blocks) == available()
    assert any(isinstance(b, Table) for b in results.blocks)


def test_exhibits_can_be_left_out_entirely() -> None:
    built = build(evidence(), style="journal", exhibits="none")
    assert not any(isinstance(b, Figure) for s in built.report.sections for b in s.blocks)
    assert built.missing() == ()


def test_an_unknown_exhibit_mode_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown exhibits"):
        build(evidence(), exhibits="appendix")  # type: ignore[arg-type]


def test_the_apa_order_matches_the_guide() -> None:
    assert APA_SECTIONS[:3] == ("title_page", "abstract", "introduction")
    assert APA_SECTIONS.index("discussion") > APA_SECTIONS.index("results")


def test_an_apa_manuscript_renders(tmp_path) -> None:  # type: ignore[no-untyped-def]
    out = build(evidence(), style="apa").write(str(tmp_path / "m.pdf"))
    assert not isinstance(out, Unsupported), out
    assert (tmp_path / "m.pdf").read_bytes().startswith(b"%PDF")


@needs_plotly
def test_the_html_keeps_the_figure_interactive(tmp_path) -> None:
    """A rasterised chart in HTML would be a regression, not a rendering choice."""
    out = build(evidence(), style="apa").write(str(tmp_path / "m.html"))
    assert not isinstance(out, Unsupported), out
    html = (tmp_path / "m.html").read_text()
    assert "newPlot" in html, "the figure was not drawn by plotly at open time"
    assert "hovertemplate" in html


@needs_plotly
def test_the_library_can_be_left_out_of_the_html(tmp_path) -> None:
    built = build(evidence(), style="apa")
    built.write(str(tmp_path / "big.html"))
    built.write(str(tmp_path / "small.html"), inline_plotly=False)
    assert (tmp_path / "small.html").stat().st_size < (tmp_path / "big.html").stat().st_size
