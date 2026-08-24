"""The pipeline: what it reads, what it runs, and what it refuses to do."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest
from axiom.core import Unsupported

from axiom_dossier.agent import (
    ExecutionRefused,
    describe_namespace,
    execute,
    harvest,
    read_notebook,
    read_series,
    run_pipeline,
    run_python,
)
from axiom_dossier.agent.notebooks import describe_figure

pytest.importorskip("langgraph")


def notebook(tmp_path: Path, cells: list[tuple[str, str]], name: str = "01-x.ipynb") -> Path:
    """Write a tiny .ipynb from (kind, source) pairs."""
    path = tmp_path / name
    path.write_text(
        json.dumps(
            {
                "cells": [
                    {"cell_type": kind, "source": src.splitlines(keepends=True), "outputs": []}
                    for kind, src in cells
                ],
                "metadata": {},
                "nbformat": 4,
                "nbformat_minor": 5,
            }
        )
    )
    return path


# -- reading -----------------------------------------------------------------------------


def test_prose_stays_attached_to_the_code_it_introduced(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """A variable called `itt` is not a label; the sentence above it is."""
    path = notebook(
        tmp_path,
        [
            ("markdown", "## 2. What randomization identifies\nThe ITT contrast."),
            ("code", "itt = 3.5\nother = 1"),
        ],
    )
    book = read_notebook(path)
    passage = book.passages[0]
    assert passage.heading() == "2. What randomization identifies"
    assert "itt" in passage.defines and "other" in passage.defines
    assert "ITT contrast" in passage.text


def test_a_series_is_read_in_filename_order(tmp_path) -> None:  # type: ignore[no-untyped-def]
    notebook(tmp_path, [("code", "a = 1")], "02-second.ipynb")
    notebook(tmp_path, [("code", "b = 2")], "01-first.ipynb")
    assert [b.name for b in read_series(tmp_path)] == ["01-first.ipynb", "02-second.ipynb"]


def test_an_empty_directory_is_an_error_not_an_empty_report(tmp_path) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(FileNotFoundError, match="no notebooks"):
        read_series(tmp_path)


# -- running -----------------------------------------------------------------------------


def test_a_failing_cell_stops_the_notebook_and_is_reported(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """A report built from a half-run notebook describes an analysis that did not finish."""
    path = notebook(
        tmp_path,
        [("code", "good = 1"), ("code", "raise ValueError('boom')"), ("code", "never = 2")],
    )
    ns: dict = {}
    result = execute(read_notebook(path), ns)
    assert not result.complete
    assert result.cells_run == 1
    assert "ValueError: boom" in result.error
    assert "never" not in ns


def test_magics_and_shell_lines_are_stripped(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = notebook(tmp_path, [("code", "%matplotlib inline\n!echo hi\nvalue = 7")])
    ns: dict = {}
    assert execute(read_notebook(path), ns).complete
    assert ns["value"] == 7


def test_the_notebook_directory_is_importable_while_it_runs(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Jupyter puts the notebook's directory on sys.path; without that every
    `import hyper3` in the series fails on the first cell."""
    (tmp_path / "sidecar.py").write_text("VALUE = 42\n")
    path = notebook(tmp_path, [("code", "import sidecar\ngot = sidecar.VALUE")])
    ns: dict = {}
    assert execute(read_notebook(path), ns).complete, "the sidecar module was not importable"
    assert ns["got"] == 42


def test_harvest_separates_quantities_from_figures() -> None:
    from axiom.core import Interval

    plotly = pytest.importorskip("plotly.graph_objects")
    ns = {
        "band": Interval(lower=1.0, upper=2.0, definition="eti", mass=0.9),
        "fig": plotly.Figure(),
        "helper": lambda: None,
        "_private": 1,
    }
    quantities, figures = harvest(ns)
    assert list(quantities) == ["band"]
    assert list(figures) == ["fig"]


def test_a_figure_is_described_by_what_is_in_it() -> None:
    """The gap stage judging `01_fig` blind either invents work or sees none."""
    go = pytest.importorskip("plotly.graph_objects")
    fig = go.Figure(go.Scatter(x=[1], y=[2], name="dose 40"))
    fig.update_layout(xaxis_title="week", yaxis_title="mmHg")
    described = describe_figure(fig)
    assert "scatter" in described
    assert "week" in described and "mmHg" in described
    assert "dose 40" in described


# -- the executor ------------------------------------------------------------------------


def test_generated_code_does_not_run_unless_it_was_allowed() -> None:
    with pytest.raises(ExecutionRefused, match="allow_execution"):
        run_python("x = 1", {}, expect="x")


def test_a_cell_that_binds_nothing_is_a_failure_not_a_silent_pass() -> None:
    out = run_python("x = 1", {}, expect="figure", allow_execution=True)
    assert not out.ok
    assert "bound nothing" in out.error


def test_a_cell_that_binds_the_wrong_kind_is_refused() -> None:
    out = run_python("figure = 12", {}, expect="figure", allow_execution=True)
    assert not out.ok
    assert "is a int" in out.error


def test_a_traceback_comes_back_as_text_for_a_repair_attempt() -> None:
    out = run_python("raise RuntimeError('nope')", {}, expect="figure", allow_execution=True)
    assert not out.ok
    assert "RuntimeError" in out.error
    assert "nope" in out.feedback()


def test_a_failed_cell_leaves_the_namespace_as_it_found_it() -> None:
    ns = {"keep": 1}
    run_python("added = 2\nraise ValueError()", ns, expect="figure", allow_execution=True)
    assert ns == {"keep": 1}, "a failed cell must not leave half its work behind"


def test_a_good_cell_binds_its_figure() -> None:
    pytest.importorskip("plotly.graph_objects")
    ns: dict = {}
    out = run_python(
        "import plotly.graph_objects as go\nfigure = go.Figure()",
        ns,
        expect="figure",
        allow_execution=True,
    )
    assert out.ok and out.bound == "figure" and out.kind == "Figure"


def test_the_namespace_description_names_things_and_never_values() -> None:
    """A model shown numbers puts them in prose, and the gates then check it
    against its own arithmetic."""
    described = describe_namespace({"estimate": 12.4, "label": "secret", "_hidden": 1})
    assert "estimate: float" in described
    assert "12.4" not in described and "secret" not in described
    assert "_hidden" not in described


# -- the graph ---------------------------------------------------------------------------


def test_the_pipeline_runs_end_to_end_without_a_model(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from axiom.core import Interval

    notebook(
        tmp_path,
        [
            ("markdown", "## 1. The contrast\nWhat the arms differ by."),
            (
                "code",
                "from axiom.core import Interval\n"
                "band = Interval(lower=-2.0, upper=-1.0, definition='eti', mass=0.9)",
            ),
        ],
    )
    run = run_pipeline(tmp_path, title="T", question="q?")
    assert run.complete()
    assert run.evidence is not None
    assert len(run.evidence.findings) == 1
    assert run.evidence.findings[0].label.startswith("band")
    assert "no model" in " ".join(run.transcript)
    assert isinstance(run.evidence.findings[0].interval, Interval)


def test_gaps_are_found_but_not_closed_when_execution_is_off(tmp_path) -> None:  # type: ignore[no-untyped-def]
    @dataclass(frozen=True)
    class Stub:
        name: str = "stub"

        def generate(self, prompt: str, *, system: str = "", temperature: float = 0.2) -> str:
            if "name things" in system:
                return (
                    "label: A contrast\nunit: mmHg\nthreshold: 0\nbeneficial: lower\ninclude: yes"
                )
            if "which figures" in system:
                return "the dose-response curve | the prose describes it"
            return "```python\nfigure = 1\n```"

    notebook(
        tmp_path,
        [
            ("markdown", "## The curve\nA dose-response curve is described here."),
            (
                "code",
                "from axiom.core import Interval\n"
                "band = Interval(lower=-2.0, upper=-1.0, definition='eti', mass=0.9)",
            ),
        ],
    )
    run = run_pipeline(tmp_path, title="T", model=Stub(), allow_execution=False)
    assert len(run.gaps) == 1
    assert not run.gaps[0].resolved
    assert "execution not enabled" in " ".join(run.transcript)
    # the label stage still ran, and its answer reached the finding
    assert run.evidence is not None
    assert run.evidence.findings[0].label == "A contrast"
    assert run.evidence.findings[0].unit == "mmHg"


def test_an_unreachable_model_leaves_a_report_rather_than_an_exception(tmp_path) -> None:  # type: ignore[no-untyped-def]
    @dataclass(frozen=True)
    class Absent:
        name: str = "absent"

        def generate(self, prompt: str, *, system: str = "", temperature: float = 0.2):
            return Unsupported(reason="no key configured")

    notebook(tmp_path, [("code", "x = 1")])
    run = run_pipeline(tmp_path, title="T", model=Absent())
    assert run.evidence is not None
    assert run.complete()


def test_an_empty_reply_from_the_gap_stage_means_no_gaps(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """It is how a model says 'nothing missing', and it is not a failure."""

    @dataclass(frozen=True)
    class Silent:
        name: str = "silent"

        def generate(self, prompt: str, *, system: str = "", temperature: float = 0.2):
            if "which figures" in system:
                return Unsupported(reason="silent returned no text")
            return "label: X\ninclude: yes"

    notebook(tmp_path, [("code", "x = 1")])
    run = run_pipeline(tmp_path, title="T", model=Silent())
    assert run.gaps == ()
    assert "nothing reported as missing" in " ".join(run.transcript)
