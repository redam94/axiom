"""The pipeline: read, run, label, find the gaps, close them, assemble, verify.

Seven stages, wired with LangGraph so the order is a declared thing rather than
the order of statements in a function, and so the repair loop is an edge rather
than a ``while``.

    read ─▶ run ─▶ label ─▶ gaps ─▶ fill ─▶ assemble ─▶ verify ─┐
                                     ▲                          │
                                     └──────── unresolved ──────┘

Which stage does what, and which of them needs a model:

===========  ======  ==========================================================
stage        model?  what it does
===========  ======  ==========================================================
``read``     no      parse the notebook series and the planning notes
``run``      no      execute every notebook in one namespace; harvest per book
``label``    yes     name each harvested object from the prose that introduced it
``gaps``     yes     find claims with no quantity and sections with no exhibit
``fill``     yes     write and run Python that draws what is missing
``assemble`` no      build the ``Evidence`` and the document
``verify``   no      run the numeric and claim gates; loop if anything is open
===========  ======  ==========================================================

The three model stages are all **naming and noticing**, never computing. The
model labels a variable, says a picture is missing, and writes the code that
draws it; the numbers come from executing the analysis, and the gates in
``numbers`` and ``claims`` still hold over everything that reaches prose. With
no model configured the graph runs the four deterministic stages and produces a
smaller report rather than none.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from axiom.core import Unsupported

from axiom_dossier.agent.notebooks import (
    Notebook,
    Passage,
    describe_figure,
    execute,
    harvest,
    read_series,
)
from axiom_dossier.agent.state import DossierState, Gap, Harvested, Run
from axiom_dossier.agent.tools import describe_namespace, run_python
from axiom_dossier.evidence import EvidenceBuilder, quantity_from
from axiom_dossier.language import LIGHT_MODEL, PROSE_MODEL, Gemini, LanguageModel

__all__ = ["MAX_REPAIRS", "build_graph", "run_pipeline"]

MAX_REPAIRS = 2
"""How many times the graph may go back and try to close a gap it left open."""

_CODE_FENCE = re.compile(r"```(?:python)?\s*(.*?)```", re.S)


def _code_of(text: str) -> str:
    """The first fenced block, or the whole reply when a model forgot the fence."""
    found = _CODE_FENCE.search(text)
    return (found.group(1) if found else text).strip()


# -- stages ------------------------------------------------------------------------------


def stage_read(state: DossierState) -> DossierState:
    """Parse the notebook series and whatever planning notes were pointed at."""
    run = state["run"]
    run.say("read", f"{len(run.notebooks)} notebooks, {len(run.notes)} notes")
    for book in run.notebooks:
        run.say("read", f"  {book.name}: {len(book.passages)} passages")
    return state


def stage_run(state: DossierState) -> DossierState:
    """Execute the series in one namespace, harvesting after each notebook.

    Harvesting per notebook rather than once at the end is not a detail: the six
    HYPER-3 notebooks all bind ``fig`` and several bind ``estimate``, so a single
    harvest at the end returns the last one and silently loses the other five.
    """
    run = state["run"]
    namespace = state["namespace"]
    executions = []
    harvested: list[Harvested] = []
    for book in run.notebooks:
        result = execute(book, namespace)
        executions.append(result)
        run.say("run", result.summary())
        quantities, figures = harvest(namespace)
        stem = book.name.split("-")[0]
        for key, value in quantities.items():
            harvested.append(
                Harvested(
                    key=f"{stem}_{key}",
                    notebook=book.name,
                    kind=type(value).__name__,
                    value=value,
                    passage=_passage_for(book, key),
                )
            )
        for key, value in figures.items():
            run.figures[f"{stem}_{key}"] = value
        # the next notebook rebinds these names; take them out so the following
        # harvest reports what that notebook produced rather than this one's
        for key in (*quantities, *figures):
            namespace.pop(key, None)
    run.executions = tuple(executions)
    run.harvested = tuple(harvested)
    run.say("run", f"harvested {len(harvested)} quantities and {len(run.figures)} figures")
    return state


def _passage_for(book: Notebook, name: str) -> Passage | None:
    for passage in book.passages:
        if name in passage.defines:
            return passage
    return None


def stage_label(state: DossierState) -> DossierState:
    """Ask the model what each harvested object should be called in a report.

    It is shown the variable name, its type, and the prose that introduced it —
    never its value. A model given the number would put the number in the label,
    and the gates would then be checking prose against a model's own arithmetic.
    """
    run = state["run"]
    model: LanguageModel | None = state.get("model")
    if model is None:
        run.say("label", "no model; keeping variable names as labels")
        run.harvested = tuple(h if h.label else _fallback_label(h) for h in run.harvested)
        return state

    labelled: list[Harvested] = []
    for item in run.harvested:
        reply = model.generate(
            LABEL_PROMPT.format(context=item.context()),
            system=LABEL_SYSTEM,
            temperature=0.0,
        )
        if isinstance(reply, Unsupported):
            run.say("label", f"{item.key}: {reply.reason}")
            labelled.append(_fallback_label(item))
            continue
        fields = _parse_label(reply)
        labelled.append(
            Harvested(
                key=item.key,
                notebook=item.notebook,
                kind=item.kind,
                value=item.value,
                passage=item.passage,
                label=fields.get("label") or _fallback_label(item).label,
                unit=fields.get("unit", ""),
                threshold=_as_float(fields.get("threshold")),
                beneficial=fields.get("beneficial", "either"),
                include=fields.get("include", "yes").lower().startswith("y"),
            )
        )
    run.harvested = tuple(labelled)
    kept = sum(1 for h in run.harvested if h.include)
    run.say("label", f"labelled {len(labelled)}; {kept} judged reportable")
    return state


def _fallback_label(item: Harvested) -> Harvested:
    heading = item.passage.heading() if item.passage else ""
    pretty = item.key.split("_", 1)[-1].replace("_", " ")
    return Harvested(
        key=item.key,
        notebook=item.notebook,
        kind=item.kind,
        value=item.value,
        passage=item.passage,
        label=f"{pretty} ({heading})" if heading else pretty,
    )


def _parse_label(reply: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in reply.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        out[key.strip().lower()] = value.strip()
    return out


def _as_float(text: str | None) -> float | None:
    if not text or text.lower() in ("none", "-", "n/a", ""):
        return None
    try:
        return float(text)
    except ValueError:
        return None


def stage_gaps(state: DossierState) -> DossierState:
    """Find what the report should carry and does not.

    Deliberately conservative about what counts. A gap is a *picture* that the
    prose describes and the run did not draw; it is never "add a number", because
    a number that is not in the record is one nobody computed.
    """
    run = state["run"]
    model: LanguageModel | None = state.get("model")
    if model is None:
        run.say("gaps", "no model; skipping gap analysis")
        return state

    prose = "\n\n".join(f"### {b.name}\n{b.prose()[:1800]}" for b in run.notebooks)
    reply = model.generate(
        GAPS_PROMPT.format(
            prose=prose[:12000],
            figures="\n".join(
                f"{key}: {describe_figure(fig)}" for key, fig in sorted(run.figures.items())
            )
            or "(none)",
            quantities="\n".join(f"{h.key}: {h.label}" for h in run.harvested if h.include),
        ),
        system=GAPS_SYSTEM,
        temperature=0.1,
    )
    if isinstance(reply, Unsupported):
        # An empty reply is how a model says "nothing is missing" when it has
        # been told to reply with nothing, which is indistinguishable from a
        # failure. The prompt asks for NONE instead, and this treats a silent
        # stop as the same answer rather than as an error.
        if "no text" in reply.reason:
            run.say("gaps", "nothing reported as missing")
            return state
        run.say("gaps", reply.reason)
        return state
    if reply.strip().upper().startswith("NONE"):
        run.say("gaps", "the drawn figures already cover the claims")
        return state
    gaps = []
    for line in reply.splitlines():
        line = line.strip().lstrip("-* ").strip()
        if not line or "|" not in line:
            continue
        what, _, why = line.partition("|")
        gaps.append(Gap(kind="figure", what=what.strip(), why=why.strip()))
    run.gaps = tuple(gaps[:4])
    run.say("gaps", f"{len(run.gaps)} figure(s) the prose describes and the run did not draw")
    return state


def stage_fill(state: DossierState) -> DossierState:
    """Write and run code that closes each open gap, once, with one repair."""
    run = state["run"]
    model: LanguageModel | None = state.get("model")
    if model is None or not run.gaps:
        return state
    if not state.get("allow_execution"):
        run.say("fill", "execution not enabled; gaps left open")
        return state

    namespace = state["namespace"]
    names = describe_namespace(namespace)
    closed: list[Gap] = []
    for gap in run.gaps:
        if gap.resolved:
            closed.append(gap)
            continue
        target = f"gap_figure_{len(closed)}"
        attempt = model.generate(
            FILL_PROMPT.format(what=gap.what, why=gap.why, names=names, target=target),
            system=FILL_SYSTEM,
            temperature=0.1,
        )
        if isinstance(attempt, Unsupported):
            closed.append(Gap(**{**gap.__dict__, "error": attempt.reason}))
            continue
        code = _code_of(attempt)
        outcome = run_python(code, namespace, expect=target, allow_execution=True)
        if not outcome.ok:
            repair = model.generate(
                REPAIR_PROMPT.format(code=code, error=outcome.feedback(), target=target),
                system=FILL_SYSTEM,
                temperature=0.0,
            )
            if not isinstance(repair, Unsupported):
                code = _code_of(repair)
                outcome = run_python(code, namespace, expect=target, allow_execution=True)
        if outcome.ok:
            run.figures[target] = namespace[target]
            run.say("fill", f"drew {gap.what[:60]}")
            closed.append(Gap(**{**gap.__dict__, "code": code, "resolved": True}))
        else:
            run.say("fill", f"could not draw {gap.what[:50]}: {outcome.feedback()[:80]}")
            closed.append(Gap(**{**gap.__dict__, "code": code, "error": outcome.feedback()}))
    run.gaps = tuple(closed)
    return state


def stage_assemble(state: DossierState) -> DossierState:
    """Turn everything harvested into one ``Evidence``."""
    run = state["run"]
    namespace = state["namespace"]
    builder = EvidenceBuilder(
        state.get("title", "Case study"),
        state.get("question", ""),
    )
    verdict = next(
        (v for v in namespace.values() if type(v).__name__ == "IdentificationVerdict"),
        None,
    )
    if verdict is not None:
        builder.verdict(verdict)

    for book in run.notebooks:
        headings = [p.heading() for p in book.passages if p.heading()]
        builder.step(
            book.name.replace(".ipynb", ""),
            _title_of(book),
            what=" ".join(book.prose().split())[:600],
            why="",
            detail={"cells": str(len(book.code_cells)), "sections": str(len(headings))},
        )

    kept = 0
    for item in run.harvested:
        if not item.include:
            continue
        try:
            quantity = quantity_from(
                item.key,
                item.value,
                label=item.label or item.key,
                unit=item.unit,
                threshold=item.threshold,
                beneficial=item.beneficial,  # type: ignore[arg-type]
                source=f"{item.notebook} · {item.kind}",
            )
        except TypeError as exc:
            run.say("assemble", f"{item.key} is not reportable: {exc}")
            continue
        builder._findings.append(quantity)  # noqa: SLF001 - the builder's own list
        kept += 1

    for name, text in run.notes.items():
        builder.remark(f"From {name}: {' '.join(text.split())[:600]}")
    builder.provenance(
        notebooks=str(len(run.notebooks)),
        cells=str(sum(len(b.code_cells) for b in run.notebooks)),
        figures=str(len(run.figures)),
        gaps_closed=str(sum(1 for g in run.gaps if g.resolved)),
    )
    run.evidence = builder.build()
    run.say("assemble", f"{kept} findings, {len(run.figures)} figures")
    return state


def _title_of(book: Notebook) -> str:
    for passage in book.passages:
        if passage.heading():
            return passage.heading()
    return book.name.replace(".ipynb", "").replace("-", " ")


def stage_verify(state: DossierState) -> DossierState:
    """Check the run rather than the prose: did everything execute, and is
    anything still open that another pass could close?"""
    run = state["run"]
    if not run.complete():
        failed = [e.summary() for e in run.executions if not e.complete]
        run.say("verify", f"{len(failed)} notebook(s) did not finish: {failed[:2]}")
    open_gaps = [g for g in run.gaps if not g.resolved and not g.error]
    run.say("verify", f"{len(open_gaps)} gap(s) still open")
    state["iterations"] = state.get("iterations", 0) + 1
    return state


def _should_retry(state: DossierState) -> str:
    run = state["run"]
    open_gaps = [g for g in run.gaps if not g.resolved and not g.error]
    if open_gaps and state.get("iterations", 0) < MAX_REPAIRS:
        return "fill"
    return "done"


# -- prompts -----------------------------------------------------------------------------

LABEL_SYSTEM = """You name things for a statistical report. You never compute.

You are shown a variable, its type, and the prose that introduced it. Reply with
exactly these lines and nothing else:

label: <what a reader should see, a short noun phrase>
unit: <the unit, or empty>
threshold: <the value a decision turns on, or none>
beneficial: <lower | higher | either>
include: <yes | no>

Say include: no for a variable that is scaffolding rather than a finding — an
intermediate, a check, a thing the prose does not report."""

LABEL_PROMPT = "{context}"

GAPS_SYSTEM = """You decide which figures a report of this analysis still needs.

You are given the prose of the analysis, the figures its notebooks already drew,
and the quantities it recorded. The notebooks were written to be *read in
order*; the report is read on its own. So a figure can be missing from the
report even when the analysis plainly computed the thing it would show.

List figures that would carry a claim the prose makes, and that the drawn list
does not already cover. One per line, exactly:

    what to draw | why the report needs it

Rules. At most four. Only pictures of quantities the analysis already computed --
never a new statistic, a new claim, or a comparison nobody ran. Prefer one
figure that carries a headline finding over three that decorate. If the drawn
figures already cover everything, reply with the single word NONE."""

GAPS_PROMPT = """PROSE:
{prose}

FIGURES ALREADY DRAWN:
{figures}

QUANTITIES RECORDED:
{quantities}"""

FILL_SYSTEM = """You write one short Python cell that draws a figure.

It runs in a namespace where the analysis has already been executed. Use the
names you are given; do not re-fit, re-simulate or invent data. Use plotly
(`import plotly.graph_objects as go`) or an axiom helper.

Bind the finished figure to the requested name. Give the axes titles with units.
Do NOT set a figure title -- the caption carries it.

Reply with one fenced python block and nothing else."""

FILL_PROMPT = """Draw: {what}
Why it is needed: {why}

Bind the figure to `{target}`.

Names available:
{names}"""

REPAIR_PROMPT = """This cell failed:

```python
{code}
```

{error}

Return a corrected cell that binds `{target}`. One fenced python block."""


# -- assembly ----------------------------------------------------------------------------


def build_graph() -> Any:
    """The compiled LangGraph. Importing langgraph is deferred to here."""
    from langgraph.graph import END, StateGraph

    graph = StateGraph(DossierState)
    graph.add_node("read", stage_read)
    graph.add_node("run", stage_run)
    graph.add_node("label", stage_label)
    graph.add_node("gaps", stage_gaps)
    graph.add_node("fill", stage_fill)
    graph.add_node("assemble", stage_assemble)
    graph.add_node("verify", stage_verify)

    graph.set_entry_point("read")
    graph.add_edge("read", "run")
    graph.add_edge("run", "label")
    graph.add_edge("label", "gaps")
    graph.add_edge("gaps", "fill")
    graph.add_edge("fill", "assemble")
    graph.add_edge("assemble", "verify")
    graph.add_conditional_edges("verify", _should_retry, {"fill": "fill", "done": END})
    return graph.compile()


def run_pipeline(
    notebooks: Path | str,
    *,
    title: str,
    question: str = "",
    notes: dict[str, str] | None = None,
    model: LanguageModel | None = None,
    allow_execution: bool = False,
    pattern: str = "*.ipynb",
) -> Run:
    """Read a notebook series, run it, and come back with everything it said.

    ``model`` absent runs the four deterministic stages. ``allow_execution``
    must be passed for generated code to run at all; without it gaps are found
    and reported but not closed.
    """
    run = Run(notebooks=read_series(notebooks, pattern), notes=dict(notes or {}))
    state: DossierState = {
        "run": run,
        "namespace": {},
        "iterations": 0,
    }
    state["model"] = model
    state["allow_execution"] = allow_execution
    state["title"] = title
    state["question"] = question
    build_graph().invoke(state, {"recursion_limit": 50})
    return run


def default_models() -> tuple[LanguageModel, LanguageModel]:
    """The pair this pipeline is tuned for: prose model, and the cheap one."""
    return Gemini(PROSE_MODEL), Gemini(LIGHT_MODEL)
