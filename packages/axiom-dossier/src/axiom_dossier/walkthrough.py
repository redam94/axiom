"""Turning an axiom example's own record into a report.

The twelve scripts under ``examples/`` narrate themselves as they run: each
``step`` carries why it exists and what a reasonable person would have reached
for instead, each table is recorded as rows rather than as printed text, each
chart is recorded as the numbers behind it, and the closing paragraphs say what
the run actually showed. Setting ``AXIOM_WALKTHROUGH_JSON`` makes the script
write that record out.

It is already almost an ``Evidence``. This module completes the mapping, so an
example that was written to be *read* becomes an example that can also be
*published* — the same run, as a paper.

    record = json.loads(Path("out/07.json").read_text())
    exhibits = exhibits_from_record(record)
    built = build(evidence_from_record(record), style="journal", extra_exhibits=exhibits)
    built.write("07.pdf")

Four kinds of thing come across, and they land in four different places,
because a report that flattens them into one is the report that produced
``readout: design : central_composite, 12 plots; detail : {'alpha': 1.41…}`` as
a row of its design table:

* the narrative — ``say`` blocks — becomes the step's prose;
* the rejected alternative becomes ``MethodStep.instead``, the most useful
  sentence in a methods section and the one most often missing;
* the printed lines become ``MethodStep.readout``, shown as they were seen;
* the tables and charts become ``Exhibit`` references, with the data itself in
  the render context beside the evidence.

One honest limit remains, and it is visible in the output. The examples record
their findings as **prose** — "2SLS bought accuracy and cost width" — not as
quantities. So a report built this way carries the narrative, the reasoning, the
rejected alternatives, every table and every chart, but has no metric blocks and
no threshold reading: there is nothing structured to read one from. Lifting the
numbers out of the prose would be guessing, and the results section says it
found none rather than inventing them. An example that records
``quantities=[...]`` gets those sections for free; nothing here needs to change
for that to work.
"""

from __future__ import annotations

from typing import Any

from axiom.core import Unsupported
from axiom.report import Theme

from axiom_dossier.charts import chart_figure
from axiom_dossier.evidence import Evidence, EvidenceBuilder, Exhibit

__all__ = [
    "evidence_from_record",
    "exhibits_from_record",
    "figures_from_record",
    "tables_from_record",
]

#: How many printed lines of one step a report carries before it says how many
#: it left. A readout is evidence of what the run showed, not a transcript.
READOUT_LINES = 12


def _prose_of(step: dict[str, Any]) -> str:
    """The narrative blocks of a step, joined — what the step said it was doing."""
    return " ".join(block["text"] for block in step.get("blocks", ()) if block.get("type") == "say")


def _readout_of(step: dict[str, Any]) -> tuple[str, ...]:
    """The printed lines of a step, verbatim, trimmed to a readable number.

    Kept as lines rather than joined with semicolons: column-aligned output is
    aligned in columns, and running it together is how ``nitrogen   : 0 to 200``
    stopped being readable.
    """
    lines: list[str] = []
    for block in step.get("blocks", ()):
        if block.get("type") == "out":
            lines.extend(line.rstrip() for line in block.get("lines", ()) if line.strip())
    if len(lines) > READOUT_LINES:
        dropped = len(lines) - READOUT_LINES
        return (*lines[:READOUT_LINES], f"… {dropped} further line(s) not shown")
    return tuple(lines)


def _blocks_of(record: dict[str, Any], kind: str) -> list[tuple[int, dict[str, Any]]]:
    """Every block of one kind, paired with the index of the step that made it."""
    out: list[tuple[int, dict[str, Any]]] = []
    for i, step in enumerate(record.get("steps", ())):
        for block in step.get("blocks", ()):
            if block.get("type") == kind:
                out.append((i, block))
    return out


def tables_from_record(record: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    """Every recorded table, keyed ``table_1``, ``table_2``, … in document order.

    Returned separately from the evidence because a table is data and an
    ``Evidence`` is a record of claims; these go into the render context beside
    it, which is the same separation the rest of the package keeps.
    """
    out: dict[str, list[dict[str, str]]] = {}
    for n, (_, block) in enumerate(_blocks_of(record, "table"), start=1):
        columns = [str(c) for c in block.get("columns", ())]
        out[f"table_{n}"] = [
            dict(zip(columns, (str(c) for c in row), strict=False)) for row in block.get("rows", ())
        ]
    return out


def figures_from_record(record: dict[str, Any], *, theme: Theme | None = None) -> dict[str, object]:
    """Every recorded chart as a plotly figure, keyed ``figure_1``, … in order.

    A chart whose kind is unknown or whose payload is incomplete is left out
    rather than drawn wrong, and the step that recorded it simply has one fewer
    exhibit — ``evidence_from_record`` names them all, and ``build`` keeps only
    the ones with data behind them.
    """
    payloads = record.get("figures", {})
    out: dict[str, object] = {}
    for n, (_, block) in enumerate(_blocks_of(record, "figure"), start=1):
        drawn = chart_figure(
            str(block.get("kind", "")),
            payloads.get(block.get("name")),
            dict(block.get("opt", {})),
            theme=theme,
        )
        if not isinstance(drawn, Unsupported):
            out[f"figure_{n}"] = drawn
    return out


def exhibits_from_record(
    record: dict[str, Any], *, theme: Theme | None = None
) -> dict[str, object]:
    """The record's tables and charts together, ready for ``build(extra_exhibits=…)``."""
    return {**tables_from_record(record), **figures_from_record(record, theme=theme)}


def _exhibits_by_step(record: dict[str, Any]) -> dict[int, list[Exhibit]]:
    """Which exhibits each step produced, under the keys the context will hold."""
    out: dict[int, list[Exhibit]] = {}
    for kind in ("table", "figure"):
        for n, (step, block) in enumerate(_blocks_of(record, kind), start=1):
            caption = str(block.get("caption") if kind == "table" else block.get("title") or "")
            note = str(block.get("note", "") if kind == "figure" else "")
            # Two sentences, so they read as two: the recorded title says what
            # the picture is and the note says what to look at in it.
            if caption and note:
                caption = caption if caption[-1] in ".?!" else f"{caption}."
            text = " ".join(part for part in (caption, note) if part)
            out.setdefault(step, []).append(
                Exhibit(
                    key=f"{kind}_{n}",
                    kind="table" if kind == "table" else "figure",
                    caption=" ".join(text.split()),
                )
            )
    return out


def evidence_from_record(record: dict[str, Any], *, title: str = "") -> Evidence:
    """An ``Evidence`` from one walkthrough record.

    ``step.why`` becomes the step's reason and ``step.instead`` the alternative
    it was chosen over, because the alternative an example rejected is the most
    useful thing in it and dropping it would leave the methods section
    describing a choice with no alternative.
    """
    builder = EvidenceBuilder(
        title or record.get("title") or "Untitled example",
        " ".join(str(record.get("question", "")).split()),
    )
    exhibits = _exhibits_by_step(record)
    for i, step in enumerate(record.get("steps", ())):
        builder.step(
            f"step_{step.get('n', i + 1)}",
            str(step.get("title", "Step")),
            what=_prose_of(step),
            why=str(step.get("why", "")),
            instead=str(step.get("instead", "") or ""),
            readout=_readout_of(step),
            exhibits=exhibits.get(i, ()),
        )
    builder.remark(*record.get("findings", ()))
    builder.provenance(
        field=str(record.get("field", "")),
        source="examples/ walkthrough record",
        steps=str(len(record.get("steps", ()))),
    )
    return builder.build()
