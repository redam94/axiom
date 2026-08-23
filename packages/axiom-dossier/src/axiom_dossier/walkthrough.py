"""Turning an axiom example's own record into a report.

The twelve scripts under ``examples/`` narrate themselves as they run: each
``step`` carries why it exists and what a reasonable person would have reached
for instead, each table is recorded as rows rather than as printed text, and the
closing paragraphs say what the run actually showed. Setting
``AXIOM_WALKTHROUGH_JSON`` makes the script write that record out.

It is already almost an ``Evidence``. This module completes the mapping, so an
example that was written to be *read* becomes an example that can also be
*published* — the same run, as a paper.

    record = json.loads(Path("out/07.json").read_text())
    built = build(evidence_from_record(record), style="journal")
    built.write("07.pdf")

One honest limit, and it is visible in the output. The examples record their
findings as **prose**, because that is what a terminal walkthrough needs: a
sentence like "2SLS bought accuracy and cost width". They do not record the
numbers behind those sentences as quantities. So a report built this way carries
the narrative, the reasoning, the rejected alternatives and every table, but has
no metric blocks and no threshold reading — there is nothing structured to read.
An example that adds `quantities=[...]` to its record gets those sections for
free; nothing here needs to change for that to work.
"""

from __future__ import annotations

from typing import Any

from axiom_dossier.evidence import Evidence, EvidenceBuilder
from axiom_dossier.sections import literal

__all__ = ["evidence_from_record", "tables_from_record"]


def _prose_of(step: dict[str, Any]) -> str:
    """The narrative blocks of a step, joined — what the step said it was doing."""
    return literal(
        " ".join(block["text"] for block in step.get("blocks", ()) if block.get("type") == "say")
    )


def _readout_of(step: dict[str, Any]) -> str:
    """The printed lines of a step, trimmed to something a detail line can hold."""
    lines: list[str] = []
    for block in step.get("blocks", ()):
        if block.get("type") == "out":
            lines.extend(line for line in block.get("lines", ()) if line.strip())
    return literal("; ".join(lines[:4]))


def tables_from_record(record: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    """Every recorded table, keyed ``table_1``, ``table_2``, … in document order.

    Returned separately from the evidence because a table is data and an
    ``Evidence`` is a record of claims; these go into the render context beside
    it, which is the same separation the rest of the package keeps.
    """
    out: dict[str, list[dict[str, str]]] = {}
    n = 0
    for step in record.get("steps", ()):
        for block in step.get("blocks", ()):
            if block.get("type") != "table":
                continue
            n += 1
            columns = [str(c) for c in block.get("columns", ())]
            out[f"table_{n}"] = [
                dict(zip(columns, (str(c) for c in row), strict=False))
                for row in block.get("rows", ())
            ]
    return out


def evidence_from_record(record: dict[str, Any], *, title: str = "") -> Evidence:
    """An ``Evidence`` from one walkthrough record.

    ``step.why`` becomes the step's reason and ``step.instead`` is kept as a
    detail, because the alternative an example rejected is the most useful thing
    in it and dropping it would leave the methods section describing a choice
    with no alternative.
    """
    builder = EvidenceBuilder(
        title or record.get("title") or "Untitled example",
        literal(" ".join(str(record.get("question", "")).split())),
    )
    for step in record.get("steps", ()):
        detail: dict[str, str] = {}
        if step.get("instead"):
            detail["considered instead"] = literal(str(step["instead"]))
        readout = _readout_of(step)
        if readout:
            detail["readout"] = readout
        builder.step(
            f"step_{step.get('n', len(detail) + 1)}",
            literal(str(step.get("title", "Step"))),
            what=_prose_of(step),
            why=literal(str(step.get("why", ""))),
            detail=detail,
        )
    builder.remark(*record.get("findings", ()))
    builder.provenance(
        field=str(record.get("field", "")),
        source="examples/ walkthrough record",
        steps=str(len(record.get("steps", ()))),
    )
    return builder.build()
