"""The data tables a report carries, built from the record rather than typed.

A table is the part of a report a reader checks numbers against, so every one
here is derived: ``findings_rows`` cannot disagree with the metric blocks
because both read the same ``Quantity``, and ``design_rows`` cannot describe a
protocol nobody ran because it is the ``detail`` the method steps recorded.

Three tables, matching the three questions a reader asks of a study:

* **Design** — what was done, as parameters. The experimental-design half of a
  report is a table of settings, and prose is the wrong shape for it.
* **Findings** — every estimate with its interval, its threshold, and where it
  landed relative to that threshold.
* **Diagnostics** — what was checked about the analysis rather than the world.

APA asks that a table's number and title sit above it and any note below; that
placement is the renderer's business, so what these return is rows.
"""

from __future__ import annotations

from axiom_dossier.evidence import Evidence, GraphRecord, Quantity

__all__ = [
    "TABLE_SOURCES",
    "design_rows",
    "graph_rows",
    "graph_table_rows",
    "diagnostics_rows",
    "findings_rows",
    "tables_for",
]

#: How the threshold comparison reads in a table cell. ``spans`` deliberately
#: does not say "no effect" — see ``interpret.reading_of``.
_WHERE = {
    "below": "below threshold",
    "above": "above threshold",
    "spans": "spans threshold",
    "no threshold": "—",
    "no interval": "no interval",
}


def _interval_cell(q: Quantity) -> str:
    if q.interval is None:
        return "—"
    mass = f"{q.interval.mass:.0%}" if q.interval.mass is not None else "?"
    return (
        f"{q.interval.lower:.{q.precision}f} to "
        f"{q.interval.upper:.{q.precision}f} ({mass} {q.interval.definition.upper()})"
    )


def findings_rows(evidence: Evidence) -> list[dict[str, str]]:
    """One row per finding: the estimate, its interval, and where it landed."""
    return [
        {
            "Quantity": q.label,
            "Estimate": f"{q.value:.{q.precision}f}",
            "Unit": q.unit or "—",
            "Interval": _interval_cell(q),
            "Threshold": ("—" if q.threshold is None else f"{q.threshold:.{q.precision}f}"),
            "Relative to threshold": _WHERE.get(q.against_threshold(), q.against_threshold()),
        }
        for q in evidence.findings
    ]


def diagnostics_rows(evidence: Evidence) -> list[dict[str, str]]:
    """One row per diagnostic. These describe the fit, never the effect."""
    return [
        {
            "Check": q.label,
            "Value": f"{q.value:.{q.precision}f}",
            "Unit": q.unit or "—",
            "Interval": _interval_cell(q),
        }
        for q in evidence.diagnostics
    ]


def design_rows(evidence: Evidence) -> list[dict[str, str]]:
    """The experimental design as parameters, gathered from every method step.

    A step's ``detail`` is where a protocol's settings live — arms, allocation,
    strata, windows, the estimator. Collected into one table they are the design
    section a reader can check a replication against, which is what prose alone
    never gives them.
    """
    return [
        {"Step": step.title, "Parameter": key, "Value": value}
        for step in evidence.steps
        for key, value in sorted(step.detail.items())
    ]


#: Context key -> the function that fills it. A key is present only when its
#: table has rows, so a section naming one can rely on it having content.
def graph_rows(graph: GraphRecord) -> list[dict[str, str]]:
    """The identification graph as a table of arrows — the checkable form.

    A picture of a DAG needs plotly and a reader willing to squint. The edge
    list needs neither, and it is the form in which someone can disagree with a
    specific arrow rather than with the conclusion drawn from all of them.
    """
    rows = [{"From": a, "Arrow": "→", "To": b, "Kind": "direct effect"} for a, b in graph.edges]
    rows.extend(
        {"From": a, "Arrow": "↔", "To": b, "Kind": "unmeasured common cause"}
        for a, b in graph.bidirected
    )
    for node in graph.unmeasured:
        rows.append({"From": node, "Arrow": "", "To": "", "Kind": "unmeasured node"})
    return rows


def graph_table_rows(evidence: Evidence) -> list[dict[str, str]]:
    """The identification graph as arrows, or nothing when no graph was recorded."""
    if evidence.graph is None:
        return []
    return graph_rows(evidence.graph)


TABLE_SOURCES = {
    "graph_table": graph_table_rows,
    "design_table": design_rows,
    "findings_table": findings_rows,
    "diagnostics_table": diagnostics_rows,
}


def tables_for(evidence: Evidence) -> dict[str, list[dict[str, str]]]:
    """Every non-empty table this evidence supports, keyed for the context."""
    out: dict[str, list[dict[str, str]]] = {}
    for key, build in TABLE_SOURCES.items():
        rows = build(evidence)
        if rows:
            out[key] = rows
    return out
