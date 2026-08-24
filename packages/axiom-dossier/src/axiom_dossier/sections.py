"""Methods, results, diagnostics and limitations, written from the evidence.

Every function here is deterministic and needs no network, no key and no extra.
That is the point of the split: the document a language model narrates is not
the document this package can produce, it is an *improvement* on it. If the
model is unavailable, unaffordable or wrong, these sections still render, still
carry every interval, and still say what was assumed.

What makes them worth generating rather than writing is that they are derived
from typed results. A methods section produced from ``Verdict`` and
``Assumption`` cannot describe an identification route the analysis did not
take, and a limitations section produced from ``Evidence.unresolved`` cannot
quietly omit the assumption that is doing the most work.

**Verbosity is a property of the draft, not only of the narration.** Asking a
model to "write more" about a three-line draft is asking it to pad. What a
longer report needs is more *content*, so ``brief`` / ``standard`` / ``full``
change what the generated draft includes — per-step detail, per-finding
headings, the ledger, the recaps — and the narration inherits the richer draft
along with a longer target.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from axiom.core import Assumption
from axiom.report import (
    Divider,
    Figure,
    Heading,
    LedgerBlock,
    Metric,
    Paragraph,
    Section,
    Table,
)

from axiom_dossier.evidence import Evidence

__all__ = [
    "VERBOSITY",
    "Verbosity",
    "assumption_rows",
    "equation_text",
    "literal",
    "plural",
    "readout_text",
    "sentence",
    "diagnostics_section",
    "limitations_section",
    "methods_section",
    "provenance_section",
    "remarks_section",
    "results_section",
    "standing_assumptions",
]

Verbosity = Literal["brief", "standard", "full"]
"""How much the generated draft says. Not a font size — it changes the content."""

#: What each level includes. ``sentences`` is the target handed to the narrator,
#: so a longer report is longer because it covers more, not because it repeats.
VERBOSITY: dict[str, dict[str, bool | int]] = {
    "brief": {
        "include_step_detail": False,
        "include_assumption_table": False,
        "include_ledger": False,
        "include_notes": False,
        "include_next_steps": False,
        "include_assumption_recap": False,
        "per_finding_heading": False,
        "per_assumption_paragraph": False,
        "sentences": 3,
    },
    "standard": {
        "include_step_detail": True,
        "include_assumption_table": True,
        "include_ledger": True,
        "include_notes": True,
        "include_next_steps": True,
        "include_assumption_recap": False,
        "per_finding_heading": False,
        "per_assumption_paragraph": True,
        "sentences": 6,
    },
    "full": {
        "include_step_detail": True,
        "include_assumption_table": True,
        "include_ledger": True,
        "include_notes": True,
        "include_next_steps": True,
        "include_assumption_recap": True,
        "per_finding_heading": True,
        "per_assumption_paragraph": True,
        "sentences": 12,
    },
}

#: How each ``core.Assumption`` state reads in a sentence. The four keys are the
#: whole of that Literal; a state outside them falls through to its own name.
_STATE_WORD = {
    "unverified": "is assumed and has not been checked",
    "satisfied": "was checked and holds",
    "violated": "is known to fail",
    "asserted": "was asserted without a check",
}


def _detail(verbosity: Verbosity) -> dict[str, bool | int]:
    if verbosity not in VERBOSITY:
        raise ValueError(f"unknown verbosity {verbosity!r}; have {sorted(VERBOSITY)}")
    return VERBOSITY[verbosity]


def sentence(text: str) -> str:
    """Free text made into a sentence: capitalised, and closed with a stop.

    A recorded ``reason`` or ``statement`` is written as a clause — "age blocks
    the only back-door path" — because that is how it reads inside the object
    that holds it. Every section that appends one after a full stop was printing
    it as a sentence starting in lower case, which reads as a typo in a document
    whose whole claim is that it was assembled carefully.
    """
    stripped = text.strip()
    if not stripped:
        return ""
    opened = stripped[0].upper() + stripped[1:]
    return opened if opened[-1] in ".?!" else opened + "."


def plural(count: int, noun: str, *, verb: str = "") -> str:
    """``2 assumptions``, ``1 assumption`` — and optionally the verb to match.

    A paper does not contain "assumption(s)". The bracketed plural is a template
    showing through, and in a document whose whole argument is that it was
    generated from a record rather than written by a person, it is the detail
    that makes a reader believe the rest was generated carelessly too.
    """
    word = noun if count == 1 else _plural_of(noun)
    if not verb:
        return f"{count} {word}"
    return f"{count} {word} {verb if count == 1 else _AGREES.get(verb, verb)}"


#: Third-person singular -> plural, for the handful of verbs these sections use.
_AGREES = {"is": "are", "was": "were", "carries": "carry", "holds": "hold"}


def _plural_of(noun: str) -> str:
    """English plural for the last word of a noun phrase. Enough English for this.

    "further quantity" pluralises on *quantity*, not on the phrase, and a naive
    ``+ "s"`` produced "4 further quantitys" in the abstract of the flagship
    sample.
    """
    head, _, last = noun.rpartition(" ")
    if last.endswith("y") and last[-2:-1] not in "aeiou":
        last = last[:-1] + "ies"
    elif last.endswith("is"):
        last = last[:-2] + "es"  # analysis -> analyses, basis -> bases
    elif last.endswith(("s", "x", "z", "ch", "sh")):
        last = last + "es"
    else:
        last = last + "s"
    return f"{head} {last}".strip()


def literal(text: str) -> str:
    """Escape braces so free prose cannot be read as a template placeholder.

    ``report.Paragraph`` treats ``{name}`` as a context key. That is right for
    text this package writes and wrong for text it merely carries: an analyst's
    sentence mentioning ``{}`` or a set literal would otherwise fail the render
    with a missing key. Three of the twelve axiom examples do exactly that.

    **An ``Evidence`` holds raw text and the escape happens where the paragraph
    is built.** The other order — escaping on the way in — is what put
    ``{{'alpha': 1.41…}}`` in a design table, because the same value is a table
    cell too, and a cell is not a template.
    """
    return text.replace("{", "{{").replace("}", "}}")


def readout_text(lines: Sequence[str]) -> str:
    """Printed output as one paragraph of monospaced lines, escaped and marked.

    ``report`` has no preformatted block, and adding one to carry six lines of
    terminal output would be the wrong place to put it: what a readout needs is
    a fixed pitch and its own line breaks, and the inline code mark gives both
    in every renderer that has a monospaced face.
    """
    return "\n".join("`" + literal(line).replace("`", "'") + "`" for line in lines)


def equation_text(lines: Sequence[str]) -> str:
    """Equations as their own monospaced block, one to a line.

    ``report`` has no math block and the three renderers have no LaTeX between
    them, so an equation is set the way the readouts are: fixed pitch, its own
    line breaks, escaped. That is worse than typeset mathematics and much better
    than the alternative the package had, which was to name the estimator and
    never write it down.
    """
    return readout_text(lines)


def assumption_rows(assumptions: tuple[Assumption, ...]) -> list[dict[str, str]]:
    """The standing-assumptions table, with each state written as a sentence."""
    return [
        {
            "Assumption": a.name.replace("_", " "),
            "Statement": a.statement,
            "Status": _STATE_WORD.get(a.state, a.state),
            "Challenged by": a.challenged_by or "—",
        }
        for a in assumptions
    ]


def _all_assumptions(evidence: Evidence) -> tuple[Assumption, ...]:
    pool = list(evidence.assumptions)
    if evidence.verdict is not None:
        pool.extend(evidence.verdict.assumptions)
    for step in evidence.steps:
        pool.extend(step.assumptions)
    seen: dict[str, Assumption] = {}
    for a in pool:
        seen.setdefault(a.name, a)
    return tuple(seen[name] for name in sorted(seen))


def standing_assumptions(evidence: Evidence) -> tuple[Assumption, ...]:
    """The unresolved assumptions, in a stable order — the methods table's data."""
    unresolved = set(evidence.unresolved())
    return tuple(a for a in _all_assumptions(evidence) if a.name in unresolved)


def methods_section(
    evidence: Evidence, *, title: str = "Methods", verbosity: Verbosity = "standard"
) -> Section:
    """What was done, in the order it was done, and what each step rests on.

    One paragraph per recorded step. A step contributes its assumptions to the
    table at the end, so the reader sees the whole standing set in one place
    rather than scattered through the prose.
    """
    detail = _detail(verbosity)
    blocks: list[object] = []
    if evidence.question:
        blocks.append(Paragraph(text=f"Question: {literal(evidence.question)}"))

    for step in evidence.steps:
        blocks.append(Heading(text=literal(step.title), level=3))
        sentences = [s for s in (step.what, step.why) if s]
        if sentences:
            blocks.append(Paragraph(text=literal(" ".join(sentences))))
        if step.instead and detail["include_step_detail"]:
            blocks.append(Paragraph(text=f"Considered instead: {literal(step.instead)}"))
        if step.detail and detail["include_step_detail"]:
            # Escaped here rather than on the way in: the same value is a table
            # cell in the design table, and a cell is not a template.
            line = "; ".join(f"{k}: {literal(v)}" for k, v in sorted(step.detail.items()))
            blocks.append(Paragraph(text=line, emphasis=True))
        # Equations before the readout: the reader wants the estimator written
        # down before they are shown what it printed.
        if step.equations:
            blocks.append(Paragraph(text=equation_text(step.equations)))
        if step.readout and detail["include_step_detail"]:
            blocks.append(Paragraph(text=readout_text(step.readout)))
        for exhibit in step.exhibits:
            if exhibit.kind == "figure":
                blocks.append(Figure(source=exhibit.key, caption=literal(exhibit.caption)))
            else:
                blocks.append(
                    Table(source=exhibit.key, caption=literal(exhibit.caption), max_rows=40)
                )

    # The graph itself, as the arrows it is made of. It goes with the methods
    # because that is where the identification is argued, and it is a table
    # rather than only a picture because a picture needs plotly and an arrow a
    # reader wants to argue with needs a row.
    if evidence.graph is not None and evidence.graph.edges:
        blocks.append(Heading(text="The identification graph", level=3))
        blocks.append(
            Paragraph(
                text=(
                    "Every arrow below is a claim about the world that the data does not "
                    "make on its own. The route named above is read off this graph, so a "
                    "reader who disagrees with the conclusion should find the arrow they "
                    "disagree with here."
                )
            )
        )
        blocks.append(Table(source="graph_table", caption="The identification graph, as arrows"))

    standing = standing_assumptions(evidence)
    if standing and detail["include_assumption_table"]:
        blocks.append(Heading(text="Assumptions this analysis rests on", level=3))
        blocks.append(
            Paragraph(
                text=(
                    "Each of the following licenses a step above. None is a property of "
                    "the data alone; each is a claim about the world that the data cannot "
                    "settle, listed here so a reader can disagree with it specifically."
                )
            )
        )
        blocks.append(Table(source="assumption_table", caption="Standing assumptions"))
    if evidence.ledger and detail["include_ledger"]:
        blocks.append(LedgerBlock(source="ledger", caption="Assumption ledger"))
    if not blocks:
        blocks.append(Paragraph(text="No method steps were recorded for this analysis."))
    return Section(title=title, blocks=tuple(blocks))  # type: ignore[arg-type]


def results_section(
    evidence: Evidence, *, title: str = "Results", verbosity: Verbosity = "standard"
) -> Section:
    """Every finding as a metric block, so no interval can be lost on the way out.

    The prose states the quantity through ``{key_stated}``, which the evidence
    resolves to the point *and* its interval. Writing the point alone is not
    available here, which is the reporting half of axiom's fourth rule.
    """
    detail = _detail(verbosity)
    blocks: list[object] = []
    if not evidence.findings:
        blocks.append(Paragraph(text="No findings were recorded."))
        return Section(title=title, blocks=tuple(blocks))  # type: ignore[arg-type]

    # No per-finding heading here even at full verbosity: a Metric block renders
    # its own label, and a heading above it prints the same words twice. The
    # discussion does use them, because there the label introduces prose.
    for q in evidence.findings:
        blocks.append(Metric(source=q.key, label=q.label, unit=q.unit, precision=q.precision))
    # A sentence for every finding, not only the lead. The metric blocks carry
    # the numbers for a reader scanning the page; the prose carries them for the
    # narrator, which sees the paragraphs and nothing else. A finding absent
    # from the draft is absent from the narrated results section -- which is how
    # four of five estimates came to be stated first in the discussion.
    stated = " ".join(f"{q.label} is {{{q.key}_stated}}." for q in evidence.findings)
    blocks.append(
        Paragraph(
            text=(
                f"{stated} Intervals are reported with the definition and mass "
                "they were computed at."
            )
        )
    )
    if detail["include_notes"]:
        for q in evidence.findings:
            if q.note:
                blocks.append(Paragraph(text=f"*{q.label}.* {q.note}", emphasis=True))
    return Section(title=title, blocks=tuple(blocks))  # type: ignore[arg-type]


def diagnostics_section(
    evidence: Evidence, *, title: str = "Model checking", verbosity: Verbosity = "standard"
) -> Section:
    """What was checked about the analysis itself, rather than about the world."""
    _detail(verbosity)
    if not evidence.diagnostics:
        return Section(
            title=title,
            blocks=(
                Paragraph(
                    text=(
                        "No diagnostics were recorded. An analysis with no model checking "
                        "is not therefore sound; it is unexamined."
                    )
                ),
            ),
        )
    blocks: list[object] = [
        Paragraph(
            text=(
                "These describe the fit rather than the effect. They can only ever "
                "reduce confidence in the numbers above, never establish them."
            )
        )
    ]
    for q in evidence.diagnostics:
        blocks.append(Metric(source=q.key, label=q.label, unit=q.unit, precision=q.precision))
    blocks.append(
        Paragraph(text=" ".join(f"{q.label} is {{{q.key}_stated}}." for q in evidence.diagnostics))
    )
    return Section(title=title, blocks=tuple(blocks))  # type: ignore[arg-type]


def limitations_section(
    evidence: Evidence, *, title: str = "Limitations", verbosity: Verbosity = "standard"
) -> Section:
    """The assumptions still standing, stated as limitations rather than buried.

    Generated from ``Evidence.unresolved``, so this section cannot be shorter
    than the truth: every assumption that is not ``satisfied`` appears.
    """
    detail = _detail(verbosity)
    standing = standing_assumptions(evidence)
    blocks: list[object] = []
    if evidence.verdict is not None and evidence.verdict.status != "identified":
        blocks.append(
            Paragraph(
                text=(
                    f"The effect is {evidence.verdict.status}, not identified. "
                    f"{sentence(evidence.verdict.reason)}"
                )
            )
        )
    if not standing:
        blocks.append(
            Paragraph(
                text=(
                    "No assumption in the record is left unresolved. That is a statement "
                    "about the record, not a guarantee about the world."
                )
            )
        )
        return Section(title=title, blocks=tuple(blocks))  # type: ignore[arg-type]

    blocks.append(
        Paragraph(
            text=(
                f"{plural(len(standing), 'assumption')} below do work that the data does "
                "not do. Each is stated with what would challenge it, so disagreement can "
                "be specific rather than general."
            )
        )
    )
    if detail["per_assumption_paragraph"]:
        for a in standing:
            state = _STATE_WORD.get(a.state, a.state)
            challenged = f" It would be challenged by {a.challenged_by}." if a.challenged_by else ""
            blocks.append(
                Paragraph(
                    text=(
                        f"{a.name.replace('_', ' ')}: {a.statement} " f"This {state}.{challenged}"
                    )
                )
            )
    else:
        blocks.append(Table(source="assumption_table", caption="Standing assumptions"))
    return Section(title=title, blocks=tuple(blocks))  # type: ignore[arg-type]


def remarks_section(
    evidence: Evidence, *, title: str = "What the run showed", verbosity: Verbosity = "standard"
) -> Section:
    """The analyst's own words about the run, carried verbatim.

    Every other section here is generated. This one is not, and that is its
    value: it is the part written after looking at the output, and the part that
    most often disagrees with the setup. It is never narrated — a model
    rewriting a person's conclusion and leaving it attributed to them is the one
    lie this package must not tell.
    """
    _detail(verbosity)
    if not evidence.remarks:
        return Section(
            title=title,
            blocks=(Paragraph(text="No closing remarks were recorded for this run."),),
        )
    blocks: list[object] = [
        Paragraph(
            text=(
                "Written by the analyst after seeing the output, and reproduced "
                "unchanged. Nothing in this section is generated."
            ),
            emphasis=True,
        )
    ]
    blocks.extend(Paragraph(text=literal(text)) for text in evidence.remarks)
    return Section(title=title, blocks=tuple(blocks))  # type: ignore[arg-type]


def provenance_section(
    evidence: Evidence, *, title: str = "Provenance", verbosity: Verbosity = "standard"
) -> Section:
    """Where every number came from, and what wrote the prose.

    A dossier that was narrated says so here, naming the model and whether the
    generated text passed both checks. A reader who distrusts a language model
    can then read the deterministic sections and ignore the rest, which is only
    possible because the distinction is recorded rather than blurred.
    """
    _detail(verbosity)
    blocks: list[object] = [
        Paragraph(
            text=(
                "Every number in this document resolves to a quantity in the evidence "
                "record below. Prose was checked against that record: any numeral not "
                "traceable to it, and any claim the record does not itself make, is "
                "reported rather than published."
            )
        ),
        Divider(),
    ]
    # A record with no quantities has no table to draw, and an empty one reads
    # as a rendering fault rather than as an absence.
    if evidence.quantities():
        blocks.append(Table(source="provenance_table", caption="Quantities and their sources"))
    if evidence.provenance:
        blocks.append(Table(source="run_table", caption="Run"))
    return Section(title=title, blocks=tuple(blocks))  # type: ignore[arg-type]
