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

from typing import Literal

from axiom.core import Assumption
from axiom.report import Divider, Heading, LedgerBlock, Metric, Paragraph, Section, Table

from axiom_dossier.evidence import Evidence

__all__ = [
    "VERBOSITY",
    "Verbosity",
    "assumption_rows",
    "diagnostics_section",
    "limitations_section",
    "methods_section",
    "provenance_section",
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
        blocks.append(Paragraph(text=f"**Question.** {evidence.question}"))

    for step in evidence.steps:
        blocks.append(Heading(text=step.title, level=3))
        sentences = [s for s in (step.what, step.why) if s]
        if sentences:
            blocks.append(Paragraph(text=" ".join(sentences)))
        if step.detail and detail["include_step_detail"]:
            line = "; ".join(f"{k}: {v}" for k, v in sorted(step.detail.items()))
            blocks.append(Paragraph(text=line, emphasis=True))

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
    lead = evidence.findings[0]
    blocks.append(
        Paragraph(
            text=(
                f"{lead.label} is {{{lead.key}_stated}}. "
                "Intervals are reported with the definition and mass they were computed at."
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
                    f"**The effect is {evidence.verdict.status}, not identified.** "
                    f"{evidence.verdict.reason}"
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
                f"{len(standing)} assumption(s) below are doing work that the data does "
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
                        f"**{a.name.replace('_', ' ')}** — {a.statement} "
                        f"This {state}.{challenged}"
                    )
                )
            )
    else:
        blocks.append(Table(source="assumption_table", caption="Standing assumptions"))
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
        Table(source="provenance_table", caption="Quantities and their sources"),
    ]
    if evidence.provenance:
        blocks.append(Table(source="run_table", caption="Run"))
    return Section(title=title, blocks=tuple(blocks))  # type: ignore[arg-type]
