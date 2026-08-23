"""What the graph carries between stages.

Kept as one flat, inspectable structure rather than hidden inside the nodes,
because the point of running this as a graph is that a person can look at what
each stage did. ``Run.transcript`` is the record of that, and every stage
appends to it whether it succeeded or not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypedDict

from axiom_dossier.agent.notebooks import Execution, Notebook, Passage
from axiom_dossier.evidence import Evidence

__all__ = ["Gap", "Harvested", "Run", "DossierState"]


@dataclass(frozen=True)
class Harvested:
    """One object a notebook left behind, with the prose that introduced it.

    ``label`` starts empty. A variable is called ``itt`` or ``band``, which is
    not a thing to put in a report, and the passage is what a labelling stage
    reads to do better.
    """

    key: str
    notebook: str
    kind: str
    value: Any = field(repr=False, default=None)
    passage: Passage | None = field(repr=False, default=None)
    label: str = ""
    unit: str = ""
    threshold: float | None = None
    beneficial: str = "either"
    include: bool = True

    def context(self) -> str:
        """What a model is shown about this object: never the number alone."""
        prose = " ".join(self.passage.text.split())[:400] if self.passage else ""
        heading = self.passage.heading() if self.passage else ""
        return (
            f"variable `{self.key}` ({self.kind}) from {self.notebook}\n"
            f"under heading: {heading or '(none)'}\n"
            f"introduced by: {prose or '(no prose)'}"
        )


@dataclass(frozen=True)
class Gap:
    """Something the report should carry and does not yet.

    ``kind`` is ``figure`` when an exhibit is missing and ``quantity`` when a
    claim in the prose has no number behind it. ``code`` is filled in by the
    stage that closes the gap, so a reader can see what was run.
    """

    kind: str
    what: str
    why: str
    section: str = ""
    code: str = ""
    resolved: bool = False
    error: str = ""


@dataclass
class Run:
    """Everything one pass of the graph produced. Inspect this, not the nodes."""

    notebooks: tuple[Notebook, ...] = ()
    executions: tuple[Execution, ...] = ()
    harvested: tuple[Harvested, ...] = ()
    figures: dict[str, Any] = field(default_factory=dict, repr=False)
    notes: dict[str, str] = field(default_factory=dict, repr=False)
    gaps: tuple[Gap, ...] = ()
    evidence: Evidence | None = None
    transcript: list[str] = field(default_factory=list)

    def say(self, stage: str, message: str) -> None:
        self.transcript.append(f"[{stage}] {message}")

    def complete(self) -> bool:
        """Whether every notebook ran to its last cell."""
        return bool(self.executions) and all(e.complete for e in self.executions)

    def summary(self) -> str:
        ran = sum(1 for e in self.executions if e.complete)
        closed = sum(1 for g in self.gaps if g.resolved)
        return (
            f"{ran}/{len(self.executions)} notebooks ran, "
            f"{len(self.harvested)} quantities, {len(self.figures)} figures, "
            f"{closed}/{len(self.gaps)} gaps closed"
        )


class DossierState(TypedDict, total=False):
    """The LangGraph channel. One key, because the stages share one object.

    LangGraph wants a mapping; the work wants a mutable record with a
    transcript. Carrying ``Run`` under a single key keeps both without
    scattering the state across a dozen reducers that would each need their own
    merge rule.
    """

    run: Run
    namespace: dict[str, Any]
    iterations: int
    # Declared rather than passed as extras: LangGraph filters the state to the
    # keys this class names, so an undeclared key is silently dropped and every
    # stage that reads it behaves as though it were never configured. That is
    # exactly how the model stages came to report "no model" while a model was
    # sitting in the call.
    model: Any
    allow_execution: bool
    title: str
    question: str
