"""The record a report is written from — and the only place its numbers may come from.

A report is prose wrapped around numbers, and the failure mode of writing one is
that the prose and the numbers drift apart: a figure is refreshed and the
sentence describing it is not, or a number is retyped one digit short. Handing
the prose to a language model makes that failure mode worse, not better, because
a model will produce a fluent sentence containing a number it invented.

``Evidence`` is the answer. It is a ``Spec``, so it hashes and round-trips like
anything else in axiom, and it holds every quantity the report is *allowed* to
state, each with its interval, its unit and what produced it. Sections are
generated from it; narration rewrites those sections and is then checked back
against it (see ``numbers.unverified``). A number that is not in here cannot
legitimately appear in the document.

The collectors below turn axiom's typed results into that record, so the usual
path is not to build one by hand:

    ev = (
        EvidenceBuilder("HYPER-3", "Does the 40 mg arm lower blood pressure?")
        .verdict(identification)          # axiom.identify -> route + assumptions
        .finding("contrast", result, label="40 mg vs control", unit="mmHg")
        .diagnostic("coverage", 0.94, label="Interval coverage")
        .build()
    )
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, ClassVar, Literal

from axiom.core import Assumption, Interval, LedgerLine, NonEmptyStr, Spec, Verdict

__all__ = [
    "Evidence",
    "EvidenceBuilder",
    "Exhibit",
    "GraphRecord",
    "graph_from",
    "MethodStep",
    "Quantity",
    "quantity_from",
]


def _join(names: Sequence[str]) -> str:
    """``a``, ``a and b``, ``a, b and c`` — a list a sentence can contain."""
    items = list(names)
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + f" and {items[-1]}"


class Quantity(Spec):
    """One number the report may state, with what it takes to state it honestly.

    ``interval`` is not optional by accident: it is optional because some
    quantities genuinely have none (a count, a sample size). Anything estimated
    should carry one, and ``stated()`` prints it with its definition and mass
    because a point estimate that lost its interval on the way to a slide is
    the failure axiom's rule 4 exists to prevent.
    """

    key: NonEmptyStr
    label: NonEmptyStr
    value: float
    unit: str = ""
    interval: Interval | None = None
    source: str = ""
    note: str = ""
    precision: int = 2
    #: The value a decision turns on — a null of no effect, a minimum worthwhile
    #: difference, a budget. Without one, a conclusions section can only report
    #: the estimate; with one it can say whether the interval settles the
    #: question, which is the sentence a reader actually wants.
    threshold: float | None = None
    #: Which direction counts as a good outcome, for reading the comparison.
    beneficial: Literal["lower", "higher", "either"] = "either"

    def stated(self) -> str:
        """The quantity as prose: point, unit, and interval with its provenance."""
        unit = f" {self.unit}" if self.unit else ""
        text = f"{self.value:.{self.precision}f}{unit}"
        if self.interval is None:
            return text
        mass = f"{self.interval.mass:.0%}" if self.interval.mass is not None else "?"
        lower = f"{self.interval.lower:.{self.precision}f}"
        upper = f"{self.interval.upper:.{self.precision}f}"
        return f"{text} ({mass} {self.interval.definition.upper()} {lower} to {upper}{unit})"

    @property
    def mean(self) -> float:
        """The point, under the name a report's metric block looks for it by.

        ``axiom.report`` resolves a metric from an ``Interval`` by taking its
        midpoint, and an HDI is not symmetric about the estimate it summarizes.
        Exposing the point here — the one name ``_metric_of`` reads a summary's
        value from — is what lets ``context`` hand over the whole quantity, so
        the page prints the number the record holds rather than a midpoint
        nobody computed.
        """
        return self.value

    def numbers(self) -> tuple[float, ...]:
        """Every number this quantity licenses, for the provenance check."""
        out = [self.value]
        if self.interval is not None:
            out.extend([self.interval.lower, self.interval.upper])
            if self.interval.mass is not None:
                out.append(self.interval.mass)
        if self.threshold is not None:
            out.append(self.threshold)
        return tuple(out)

    def against_threshold(self) -> str:
        """Where the interval sits relative to ``threshold``: the checkable part.

        One of ``"below"``, ``"above"``, ``"spans"`` (the interval contains the
        threshold, so the data does not settle which side it is on), or
        ``"no threshold"`` / ``"no interval"`` when the comparison cannot be
        made at all. Deliberately a statement about the interval and not about
        significance: a spanning interval is an unsettled question, not a
        null result.
        """
        if self.threshold is None:
            return "no threshold"
        if self.interval is None:
            return "no interval"
        if self.interval.upper < self.threshold:
            return "below"
        if self.interval.lower > self.threshold:
            return "above"
        return "spans"

    def settles(self) -> bool:
        """Whether the interval lies wholly on one side of the threshold."""
        return self.against_threshold() in ("below", "above")

    def is_beneficial(self) -> bool | None:
        """Whether the interval sits wholly on the good side. ``None`` if unsettled."""
        side = self.against_threshold()
        if side not in ("below", "above") or self.beneficial == "either":
            return None
        return (side == "below") if self.beneficial == "lower" else (side == "above")


class Exhibit(Spec):
    """A figure or a table a step produced, named by the context key holding it.

    The exhibit itself is data and lives in the render context; what the record
    keeps is which step made it, what it is, and what to call it. That is the
    same separation ``Evidence`` keeps everywhere else: claims here, data beside
    it.
    """

    key: NonEmptyStr
    kind: Literal["figure", "table"]
    caption: str = ""


class GraphRecord(Spec):
    """The identification graph itself: which variables, and which arrows.

    A verdict says "identified via the back door adjusting for age". That is the
    *conclusion*; this is the thing it was concluded from, and without it a
    reader cannot check the conclusion or disagree with the graph rather than
    with the number. It is the single most load-bearing object in a causal
    report and it was the one thing the record did not keep.

    Held structurally — nodes, edges, and the two distinctions that decide
    whether a graph is identifiable at all — rather than as an
    ``identify.CausalGraph``, for the same reason ``EvidenceBuilder.verdict``
    reads a verdict structurally: reporting on a result must not require
    importing the subpackage that produced it. Anything with ``nodes`` and
    ``edges`` can be recorded, and the shape is exactly what
    ``viz.causal_graph`` draws.
    """

    name: str = ""
    nodes: tuple[str, ...] = ()
    #: Directed arrows, ``(parent, child)``.
    edges: tuple[tuple[str, str], ...] = ()
    #: Unmeasured confounding, drawn as a dashed double-headed arrow.
    bidirected: tuple[tuple[str, str], ...] = ()
    #: Nodes that exist in the graph and not in the data. Drawn hollow, and the
    #: reason a graph can be written down and still not be identifiable.
    unmeasured: tuple[str, ...] = ()
    treatment: str = ""
    outcome: str = ""
    #: The graph's own content hash, so a report names the graph it used rather
    #: than a graph that looks like it.
    graph_hash: str = ""

    def to_text(self) -> str:
        """The edge list as one line — the graph in a form that needs no plotly.

        A report whose figures cannot be drawn still has to say which graph was
        used, and an edge list is the whole of that. Bidirected edges are written
        ``a <-> b`` because the distinction between a common cause and an arrow
        is the distinction the identification turns on.
        """
        parts = [f"{a} -> {b}" for a, b in self.edges]
        parts.extend(f"{a} <-> {b}" for a, b in self.bidirected)
        return ", ".join(parts)


def graph_from(graph: object, *, treatment: str = "", outcome: str = "") -> GraphRecord:
    """A ``GraphRecord`` off anything carrying ``nodes`` and ``edges``.

    Read structurally, so an ``identify.CausalGraph`` works and so does a
    hand-built stand-in. Anything missing is simply absent from the record
    rather than guessed at.
    """
    nodes = tuple(str(x) for x in getattr(graph, "nodes", ()) or ())
    edges = tuple((str(a), str(b)) for a, b in (getattr(graph, "edges", ()) or ()) if a is not None)
    if not nodes and not edges:
        raise TypeError(
            f"cannot read a graph out of {type(graph).__name__}; "
            "pass something with .nodes and .edges"
        )
    hasher = getattr(graph, "content_hash", None)
    return GraphRecord(
        name=str(getattr(graph, "name", "") or ""),
        nodes=nodes,
        edges=edges,
        bidirected=tuple((str(a), str(b)) for a, b in (getattr(graph, "bidirected", ()) or ())),
        unmeasured=tuple(str(x) for x in (getattr(graph, "unmeasured", ()) or ())),
        treatment=treatment,
        outcome=outcome,
        graph_hash=str(hasher()) if callable(hasher) else "",
    )


class MethodStep(Spec):
    """One thing that was done, and what it rests on.

    ``what`` is the action, ``why`` is the reason it was the right one. Both are
    written by the collector from a typed axiom result rather than by a person,
    which is what stops a methods section describing an analysis nobody ran.

    Four kinds of thing a step can carry, deliberately kept apart because they
    are read differently and belong in different places:

    * ``detail`` — the protocol's *settings*: arms, allocation, the estimator.
      Short keys and short values, because they are gathered into the design
      table and a table cell holding a paragraph is not a table.
    * ``instead`` — the alternative that was rejected, in prose. The most useful
      sentence in a methods section and the one most often missing; it is not a
      setting and putting it in ``detail`` made the design table a wall of text.
    * ``readout`` — what the step *printed*, verbatim. Terminal output is not a
      parameter either; it is shown as it was seen, in a monospaced block.
    * ``equations`` — the mathematics the step is, written out. A methods
      section that says "fitted by ANCOVA" and never writes the equation has
      told the reader the name of a thing rather than the thing; these are the
      lines a reader checks the estimator against.
    * ``exhibits`` — the figures and tables the step produced.
    """

    key: NonEmptyStr
    title: NonEmptyStr
    what: str = ""
    why: str = ""
    instead: str = ""
    assumptions: tuple[Assumption, ...] = ()
    detail: dict[str, str] = {}
    readout: tuple[str, ...] = ()
    equations: tuple[str, ...] = ()
    exhibits: tuple[Exhibit, ...] = ()


class Evidence(Spec):
    """Everything a report is entitled to say, and nothing else.

    Hashing this and storing the hash beside the rendered document is what makes
    a report reproducible: the same evidence renders the same claims, and a
    changed hash says the underlying analysis moved.
    """

    title: NonEmptyStr
    question: str = ""
    findings: tuple[Quantity, ...] = ()
    diagnostics: tuple[Quantity, ...] = ()
    steps: tuple[MethodStep, ...] = ()
    assumptions: tuple[Assumption, ...] = ()
    ledger: tuple[LedgerLine, ...] = ()
    verdict: Verdict | None = None
    #: The graph the identification was read off. Without it a reader can
    #: disagree with the verdict and not with the thing that produced it.
    graph: GraphRecord | None = None
    #: What the analyst wrote after looking at the output, carried verbatim.
    #: Distinct from a finding, which is a number: these are sentences, they are
    #: not generated, and they are never rewritten by a model — a remark is the
    #: one part of a report whose author is a person, and relabelling it would
    #: be the one lie this package must not tell.
    remarks: tuple[str, ...] = ()
    provenance: dict[str, str] = {}

    def quantities(self) -> tuple[Quantity, ...]:
        """Findings and diagnostics together — every number in the record."""
        return (*self.findings, *self.diagnostics)

    def by_key(self, key: str) -> Quantity:
        for q in self.quantities():
            if q.key == key:
                return q
        raise KeyError(f"no quantity {key!r}; have {[q.key for q in self.quantities()]}")

    def context(self) -> dict[str, object]:
        """The render context: each quantity under its key, whole or as a bare float.

        ``axiom.report`` resolves a metric block from anything carrying an
        ``interval`` and a point, and shows the interval with its definition and
        mass. Handing it the quantity itself rather than the float is what keeps
        the uncertainty attached through rendering — and rather than the bare
        interval, so that the point printed is the estimate and not the
        interval's midpoint.
        """
        out: dict[str, object] = {}
        for q in self.quantities():
            out[q.key] = q if q.interval is not None else q.value
            out[f"{q.key}_stated"] = q.stated()
        if self.ledger:
            out["ledger"] = list(self.ledger)
        return out

    #: The one assumption state that counts as settled. ``core.Assumption`` is
    #: only ever moved to ``satisfied`` by code that actually checked it —
    #: ``asserted`` means a person said so, which is not the same thing and is
    #: reported as standing.
    RESOLVED_STATE: ClassVar[str] = "satisfied"

    def unresolved(self) -> tuple[str, ...]:
        """Assumptions still standing between this evidence and a causal claim.

        Everything not ``satisfied``: ``unverified`` (assumed, unchecked),
        ``violated`` (known to fail) and ``asserted`` (a person's word). A
        limitations section that does not mention these is not a limitations
        section.
        """
        seen: dict[str, Assumption] = {}
        pool: list[Assumption] = list(self.assumptions)
        if self.verdict is not None:
            pool.extend(self.verdict.assumptions)
        for step in self.steps:
            pool.extend(step.assumptions)
        for a in pool:
            if a.state != self.RESOLVED_STATE:
                seen[a.name] = a
        return tuple(sorted(seen))


def quantity_from(
    key: str,
    value: object,
    *,
    label: str,
    unit: str = "",
    source: str = "",
    note: str = "",
    precision: int = 2,
    threshold: float | None = None,
    beneficial: Literal["lower", "higher", "either"] = "either",
    mass: float = 0.9,
) -> Quantity:
    """Build a ``Quantity`` from whatever axiom handed back.

    Five shapes are understood, which between them cover what axiom returns:

    * a plain number — a count, a share, something with no interval;
    * an ``Interval`` — the point is taken as its midpoint and the note says so;
    * a ``LinearEstimate`` and friends: ``.estimate`` plus ``.ci(mass)``, which
      is what ``identify.ols`` and the other estimators hand back, and where
      ``mass`` chooses the interval;
    * a ``core.Summary`` — ``.mean`` and ``.interval``, the shape a posterior
      reports itself in;
    * anything wrapping one of those in ``.summary`` — ``estimands.realize``
      returns an ``EstimandResult``, and an estimand realized against a fit is
      the usual headline finding of a report.

    Anything else is a ``TypeError`` naming what arrived, because guessing at an
    unfamiliar result type is how a report ends up stating the wrong field.
    """
    if isinstance(value, Interval):
        point = (value.lower + value.upper) / 2.0
        return Quantity(
            key=key,
            label=label,
            value=point,
            unit=unit,
            interval=value,
            source=source or "interval",
            note=note or "point is the interval midpoint; no separate estimate was given",
            precision=precision,
            threshold=threshold,
            beneficial=beneficial,
        )
    if isinstance(value, bool):
        raise TypeError(f"{key!r}: a boolean is not a reportable quantity")
    # axiom's own estimators return `.estimate` and build the interval on demand
    # through `.ci(mass)` rather than carrying one. Reading that shape here is
    # what lets `ols(...)` be handed straight to a report; without it the caller
    # has to take the estimate apart and put it back together.
    estimate = getattr(value, "estimate", None)
    maker = getattr(value, "ci", None)
    if estimate is not None and callable(maker):
        return Quantity(
            key=key,
            label=label,
            value=float(estimate),
            unit=unit,
            interval=maker(mass),
            source=source or f"{type(value).__name__}.{getattr(value, 'method', 'estimate')}",
            note=note,
            precision=precision,
            threshold=threshold,
            beneficial=beneficial,
        )
    if isinstance(value, (int, float)):
        return Quantity(
            key=key,
            label=label,
            value=float(value),
            unit=unit,
            interval=None,
            source=source,
            note=note,
            precision=precision,
            threshold=threshold,
            beneficial=beneficial,
        )
    origin = type(value).__name__
    # ``realize`` returns an ``EstimandResult``, which keeps its point and its
    # interval together inside a ``core.Summary`` rather than beside each other.
    # Unwrap it, so the thing a decision is actually about can be handed to a
    # report the same way an estimator's output can.
    summary = getattr(value, "summary", None)
    if summary is not None and isinstance(getattr(summary, "interval", None), Interval):
        value = summary
    raw: object = getattr(value, "value", None)
    if raw is None:
        # A ``Summary`` states its point as ``mean``. Reading it is the
        # difference between reporting the posterior mean and reporting the
        # midpoint of an interval that need not be symmetric about it.
        mean = getattr(value, "mean", None)
        raw = None if callable(mean) else mean
    interval = getattr(value, "interval", None)
    if raw is None and interval is None:
        raise TypeError(
            f"{key!r}: cannot read a quantity out of {origin}; "
            "pass a float, an Interval, or a result carrying .value / .interval"
        )
    if raw is None and isinstance(interval, Interval):
        raw = (interval.lower + interval.upper) / 2.0
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise TypeError(
            f"{key!r}: the point read out of {origin} is "
            f"{type(raw).__name__}, which is not a number"
        )
    return Quantity(
        key=key,
        label=label,
        value=float(raw),
        unit=unit,
        interval=interval if isinstance(interval, Interval) else None,
        source=source or origin,
        note=note,
        precision=precision,
        threshold=threshold,
        beneficial=beneficial,
    )


class EvidenceBuilder:
    """Fluent assembly of an ``Evidence``, in the order the work happened.

    Mutable on purpose — it is scaffolding, and ``build()`` freezes it into the
    ``Spec`` that everything downstream reads.
    """

    def __init__(self, title: str, question: str = "") -> None:
        self._title = title
        self._question = question
        self._findings: list[Quantity] = []
        self._diagnostics: list[Quantity] = []
        self._steps: list[MethodStep] = []
        self._assumptions: list[Assumption] = []
        self._ledger: list[LedgerLine] = []
        self._verdict: Verdict | None = None
        self._graph: GraphRecord | None = None
        self._remarks: list[str] = []
        self._provenance: dict[str, str] = {}

    def finding(self, key: str, value: object, *, label: str, **kw: Any) -> EvidenceBuilder:
        """A headline quantity — something the report is about."""
        self._findings.append(quantity_from(key, value, label=label, **kw))
        return self

    def diagnostic(self, key: str, value: object, *, label: str, **kw: Any) -> EvidenceBuilder:
        """A quantity about the analysis rather than the world: coverage, ESS, R-hat."""
        self._diagnostics.append(quantity_from(key, value, label=label, **kw))
        return self

    def step(
        self,
        key: str,
        title: str,
        *,
        what: str = "",
        why: str = "",
        instead: str = "",
        assumptions: Iterable[Assumption] = (),
        detail: Mapping[str, str] | None = None,
        readout: Iterable[str] = (),
        equations: Iterable[str] = (),
        exhibits: Iterable[Exhibit] = (),
    ) -> EvidenceBuilder:
        self._steps.append(
            MethodStep(
                key=key,
                title=title,
                what=what,
                why=why,
                instead=instead,
                assumptions=tuple(assumptions),
                detail=dict(detail or {}),
                readout=tuple(readout),
                equations=tuple(equations),
                exhibits=tuple(exhibits),
            )
        )
        return self

    def graph(self, graph: object, *, treatment: str = "", outcome: str = "") -> EvidenceBuilder:
        """Record the identification graph. Usually reached through ``verdict``."""
        self._graph = graph_from(graph, treatment=treatment, outcome=outcome)
        return self

    def verdict(
        self, verdict: object, *, key: str = "identification", graph: object = None
    ) -> EvidenceBuilder:
        """Record an identification verdict, and the step it stands for.

        Accepts a ``core.Verdict`` or an ``identify.IdentificationVerdict``. The
        latter is read for everything it knows — the route, the adjustment set,
        the mediators, the instrument, the routes it rejected — because that is
        the difference between a methods section that says "the effect is
        identified" and one that says which variables were adjusted for.

        ``graph`` records the graph the verdict was read off. Pass it, or pass
        an ``IdentificationVerdict`` that carries one: a verdict without its
        graph is a conclusion a reader cannot check, and the edge list is what
        lets them disagree with the graph rather than with the number.

        Read structurally rather than by import, so this package does not need to
        depend on ``axiom.identify`` to report on its results.
        """
        inner = getattr(verdict, "verdict", verdict)
        if not isinstance(inner, Verdict):
            raise TypeError(
                f"expected a core.Verdict or something wrapping one, got "
                f"{type(verdict).__name__}"
            )
        self._verdict = inner
        route = str(getattr(verdict, "route", "") or inner.route or "no route")

        what = [f"The effect is {inner.status} via the {route.replace('_', '-')} route."]
        detail: dict[str, str] = {"status": inner.status, "route": route}

        treatment = str(getattr(verdict, "treatment", "") or "")
        outcome = str(getattr(verdict, "outcome", "") or "")
        if treatment and outcome:
            what.insert(0, f"The target is the effect of {treatment} on {outcome}.")

        # The graph, from wherever it can be had: handed in, or carried by the
        # verdict itself. A `graph_hash` with no graph beside it names a thing
        # the reader has no way to look at.
        source = graph if graph is not None else getattr(verdict, "graph", None)
        if source is not None:
            self._graph = graph_from(source, treatment=treatment, outcome=outcome)
        if self._graph is not None:
            what.append(f"The graph is {self._graph.to_text()}.")
            detail["graph"] = self._graph.to_text()
            if self._graph.unmeasured:
                detail["unmeasured nodes"] = ", ".join(self._graph.unmeasured)
        graph_hash = str(getattr(verdict, "graph_hash", "") or "")
        if graph_hash:
            detail["graph hash"] = graph_hash[:12]
            if self._graph is not None and not self._graph.graph_hash:
                self._graph = self._graph.model_copy(update={"graph_hash": graph_hash})

        adjustment = tuple(getattr(verdict, "adjustment_set", ()) or ())
        if adjustment:
            what.append(f"Estimation adjusts for {_join(adjustment)}.")
            detail["adjustment set"] = ", ".join(adjustment)
        mediators = tuple(getattr(verdict, "mediators", ()) or ())
        if mediators:
            what.append(f"The effect is carried through {_join(mediators)}.")
            detail["mediators"] = ", ".join(mediators)
        instrument = str(getattr(verdict, "instrument", "") or "")
        if instrument:
            what.append(f"{instrument} is used as an instrument.")
            detail["instrument"] = instrument
        required = tuple(getattr(verdict, "unmeasured_required", ()) or ())
        if required:
            what.append(
                f"The route would additionally require {_join(required)}, which was "
                "not measured."
            )
            detail["unmeasured but required"] = ", ".join(required)
        alternatives = tuple(getattr(verdict, "alternatives", ()) or ())
        if alternatives:
            detail["routes also available"] = ", ".join(str(a) for a in alternatives)
        limits = tuple(getattr(verdict, "search_limits_hit", ()) or ())
        if limits:
            detail["search limits hit"] = "; ".join(str(x) for x in limits)

        self._steps.append(
            MethodStep(
                key=key,
                title="Identification",
                what=" ".join(what),
                why=inner.reason,
                assumptions=tuple(inner.assumptions),
                detail=detail,
            )
        )
        return self

    def assume(self, *assumptions: Assumption) -> EvidenceBuilder:
        self._assumptions.extend(assumptions)
        return self

    def ledger_lines(self, lines: Sequence[LedgerLine]) -> EvidenceBuilder:
        self._ledger.extend(lines)
        return self

    def remark(self, *texts: str) -> EvidenceBuilder:
        """Prose the analyst wrote about the run. Carried verbatim, never narrated."""
        self._remarks.extend(" ".join(t.split()) for t in texts if t.strip())
        return self

    def provenance(self, **entries: str) -> EvidenceBuilder:
        """Content hashes, versions, seeds — whatever makes the run findable again."""
        self._provenance.update(entries)
        return self

    def build(self) -> Evidence:
        keys = [q.key for q in (*self._findings, *self._diagnostics)]
        duplicated = sorted({k for k in keys if keys.count(k) > 1})
        if duplicated:
            raise ValueError(
                f"two quantities share a key: {duplicated}; keys are how the report "
                "cites a number, so they have to be unique"
            )
        return Evidence(
            title=self._title,
            question=self._question,
            findings=tuple(self._findings),
            diagnostics=tuple(self._diagnostics),
            steps=tuple(self._steps),
            assumptions=tuple(self._assumptions),
            ledger=tuple(self._ledger),
            verdict=self._verdict,
            graph=self._graph,
            remarks=tuple(self._remarks),
            provenance=dict(self._provenance),
        )
