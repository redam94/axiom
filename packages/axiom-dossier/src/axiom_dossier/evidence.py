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


class MethodStep(Spec):
    """One thing that was done, and what it rests on.

    ``what`` is the action, ``why`` is the reason it was the right one. Both are
    written by the collector from a typed axiom result rather than by a person,
    which is what stops a methods section describing an analysis nobody ran.
    """

    key: NonEmptyStr
    title: NonEmptyStr
    what: str = ""
    why: str = ""
    assumptions: tuple[Assumption, ...] = ()
    detail: dict[str, str] = {}


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
        """The render context: each quantity under its key, as its interval or value.

        ``axiom.report`` resolves an ``Interval`` into a metric that shows its
        definition and mass, so handing it the interval rather than the float is
        what keeps the uncertainty attached through rendering.
        """
        out: dict[str, object] = {}
        for q in self.quantities():
            out[q.key] = q.interval if q.interval is not None else q.value
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
) -> Quantity:
    """Build a ``Quantity`` from whatever axiom handed back.

    Accepts a float, an ``Interval``, or any object carrying a ``value`` and an
    ``interval`` attribute — which is the shape of ``estimands.EstimandResult``
    and of the summaries the estimators return. Anything else is a ``TypeError``
    naming what arrived, because guessing at an unfamiliar result type is how a
    report ends up stating the wrong field.
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
    raw: object = getattr(value, "value", None)
    interval = getattr(value, "interval", None)
    if raw is None and interval is None:
        raise TypeError(
            f"{key!r}: cannot read a quantity out of {type(value).__name__}; "
            "pass a float, an Interval, or a result carrying .value / .interval"
        )
    if raw is None and isinstance(interval, Interval):
        raw = (interval.lower + interval.upper) / 2.0
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise TypeError(
            f"{key!r}: .value on {type(value).__name__} is "
            f"{type(raw).__name__}, which is not a number"
        )
    return Quantity(
        key=key,
        label=label,
        value=float(raw),
        unit=unit,
        interval=interval if isinstance(interval, Interval) else None,
        source=source or type(value).__name__,
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
        assumptions: Iterable[Assumption] = (),
        detail: Mapping[str, str] | None = None,
    ) -> EvidenceBuilder:
        self._steps.append(
            MethodStep(
                key=key,
                title=title,
                what=what,
                why=why,
                assumptions=tuple(assumptions),
                detail=dict(detail or {}),
            )
        )
        return self

    def verdict(self, verdict: object, *, key: str = "identification") -> EvidenceBuilder:
        """Record an identification verdict, and the step it stands for.

        Accepts a ``core.Verdict`` or an ``identify.IdentificationVerdict``. The
        latter is read for everything it knows — the route, the adjustment set,
        the mediators, the instrument, the routes it rejected — because that is
        the difference between a methods section that says "the effect is
        identified" and one that says which variables were adjusted for.

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
            provenance=dict(self._provenance),
        )
