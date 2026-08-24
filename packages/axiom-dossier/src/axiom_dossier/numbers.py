"""Checking prose back against the record: which numbers in this text are invented?

This is the module that makes it defensible to let a language model write any of
the document. The model is given the evidence and asked to narrate it, which is
a request it can satisfy fluently and wrongly — the characteristic failure is a
sentence that reads perfectly and contains a number nobody computed.

So the prose is checked afterwards, mechanically. Every numeric literal in the
text is extracted and matched against the numbers the evidence licenses: point
estimates, interval bounds, interval masses, and any number already appearing in
the evidence's own strings. A literal matching nothing is returned. A caller that
gets a non-empty result should not publish the sentence.

Matching allows for **rounding, in the direction a writer actually rounds**: a
literal printed to two decimals matches any licensed value that rounds to it at
two decimals, so ``12.43`` licenses "12.4" and "12". It does not work the other
way — "12.4321" is not licensed by a value of 12.43, because that is a claim to
precision the analysis never made.

Percentages are matched both ways: an interval mass of ``0.9`` licenses "90 %",
and a licensed value of ``0.9`` also licenses the bare literal ``0.9``. So is
magnitude: an estimate of ``-12.4`` licenses "12.4", because a report writes that
effect as "lowered by 12.4" and a check that rejected the sentences reports are
actually made of would simply be switched off.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from axiom_dossier.evidence import Evidence

__all__ = ["Literal", "licensed_numbers", "literals", "strings_of", "unverified"]

# A signed decimal with optional thousands separators and optional exponent.
# The second lookbehind rejects a numeral inside an identifier -- the "3" of
# HYPER-3, the "19" of COVID-19 -- which is a name, not a claim about a quantity.
_NUMBER = re.compile(
    r"(?<![\w.])(?<!\w-)"
    r"(-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)(?![\w])"
)
_PERCENT_AFTER = re.compile(r"\s*(?:%|per\s?cent\b|percent\b)")


class Literal:
    """One number as it was written: its value, its printed precision, its span."""

    __slots__ = ("text", "value", "decimals", "start", "end", "percent")

    def __init__(self, text: str, value: float, decimals: int, start: int, end: int, percent: bool):
        self.text = text
        self.value = value
        self.decimals = decimals
        self.start = start
        self.end = end
        self.percent = percent

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Literal({self.text!r}, value={self.value}, decimals={self.decimals})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Literal) and (self.text, self.start) == (other.text, other.start)

    def __hash__(self) -> int:
        return hash((self.text, self.start))


def literals(text: str) -> tuple[Literal, ...]:
    """Every numeric literal in ``text``, with the precision it was printed at."""
    out: list[Literal] = []
    for m in _NUMBER.finditer(text):
        raw = m.group(1)
        cleaned = raw.replace(",", "")
        try:
            value = float(cleaned)
        except ValueError:  # pragma: no cover - the pattern already guarantees this parses
            raise ValueError(f"matched {raw!r} as a number but could not parse it") from None
        decimals = len(cleaned.split(".")[1].split("e")[0]) if "." in cleaned else 0
        percent = bool(_PERCENT_AFTER.match(text, m.end()))
        out.append(Literal(raw, value, decimals, m.start(1), m.end(1), percent))
    return tuple(out)


def strings_of(evidence: Evidence) -> list[str]:
    """Every string the evidence itself carries — its numbers are licensed too."""
    parts: list[str] = [evidence.title, evidence.question]
    for q in evidence.quantities():
        parts.extend([q.label, q.unit, q.note, q.source])
    for step in evidence.steps:
        parts.extend([step.title, step.what, step.why])
        parts.extend(step.detail.values())
        for a in step.assumptions:
            parts.extend([a.statement, a.challenged_by])
    for a in evidence.assumptions:
        parts.extend([a.statement, a.challenged_by])
    for line in evidence.ledger:
        parts.append(line.statement)
    if evidence.verdict is not None:
        parts.append(evidence.verdict.reason)
        for a in evidence.verdict.assumptions:
            parts.extend([a.statement, a.challenged_by])
    parts.extend(evidence.remarks)
    parts.extend(evidence.provenance.values())
    return [p for p in parts if p]


def licensed_numbers(evidence: Evidence, *, allow: Iterable[float] = ()) -> tuple[float, ...]:
    """Every number this evidence entitles the prose to contain.

    Point estimates and interval bounds and masses, plus a mass expressed as a
    percentage, plus any number already written into the evidence's own text
    (an assumption statement that says "two pre-periods" licenses "2").
    """
    out: set[float] = set(allow)
    for q in evidence.quantities():
        for n in q.numbers():
            out.add(n)
            # The magnitude, because prose carries the sign in the verb: an effect
            # of -12.4 is written "lowered by 12.4", and refusing that would make
            # the check unusable on exactly the sentences reports are made of.
            # This is a real weakening -- a sign error passes -- and it is the
            # reason this function is documented as catching invented numbers
            # rather than wrong claims.
            out.add(abs(n))
            if 0.0 < abs(n) <= 1.0:
                out.add(abs(n) * 100.0)
    for text in strings_of(evidence):
        for lit in literals(text):
            out.add(lit.value)
    return tuple(sorted(out))


def _matches(lit: Literal, value: float) -> bool:
    """Does ``lit`` round-trip to ``value`` at the precision it was printed at?"""
    for candidate in (value, value * 100.0) if lit.percent else (value,):
        if lit.decimals == 0 and candidate == 0 and lit.value == 0:
            return True
        if round(candidate, lit.decimals) == round(lit.value, lit.decimals):
            return True
        # tolerate the half-ulp a writer's rounding introduces
        if abs(candidate - lit.value) <= 0.5 * 10.0 ** (-lit.decimals) + 1e-12:
            return True
    return False


def unverified(
    text: str,
    evidence: Evidence,
    *,
    allow: Iterable[float] = (),
) -> tuple[Literal, ...]:
    """The literals in ``text`` that the evidence does not license, in order.

    An empty result means every number in the prose traces to the record. It does
    **not** mean the prose is true — a model can still attach a licensed number to
    the wrong noun, which is why narration is reviewed and why the deterministic
    sections exist as the fallback. It means no number was conjured.
    """
    permitted = licensed_numbers(evidence, allow=allow)
    return tuple(lit for lit in literals(text) if not any(_matches(lit, v) for v in permitted))
