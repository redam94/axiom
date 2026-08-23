"""The other way narration goes wrong: a claim with no number in it.

``numbers.unverified`` catches a fabricated quantity. It cannot catch "the
effect is robust", "this demonstrates that the drug works", or "the result is
statistically significant" — sentences with no numeral at all, which are exactly
the sentences a conclusions section invites. Adding an interpretation section
without adding this check would be adding the one place a model is most likely
to overreach and leaving it unguarded.

The rule is the same as for numbers, and just as blunt: **a claim word is
licensed only if the evidence record already uses it.** If the analysis
concluded something was significant, the word is in the record and the prose may
repeat it. If it did not, the word is the model's own and the narration is
rejected.

That is deliberately strict. It rejects some sentences a careful author would
have been entitled to write, and the remedy is the right one: put the claim in
the evidence, where it is a recorded finding rather than a flourish.

Not every strong word is here. The list is confined to four kinds of claim that
a causal analysis specifically must not make for free — established causation,
statistical significance, robustness, and universal generalization.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from axiom_dossier.evidence import Evidence
from axiom_dossier.numbers import strings_of

__all__ = ["CLAIM_WORDS", "Claim", "licensed_claims", "unlicensed"]

#: Claim vocabulary, by what the word asserts. Matched on word boundaries and
#: case-insensitively; a phrase matches across whitespace.
CLAIM_WORDS: dict[str, tuple[str, ...]] = {
    "causation": (
        "proves",
        "proven",
        "demonstrates that",
        "establishes that",
        "shows conclusively",
        "causal proof",
    ),
    "significance": (
        "statistically significant",
        "significant",
        "significantly",
        "insignificant",
        "non-significant",
    ),
    "robustness": (
        "robust",
        "robustly",
        "conclusive",
        "conclusively",
        "definitive",
        "definitively",
        "confirms",
        "confirmed",
        "validates",
        "validated",
    ),
    "generalization": (
        "always",
        "never fails",
        "in all cases",
        "universally",
        "guarantees",
        "guaranteed",
    ),
}


class Claim:
    """One unlicensed claim: the phrase, what kind of claim it is, where it sat."""

    __slots__ = ("text", "kind", "start", "end")

    def __init__(self, text: str, kind: str, start: int, end: int) -> None:
        self.text = text
        self.kind = kind
        self.start = start
        self.end = end

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Claim({self.text!r}, kind={self.kind!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Claim) and (self.text, self.start) == (other.text, other.start)

    def __hash__(self) -> int:
        return hash((self.text, self.start))


def _pattern(phrase: str) -> re.Pattern[str]:
    return re.compile(r"\b" + r"\s+".join(map(re.escape, phrase.split())) + r"\b", re.I)


_COMPILED = {
    kind: [(phrase, _pattern(phrase)) for phrase in phrases]
    for kind, phrases in CLAIM_WORDS.items()
}


def licensed_claims(evidence: Evidence, *, allow: Iterable[str] = ()) -> frozenset[str]:
    """The claim phrases the evidence already makes, and may therefore repeat."""
    haystack = " ".join(strings_of(evidence)).lower()
    found = {phrase.lower() for phrase in allow}
    for _, phrases in _COMPILED.items():
        for phrase, pattern in phrases:
            if pattern.search(haystack):
                found.add(phrase.lower())
    return frozenset(found)


def unlicensed(text: str, evidence: Evidence, *, allow: Iterable[str] = ()) -> tuple[Claim, ...]:
    """Claim phrases in ``text`` that the evidence never makes, in order.

    Empty means the prose asserts nothing stronger than the record does. As with
    the numeric check this is a check on provenance, not on truth: prose can
    still be wrong while using only licensed words.
    """
    permitted = licensed_claims(evidence, allow=allow)
    out: list[Claim] = []
    for kind, phrases in _COMPILED.items():
        for phrase, pattern in phrases:
            if phrase.lower() in permitted:
                continue
            for m in pattern.finditer(text):
                out.append(Claim(m.group(0), kind, m.start(), m.end()))
    # longest match wins where two phrases overlap ("significant" inside
    # "statistically significant"), so a single overreach is reported once
    kept: list[Claim] = []
    for claim in sorted(out, key=lambda c: (c.start, -(c.end - c.start))):
        if kept and claim.start < kept[-1].end:
            continue
        kept.append(claim)
    return tuple(kept)
