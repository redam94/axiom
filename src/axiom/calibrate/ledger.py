"""The assumption ledger: every transfer of evidence, one checkable line at a time.

A ``Ledger`` is an immutable, content-hashed tuple of ``core.LedgerLine``.
Nothing is ever edited in place: ``append`` returns a new ledger, so the
ledger attached to a ``ResolvedTransfer`` is exactly the one its hash says.

Two kinds of line matter for completeness:

* **facet lines** — ``kind == "facet:<facet>"`` — one per estimand facet
  that differs between the source and the target. ``Estimand.transfer_to``
  writes the structural version (the licensing ``Assumption`` and a status
  of ``assumed`` or ``blocked``); the correction operators in
  ``calibrate.transfer`` write the quantitative version (the same facet,
  plus the number before and after the correction).
* **other lines** — ``transport`` (the selection-diagram verdict), unit
  conversions, overrides — which carry context but do not stand in for a
  facet.

**The completeness rule** (decision D6.5, made explicit here). A plan is
completely accounted for when, for every facet in ``plan.differing``,

1. exactly one line in the ledger has ``kind == f"facet:{facet}"``;
2. that line's ``assumption`` is an ``Assumption`` (typed, not free text);
3. that line's ``detail`` has a ``"counterfactual"`` key — the reading
   without the correction — and a ``"value"`` key — the reading with it;
4. that line's ``detail["status"]`` is not ``"blocked"``.

The lines ``transfer_to`` produces do not carry a counterfactual, because
at that point no number exists yet. ``Ledger.from_plan`` therefore stamps
each plan line with ``detail["counterfactual"] = detail["value"] =
UNCORRECTED`` and ``detail["correction"] = "none"``: the counterfactual of
an uncorrected facet *is* the source estimate read as-is, and recording
that literally is what makes "we applied no correction here" auditable
instead of silent. A facet whose plan line has been replaced by a
correction's line (``calibrate.transfer.resolve`` does this) carries the
actual numbers instead.

``check_complete`` returns ``identified`` (no assumptions attached — the
ledger's own lines carry them) when the rule holds, and ``blocked`` with a
reason that names every missing, duplicated, blocked, or malformed facet
otherwise. It is the gate ``tests/contracts/test_ledger_completeness.py``
runs.
"""

from __future__ import annotations

from collections import Counter

import pandas as pd

from axiom.core import LedgerLine, Spec, Verdict
from axiom.estimands import TransferPlan

__all__ = [
    "FACET_PREFIX",
    "UNCORRECTED",
    "Ledger",
    "facet_of",
]

FACET_PREFIX = "facet:"
"""``kind`` prefix of the lines that stand in for an estimand facet."""

UNCORRECTED = "uncorrected: source estimate read as-is"
"""The literal counterfactual of a facet no operator corrected (see the module docstring)."""


def facet_of(line: LedgerLine) -> str | None:
    """The facet a line stands in for, or ``None`` for a non-facet line."""
    if line.kind.startswith(FACET_PREFIX):
        return line.kind[len(FACET_PREFIX) :]
    return None


class Ledger(Spec):
    """An immutable sequence of ``LedgerLine``; see the module docstring for the rule."""

    lines: tuple[LedgerLine, ...] = ()

    # -- construction ----------------------------------------------------------------

    def append(self, *lines: LedgerLine) -> Ledger:
        """A new ledger with ``lines`` added at the end; ``self`` is unchanged."""
        return Ledger(lines=(*self.lines, *lines))

    @classmethod
    def from_plan(cls, plan: TransferPlan) -> Ledger:
        """The plan's facet lines, each stamped with the ``UNCORRECTED`` counterfactual."""
        return cls(
            lines=tuple(
                line.model_copy(
                    update={
                        "detail": {
                            **line.detail,
                            "counterfactual": UNCORRECTED,
                            "value": UNCORRECTED,
                            "correction": "none",
                        }
                    }
                )
                for line in plan.ledger_lines
            )
        )

    # -- queries ---------------------------------------------------------------------

    def facet_lines(self) -> tuple[LedgerLine, ...]:
        return tuple(line for line in self.lines if facet_of(line) is not None)

    def facets_covered(self) -> dict[str, int]:
        """How many facet lines name each facet (``{"window": 1, "level": 2}``)."""
        counts = Counter(f for line in self.lines if (f := facet_of(line)) is not None)
        return dict(sorted(counts.items()))

    def check_complete(self, plan: TransferPlan) -> Verdict:
        """``identified`` iff every differing facet meets the four-point rule; else ``blocked``.

        The reason of a ``blocked`` verdict names every facet that fails and
        why (``missing``, ``duplicated`` with the count, ``no assumption``,
        ``no counterfactual``, ``blocked``), so a failing gate says which
        line to write.
        """
        counts = self.facets_covered()
        problems: list[str] = []
        for facet in plan.differing:
            n = counts.get(facet, 0)
            if n == 0:
                problems.append(f"{facet}: missing")
                continue
            if n > 1:
                problems.append(f"{facet}: duplicated ({n} lines)")
                continue
            (line,) = (ln for ln in self.lines if facet_of(ln) == facet)
            if line.assumption is None:
                problems.append(f"{facet}: no assumption")
            if "counterfactual" not in line.detail:
                problems.append(f"{facet}: no counterfactual")
            if "value" not in line.detail:
                problems.append(f"{facet}: no value")
            if line.detail.get("status") == "blocked":
                problems.append(f"{facet}: blocked")
        if problems:
            return Verdict(
                status="blocked",
                reason="ledger incomplete — " + "; ".join(problems),
                route="ledger",
            )
        covered = ", ".join(plan.differing) if plan.differing else "none differ"
        return Verdict(
            status="identified",
            reason=f"every differing facet accounted for exactly once ({covered})",
            route="ledger",
        )

    # -- reporting -------------------------------------------------------------------

    def to_frame(self) -> pd.DataFrame:
        """One row per line: kind, facet, statement, assumption, state, counterfactual, value."""
        rows = [
            {
                "kind": line.kind,
                "facet": facet_of(line) or "",
                "statement": line.statement,
                "assumption": line.assumption.name if line.assumption is not None else "",
                "state": line.assumption.state if line.assumption is not None else "",
                "counterfactual": line.detail.get("counterfactual", ""),
                "value": line.detail.get("value", ""),
                "correction": line.detail.get("correction", ""),
                "source": line.source,
                "target": line.target,
            }
            for line in self.lines
        ]
        columns = [
            "kind",
            "facet",
            "statement",
            "assumption",
            "state",
            "counterfactual",
            "value",
            "correction",
            "source",
            "target",
        ]
        return pd.DataFrame(rows, columns=columns)

    def summary(self) -> str:
        """A human-readable listing, one line per entry."""
        if not self.lines:
            return "ledger: empty"
        out = [f"ledger: {len(self.lines)} line(s)"]
        for i, line in enumerate(self.lines, start=1):
            who = line.assumption.name if line.assumption is not None else "-"
            state = f" [{line.assumption.state}]" if line.assumption is not None else ""
            cf = line.detail.get("counterfactual")
            val = line.detail.get("value")
            numbers = f"  {cf} -> {val}" if cf is not None and val is not None else ""
            out.append(f"{i:>3}. {line.kind:<24} {who}{state}: {line.statement}{numbers}")
        return "\n".join(out)
