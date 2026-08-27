"""Whether a corpus is pooling one quantity, or averaging several that look alike.

``meta.pool`` selects records by ``family`` and ``quantity`` and pools them.
Neither of those is an estimand. A ``StudyRecord`` carries a number, a standard
error, a quantity name and an optional ``estimand_hash`` that defaults to the
empty string, so nothing has ever compared what two records were actually
measuring — the eight-facet key exists one layer down and the pool never asked
for it.

Across parties that is not a theoretical worry. The intention-to-treat effect
and the complier effect of the same experiment differ on ``intervention`` and
``population`` and are both reported as "the lift"
(``docs/notes/0031-assigned-is-not-received.md``); a party whose operations
deliver at 95 % and one that delivers at 60 % will file one of each without
anybody choosing to.

``commensurable`` is the check: give it the ``Estimand`` behind each record and
it transfers every one to a reference and reports what differs.

**Pooling is not transferring, and the rule here is stricter than
``transfer_to``.** A transfer attaches a named assumption to *one* number and
carries it in the ledger, so a reader sees what was assumed. A pool takes a
precision-weighted average: the weights come from standard errors, nothing in
them refers to the populations, and the assumption — if anyone had named it —
would apply to a quantity that never appears in the output. So:

* a ``blocked`` transfer blocks the pool, as it must;
* a **latent subpopulation** on either side blocks the pool even though
  ``transfer_to`` only downgrades it. The compliers of one instrument are a set
  of units no covariate describes; averaging their effect with an effect over
  everybody produces a number that is over no population at all;
* every other downgrade is allowed and *named*. Between-population
  heterogeneity is what a random-effects pool is for; the assumptions ride on
  the result rather than blocking it.

Records with no estimand supplied are reported in ``unchecked`` and never
counted as passing. A pool run without estimands says ``unchecked`` in its
detail rather than nothing at all.
"""

from __future__ import annotations

from collections.abc import Mapping

from axiom.core import Assumption, LedgerLine, NonEmptyStr, Spec, Verdict
from axiom.core.verdict import Status
from axiom.estimands import Estimand, Facet, TransferPlan
from axiom.meta.schema import Corpus

__all__ = [
    "Commensurability",
    "Incompatibility",
    "commensurable",
]


class Incompatibility(Spec):
    """One record that cannot be pooled with the reference, and why.

    ``facet`` names the facet that did it. ``status`` is the transfer's own
    status — ``blocked`` when the estimand machinery refused, ``downgraded``
    when it licensed a transfer that a *pool* still may not make (a latent
    subpopulation).
    """

    study: NonEmptyStr
    reference: NonEmptyStr
    facet: Facet
    status: Status
    reason: NonEmptyStr


class Commensurability(Spec):
    """What the records in a corpus are measuring, relative to one of them.

    ``entries`` is empty when every checked record may be pooled with the
    reference. ``assumptions`` are the ones the pool would be making silently —
    they do not block, and they belong on the result so that a pooled mean
    carries what was assumed to produce it.
    """

    family: str
    reference: NonEmptyStr
    checked: tuple[str, ...]
    unchecked: tuple[str, ...] = ()
    entries: tuple[Incompatibility, ...] = ()
    assumptions: tuple[Assumption, ...] = ()

    @property
    def poolable(self) -> bool:
        return not self.entries

    def verdict(self) -> Verdict:
        """``identified`` when nothing differs, ``downgraded`` under named assumptions,
        ``blocked`` when a record measures something the pool may not average."""
        if self.entries:
            named = "; ".join(f"{e.study} ({e.facet}): {e.reason}" for e in self.entries)
            return Verdict(
                status="blocked",
                reason=f"{len(self.entries)} record(s) are not commensurable with "
                f"{self.reference!r}: {named}",
                route="commensurability",
            )
        if self.assumptions:
            facets = sorted({a.facet for a in self.assumptions})
            return Verdict(
                status="downgraded",
                reason=(
                    f"the pool averages over records differing on {facets}; the assumptions "
                    "that licenses are named and unverified"
                ),
                assumptions=self.assumptions,
                route="commensurability",
            )
        return Verdict(status="identified", route="commensurability")

    def ledger_line(self) -> LedgerLine:
        verdict = self.verdict()
        return LedgerLine(
            kind="commensurability",
            statement=(
                f"{len(self.checked)} record(s) checked against {self.reference!r}"
                + (f", {len(self.unchecked)} unchecked" if self.unchecked else "")
                + f": {verdict.status}"
                + (f" — {verdict.reason}" if verdict.reason else "")
            ),
            assumption=self.assumptions[0] if self.assumptions else None,
            detail={
                "family": self.family,
                "reference": self.reference,
                "checked": str(len(self.checked)),
                "unchecked": ", ".join(self.unchecked),
                "status": verdict.status,
            },
        )


def _latent_block(plan: TransferPlan, source: Estimand, target: Estimand) -> str | None:
    """The pool-specific rule: a latent subpopulation cannot be averaged into one that is not."""
    if "population" not in plan.differing:
        return None
    left, right = source.population.latent, target.population.latent
    if left is None and right is None:
        return None
    if left is not None and right is not None and left.same_stratum(right):
        return None
    described = left if left is not None else right
    if described is None:  # pragma: no cover - both None is handled above
        return None
    return (
        f"one side is the {described.kind}s of {described.instrument} — a subpopulation no "
        "covariate describes — and a precision-weighted average over it and a different "
        "population is a number over no population at all"
    )


def commensurable(
    corpus: Corpus,
    estimands: Mapping[str, Estimand],
    *,
    family: str = "",
    reference: str = "",
) -> Commensurability:
    """Transfer every record's estimand to a reference and report what a pool would assume.

    ``estimands`` maps ``StudyRecord.study`` to the ``Estimand`` behind it; a
    record with no entry is listed in ``unchecked``. ``family`` restricts the
    corpus the way ``PoolSpec.family`` does; ``reference`` names the study whose
    estimand the others are read as, and defaults to the first checkable record
    in corpus order.
    """
    records = corpus.by_family(family).records if family else corpus.records
    if not records:
        raise ValueError(f"no records to check{f' in family {family!r}' if family else ''}")
    known = [r for r in records if r.study in estimands]
    unchecked = tuple(r.study for r in records if r.study not in estimands)
    if not known:
        raise ValueError(
            f"no estimand supplied for any of the {len(records)} record(s); "
            "commensurability is a claim about estimands, not about record ids"
        )
    if reference and reference not in estimands:
        raise KeyError(f"no estimand supplied for the reference study {reference!r}")
    anchor = reference or known[0].study
    target = estimands[anchor]

    entries: list[Incompatibility] = []
    assumptions: list[Assumption] = []
    seen: set[tuple[str, str]] = set()
    for record in known:
        if record.study == anchor:
            continue
        source = estimands[record.study]
        plan = source.transfer_to(target)
        latent = _latent_block(plan, source, target)
        if latent is not None:
            entries.append(
                Incompatibility(
                    study=record.study,
                    reference=anchor,
                    facet="population",
                    status=plan.status,
                    reason=latent,
                )
            )
            continue
        if plan.status == "blocked":
            entry = next(e for e in plan.entries if e.blocked is not None)
            entries.append(
                Incompatibility(
                    study=record.study,
                    reference=anchor,
                    facet=entry.facet,
                    status="blocked",
                    reason=entry.blocked.reason if entry.blocked else "blocked",
                )
            )
            continue
        for assumption in plan.assumptions:
            key = (assumption.name, assumption.facet)
            if key not in seen:
                seen.add(key)
                assumptions.append(assumption)
    return Commensurability(
        family=family,
        reference=anchor,
        checked=tuple(r.study for r in known),
        unchecked=unchecked,
        entries=tuple(entries),
        assumptions=tuple(assumptions),
    )
