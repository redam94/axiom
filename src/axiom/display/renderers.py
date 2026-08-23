"""How each kind of result turns into a ``Card``.

Registered by type rather than implemented as methods, for the reason the rest
of the codebase gives: a ``Verdict`` lives in ``core`` and this module lives at
the top of the stack, so the arrow has to point this way. It also means a result
type gains a good rendering without gaining a dependency, and anything can add
one for its own types with ``@renders``.

The **generic renderer matters more than the specific ones**. There are sixty-one
result-shaped public types in axiom and none of them rendered at all before
this; hand-writing sixty-one renderers would be a way of shipping thirty. So
every ``Spec`` gets a decent card from its own fields, and the dozen results a
reader actually looks at get one that knows what the fields mean — which
interval to lead with, which status the verdict implies, what to say when a
number is missing.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from axiom.display.card import Card, Status, status_from

__all__ = ["REGISTRY", "card_for", "generic_card", "renders"]

REGISTRY: dict[type, Callable[[Any], Card]] = {}


def renders(*types: type) -> Callable[[Callable[[Any], Card]], Callable[[Any], Card]]:
    """Register a renderer for one or more result types."""

    def register(fn: Callable[[Any], Card]) -> Callable[[Any], Card]:
        for kind in types:
            REGISTRY[kind] = fn
        return fn

    return register


def card_for(obj: object) -> Card:
    """The card for ``obj``: its own renderer, an ancestor's, or the generic one."""
    for kind in type(obj).__mro__:
        if kind in REGISTRY:
            return REGISTRY[kind](obj)
    return generic_card(obj)


# -- the fallback, which is what makes this cover everything -----------------------------


def _short(value: object, *, limit: int = 68) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, (list, tuple)):
        if not value:
            return ""
        shown = ", ".join(_short(v, limit=20) for v in list(value)[:4])
        return shown + (f" … ({len(value)})" if len(value) > 4 else "")
    if isinstance(value, dict):
        if not value:
            return ""
        return ", ".join(f"{k}={_short(v, limit=16)}" for k, v in list(value.items())[:3])
    text = str(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def generic_card(obj: object) -> Card:
    """A card from whatever fields the object declares.

    Pydantic models expose ``model_fields``; dataclasses expose ``__dict__``.
    Anything else gets its ``repr``, which is what it had before.
    """
    title = type(obj).__name__
    fields = getattr(type(obj), "model_fields", None)
    card = Card(title=title)
    if fields is not None:
        for name in fields:
            if name.startswith("_"):
                continue
            card.add(name.replace("_", " "), _short(getattr(obj, name, None)))
        return card
    values = getattr(obj, "__dict__", None)
    if values:
        for name, value in values.items():
            if not name.startswith("_"):
                card.add(name.replace("_", " "), _short(value))
        return card
    card.add("value", _short(obj))
    return card


# -- the results a reader actually looks at ----------------------------------------------


def _interval_text(interval: Any) -> str:
    if interval is None:
        return ""
    mass = f"{interval.mass:.0%}" if getattr(interval, "mass", None) is not None else "?"
    return f"{interval.lower:.4g} to {interval.upper:.4g}  ({mass} {interval.definition.upper()})"


def register_core() -> None:
    """Renderers for the core vocabulary. Imported lazily to keep the arrow down."""
    from axiom.core import (
        Assumption,
        Interval,
        LedgerLine,
        Summary,
        Unsupported,
        Unverified,
        Verdict,
    )

    @renders(Interval)
    def _interval(obj: Any) -> Card:
        card = Card(title="Interval", status="neutral")
        card.add("range", f"{obj.lower:.4g} to {obj.upper:.4g}", emphasis=True)
        card.add("mass", f"{obj.mass:.0%}" if obj.mass is not None else "unstated")
        card.add("definition", obj.definition.upper())
        return card

    @renders(Summary)
    def _summary(obj: Any) -> Card:
        card = Card(title="Summary")
        card.add("mean", f"{obj.mean:.4g}", emphasis=True)
        card.add("median", f"{obj.median:.4g}")
        card.add("sd", f"{obj.sd:.4g}")
        card.add("interval", _interval_text(obj.interval))
        card.add("draws", obj.n)
        return card

    @renders(Verdict)
    def _verdict(obj: Any) -> Card:
        by_status: dict[str, Status] = {
            "identified": "good",
            "downgraded": "assumed",
            "blocked": "bad",
            "unsupported": "bad",
        }
        status = status_from(obj.status, by_status)
        card = Card(title=f"Verdict — {obj.status}", status=status, note=obj.reason)
        card.add("route", obj.route)
        for assumption in obj.assumptions:
            card.add(f"assumes {assumption.name}", assumption.state)
        return card

    @renders(Assumption)
    def _assumption(obj: Any) -> Card:
        by_state: dict[str, Status] = {"satisfied": "good", "violated": "bad"}
        status = status_from(obj.state, by_state, "assumed")
        card = Card(title=f"Assumption — {obj.name}", status=status, note=obj.statement)
        card.add("facet", obj.facet)
        card.add("state", obj.state)
        card.add("challenged by", obj.challenged_by)
        return card

    @renders(LedgerLine)
    def _ledger(obj: Any) -> Card:
        card = Card(title=f"Ledger — {obj.kind}", note=obj.statement)
        card.add("source", obj.source)
        card.add("target", obj.target)
        if obj.assumption is not None:
            card.add("assumption", f"{obj.assumption.name} ({obj.assumption.state})")
        return card

    @renders(Unsupported)
    def _unsupported(obj: Any) -> Card:
        card = Card(title="Unsupported", status="bad", note=obj.reason)
        card.add("missing", ", ".join(obj.missing))
        for key, value in obj.detail.items():
            card.add(key, value)
        return card

    @renders(Unverified)
    def _unverified(obj: Any) -> Card:
        card = Card(title="Unverified", status="assumed", note=obj.reason)
        for key, value in obj.detail.items():
            card.add(key, value)
        return card


def register_results() -> None:
    """Renderers for the results the pillars return."""
    try:
        from axiom.identify import IdentificationVerdict, LinearEstimate
    except ImportError:  # pragma: no cover - identify is always present
        return

    @renders(LinearEstimate)
    def _estimate(obj: Any) -> Card:
        card = Card(
            title=f"{obj.method or 'estimate'}: {obj.treatment} → {obj.outcome}".strip(": "),
        )
        card.add("estimate", f"{obj.estimate:.4g}", emphasis=True)
        card.add("standard error", f"{obj.se:.4g}")
        card.add("90% interval", _interval_text(obj.ci(0.9)))
        card.add("n", obj.n)
        card.add("adjusted for", ", ".join(obj.covariates) or "nothing")
        return card

    @renders(IdentificationVerdict)
    def _identification(obj: Any) -> Card:
        inner = obj.verdict
        by_status: dict[str, Status] = {"identified": "good", "downgraded": "assumed"}
        status = status_from(inner.status, by_status, "bad")
        card = Card(
            title=f"Identification — {obj.treatment} → {obj.outcome}",
            status=status,
            note=inner.reason,
        )
        card.add("status", inner.status, emphasis=True)
        card.add("route", obj.route)
        card.add("adjustment set", ", ".join(obj.adjustment_set))
        card.add("mediators", ", ".join(obj.mediators))
        card.add("instrument", obj.instrument)
        card.add("unmeasured but required", ", ".join(obj.unmeasured_required))
        card.add("other routes", ", ".join(str(a) for a in obj.alternatives))
        for limit in obj.search_limits_hit:
            card.add("limit", limit)
        return card

    try:
        from axiom.discover import PAG, DiscoveryResult, StabilityReport
    except ImportError:  # pragma: no cover
        return

    @renders(DiscoveryResult)
    def _discovery(obj: Any) -> Card:
        card = Card(title="Discovered graph", note=obj.essential.summary())
        card.add("score", f"{obj.score:.6g}", emphasis=True)
        card.add("observations", obj.n_observations)
        card.add("phases", " → ".join(obj.phases))
        card.add("steps", obj.steps)
        card.add("targets", "; ".join(", ".join(t) for t in obj.targets) or "observational")
        for limit in obj.limits_hit:
            card.add("limit", limit)
        return card

    @renders(StabilityReport)
    def _stability(obj: Any) -> Card:
        card = Card(title="Edge stability", note=obj.summary())
        card.add("resamples", obj.n_bootstrap)
        card.add("rows", obj.n_rows)
        for edge in obj.edges[:8]:
            card.add(f"{edge.a} – {edge.b}", edge.describe())
        return card

    @renders(PAG)
    def _pag(obj: Any) -> Card:
        card = Card(title="PAG", note=obj.summary())
        for edge in obj.edges[:10]:
            card.add("edge", edge.to_text())
        for limit in obj.limits_hit:
            card.add("limit", limit)
        return card


def register_all() -> None:
    """Register every renderer. Safe to call more than once."""
    register_core()
    register_results()
