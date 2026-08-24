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

import math
from collections.abc import Callable
from typing import Any

from axiom.core.rounding import format_interval, format_measured
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


#: Field names that hold the number an object's own uncertainty is *about*.
#: A card rounds these to the place that uncertainty reaches and leaves every
#: other float alone, because a p-value, a mass and a coefficient sitting on
#: one card do not share a resolution.
_MEASURED = frozenset(
    {
        "effect",
        "estimate",
        "expected_outcome",
        "intercept",
        "mean",
        "median",
        "point",
        "pooled",
        "slope",
        "theta",
        "value",
    }
)

#: The suffixes that name a *specific* field's uncertainty. ``slope_se`` beats
#: the object's own ``se``, which belongs to something else — a result with two
#: coefficients has two resolutions and one of them is not both.
_SUFFIXES = ("_se", "_sd", "_interval")

#: Where a card looks when a field names no uncertainty of its own, in order.
#: A standard error wins over an interval's half-width: a band is one or two
#: sigma wide, so rounding to the band throws away a digit the sigma supports,
#: and a card that has the sigma should use it for everything it prints.
_UNCERTAINTY = ("se", "sd", "standard_error", "interval")


def _scale(found: object) -> float | None:
    """A usable resolution from an ``Interval``, a float, or neither."""
    half = getattr(found, "half_width", found)
    if isinstance(half, (int, float)) and not isinstance(half, bool):
        u = abs(float(half))
        if math.isfinite(u) and u > 0.0:
            return u
    return None


def _uncertainty_of(obj: object, field: str = "") -> float | None:
    """The resolution stated for ``field``, or for the object as a whole."""
    for suffix in _SUFFIXES if field else ():
        scale = _scale(getattr(obj, field + suffix, None))
        if scale is not None:
            return scale
    if field and field not in _MEASURED:
        return None
    for name in _UNCERTAINTY:
        scale = _scale(getattr(obj, name, None))
        if scale is not None:
            return scale
    return None


def _short(value: object, *, limit: int = 68, uncertainty: float | None = None) -> str:
    if isinstance(value, float):
        return format_measured(value, uncertainty, fallback=4)
    if uncertainty is not None and hasattr(value, "text") and hasattr(value, "half_width"):
        return str(value.text(uncertainty))  # an Interval, at the card's one resolution
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

    def cell(name: str, value: object) -> str:
        # the uncertainty prints at its own resolution — two significant digits
        # of itself — and the number it qualifies prints at that same place
        if isinstance(value, float) and (name in _UNCERTAINTY or name.endswith(_SUFFIXES)):
            return _short(value, uncertainty=value)
        if hasattr(value, "half_width"):
            # an interval prints at whatever resolution the rest of the card uses
            return _short(value, uncertainty=_uncertainty_of(obj))
        return _short(value, uncertainty=_uncertainty_of(obj, name))

    if fields is not None:
        for name in fields:
            if name.startswith("_"):
                continue
            card.add(name.replace("_", " "), cell(name, getattr(obj, name, None)))
        return card
    values = getattr(obj, "__dict__", None)
    if values:
        for name, value in values.items():
            if not name.startswith("_"):
                card.add(name.replace("_", " "), cell(name, value))
        return card
    card.add("value", _short(obj))
    return card


# -- the results a reader actually looks at ----------------------------------------------


def _bounds(interval: Any, uncertainty: float | None = None) -> str:
    """``lower to upper``, both at the precision the card is working at."""
    bounds = format_interval(interval.lower, interval.upper, uncertainty=uncertainty)
    return bounds.strip("[]").replace(", ", " to ")


def _interval_text(interval: Any, uncertainty: float | None = None) -> str:
    if interval is None:
        return ""
    mass = f"{interval.mass:.0%}" if getattr(interval, "mass", None) is not None else "?"
    return f"{_bounds(interval, uncertainty)}  ({mass} {interval.definition.upper()})"


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
        card.add("range", _bounds(obj), emphasis=True)
        card.add("mass", f"{obj.mass:.0%}" if obj.mass is not None else "unstated")
        card.add("definition", obj.definition.upper())
        return card

    @renders(Summary)
    def _summary(obj: Any) -> Card:
        card = Card(title="Summary")
        # every point summary at the resolution the spread states: a mean of
        # 4000 draws has fifteen digits and two of them are the posterior
        card.add("mean", format_measured(obj.mean, obj.sd), emphasis=True)
        card.add("median", format_measured(obj.median, obj.sd))
        card.add("sd", format_measured(obj.sd, obj.sd))
        card.add("interval", _interval_text(obj.interval, obj.sd))
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
        card.add("estimate", format_measured(obj.estimate, obj.se), emphasis=True)
        card.add("standard error", format_measured(obj.se, obj.se))
        card.add("90% interval", _interval_text(obj.ci(0.9), obj.se))
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

    from axiom.estimands import TransferPlan

    @renders(TransferPlan)
    def _transfer(obj: Any) -> Card:
        """What moving a result costs, which is the only question a plan answers.

        The generic card led with two 64-character content hashes and truncated
        the three fields a reader wants. The costs are what belong at the top,
        and *no correction but one assumption* — the common case — has to be
        legible rather than an empty tuple.
        """
        by_status: dict[str, Status] = {"transportable": "good", "downgraded": "assumed"}
        card = Card(
            title="Transfer plan",
            status=status_from(obj.status, by_status, "bad"),
            note=obj.reason or "; ".join(e.statement for e in obj.entries),
        )
        card.add("status", obj.status, emphasis=True)
        card.add("facets that differ", ", ".join(obj.differing) or "none")
        card.add("corrections", len(obj.corrections))
        for correction in obj.corrections[:4]:
            card.add(f"  {correction.name}", f"×{correction.value:.4g}")
        card.add("assumptions", len(obj.assumptions))
        for assumption in obj.assumptions[:4]:
            card.add(f"  {assumption.name}", assumption.statement)
        card.add("ledger lines", len(obj.ledger_lines))
        return card


def register_containers() -> None:
    """Renderers for the three types a reader meets constantly.

    All three rendered *worse* than ``print`` before this, which is the whole
    argument for looking at the output rather than trusting the fallback.
    ``Posterior`` and ``Panel`` are not ``Spec`` subclasses, so the generic
    renderer had no fields to walk and produced a card with a title and nothing
    else. ``CausalGraph`` is a ``Spec``, but its edges are bare 2-tuples, so the
    generic flattener turned ``X -> Y, Z -> X`` into ``X, Y, Z, X`` — every
    arrow gone.
    """
    from axiom.core import Posterior
    from axiom.data import Panel
    from axiom.identify import CausalGraph

    @renders(Posterior)
    def _posterior(obj: Any) -> Card:
        names = sorted(obj.names())  # a frozenset; unsorted reorders per run
        card = Card(title="Posterior")
        card.add("parameters", ", ".join(names) or "none", emphasis=True)
        card.add("chains", obj.n_chains)
        card.add("draws", obj.n_draws())
        for name in names[:6]:
            # the per-draw shape, bracketed: a bare "2" reads as a count next to
            # `chains 4` and `draws 2000`
            shape = obj.draws(name).shape[2:]
            card.add(f"  {name}", "scalar" if not shape else f"[{'×'.join(map(str, shape))}]")
        if len(names) > 6:
            card.add("  …", f"{len(names) - 6} more")
        for axis, labels in obj.coords().items():
            card.add(f"coord {axis}", _short(list(labels)))
        seed = obj.provenance.get("seed")
        card.note = "no seed recorded" if seed is None else f"seed {seed}"
        return card

    @renders(Panel)
    def _panel(obj: Any) -> Card:
        report = obj.completeness()
        card = Card(
            title="Panel",
            # There is no "warn": the vocabulary is holds / taken-on-trust /
            # failed, and an unbalanced panel is none of those — it is a fact
            # with consequences. So the note carries it and the mark stays off.
            status="good" if report.balanced else "neutral",
            note="balanced" if report.balanced else f"{report.missing_cells} cell(s) missing",
        )
        card.add("units", len(obj.units), emphasis=True)
        card.add("periods", len(obj.periods))
        card.add("rows", len(obj.frame))
        card.add("outcome", obj.roles.outcome[0])
        # `treatments` is a dict, so iterating yields names, not pairs
        card.add("treatments", ", ".join(obj.roles.treatments) or "none")
        if report.gaps:
            card.add("gaps", _short(report.gaps))
        return card

    @renders(CausalGraph)
    def _graph(obj: Any) -> Card:
        card = Card(
            title=f"CausalGraph — {obj.name}" if obj.name else "CausalGraph",
            # feedback is not a failure — `dynamics` is built for it — so it is
            # a row rather than a mark
            status="neutral",
            # to_text keeps the arrows; the generic flattener drops them
            note=obj.to_text(),
        )
        card.add("nodes", ", ".join(obj.nodes), emphasis=True)
        card.add("edges", len(obj.edges) + len(obj.bidirected))
        card.add("unmeasured", ", ".join(obj.unmeasured) or "none")
        if obj.bidirected:
            card.add("bidirected", ", ".join(f"{a} <-> {b}" for a, b in obj.bidirected))
        if obj.selection:
            card.add("selection", ", ".join(obj.selection))
        card.add("feedback", "yes" if obj.feedback else "no")
        return card


def register_all() -> None:
    """Register every renderer. Safe to call more than once."""
    register_core()
    register_results()
    register_containers()
