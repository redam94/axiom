"""``identify()``: the honest verdict on whether ``x -> y`` is identified from a graph.

Routes are tried in preference order and the verdict reports the best one
*and* the alternatives, so a downstream estimator can pick a different route
for a stated reason.

* ``backdoor`` — a measured admissible set exists: ``identified``.
* ``frontdoor`` — a measured mediator set satisfies the front-door criterion:
  ``identified``.
* ``instrument`` — a (conditional) instrument exists: ``downgraded``. The
  graph establishes exclusion; relevance is checked from data; and an IV
  identifies the effect only under effect homogeneity (linear SCM) or
  monotonicity (LATE), which the graph cannot establish. Those are named.
* ``backdoor_unmeasured`` — an admissible set exists but needs an unmeasured
  node: ``downgraded``, naming the node. A point estimate is not reported
  without ``assume_identified=True``, which writes a ledger line (Phase 4).
* otherwise ``blocked``, with the reason.

Review B4: when the graph is flagged ``feedback=True`` and the caller says
the treatment has carryover, every licensed verdict is downgraded with
``no_time_varying_confounding`` unverified — static adjustment on a summary
graph is not valid under treatment–outcome feedback.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from axiom.core.result import Unsupported
from axiom.core.spec import Spec
from axiom.core.verdict import Assumption, Status, Verdict
from axiom.identify.backdoor import minimal_adjustment_sets, requires_unmeasured
from axiom.identify.frontdoor import conditional_instruments, frontdoor_sets, instruments
from axiom.identify.graph import CausalGraph
from axiom.identify.transport import TransportVerdict, transport_verdict

__all__ = ["IdentificationVerdict", "Route", "identify"]

Route = Literal[
    "no_causal_path",
    "backdoor",
    "frontdoor",
    "instrument",
    "backdoor_unmeasured",
    "none",
]
DEFAULT_PREFERENCE: tuple[Route, ...] = (
    "backdoor",
    "frontdoor",
    "instrument",
    "backdoor_unmeasured",
)


class IdentificationVerdict(Spec):
    """What the graph says about ``treatment -> outcome``, and by which route.

    ``verdict`` carries the status and assumptions; the route-specific
    fields say what to adjust for, which mediators, or which instrument.
    ``alternatives`` lists every other route the graph supports, so a refusal
    of the preferred route is not a dead end. ``transport`` is the selection-
    diagram verdict when the graph has S-nodes.
    """

    treatment: str
    outcome: str
    graph_hash: str
    verdict: Verdict
    route: Route
    adjustment_set: tuple[str, ...] = ()
    mediators: tuple[str, ...] = ()
    instrument: str = ""
    conditioning: tuple[str, ...] = ()
    unmeasured_required: tuple[str, ...] = ()
    alternatives: tuple[Route, ...] = ()
    search_limits_hit: tuple[str, ...] = ()
    transport: TransportVerdict | None = None

    @property
    def status(self) -> Status:
        return self.verdict.status

    @property
    def licensed(self) -> bool:
        return self.verdict.licensed


def _feedback_assumption() -> Assumption:
    return Assumption(
        name="no_time_varying_confounding",
        facet="identification",
        statement=(
            "the static summary graph is adequate despite treatment–outcome feedback over time; "
            "no lagged outcome confounds the lagged dose"
        ),
        challenged_by="carryover in the treatment; g-methods or an unrolled graph are needed",
    )


def identify(
    graph: CausalGraph,
    treatment: str,
    outcome: str,
    *,
    prefer: Sequence[Route] = DEFAULT_PREFERENCE,
    has_carryover: bool = False,
    max_candidates: int = 16,
) -> IdentificationVerdict:
    """Decide how (and whether) ``treatment -> outcome`` is identified in ``graph``."""
    x, y = treatment, outcome
    graph._require(x)
    graph._require(y)
    transport = (
        transport_verdict(graph, x, y, max_candidates=max_candidates) if graph.selection else None
    )
    graph_hash = graph.content_hash()

    def make(
        verdict: Verdict,
        route: Route,
        *,
        adjustment_set: tuple[str, ...] = (),
        mediators: tuple[str, ...] = (),
        instrument: str = "",
        conditioning: tuple[str, ...] = (),
        unmeasured_required: tuple[str, ...] = (),
        alternatives: tuple[Route, ...] = (),
        limits: tuple[str, ...] = (),
    ) -> IdentificationVerdict:
        return IdentificationVerdict(
            treatment=x,
            outcome=y,
            graph_hash=graph_hash,
            verdict=verdict,
            route=route,
            adjustment_set=adjustment_set,
            mediators=mediators,
            instrument=instrument,
            conditioning=conditioning,
            unmeasured_required=unmeasured_required,
            alternatives=alternatives,
            search_limits_hit=limits,
            transport=transport,
        )

    extra: list[Assumption] = []
    if graph.feedback and has_carryover:
        extra.append(_feedback_assumption())

    if x in graph.unmeasured or y in graph.unmeasured:
        which = [n for n in (x, y) if n in graph.unmeasured]
        reason = f"{which} unmeasured: an unobserved treatment or outcome cannot be estimated"
        return make(Verdict(status="blocked", reason=reason, route="none"), "none")
    if y not in graph.descendants(x):
        return make(Verdict(status="identified", route="no_causal_path"), "no_causal_path")

    limits: list[str] = []
    found = _Routes()

    bd = minimal_adjustment_sets(graph, x, y, measured_only=True, max_candidates=max_candidates)
    if isinstance(bd, Unsupported):
        limits.append(f"backdoor: {bd.reason}")
    elif bd:
        found.backdoor = tuple(sorted(bd[0]))

    fd = frontdoor_sets(graph, x, y, measured_only=True, max_candidates=min(max_candidates, 12))
    if isinstance(fd, Unsupported):
        limits.append(f"frontdoor: {fd.reason}")
    elif fd:
        found.frontdoor = tuple(sorted(fd[0]))

    ivs = instruments(graph, x, y, measured_only=True)
    if ivs:
        found.instrument = (ivs[0], ())
    else:
        civ = conditional_instruments(
            graph, x, y, measured_only=True, max_candidates=min(max_candidates, 12)
        )
        if isinstance(civ, Unsupported):
            limits.append(f"conditional_instruments: {civ.reason}")
        elif civ:
            z0, w0 = civ[0]
            found.instrument = (z0, tuple(sorted(w0)))

    if found.backdoor is None and requires_unmeasured(graph, x, y):
        full = minimal_adjustment_sets(
            graph, x, y, measured_only=False, max_candidates=max_candidates
        )
        if isinstance(full, Unsupported):
            limits.append(f"backdoor_unmeasured: {full.reason}")
        elif full:
            chosen = tuple(sorted(full[0]))
            found.backdoor_unmeasured = (
                chosen,
                tuple(n for n in chosen if n in graph.unmeasured),
            )

    available = found.available()
    order = [r for r in prefer if r in available]
    order += [r for r in DEFAULT_PREFERENCE if r in available and r not in order]
    if not order:
        reason = "no back-door, front-door, or instrumental route exists in the graph"
        if limits:
            reason += f"; search limits hit: {limits}"
        return make(
            Verdict(status="blocked", reason=reason, route="none", assumptions=tuple(extra)),
            "none",
            limits=tuple(limits),
        )

    chosen_route = order[0]
    alternatives = tuple(order[1:])
    assumptions = list(extra)
    if chosen_route == "backdoor" and found.backdoor is not None:
        status: Status = "downgraded" if assumptions else "identified"
        return make(
            Verdict(status=status, route="backdoor", assumptions=tuple(assumptions)),
            "backdoor",
            adjustment_set=found.backdoor,
            alternatives=alternatives,
            limits=tuple(limits),
        )
    if chosen_route == "frontdoor" and found.frontdoor is not None:
        status = "downgraded" if assumptions else "identified"
        return make(
            Verdict(status=status, route="frontdoor", assumptions=tuple(assumptions)),
            "frontdoor",
            mediators=found.frontdoor,
            alternatives=alternatives,
            limits=tuple(limits),
        )
    if chosen_route == "instrument" and found.instrument is not None:
        z, w = found.instrument
        assumptions += _instrument_assumptions(z, x, y)
        return make(
            Verdict(status="downgraded", route="instrument", assumptions=tuple(assumptions)),
            "instrument",
            instrument=z,
            conditioning=w,
            alternatives=alternatives,
            limits=tuple(limits),
        )
    assert found.backdoor_unmeasured is not None
    adj, unmeasured = found.backdoor_unmeasured
    assumptions.append(
        Assumption(
            name="unmeasured_adjustment_set",
            facet="identification",
            statement=(
                f"the adjustment set {list(adj)} is available, but {list(unmeasured)} is unmeasured"
            ),
            challenged_by="no proxy or sensitivity bound for the unmeasured node",
        )
    )
    return make(
        Verdict(status="downgraded", route="backdoor_unmeasured", assumptions=tuple(assumptions)),
        "backdoor_unmeasured",
        adjustment_set=adj,
        unmeasured_required=unmeasured,
        alternatives=alternatives,
        limits=tuple(limits),
    )


class _Routes:
    """What the searches found, one slot per route (a scratch record, not a Spec)."""

    def __init__(self) -> None:
        self.backdoor: tuple[str, ...] | None = None
        self.frontdoor: tuple[str, ...] | None = None
        self.instrument: tuple[str, tuple[str, ...]] | None = None
        self.backdoor_unmeasured: tuple[tuple[str, ...], tuple[str, ...]] | None = None

    def available(self) -> set[Route]:
        out: set[Route] = set()
        if self.backdoor is not None:
            out.add("backdoor")
        if self.frontdoor is not None:
            out.add("frontdoor")
        if self.instrument is not None:
            out.add("instrument")
        if self.backdoor_unmeasured is not None:
            out.add("backdoor_unmeasured")
        return out


def _instrument_assumptions(z: str, x: str, y: str) -> list[Assumption]:
    return [
        Assumption(
            name="exclusion",
            facet="identification",
            statement=f"{z} affects {y} only through {x}",
            challenged_by="a direct or confounded path in the graph",
            state="satisfied",
        ),
        Assumption(
            name="relevance",
            facet="identification",
            statement=f"{z} is associated with {x} (first-stage strength)",
            challenged_by="first-stage F below 10",
        ),
        Assumption(
            name="effect_homogeneity_or_monotonicity",
            facet="identification",
            statement=(
                "the IV estimand equals the average effect (homogeneous/linear effect) "
                "or is read as a LATE under monotonicity"
            ),
            challenged_by="heterogeneous effects; defiers",
        ),
    ]
