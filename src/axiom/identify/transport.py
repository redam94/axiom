"""Selection diagrams and transportability (Bareinboim & Pearl 2014).

A study estimates ``P(y | do(x))`` in a *source* population; the decision is
taken in a *target* population whose mechanisms may differ. A **selection
diagram** records where they differ: for every node ``v`` whose mechanism
``f_v`` (or exogenous distribution) is not the same in both populations, an
extra node ``S[v]`` with the single edge ``S[v] -> v``. ``CausalGraph`` stores
that set in its ``selection`` field; :func:`selection_diagram` materialises the
``S[v]`` nodes so that ordinary d-separation queries can be run on them.

All numbering below is from Pearl & Bareinboim, "External validity: from
do-calculus to transportability across populations", *Statistical Science*
29(4), 2014 (cited as B&P 2014). Three sufficient conditions are implemented:

* **Trivial transportability** (Def. 6): ``P*(y | do(x))`` is identifiable
  from the target population's own graph and data, so no source data are
  needed. Decided with the target's back-door and front-door criteria on the
  graph with the S-nodes stripped; see :func:`trivially_transportable`.
* **Direct transportability** (Def. 7): if ``(Y ⊥ S | X)`` holds in
  ``G_{\\bar X}`` of the selection diagram, then
  ``P*(y | do(x)) = P(y | do(x))`` — the source effect is the target effect.
* **S-admissibility** (Def. 8, Thm. 2 / Cor. 1): if ``Z`` contains no
  descendant of ``X`` and ``(Y ⊥ S | X, Z)`` holds in ``G_{\\bar X}``, then
  ``P*(y | do(x)) = Σ_z P(y | do(x), z) P*(z)`` — the *transport formula*:
  the source's ``z``-specific effects re-weighted by the target's ``P*(z)``.

Not implemented: the complete recursive procedure of B&P 2014 Thm. 3 (the
sID algorithm), which decides transportability for every selection diagram.
When none of the three sufficient conditions holds, :func:`transport_verdict`
returns ``unsupported`` naming that gap; it never claims the effect is not
transportable. ``blocked`` is reserved for a caller-proposed set ``given``
that the graph shows is *not* S-admissible.

:func:`transport_verdict` turns the conditions into a ``TransportVerdict``
that shares ``axiom.core.Verdict``'s status vocabulary, so ``calibrate`` can
fold it into a transfer plan and the ledger by assumption name.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from itertools import combinations
from typing import Literal

from axiom.core.result import NonEmptyStr, Unsupported
from axiom.core.spec import Spec
from axiom.core.verdict import Assumption, Verdict
from axiom.identify.backdoor import (
    backdoor_admissible,
    canonical_adjustment_set,
    minimal_adjustment_sets,
)
from axiom.identify.frontdoor import frontdoor_sets
from axiom.identify.graph import CausalGraph, Edge, GraphError

__all__ = [
    "TransportVerdict",
    "directly_transportable",
    "minimal_s_admissible_sets",
    "s_admissible",
    "s_admissible_sets",
    "selection_diagram",
    "transport_verdict",
    "trivially_transportable",
]

_S_PREFIX = "S["
_SID_GAP = (
    "no S-admissible pre-treatment set found; the complete sID algorithm "
    "(Bareinboim & Pearl 2014 Thm 3 recursion) is not implemented"
)


def _s_node(v: str) -> str:
    return f"{_S_PREFIX}{v}]"


def _s_nodes(graph: CausalGraph) -> tuple[str, ...]:
    """The ``S[v]`` names for ``graph.selection`` (sorted like ``selection``)."""
    return tuple(_s_node(v) for v in graph.selection)


def _require(graph: CausalGraph, *nodes: str) -> None:
    for n in nodes:
        if n not in graph.nodes:
            raise GraphError(f"unknown node {n!r}; nodes are {list(graph.nodes)}")


def _require_pair(graph: CausalGraph, x: str, y: str) -> None:
    _require(graph, x, y)
    if x == y:
        raise GraphError(f"treatment and outcome are the same node {x!r}")


def _node_set(nodes: Iterable[str]) -> frozenset[str]:
    return frozenset((nodes,)) if isinstance(nodes, str) else frozenset(nodes)


# -- the diagram -------------------------------------------------------------------


def selection_diagram(graph: CausalGraph) -> CausalGraph:
    """The selection diagram of ``graph`` as a plain DAG (B&P 2014 Def. 4).

    For every ``v`` in ``graph.selection`` an ordinary node ``S[v]`` is added
    with the single edge ``S[v] -> v``. The ``S`` nodes are marked
    ``unmeasured`` (they are population indicators, not variables a panel
    records) and ``selection`` is cleared, so the result is a graph on which
    ``d_separated``, ``remove_edges_into`` and the rest behave with no special
    cases. The bidirected edges, ``unmeasured``, ``feedback`` and ``name`` of
    the input are kept.

    Raises ``GraphError`` if a node named ``S[v]`` already exists, which would
    make the S-node indistinguishable from a variable.
    """
    if not graph.selection:
        return graph
    s_nodes = _s_nodes(graph)
    clash = sorted(set(s_nodes) & set(graph.nodes))
    if clash:
        raise GraphError(f"nodes {clash} collide with selection-node names; rename them")
    extra: tuple[Edge, ...] = tuple((_s_node(v), v) for v in graph.selection)
    return CausalGraph(
        nodes=(*graph.nodes, *s_nodes),
        edges=(*graph.edges, *extra),
        bidirected=graph.bidirected,
        unmeasured=(*graph.unmeasured, *s_nodes),
        selection=(),
        feedback=graph.feedback,
        name=graph.name,
    )


def _diagram_bar_x(graph: CausalGraph, x: str) -> CausalGraph:
    """``G_{\\bar X}`` of the selection diagram: the object every criterion queries."""
    return selection_diagram(graph).remove_edges_into(x)


def _target_graph(graph: CausalGraph) -> CausalGraph:
    """The target population's causal graph ``G*``: the diagram with its S-nodes stripped."""
    if not graph.selection:
        return graph
    return graph.model_copy(update={"selection": ()})


# -- criteria ------------------------------------------------------------------------


def directly_transportable(graph: CausalGraph, x: str, y: str) -> bool:
    """Whether ``P*(y | do(x)) = P(y | do(x))`` (B&P 2014 Def. 7).

    The sufficient criterion is ``(Y ⊥ S | X)`` in ``G_{\\bar X}`` of the
    selection diagram, with ``S`` the set of *all* S-nodes jointly — the
    ``Z = ∅`` case of S-admissibility (Thm. 2). A graph with no selection
    nodes is directly transportable by convention: the populations are
    declared identical.
    """
    _require_pair(graph, x, y)
    if not graph.selection:
        return True
    return _diagram_bar_x(graph, x).d_separated(y, _s_nodes(graph), (x,))


TrivialRoute = tuple[Literal["backdoor", "frontdoor"], tuple[str, ...]]


def _greedy_minimal(
    start: frozenset[str], admissible: Callable[[frozenset[str]], bool]
) -> frozenset[str]:
    """Drop members of ``start`` one at a time while ``admissible`` still holds.

    The result is inclusion-minimal with respect to single removals, which is
    what a caller wants when the exhaustive search for the smallest set was
    refused.
    """
    current = start
    for node in sorted(start):
        trial = current - {node}
        if admissible(trial):
            current = trial
    return current


def _trivial_route(
    graph: CausalGraph, x: str, y: str, *, max_candidates: int
) -> TrivialRoute | None | Unsupported:
    """A target-only identification route, or why none was found.

    ``("backdoor", z)`` when a measured back-door set exists in ``G*``
    (``z`` is the smallest one, or an inclusion-minimal one when enumeration
    overflows ``max_candidates``); otherwise ``("frontdoor", m)`` for the
    smallest measured front-door set; ``None`` when neither exists;
    ``Unsupported`` when the front-door search overflowed, so the question
    is open rather than settled. An unmeasured ``x`` or ``y`` gives ``None``:
    the target cannot identify an effect it does not observe.
    """
    if x in graph.unmeasured or y in graph.unmeasured:
        return None
    target = _target_graph(graph)
    canonical = canonical_adjustment_set(target, x, y, measured_only=True)
    if canonical is not None:
        sets = minimal_adjustment_sets(
            target, x, y, measured_only=True, max_candidates=max_candidates
        )
        if not isinstance(sets, Unsupported):
            return ("backdoor", tuple(sorted(sets[0])))
        z = _greedy_minimal(canonical, lambda s: backdoor_admissible(target, x, y, s))
        return ("backdoor", tuple(sorted(z)))
    fd = frontdoor_sets(target, x, y, measured_only=True, max_candidates=max_candidates)
    if isinstance(fd, Unsupported):
        return fd
    if fd:
        return ("frontdoor", tuple(sorted(fd[0])))
    return None


def trivially_transportable(
    graph: CausalGraph, x: str, y: str, *, max_candidates: int = 14
) -> bool:
    """Whether ``P*(y | do(x))`` is identifiable from the target alone (B&P 2014 Def. 6).

    A relation is *trivially transportable* when it is identifiable from the
    target population's own graph ``G*`` and distribution ``P*``, so the
    source contributes nothing. ``G*`` is the selection diagram with the
    S-nodes stripped. Deciding identifiability in general needs the complete
    ID algorithm; this implements the sufficient test "a measured back-door
    set (Pearl 2009 Def. 3.3.1) or a measured front-door set (Def. 3.3.3)
    exists in ``G*``". When the front-door enumeration overflows
    ``max_candidates`` the route is not established and the function returns
    ``False``; :func:`transport_verdict` reports the overflow in its reason.

    Trivial and direct transportability are independent notions: B&P 2014
    Example 4, ``X -> Y`` with ``S -> Y``, is trivially but not directly
    transportable, while ``X -> Y, X <-> Y, Y -> Q`` with ``S -> Q`` is
    directly but not trivially transportable. With no selection nodes the
    test is simply whether the graph identifies the effect by one of the two
    criteria.
    """
    _require_pair(graph, x, y)
    route = _trivial_route(graph, x, y, max_candidates=max_candidates)
    return isinstance(route, tuple)


def s_admissible(graph: CausalGraph, x: str, y: str, z: Iterable[str]) -> bool:
    """Whether ``Z`` is S-admissible for the effect of ``x`` on ``y`` (B&P 2014 Def. 8).

    ``Z`` is S-admissible iff (i) ``Z`` contains no descendant of ``x`` and
    (ii) ``(Y ⊥ S | X, Z)`` holds in ``G_{\\bar X}`` of the selection diagram.
    Then the transport formula ``P*(y | do(x)) = Σ_z P(y | do(x), z) P*(z)``
    holds (Thm. 2 / Cor. 1): the source supplies the ``z``-specific effects
    and the target supplies only ``P*(z)``, which is why ``Z`` must be
    measured in *both* populations for the formula to be usable.

    ``Z`` may not contain ``x`` or ``y`` (returns ``False``); names must be
    nodes of ``graph`` — S-nodes are not variables and raise ``GraphError``.
    With no selection nodes every such ``Z`` is admissible, including the
    empty set. Admissibility is not monotone: conditioning on a collider can
    open a path that a smaller set left closed.
    """
    _require_pair(graph, x, y)
    zs = _node_set(z)
    _require(graph, *sorted(zs))
    if x in zs or y in zs:
        return False
    if zs & graph.descendants(x):
        return False
    if not graph.selection:
        return True
    return _diagram_bar_x(graph, x).d_separated(y, _s_nodes(graph), (x, *sorted(zs)))


def _candidates(graph: CausalGraph, x: str, y: str, *, measured_only: bool) -> tuple[str, ...]:
    pool = set(graph.nodes) - {x, y} - graph.descendants(x)
    if measured_only:
        pool &= graph.measured
    return tuple(sorted(pool))


def _canonical_s_admissible(
    graph: CausalGraph, x: str, y: str, *, measured_only: bool
) -> frozenset[str] | None:
    """An S-admissible set among the candidates, found in polynomial time, or ``None``.

    Some ``Z`` drawn from candidate pool ``R`` satisfies ``(Y ⊥ S | X, Z)``
    in ``G_{\\bar X}`` iff the canonical set ``Z_0 = R ∩ An({X, Y} ∪ S)``
    does (van der Zander, Liśkiewicz & Textor 2014, Lemma 3, with ``X`` as
    the forced part of the conditioning set): a path from ``Y`` to ``S`` in
    the moral graph of the ancestral set of ``{X, Y} ∪ S`` that avoids
    ``X ∪ Z_0`` also avoids any ``Z ⊆ R``, since ``Z`` meets that ancestral
    set only inside ``Z_0``. The set returned is ``Z_0`` trimmed greedily to
    an inclusion-minimal one; it need not be the smallest.
    """
    cands = frozenset(_candidates(graph, x, y, measured_only=measured_only))
    if not graph.selection:
        return frozenset()
    bar_x = _diagram_bar_x(graph, x)
    s = _s_nodes(graph)
    canonical = cands & bar_x.ancestors((x, y, *s))
    if not bar_x.d_separated(y, s, (x, *sorted(canonical))):
        return None
    return _greedy_minimal(canonical, lambda zs: bar_x.d_separated(y, s, (x, *sorted(zs))))


def _sort_key(s: frozenset[str]) -> tuple[int, tuple[str, ...]]:
    return (len(s), tuple(sorted(s)))


def s_admissible_sets(
    graph: CausalGraph,
    x: str,
    y: str,
    *,
    measured_only: bool = True,
    max_candidates: int = 14,
) -> tuple[frozenset[str], ...] | Unsupported:
    """Every S-admissible set, smallest first, then lexicographic.

    Candidates are the nodes other than ``x`` and ``y`` that are not
    descendants of ``x`` — and, with ``measured_only``, not in
    ``graph.unmeasured``. The search is exhaustive over subsets, so when the
    candidate pool exceeds ``max_candidates`` the function returns
    ``Unsupported`` naming the pool size rather than silently truncating —
    unless the polynomial existence test shows that no S-admissible set
    exists at all, in which case the empty tuple is the complete answer.
    """
    _require_pair(graph, x, y)
    cands = _candidates(graph, x, y, measured_only=measured_only)
    if len(cands) > max_candidates:
        if _canonical_s_admissible(graph, x, y, measured_only=measured_only) is None:
            return ()
        return Unsupported(
            reason=(
                f"{len(cands)} candidate nodes exceed max_candidates={max_candidates}; "
                "exhaustive S-admissible search is not attempted"
            ),
            detail={"candidates": ",".join(cands)},
            missing=("bounded_s_admissible_search",),
        )
    if not graph.selection:
        found = [frozenset(c) for r in range(len(cands) + 1) for c in combinations(cands, r)]
        return tuple(sorted(found, key=_sort_key))
    bar_x = _diagram_bar_x(graph, x)
    s = _s_nodes(graph)
    out: list[frozenset[str]] = []
    for r in range(len(cands) + 1):
        for combo in combinations(cands, r):
            if bar_x.d_separated(y, s, (x, *combo)):
                out.append(frozenset(combo))
    return tuple(sorted(out, key=_sort_key))


def minimal_s_admissible_sets(
    graph: CausalGraph,
    x: str,
    y: str,
    *,
    measured_only: bool = True,
    max_candidates: int = 14,
) -> tuple[frozenset[str], ...] | Unsupported:
    """The S-admissible sets with no S-admissible proper subset, smallest first."""
    all_sets = s_admissible_sets(
        graph, x, y, measured_only=measured_only, max_candidates=max_candidates
    )
    if isinstance(all_sets, Unsupported):
        return all_sets
    minimal: list[frozenset[str]] = []
    for s in all_sets:  # already sorted by size, so subsets precede supersets
        if not any(m < s for m in minimal):
            minimal.append(s)
    return tuple(minimal)


# -- verdict -------------------------------------------------------------------------


class TransportVerdict(Spec):
    """Whether, and by what formula, an effect transports from source to target.

    ``verdict`` carries the shared status vocabulary; ``route`` is one of
    ``same_population``, ``direct``, ``s_admissible_adjustment``,
    ``trivial``, ``s_admissible_adjustment_unmeasured``, or ``""`` when
    blocked or unsupported. ``s_admissible_set`` is the ``Z`` of the
    transport formula (empty for direct transport) or, on the ``trivial``
    route, the target-only back-door set; ``mediators`` is the front-door set
    of a ``trivial`` route. ``given`` echoes a caller-proposed set (``None``
    when the verdict searched). ``selection`` echoes the graph's S-node
    targets; ``graph_hash`` ties the verdict to the graph it was read from;
    ``formula`` is the human-readable formula, empty when none is licensed;
    ``missing`` names the capability an ``unsupported`` verdict lacks.
    """

    treatment: NonEmptyStr
    outcome: NonEmptyStr
    verdict: Verdict
    s_admissible_set: tuple[str, ...] = ()
    mediators: tuple[str, ...] = ()
    given: tuple[str, ...] | None = None
    selection: tuple[str, ...] = ()
    graph_hash: str = ""
    formula: str = ""
    missing: tuple[str, ...] = ()

    @property
    def status(self) -> str:
        return self.verdict.status

    @property
    def route(self) -> str:
        return self.verdict.route

    @property
    def licensed(self) -> bool:
        return self.verdict.licensed


def _formula(x: str, y: str, z: Iterable[str]) -> str:
    """The transport formula of Thm. 2; with ``Z = ∅`` it is direct transport."""
    zs = ",".join(sorted(z))
    if not zs:
        return f"P*({y}|do({x})) = P({y}|do({x}))"
    return f"P*({y}|do({x})) = Σ_{{{zs}}} P({y}|do({x}),{zs}) P*({zs})"


def _target_backdoor_formula(x: str, y: str, z: Iterable[str]) -> str:
    zs = ",".join(sorted(z))
    if not zs:
        return f"P*({y}|do({x})) = P*({y}|{x})"
    return f"P*({y}|do({x})) = Σ_{{{zs}}} P*({y}|{x},{zs}) P*({zs})"


def _target_frontdoor_formula(x: str, y: str, m: Iterable[str]) -> str:
    ms = ",".join(sorted(m))
    return f"P*({y}|do({x})) = Σ_{{{ms}}} P*({ms}|{x}) Σ_{{{x}'}} P*({y}|{x}',{ms}) P*({x}')"


def _s_admissibility_assumption(x: str, y: str, z: tuple[str, ...]) -> Assumption:
    return Assumption(
        name="s_admissibility",
        facet="population",
        statement=(
            f"Z={list(z)} is S-admissible for the effect of {x} on {y}: "
            f"({y} ⊥ S | {x}, Z) in G_bar({x}) of the selection diagram, so "
            f"{_formula(x, y, z)}"
        ),
        challenged_by="a mechanism difference between the populations the diagram omits",
        state="satisfied",
        detail={"set": ",".join(z)},
    )


def _trivial_assumption(x: str, y: str, route: TrivialRoute, formula: str) -> Assumption:
    kind, nodes = route
    how = (
        f"back-door adjustment for Z={list(nodes)}"
        if kind == "backdoor"
        else f"the front-door criterion through M={list(nodes)}"
    )
    return Assumption(
        name="trivial_transportability",
        facet="population",
        statement=(
            f"P*({y}|do({x})) is identifiable from the target population alone by {how} "
            f"(Bareinboim & Pearl 2014 Def. 6); no source data are used: {formula}"
        ),
        challenged_by="a confounder or mediator the target graph omits",
        state="satisfied",
        detail={"route": kind, "set": ",".join(nodes)},
    )


def _unmeasured_assumption(z: tuple[str, ...], unmeasured: tuple[str, ...]) -> Assumption:
    return Assumption(
        name="unmeasured_s_admissible_set",
        facet="population",
        statement=(
            f"the only S-admissible sets need unmeasured nodes; Z={list(z)} requires "
            f"{list(unmeasured)} observed in both populations"
        ),
        challenged_by="the unmeasured nodes stay unrecorded",
        state="unverified",
        detail={"set": ",".join(z), "unmeasured": ",".join(unmeasured)},
    )


def _feedback_assumption() -> Assumption:
    return Assumption(
        name="no_time_varying_confounding",
        facet="identification",
        statement=(
            "the static summary graph is adequate despite treatment–outcome feedback over time; "
            "the transport formula is applied to a single-period effect"
        ),
        challenged_by="carryover in the treatment; g-methods or an unrolled graph are needed",
        state="unverified",
    )


def _with_feedback(graph: CausalGraph, verdict: Verdict) -> Verdict:
    """Review B4: a licensed verdict on a feedback graph is downgraded, never identified."""
    if not graph.feedback or not verdict.licensed:
        return verdict
    return Verdict(
        status="downgraded",
        reason=verdict.reason,
        assumptions=(*verdict.assumptions, _feedback_assumption()),
        route=verdict.route,
    )


def _connected_s_nodes(
    graph: CausalGraph, x: str, y: str, z: Iterable[str] = ()
) -> tuple[str, ...]:
    """The S-nodes still d-connected to ``y`` given ``{x} ∪ Z`` in ``G_{\\bar X}``."""
    bar_x = _diagram_bar_x(graph, x)
    given = (x, *sorted(_node_set(z)))
    return tuple(s for s in _s_nodes(graph) if not bar_x.d_separated(y, s, given))


def _overflow_note(failure: Unsupported, what: str) -> str:
    return (
        f"the {what} was not attempted ({failure.reason}); "
        "the set named is inclusion-minimal but may not be the smallest"
    )


def _s_admissible_verdict(
    graph: CausalGraph, x: str, y: str, z: tuple[str, ...], *, note: str = ""
) -> Verdict:
    """``identified`` for a measured S-admissible ``z``; ``downgraded`` if it needs unmeasured."""
    hidden = tuple(n for n in z if n in graph.unmeasured)
    if not hidden:
        return Verdict(
            status="identified",
            reason=note,
            assumptions=(_s_admissibility_assumption(x, y, z),),
            route="s_admissible_adjustment",
        )
    return Verdict(
        status="downgraded",
        reason=note,
        assumptions=(_s_admissibility_assumption(x, y, z), _unmeasured_assumption(z, hidden)),
        route="s_admissible_adjustment_unmeasured",
    )


def _search_s_admissible(
    graph: CausalGraph, x: str, y: str, *, measured_only: bool, max_candidates: int
) -> tuple[tuple[str, ...], str] | None:
    """The smallest S-admissible set and an overflow note (empty when the search ran)."""
    sets = s_admissible_sets(
        graph, x, y, measured_only=measured_only, max_candidates=max_candidates
    )
    if isinstance(sets, Unsupported):
        canonical = _canonical_s_admissible(graph, x, y, measured_only=measured_only)
        if canonical is None:  # unreachable: s_admissible_sets returns () in that case
            return None
        which = "measured" if measured_only else "unmeasured"
        return tuple(sorted(canonical)), _overflow_note(sets, f"{which} S-admissible search")
    if not sets:
        return None
    return tuple(sorted(sets[0])), ""


def transport_verdict(
    graph: CausalGraph,
    x: str,
    y: str,
    *,
    given: Iterable[str] | None = None,
    max_candidates: int = 14,
) -> TransportVerdict:
    """Read the transportability of ``P*(y | do(x))`` off ``graph``'s selection diagram.

    With ``given=None`` the verdict searches, and is, in order of preference:

    1. ``identified`` / ``same_population`` — no selection nodes, so the
       populations are declared identical and ``P(y | do(x))`` is the answer.
    2. ``identified`` / ``direct`` — :func:`directly_transportable` (Def. 7).
    3. ``identified`` / ``s_admissible_adjustment`` — the smallest measured
       S-admissible ``Z`` exists (Def. 8, Thm. 2). The ``s_admissibility``
       assumption is ``satisfied`` because the graph established it;
       positivity of ``Z`` in the target is a data diagnostic, not a graph
       assumption, and is not listed here.
    4. ``identified`` / ``trivial`` — the target alone identifies the effect
       (Def. 6) by a measured back-door or front-door set; the formula uses
       target data only.
    5. ``downgraded`` / ``s_admissible_adjustment_unmeasured`` — an
       S-admissible ``Z`` exists only using ``unmeasured`` nodes; the
       assumption names them and is ``unverified``.
    6. ``unsupported`` — no sufficient condition holds. The reason names the
       S-nodes that stay d-connected to ``y`` and the gap: the complete sID
       recursion (Thm. 3) is not implemented, so this is *not* a claim that
       the effect fails to transport. ``missing`` carries ``sid_recursion``.

    With ``given`` a set of node names (the empty set included) the verdict
    checks exactly that set: ``identified`` with route
    ``s_admissible_adjustment`` when it is S-admissible (``downgraded`` on
    the unmeasured route when it contains unmeasured nodes; ``same_population``
    when the graph has no selection nodes), otherwise ``blocked`` with a
    reason naming the descendants of ``x`` in the set or the S-nodes still
    d-connected to ``y`` given ``{x} ∪ Z``. A set containing ``x`` or ``y``
    is a malformed query and raises ``GraphError``.

    ``max_candidates`` bounds every exhaustive enumeration. When an
    S-admissible enumeration overflows, the polynomial existence test still
    decides the route and an inclusion-minimal set is named, with the reason
    saying the smallest set was not searched for. On a graph flagged
    ``feedback=True`` every licensed verdict is downgraded with the
    ``no_time_varying_confounding`` assumption (review B4).
    """
    _require_pair(graph, x, y)
    graph_hash = graph.content_hash()
    proposed = None if given is None else tuple(sorted(_node_set(given)))

    def make(
        v: Verdict,
        *,
        z: tuple[str, ...] = (),
        mediators: tuple[str, ...] = (),
        formula: str = "",
        missing: tuple[str, ...] = (),
    ) -> TransportVerdict:
        return TransportVerdict(
            treatment=x,
            outcome=y,
            verdict=_with_feedback(graph, v),
            s_admissible_set=z,
            mediators=mediators,
            given=proposed,
            selection=graph.selection,
            graph_hash=graph_hash,
            formula=formula,
            missing=missing,
        )

    if proposed is not None:
        v, z, formula = _check_proposed(graph, x, y, proposed)
        return make(v, z=z, formula=formula)

    if not graph.selection:
        v = Verdict(status="identified", route="same_population")
        return make(v, formula=f"P({y}|do({x}))")
    if directly_transportable(graph, x, y):
        v = Verdict(status="identified", route="direct")
        return make(v, formula=_formula(x, y, ()))
    measured = _search_s_admissible(graph, x, y, measured_only=True, max_candidates=max_candidates)
    if measured is not None:
        z, note = measured
        return make(
            _s_admissible_verdict(graph, x, y, z, note=note), z=z, formula=_formula(x, y, z)
        )
    trivial = _trivial_route(graph, x, y, max_candidates=max_candidates)
    if isinstance(trivial, tuple):
        kind, nodes = trivial
        formula = (
            _target_backdoor_formula(x, y, nodes)
            if kind == "backdoor"
            else _target_frontdoor_formula(x, y, nodes)
        )
        v = Verdict(
            status="identified",
            assumptions=(_trivial_assumption(x, y, trivial, formula),),
            route="trivial",
        )
        if kind == "backdoor":
            return make(v, z=nodes, formula=formula)
        return make(v, mediators=nodes, formula=formula)
    unmeasured = _search_s_admissible(
        graph, x, y, measured_only=False, max_candidates=max_candidates
    )
    if unmeasured is not None:
        z, note = unmeasured
        return make(
            _s_admissible_verdict(graph, x, y, z, note=note), z=z, formula=_formula(x, y, z)
        )
    connected = _connected_s_nodes(graph, x, y)
    reason = (
        f"selection nodes {list(connected)} stay d-connected to {y} given {x} in G_bar({x}) "
        f"and the target graph alone has no measured back-door or front-door set; {_SID_GAP}"
    )
    gaps: tuple[str, ...] = ("sid_recursion",)
    if isinstance(trivial, Unsupported):
        reason += f"; the target front-door search was not attempted ({trivial.reason})"
        gaps = (*gaps, "bounded_frontdoor_search")
    return make(Verdict(status="unsupported", reason=reason), missing=gaps)


def _check_proposed(
    graph: CausalGraph, x: str, y: str, z: tuple[str, ...]
) -> tuple[Verdict, tuple[str, ...], str]:
    """Verdict for a caller-proposed set: ``identified`` if S-admissible, else ``blocked``.

    Returns the verdict, the set to record, and the formula (empty when blocked).
    """
    _require(graph, *z)
    if x in z or y in z:
        raise GraphError(
            f"a proposed set may not contain the treatment {x!r} or the outcome {y!r}; "
            f"got {list(z)}"
        )
    descendants = tuple(n for n in z if n in graph.descendants(x))
    if descendants:
        v = Verdict(
            status="blocked",
            reason=(
                f"proposed set Z={list(z)} is not S-admissible: {list(descendants)} are "
                f"descendants of the treatment {x} (Bareinboim & Pearl 2014 Def. 8(i))"
            ),
        )
        return v, (), ""
    if not graph.selection:
        return Verdict(status="identified", route="same_population"), z, f"P({y}|do({x}))"
    connected = _connected_s_nodes(graph, x, y, z)
    if connected:
        v = Verdict(
            status="blocked",
            reason=(
                f"proposed set Z={list(z)} is not S-admissible: selection nodes "
                f"{list(connected)} stay d-connected to {y} given {x} and Z in G_bar({x})"
            ),
        )
        return v, (), ""
    return _s_admissible_verdict(graph, x, y, z), z, _formula(x, y, z)
