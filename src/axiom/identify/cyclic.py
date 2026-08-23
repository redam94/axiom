"""Graphs with cycles: sigma-separation, and the acyclification that makes it computable.

``CausalGraph`` refuses a cycle, and it is right to: **d-separation is unsound
on a cyclic graph.** Conditioning on a variable inside a feedback loop does not
block the flow of information around the loop, because the loop's members are
functionally intertwined — so a d-separation claim read off a cyclic diagram
can simply be false.

The repair is not to forbid the diagram but to change the separation criterion.
Forré and Mooij (2017) introduced **sigma-separation**, which is sound for the
global Markov property of structural causal models with cycles. It differs from
d-separation in exactly one place, and the difference is the whole idea:

    a non-collider blocks a walk only if it *points to a node in a different
    strongly connected component*.

A non-collider whose walk edges all point inside its own feedback loop cannot
block, however hard you condition on it. On an acyclic graph every strongly
connected component is a singleton, so every non-collider points outside its
own component and sigma-separation collapses to d-separation — the acyclic case
is not bolted on, it falls out.

Two implementations, deliberately
---------------------------------
This module computes sigma-separation twice.

* ``sigma_separated`` walks the graph directly, applying the rule above to
  each triple.
* ``acyclify`` builds the **acyclification** (Mooij & Claassen 2020, Def. 3):
  each strongly connected component becomes a bidirected clique, and every
  parent of a component points at *every* member of it. Their Proposition 2
  gives the equivalence — sigma-separation in ``G`` is d-separation in the
  acyclification of ``G`` — so ``CausalGraph.d_separated`` on the
  acyclification must agree with the direct walk.

Keeping both is not redundancy for its own sake. The rule is subtle enough that
one implementation is a hypothesis; two written from different definitions that
agree over tens of thousands of random queries is evidence
(``tests/unit/test_identify_cyclic.py``).

The acyclification is also what a simultaneous system's *reduced form* wanted
to be all along: solve a block and every member depends on the block's parents
and shares the block's disturbances — shared parents, **and** a bidirected edge
between the members. ``identify.unrolled_graph`` builds it that way.
"""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import field_validator, model_validator

from axiom.core.spec import Spec
from axiom.dynamics import strongly_connected_components
from axiom.identify.graph import CausalGraph, Edge, GraphError, parse_edges

__all__ = ["MixedGraph", "acyclify", "sigma_separated"]


class MixedGraph(Spec):
    """A directed mixed graph. Directed edges may form cycles; separation is sigma.

    The same content as ``CausalGraph`` — directed edges, bidirected edges for
    latent common causes, and unmeasured node names — with the acyclicity
    requirement dropped. Everything that reads a ``MixedGraph`` either uses
    sigma-separation or acyclifies first; nothing applies d-separation to it,
    because that would be unsound.
    """

    nodes: tuple[str, ...] = ()
    edges: tuple[Edge, ...] = ()
    bidirected: tuple[Edge, ...] = ()
    unmeasured: tuple[str, ...] = ()
    name: str = ""

    @classmethod
    def from_edges(
        cls,
        text: str,
        *,
        nodes: Iterable[str] = (),
        unmeasured: Iterable[str] = (),
        name: str = "",
    ) -> MixedGraph:
        """Parse ``"X -> Y, Y -> X, A <-> B"``; unlike ``CausalGraph`` the cycle is fine."""
        directed, bi = parse_edges(text)
        return cls(
            nodes=tuple(nodes),
            edges=directed,
            bidirected=bi,
            unmeasured=tuple(unmeasured),
            name=name,
        )

    @classmethod
    def from_causal_graph(cls, graph: CausalGraph) -> MixedGraph:
        """The same graph, read as a mixed graph. Every acyclic graph is one."""
        return cls(
            nodes=graph.nodes,
            edges=graph.edges,
            bidirected=graph.bidirected,
            unmeasured=graph.unmeasured,
            name=graph.name,
        )

    @field_validator("edges", mode="before")
    @classmethod
    def _norm_edges(cls, v: Iterable[Iterable[str]]) -> tuple[Edge, ...]:
        out = {(str(a), str(b)) for a, b in v}
        for a, b in out:
            if a == b:
                raise GraphError(f"self-edge {a} -> {a}")
        return tuple(sorted(out))

    @field_validator("bidirected", mode="before")
    @classmethod
    def _norm_bidirected(cls, v: Iterable[Iterable[str]]) -> tuple[Edge, ...]:
        out: set[Edge] = set()
        for a, b in v:
            a, b = str(a), str(b)
            if a == b:
                raise GraphError(f"self-edge {a} <-> {a}")
            out.add((min(a, b), max(a, b)))
        return tuple(sorted(out))

    @field_validator("unmeasured", "nodes", mode="before")
    @classmethod
    def _norm_names(cls, v: Iterable[str]) -> tuple[str, ...]:
        return tuple(sorted({str(x) for x in v}))

    @model_validator(mode="after")
    def _well_formed(self) -> MixedGraph:
        mentioned = {n for e in (*self.edges, *self.bidirected) for n in e}
        all_nodes = tuple(sorted(mentioned | set(self.nodes)))
        if all_nodes != self.nodes:
            object.__setattr__(self, "nodes", all_nodes)
        unknown = sorted(set(self.unmeasured) - set(self.nodes))
        if unknown:
            raise GraphError(f"unmeasured names unknown nodes: {unknown}")
        return self

    # -- adjacency ----------------------------------------------------------------

    def parents(self, node: str) -> frozenset[str]:
        self._require(node)
        return frozenset(a for a, b in self.edges if b == node)

    def children(self, node: str) -> frozenset[str]:
        self._require(node)
        return frozenset(b for a, b in self.edges if a == node)

    def siblings(self, node: str) -> frozenset[str]:
        self._require(node)
        return frozenset(b if a == node else a for a, b in self.bidirected if node in (a, b))

    def ancestors(
        self, nodes: str | Iterable[str], *, include_self: bool = False
    ) -> frozenset[str]:
        start = {nodes} if isinstance(nodes, str) else set(nodes)
        seen: set[str] = set()
        stack = list(start)
        while stack:
            n = stack.pop()
            for p in self.parents(n):
                if p not in seen:
                    seen.add(p)
                    stack.append(p)
        return frozenset(seen | start) if include_self else frozenset(seen)

    @property
    def measured(self) -> frozenset[str]:
        return frozenset(self.nodes) - frozenset(self.unmeasured)

    # -- components ---------------------------------------------------------------

    def components(self) -> tuple[tuple[str, ...], ...]:
        """The strongly connected components, each sorted; singletons included."""
        return strongly_connected_components(self.nodes, self.edges)

    def component_of(self, node: str) -> frozenset[str]:
        """``sc(v)``: the strongly connected component containing ``node``, itself included."""
        self._require(node)
        for component in self.components():
            if node in component:
                return frozenset(component)
        raise GraphError(f"node {node!r} is in no component")  # pragma: no cover

    @property
    def is_acyclic(self) -> bool:
        """True when every strongly connected component is a singleton."""
        return all(len(c) == 1 for c in self.components())

    @property
    def cyclic_components(self) -> tuple[tuple[str, ...], ...]:
        """The components with more than one member: the feedback loops."""
        return tuple(c for c in self.components() if len(c) > 1)

    # -- separation ---------------------------------------------------------------

    def sigma_separated(
        self, x: str | Iterable[str], y: str | Iterable[str], z: Iterable[str] = ()
    ) -> bool:
        """Whether ``X`` and ``Y`` are sigma-separated given ``Z`` (Forré & Mooij 2017)."""
        return sigma_separated(self, x, y, z)

    def d_separated(
        self, x: str | Iterable[str], y: str | Iterable[str], z: Iterable[str] = ()
    ) -> bool:
        """Sigma-separation, under the familiar name.

        A caller who reaches for ``d_separated`` on a graph with a cycle wants
        an answer about that graph; d-separation is not a sound one, so this
        gives the criterion that is. On an acyclic graph the two coincide.
        """
        return sigma_separated(self, x, y, z)

    def acyclify(self) -> CausalGraph:
        """The acyclification: an acyclic graph whose d-separation is this graph's sigma."""
        return acyclify(self)

    # -- helpers ------------------------------------------------------------------

    def _require(self, node: str) -> None:
        if node not in self.nodes:
            raise GraphError(f"unknown node {node!r}; nodes are {list(self.nodes)}")

    def to_text(self) -> str:
        parts = [f"{a} -> {b}" for a, b in self.edges] + [
            f"{a} <-> {b}" for a, b in self.bidirected
        ]
        return ", ".join(parts)

    def __str__(self) -> str:
        return self.to_text() or f"MixedGraph(nodes={list(self.nodes)})"


def acyclify(graph: MixedGraph | CausalGraph) -> CausalGraph:
    """The acyclification of a directed mixed graph (Mooij & Claassen 2020, Def. 3).

    Every strongly connected component collapses into a bidirected clique, and
    every parent of a component points at every member of it:

    * ``i -> j`` when ``i`` lies outside ``sc(j)`` and ``i -> j'`` for some
      ``j'`` in ``sc(j)``;
    * ``i <-> j`` when ``i != j`` and either they share a component, or some
      member of ``sc(i)`` has a bidirected edge to some member of ``sc(j)``.

    The result is acyclic, and ``d_separated`` on it is ``sigma_separated`` on
    the original (their Proposition 2). Read causally it is the reduced form:
    the variables in a loop are determined together by everything feeding the
    loop, and share whatever disturbs it.
    """
    mixed = MixedGraph.from_causal_graph(graph) if isinstance(graph, CausalGraph) else graph
    component = {n: mixed.component_of(n) for n in mixed.nodes}
    bidirected_pairs = set(mixed.bidirected)
    directed: set[Edge] = set()
    for node in mixed.nodes:
        for member in component[node]:
            for parent in mixed.parents(member):
                if parent not in component[node]:
                    directed.add((parent, node))
    bi: set[Edge] = set()
    for a in mixed.nodes:
        for b in mixed.nodes:
            if a >= b:
                continue
            if component[a] == component[b]:
                bi.add((a, b))
                continue
            if any(
                (min(u, v), max(u, v)) in bidirected_pairs
                for u in component[a]
                for v in component[b]
            ):
                bi.add((a, b))
    return CausalGraph(
        nodes=mixed.nodes,
        edges=tuple(sorted(directed)),
        bidirected=tuple(sorted(bi)),
        unmeasured=mixed.unmeasured,
        name=f"acyclified({mixed.name})" if mixed.name else "",
    )


# -- the direct walk ---------------------------------------------------------------------

_Arrival = tuple[int, str]
"""``(edge index, node)`` — the edge just traversed and which end we are standing on."""

_Incidence = tuple[int, str, bool, bool]
"""``(edge index, other end, arrowhead here, arrowhead there)``."""


def _incidence(graph: MixedGraph) -> dict[str, list[_Incidence]]:
    """Per node, the edges meeting it, and how each edge meets each end.

    A directed edge ``v -> w`` has a tail at ``v`` and an arrowhead at ``w``; a
    bidirected edge has an arrowhead at both. That is everything a separation
    walk needs to know about an edge, and indexing edges rather than pairs
    keeps a directed and a bidirected edge between the same two nodes apart.
    """
    out: dict[str, list[_Incidence]] = {n: [] for n in graph.nodes}
    index = 0
    for a, b in graph.edges:
        out[a].append((index, b, False, True))
        out[b].append((index, a, True, False))
        index += 1
    for a, b in graph.bidirected:
        out[a].append((index, b, True, True))
        out[b].append((index, a, True, True))
        index += 1
    return out


def _blockable(
    previous: str,
    arrowhead_from_previous: bool,
    next_node: str,
    arrowhead_towards_next: bool,
    component: frozenset[str],
) -> bool:
    """Whether a non-collider can be blocked: does it point outside its own component?

    A node points at a walk neighbour exactly when the edge to that neighbour
    carries a *tail* at the node. A non-collider always has at least one such
    neighbour, so with singleton components — an acyclic graph — this is always
    ``True`` and the criterion collapses to d-separation.
    """
    pointed_to = []
    if not arrowhead_from_previous:
        pointed_to.append(previous)
    if not arrowhead_towards_next:
        pointed_to.append(next_node)
    return any(neighbour not in component for neighbour in pointed_to)


def _walk_reachable(graph: MixedGraph, sources: set[str], given: set[str]) -> set[str]:
    """Nodes sigma-connected to ``sources`` given ``given``.

    A search over *arrivals* rather than nodes: the sigma rule for a
    non-collider looks at both of its walk edges, which a per-node state cannot
    see. Walks may reuse an edge — sigma-separation is defined over walks, not
    paths — so nothing here forbids stepping back the way it came; the arrival
    set is finite, so the search still terminates.
    """
    incident = _incidence(graph)
    ancestors_of_z = graph.ancestors(given, include_self=True) if given else frozenset()
    component = {n: graph.component_of(n) for n in graph.nodes}
    reached: set[str] = set(sources)
    seen: set[_Arrival] = set()
    queue: list[tuple[_Arrival, str, bool]] = []
    for source in sources:
        for index, other, _here, there in incident[source]:
            arrival = (index, other)
            if arrival not in seen:
                seen.add(arrival)
                queue.append((arrival, source, there))
    while queue:
        (_index, at), previous, arrowhead_here = queue.pop()
        if at not in given:
            reached.add(at)
        for next_index, next_node, tail_end_here, arrowhead_there in incident[at]:
            collider = arrowhead_here and tail_end_here
            if collider:
                if at not in ancestors_of_z:
                    continue
            elif at in given and _blockable(
                previous, arrowhead_here, next_node, tail_end_here, component[at]
            ):
                continue
            arrival = (next_index, next_node)
            if arrival not in seen:
                seen.add(arrival)
                queue.append((arrival, at, arrowhead_there))
    return reached - sources


def sigma_separated(
    graph: MixedGraph | CausalGraph,
    x: str | Iterable[str],
    y: str | Iterable[str],
    z: Iterable[str] = (),
) -> bool:
    """Sigma-separation of ``X`` and ``Y`` given ``Z`` (Forré & Mooij 2017).

    Sound for the global Markov property of structural causal models with
    cycles, and identical to d-separation when the graph is acyclic. A node in
    both ``X`` and ``Z`` (or ``Y`` and ``Z``) counts as separated, matching
    ``CausalGraph.d_separated``'s convention.
    """
    mixed = MixedGraph.from_causal_graph(graph) if isinstance(graph, CausalGraph) else graph
    xs = {x} if isinstance(x, str) else set(x)
    ys = {y} if isinstance(y, str) else set(y)
    zs = set(z)
    for n in xs | ys | zs:
        mixed._require(n)
    if xs & ys:
        return False
    if (xs & zs) or (ys & zs):
        xs, ys = xs - zs, ys - zs
        if not xs or not ys:
            return True
    return not (_walk_reachable(mixed, xs, zs) & ys)
