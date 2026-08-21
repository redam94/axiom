"""``CausalGraph``: a DAG as a ``Spec``, with ancestry and d-separation.

Own adjacency implementation; ``networkx`` is not a dependency. Latent
confounding is expressed as bidirected edges ``A <-> B`` and handled
internally by a hidden common cause, so every query works on a
semi-Markovian graph without the caller seeing the latent nodes.

Two honesty fields (review B4): ``unmeasured`` names nodes the panel does
not observe, so an adjustment set that needs one is a *downgrade*; and
``feedback`` declares that the summary graph hides treatment–outcome
feedback over time, in which case static adjustment is insufficient and the
verdict says so.

The ``selection`` field lists nodes whose mechanism differs between the
study and target populations — the S-nodes of a selection diagram. The graph
is then both the causal diagram and the selection diagram; ``transport``
reads ``selection``.
"""

from __future__ import annotations

import re
from collections import deque
from collections.abc import Iterable, Iterator
from typing import Literal

from pydantic import field_validator, model_validator

from axiom.core.spec import Spec, SpecError

__all__ = ["CausalGraph", "Edge", "GraphError", "parse_edges"]

Edge = tuple[str, str]
_EDGE_RE = re.compile(r"^\s*([A-Za-z_][\w.\-]*)\s*(->|<->|<-)\s*([A-Za-z_][\w.\-]*)\s*$")


class GraphError(SpecError):
    """The graph is malformed (cycle, unknown node, self-edge)."""


def parse_edges(text: str) -> tuple[tuple[Edge, ...], tuple[Edge, ...]]:
    """Parse ``"Z -> X, Z -> Y, X -> Y, A <-> B"`` into (directed, bidirected) edge tuples.

    Separators are commas, semicolons, or newlines. ``A <- B`` means ``B -> A``.
    """
    directed: list[Edge] = []
    bidirected: list[Edge] = []
    for chunk in re.split(r"[,;\n]", text):
        if not chunk.strip():
            continue
        m = _EDGE_RE.match(chunk)
        if not m:
            raise GraphError(
                f"cannot parse edge {chunk.strip()!r}; expected 'A -> B', 'A <- B', or 'A <-> B'"
            )
        a, op, b = m.groups()
        if op == "->":
            directed.append((a, b))
        elif op == "<-":
            directed.append((b, a))
        else:
            bidirected.append((a, b))
    return tuple(directed), tuple(bidirected)


def _latent_name(a: str, b: str) -> str:
    x, y = sorted((a, b))
    return f"U[{x},{y}]"


class CausalGraph(Spec):
    """A directed acyclic graph over named nodes, with optional bidirected edges.

    ``nodes`` may be omitted when every node appears in an edge; isolated
    nodes must be listed. Edge order is not significant: edges are sorted
    on construction so equal graphs hash equal.
    """

    nodes: tuple[str, ...] = ()
    edges: tuple[Edge, ...] = ()
    bidirected: tuple[Edge, ...] = ()
    unmeasured: tuple[str, ...] = ()
    selection: tuple[str, ...] = ()
    feedback: bool = False
    name: str = ""

    # -- construction -----------------------------------------------------------

    @classmethod
    def from_edges(
        cls,
        text: str,
        *,
        nodes: Iterable[str] = (),
        unmeasured: Iterable[str] = (),
        selection: Iterable[str] = (),
        feedback: bool = False,
        name: str = "",
    ) -> CausalGraph:
        directed, bi = parse_edges(text)
        return cls(
            nodes=tuple(nodes),
            edges=directed,
            bidirected=bi,
            unmeasured=tuple(unmeasured),
            selection=tuple(selection),
            feedback=feedback,
            name=name,
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

    @field_validator("unmeasured", "selection", "nodes", mode="before")
    @classmethod
    def _norm_names(cls, v: Iterable[str]) -> tuple[str, ...]:
        return tuple(sorted({str(x) for x in v}))

    @model_validator(mode="after")
    def _well_formed(self) -> CausalGraph:
        mentioned = {n for e in (*self.edges, *self.bidirected) for n in e}
        all_nodes = tuple(sorted(mentioned | set(self.nodes)))
        if all_nodes != self.nodes:
            object.__setattr__(self, "nodes", all_nodes)
        for label, names in (("unmeasured", self.unmeasured), ("selection", self.selection)):
            unknown = sorted(set(names) - set(self.nodes))
            if unknown:
                raise GraphError(f"{label} names unknown nodes: {unknown}")
        cycle = self._find_cycle()
        if cycle:
            raise GraphError("graph has a cycle: " + " -> ".join(cycle))
        return self

    # -- adjacency ----------------------------------------------------------------

    def parents(self, node: str) -> frozenset[str]:
        self._require(node)
        return frozenset(a for a, b in self.edges if b == node)

    def children(self, node: str) -> frozenset[str]:
        self._require(node)
        return frozenset(b for a, b in self.edges if a == node)

    def siblings(self, node: str) -> frozenset[str]:
        """Nodes sharing a bidirected edge (a latent common cause) with ``node``."""
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

    def descendants(
        self, nodes: str | Iterable[str], *, include_self: bool = False
    ) -> frozenset[str]:
        start = {nodes} if isinstance(nodes, str) else set(nodes)
        seen: set[str] = set()
        stack = list(start)
        while stack:
            n = stack.pop()
            for c in self.children(n):
                if c not in seen:
                    seen.add(c)
                    stack.append(c)
        return frozenset(seen | start) if include_self else frozenset(seen)

    def is_ancestor(self, a: str, b: str) -> bool:
        return a in self.ancestors(b)

    def topological_order(self) -> tuple[str, ...]:
        indeg = {n: len(self.parents(n)) for n in self.nodes}
        ready = sorted(n for n, d in indeg.items() if d == 0)
        out: list[str] = []
        while ready:
            n = ready.pop(0)
            out.append(n)
            for c in sorted(self.children(n)):
                indeg[c] -= 1
                if indeg[c] == 0:
                    ready.append(c)
                    ready.sort()
        return tuple(out)

    @property
    def measured(self) -> frozenset[str]:
        return frozenset(self.nodes) - frozenset(self.unmeasured)

    def is_measured(self, node: str) -> bool:
        self._require(node)
        return node not in self.unmeasured

    def directed_paths(self, source: str, target: str) -> tuple[tuple[str, ...], ...]:
        """All directed paths ``source -> ... -> target``, in lexicographic order."""
        self._require(source)
        self._require(target)
        out: list[tuple[str, ...]] = []

        def go(path: list[str]) -> None:
            if path[-1] == target:
                out.append(tuple(path))
                return
            for c in sorted(self.children(path[-1])):
                go([*path, c])

        go([source])
        return tuple(out)

    # -- d-separation ---------------------------------------------------------------

    def d_separated(
        self, x: str | Iterable[str], y: str | Iterable[str], z: Iterable[str] = ()
    ) -> bool:
        """Whether ``X`` and ``Y`` are d-separated given ``Z`` (Bayes-ball on the augmented DAG)."""
        xs = {x} if isinstance(x, str) else set(x)
        ys = {y} if isinstance(y, str) else set(y)
        zs = set(z)
        for n in xs | ys | zs:
            self._require(n)
        if xs & ys:
            return False
        if (xs & zs) or (ys & zs):
            # conditioning on a variable separates it from everything; by convention
            # treat X∩Z or Y∩Z as separated (the conditioned variable carries no info)
            xs, ys = xs - zs, ys - zs
            if not xs or not ys:
                return True
        return not (self._reachable(xs, zs) & ys)

    def _reachable(self, sources: set[str], given: set[str]) -> set[str]:
        """Nodes d-connected to ``sources`` given ``given`` (Koller & Friedman, Alg. 3.1)."""
        parents, children, latents = self._augmented()
        anc_z: set[str] = set(given)
        stack = list(given)
        while stack:
            n = stack.pop()
            for p in parents[n]:
                if p not in anc_z:
                    anc_z.add(p)
                    stack.append(p)
        visited: set[tuple[str, Literal["up", "down"]]] = set()
        reached: set[str] = set()
        todo: deque[tuple[str, Literal["up", "down"]]] = deque((s, "up") for s in sources)
        while todo:
            node, direction = todo.popleft()
            if (node, direction) in visited:
                continue
            visited.add((node, direction))
            if node not in given:
                reached.add(node)
            if direction == "up" and node not in given:
                for p in parents[node]:
                    todo.append((p, "up"))
                for c in children[node]:
                    todo.append((c, "down"))
            elif direction == "down":
                if node not in given:
                    for c in children[node]:
                        todo.append((c, "down"))
                if node in anc_z:
                    for p in parents[node]:
                        todo.append((p, "up"))
        return (reached - latents) - sources

    def _augmented(self) -> tuple[dict[str, set[str]], dict[str, set[str]], set[str]]:
        """Parents/children over the graph plus one hidden common cause per bidirected edge."""
        parents: dict[str, set[str]] = {n: set() for n in self.nodes}
        children: dict[str, set[str]] = {n: set() for n in self.nodes}
        latents: set[str] = set()
        for a, b in self.edges:
            parents[b].add(a)
            children[a].add(b)
        for a, b in self.bidirected:
            u = _latent_name(a, b)
            while u in parents and u not in latents:  # a user node happens to share the name
                u = "_" + u
            latents.add(u)
            parents.setdefault(u, set())
            children.setdefault(u, set()).update({a, b})
            parents[a].add(u)
            parents[b].add(u)
        return parents, children, latents

    # -- helpers --------------------------------------------------------------------

    def with_selection(self, *nodes: str) -> CausalGraph:
        """The selection diagram: this graph with S-nodes pointing into ``nodes``."""
        return self.model_copy(
            update={"selection": tuple(sorted(set(self.selection) | set(nodes)))}
        )

    def with_unmeasured(self, *nodes: str) -> CausalGraph:
        return self.model_copy(
            update={"unmeasured": tuple(sorted(set(self.unmeasured) | set(nodes)))}
        )

    def remove_edges_into(self, node: str) -> CausalGraph:
        """``G_{\\bar{X}}``: the graph with all edges into ``node`` removed."""
        self._require(node)
        return self.model_copy(
            update={
                "edges": tuple(e for e in self.edges if e[1] != node),
                "bidirected": tuple(e for e in self.bidirected if node not in e),
                "nodes": self.nodes,
            }
        )

    def remove_edges_out_of(self, node: str) -> CausalGraph:
        """``G_{\\underline{X}}``: the graph with all edges out of ``node`` removed."""
        self._require(node)
        return self.model_copy(
            update={"edges": tuple(e for e in self.edges if e[0] != node), "nodes": self.nodes}
        )

    def _require(self, node: str) -> None:
        if node not in self.nodes:
            raise GraphError(f"unknown node {node!r}; nodes are {list(self.nodes)}")

    def _find_cycle(self) -> tuple[str, ...]:
        color: dict[str, int] = {n: 0 for n in self.nodes}
        parent: dict[str, str | None] = {n: None for n in self.nodes}
        for root in self.nodes:
            if color[root]:
                continue
            stack: list[tuple[str, Iterator[str]]] = [(root, iter(sorted(self.children(root))))]
            color[root] = 1
            while stack:
                node, it = stack[-1]
                nxt = next(it, None)
                if nxt is None:
                    color[node] = 2
                    stack.pop()
                    continue
                if color[nxt] == 1:
                    cyc = [nxt, node]
                    while cyc[-1] != nxt:
                        cyc.append(parent[cyc[-1]] or nxt)
                    return tuple(reversed(cyc))
                if color[nxt] == 0:
                    color[nxt] = 1
                    parent[nxt] = node
                    stack.append((nxt, iter(sorted(self.children(nxt)))))
        return ()

    def to_text(self) -> str:
        parts = [f"{a} -> {b}" for a, b in self.edges] + [
            f"{a} <-> {b}" for a, b in self.bidirected
        ]
        return ", ".join(parts)

    def __str__(self) -> str:
        return self.to_text() or f"CausalGraph(nodes={list(self.nodes)})"
