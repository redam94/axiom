"""Essential graphs: which edges the data could orient, and which need an experiment.

Observational data cannot tell a DAG from its Markov equivalence class. Every
DAG in a class entails exactly the same conditional independencies, so no
amount of observation separates them; what is identified is the class, and the
**essential graph** (or CPDAG) is its canonical representative — a directed
edge where every member agrees, an undirected edge where they do not.

Interventions break ties observation cannot. Randomizing a variable severs its
incoming edges, so an edge with exactly one endpoint in the intervention target
has its direction revealed. Hauser & Bühlmann (2012) formalize this as the
**I-essential graph**: the union of the DAGs that remain indistinguishable once
a family of intervention targets has been run.

That gives the design question a precise answer. `orientation_gain` says which
currently-undirected edges a proposed experiment would orient — before it is
run, and before it is paid for. An experiment that orients nothing is one you
already know the answer to.

Everything here is checked against enumeration rather than cited: for small
graphs the CPDAG this module computes is compared with the union of *every*
DAG that entails the same independencies, tested one d-separation query at a
time (`tests/unit/test_discover_essential.py`). Meek's rules are a closure
algorithm and the enumeration is a definition; agreeing is the evidence.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterable, Sequence

from pydantic import model_validator

from axiom.core.spec import Spec
from axiom.identify.graph import CausalGraph, Edge, GraphError

__all__ = [
    "EssentialGraph",
    "consistent_extension",
    "consistent_extensions",
    "cpdag",
    "interventional_essential_graph",
    "markov_equivalent",
    "meek_closure",
    "orientation_gain",
    "v_structures",
]


def _pair(a: str, b: str) -> Edge:
    return (a, b) if a <= b else (b, a)


class EssentialGraph(Spec):
    """A partially directed graph: compelled edges directed, the rest undirected.

    ``directed`` holds edges every member of the equivalence class agrees on;
    ``undirected`` holds pairs whose orientation the data cannot settle, stored
    as sorted pairs so an equal graph hashes equal. ``targets`` records the
    intervention family the graph is essential *for* — an empty family is the
    observational CPDAG.
    """

    nodes: tuple[str, ...] = ()
    directed: tuple[Edge, ...] = ()
    undirected: tuple[Edge, ...] = ()
    targets: tuple[tuple[str, ...], ...] = ()
    name: str = ""

    @model_validator(mode="after")
    def _well_formed(self) -> EssentialGraph:
        mentioned = {n for e in (*self.directed, *self.undirected) for n in e}
        nodes = tuple(sorted(mentioned | set(self.nodes)))
        if nodes != self.nodes:
            object.__setattr__(self, "nodes", nodes)
        directed = tuple(sorted(set(self.directed)))
        if directed != self.directed:
            object.__setattr__(self, "directed", directed)
        undirected = tuple(sorted({_pair(a, b) for a, b in self.undirected}))
        if undirected != self.undirected:
            object.__setattr__(self, "undirected", undirected)
        for a, b in (*self.directed, *self.undirected):
            if a == b:
                raise GraphError(f"self-edge at {a}")
        both = {_pair(a, b) for a, b in self.directed} & set(self.undirected)
        if both:
            raise GraphError(f"edges are directed or undirected, not both: {sorted(both)}")
        opposed = {(a, b) for a, b in self.directed if (b, a) in set(self.directed)}
        if opposed:
            raise GraphError(f"a directed edge cannot point both ways: {sorted(opposed)}")
        unknown = {n for t in self.targets for n in t} - set(self.nodes)
        if unknown:
            raise GraphError(f"intervention targets name unknown nodes: {sorted(unknown)}")
        return self

    # -- adjacency ----------------------------------------------------------------

    def parents(self, node: str) -> frozenset[str]:
        return frozenset(a for a, b in self.directed if b == node)

    def children(self, node: str) -> frozenset[str]:
        return frozenset(b for a, b in self.directed if a == node)

    def neighbours(self, node: str) -> frozenset[str]:
        """Nodes joined to this one by an *undirected* edge."""
        return frozenset(b if a == node else a for a, b in self.undirected if node in (a, b))

    def adjacent(self, node: str) -> frozenset[str]:
        return self.parents(node) | self.children(node) | self.neighbours(node)

    def is_adjacent(self, a: str, b: str) -> bool:
        return b in self.adjacent(a)

    @property
    def skeleton(self) -> tuple[Edge, ...]:
        return tuple(sorted({_pair(a, b) for a, b in (*self.directed, *self.undirected)}))

    @property
    def oriented(self) -> int:
        return len(self.directed)

    @property
    def undecided(self) -> int:
        return len(self.undirected)

    @property
    def is_dag(self) -> bool:
        """True when nothing is left undirected — the class has one member."""
        return not self.undirected

    def to_graph(self) -> CausalGraph:
        """The ``CausalGraph`` this is, when it is fully directed."""
        if not self.is_dag:
            raise GraphError(
                f"{self.undecided} edges are undirected; an essential graph is a class of "
                "DAGs, not one. Use consistent_extension for a representative"
            )
        return CausalGraph(nodes=self.nodes, edges=self.directed, name=self.name)

    def to_text(self) -> str:
        parts = [f"{a} -> {b}" for a, b in self.directed]
        parts += [f"{a} - {b}" for a, b in self.undirected]
        return ", ".join(parts)

    def __str__(self) -> str:
        return self.to_text() or f"EssentialGraph(nodes={list(self.nodes)})"


# -- structure -------------------------------------------------------------------------


def v_structures(graph: CausalGraph) -> frozenset[tuple[str, str, str]]:
    """``(a, c, b)`` with ``a -> c <- b`` and ``a``, ``b`` non-adjacent; ``a < b``.

    These, with the skeleton, are what Markov equivalence preserves
    (Verma & Pearl 1990) — and the reason observation can orient anything at
    all.
    """
    out: set[tuple[str, str, str]] = set()
    for node in graph.nodes:
        parents = sorted(graph.parents(node))
        for a, b in itertools.combinations(parents, 2):
            if b not in graph.parents(a) and a not in graph.parents(b):
                out.add((a, node, b))
    return frozenset(out)


def markov_equivalent(first: CausalGraph, second: CausalGraph) -> bool:
    """Same skeleton and same v-structures — the classical characterization."""
    if set(first.nodes) != set(second.nodes):
        return False
    skeleton_a = {_pair(a, b) for a, b in first.edges}
    skeleton_b = {_pair(a, b) for a, b in second.edges}
    return skeleton_a == skeleton_b and v_structures(first) == v_structures(second)


def meek_closure(graph: EssentialGraph) -> EssentialGraph:
    """Orient every edge forced by what is already oriented (Meek 1995, rules 1-4).

    The rules are the completion step: given a skeleton, some orientations and
    the requirement that the result stay acyclic and introduce no new
    v-structure, they propagate every consequence. Applied until nothing
    changes.
    """
    directed = set(graph.directed)
    undirected = {_pair(a, b) for a, b in graph.undirected}

    def is_adjacent(a: str, b: str) -> bool:
        return (a, b) in directed or (b, a) in directed or _pair(a, b) in undirected

    changed = True
    while changed:
        changed = False
        for a, b in sorted(undirected):
            for tail, head in ((a, b), (b, a)):
                if _orient(tail, head, directed, undirected, is_adjacent):
                    undirected.discard(_pair(tail, head))
                    directed.add((tail, head))
                    changed = True
                    break
            if changed:
                break
    return EssentialGraph(
        nodes=graph.nodes,
        directed=tuple(sorted(directed)),
        undirected=tuple(sorted(undirected)),
        targets=graph.targets,
        name=graph.name,
    )


def _orient(
    tail: str,
    head: str,
    directed: set[Edge],
    undirected: set[Edge],
    adjacent: Callable[[str, str], bool],
) -> bool:
    """Whether Meek's rules force ``tail -> head`` for the currently undirected pair."""
    parents_of_tail = {a for a, b in directed if b == tail}
    parents_of_head = {a for a, b in directed if b == head}
    neighbours_of_tail = {(b if a == tail else a) for a, b in undirected if tail in (a, b)}

    # R1: a -> tail, tail - head, a and head non-adjacent
    for a in parents_of_tail:
        if not adjacent(a, head):
            return True
    # R2: tail -> c -> head with tail - head
    for c in {b for a, b in directed if a == tail}:
        if (c, head) in directed:
            return True
    # R3: tail - c, tail - d, c -> head, d -> head, c and d non-adjacent
    contributing = sorted(neighbours_of_tail & parents_of_head)
    for c, d in itertools.combinations(contributing, 2):
        if not adjacent(c, d):
            return True
    # R4: tail - c, c -> d, d -> head, with tail - head and c, head non-adjacent
    for c in sorted(neighbours_of_tail):
        for d in {b for a, b in directed if a == c}:
            if (d, head) in directed and not adjacent(c, head) and adjacent(tail, d):
                return True
    return False


def cpdag(graph: CausalGraph) -> EssentialGraph:
    """The essential graph of a DAG: what observational data could ever orient.

    Skeleton, then the v-structures — the only orientations observation
    licenses directly — then Meek's rules for everything those force.
    """
    if graph.bidirected:
        raise GraphError(
            "an essential graph is defined for a DAG without latent confounding; "
            "this graph has bidirected edges, whose equivalence class is a PAG"
        )
    directed = {(a, c) for a, c, _ in v_structures(graph)} | {
        (b, c) for _, c, b in v_structures(graph)
    }
    undirected = {_pair(a, b) for a, b in graph.edges} - {_pair(a, b) for a, b in directed}
    return meek_closure(
        EssentialGraph(
            nodes=graph.nodes,
            directed=tuple(sorted(directed)),
            undirected=tuple(sorted(undirected)),
            name=graph.name,
        )
    )


def interventional_essential_graph(
    graph: CausalGraph, targets: Sequence[Iterable[str]]
) -> EssentialGraph:
    """What the data could orient after intervening on each target in the family.

    Intervening on a set ``I`` randomizes its members, cutting their incoming
    edges; the direction of every edge with *exactly one* endpoint in ``I``
    therefore becomes visible. An edge with both endpoints inside ``I`` does
    not — both ends were randomized — which is why intervening on everything
    at once is not the most informative experiment.

    The observational case is the empty family, and gives ``cpdag``.
    """
    family = [frozenset(t) for t in targets]
    unknown = sorted({n for t in family for n in t} - set(graph.nodes))
    if unknown:
        raise GraphError(f"intervention targets name unknown nodes: {unknown}")
    base = cpdag(graph)
    directed = set(base.directed)
    undirected = set(base.undirected)
    for target in family:
        for a, b in graph.edges:
            if (a in target) != (b in target) and _pair(a, b) in undirected:
                undirected.discard(_pair(a, b))
                directed.add((a, b))
    return meek_closure(
        EssentialGraph(
            nodes=graph.nodes,
            directed=tuple(sorted(directed)),
            undirected=tuple(sorted(undirected)),
            targets=tuple(tuple(sorted(t)) for t in family),
            name=graph.name,
        )
    )


def orientation_gain(
    graph: CausalGraph, targets: Sequence[Iterable[str]], *, baseline: Sequence[Iterable[str]] = ()
) -> tuple[Edge, ...]:
    """The edges an experiment would orient that are undirected without it.

    This is the value of an experiment stated before it is run: which
    currently-ambiguous edges it settles. An empty result means the experiment
    tells you nothing you could not already have worked out — which is worth
    knowing before paying for it.
    """
    before = interventional_essential_graph(graph, baseline)
    after = interventional_essential_graph(graph, [*baseline, *targets])
    gained = set(after.directed) - set(before.directed)
    return tuple(sorted(gained))


# -- representatives -------------------------------------------------------------------


def consistent_extension(graph: EssentialGraph) -> CausalGraph:
    """One DAG from the class, by Dor & Tarsi's (1992) algorithm.

    Repeatedly take a node with no outgoing directed edge whose undirected
    neighbours are all adjacent to all its other neighbours, orient its
    undirected edges toward it, and remove it. If no such node exists the
    partially directed graph does not extend to any DAG, which is a
    malformed input rather than an unlucky one.
    """
    remaining = set(graph.nodes)
    directed = set(graph.directed)
    undirected = {_pair(a, b) for a, b in graph.undirected}
    oriented: set[Edge] = set()

    def neighbours(node: str) -> set[str]:
        return {
            b if a == node else a for a, b in undirected if node in (a, b) and {a, b} <= remaining
        }

    def adjacent(node: str) -> set[str]:
        out = neighbours(node)
        out |= {a for a, b in directed if b == node and a in remaining}
        out |= {b for a, b in directed if a == node and b in remaining}
        return out

    while remaining:
        for node in sorted(remaining):
            if any(a == node and b in remaining for a, b in directed):
                continue  # has an outgoing directed edge
            local = neighbours(node)
            others = adjacent(node)
            if all(
                other in adjacent(friend) or other == friend for friend in local for other in others
            ):
                for friend in local:
                    oriented.add((friend, node))
                    undirected.discard(_pair(friend, node))
                remaining.discard(node)
                break
        else:
            raise GraphError(
                "this partially directed graph has no consistent DAG extension; "
                "some orientation forced a cycle or a new v-structure"
            )
    return CausalGraph(
        nodes=graph.nodes,
        edges=tuple(sorted(set(graph.directed) | oriented)),
        name=graph.name,
    )


def consistent_extensions(graph: EssentialGraph, *, limit: int = 4096) -> tuple[CausalGraph, ...]:
    """Every DAG in the class, by orienting the undirected edges every way.

    Exponential in the number of undirected edges by nature — the class is
    that large. ``limit`` caps the enumeration; passing it raises rather than
    silently returning a subset, because a truncated equivalence class read as
    a complete one is a wrong answer.
    """
    pairs = list(graph.undirected)
    if 2 ** len(pairs) > limit:
        raise ValueError(
            f"the class has up to 2^{len(pairs)} orientations, past the limit of {limit}; "
            "raise limit deliberately or ask about a smaller graph"
        )
    reference = consistent_extension(graph)
    out: list[CausalGraph] = []
    for choice in itertools.product([False, True], repeat=len(pairs)):
        edges = list(graph.directed)
        for (a, b), flipped in zip(pairs, choice, strict=True):
            edges.append((b, a) if flipped else (a, b))
        try:
            candidate = CausalGraph(nodes=graph.nodes, edges=tuple(edges), name=graph.name)
        except GraphError:
            continue  # the orientation made a cycle
        if markov_equivalent(reference, candidate):
            out.append(candidate)
    return tuple(out)
