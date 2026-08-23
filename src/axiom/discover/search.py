"""GES and GIES: greedy search over equivalence classes, not over DAGs.

Searching DAGs for a structure is searching the wrong space. Markov-equivalent
DAGs have identical observational likelihoods, so a DAG-space search spends its
moves reorienting edges that no data distinguishes, and reports one arbitrary
member of a class as if it were the answer.

GES (Chickering 2002) searches the space of equivalence classes instead:
forward, adding edges while the score improves; then backward, removing them.
Each move is made on the essential graph, so a step is a step between classes,
and what it returns is a class — which is what the data identifies.

GIES (Hauser & Bühlmann 2012) is the same search over *interventional*
essential graphs, scored with interventional data. It answers the question the
observational version cannot: which edges the experiments you ran have
oriented. That is also why running it before an experiment is useful — see
``essential.orientation_gain``, which prices the experiment in edges.

What this implementation does and does not claim
------------------------------------------------
The operators are GES's Insert and Delete. Each candidate is applied to the
current graph, extended to a DAG, re-reduced to its (interventional) essential
graph, and **rescored in full** rather than by the incremental formulas. That
is slower and has no bookkeeping to get wrong; on the same candidate set it
picks the same move.

All three phases run — forward, backward and **turning** — and the three
repeat until no phase improves the score. The turning phase is not optional
once interventions are involved, and the reason is worth stating: an
interventional essential graph orients a cut edge the moment the edge is
added, so a forward step that guesses a direction wrongly is locked in, and
no single insert or delete escapes it. On a four-variable chain with three
intervention targets, forward-and-backward alone stops at the complete graph,
scoring 1,700 nats below the truth; adding turning finds the truth. That case
is a test.

Greedy is still greedy: nothing here promises the global optimum at finite
sample size. ``DiscoveryResult.steps`` records every move taken, so a
suspicious answer can be read back rather than guessed at.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterable, Sequence

from pydantic import Field

from axiom.core import Spec
from axiom.discover.essential import (
    EssentialGraph,
    consistent_extension,
    cpdag,
    interventional_essential_graph,
)
from axiom.discover.score import GaussianBIC
from axiom.identify.graph import CausalGraph, Edge, GraphError

__all__ = ["DiscoveryResult", "ges", "gies"]

MAX_SUBSET = 8
"""Largest neighbour set an Insert or Delete enumerates subsets of. Past this the
candidate list is truncated and the result says so."""


class DiscoveryResult(Spec):
    """What a structure search found, and how far it is entitled to be believed.

    ``essential`` is an equivalence class, not a DAG: a directed edge is one
    the data (and the interventions) settle, an undirected edge is one they do
    not. ``targets`` records the intervention family the class is essential
    for, which is what makes the directed edges directed.
    """

    essential: EssentialGraph
    score: float
    n_observations: int = Field(ge=1)
    targets: tuple[tuple[str, ...], ...] = ()
    phases: tuple[str, ...] = ()
    steps: tuple[str, ...] = ()
    penalty: float = Field(gt=0)
    limits_hit: tuple[str, ...] = ()
    detail: dict[str, str] = {}

    @property
    def oriented(self) -> int:
        return self.essential.oriented

    @property
    def undecided(self) -> int:
        return self.essential.undecided

    def summary(self) -> str:
        return (
            f"{self.oriented} edges oriented, {self.undecided} left undecided "
            f"by {self.n_observations} rows under {len(self.targets)} intervention target(s)"
        )


def _parents_of(dag: CausalGraph) -> dict[str, frozenset[str]]:
    return {node: dag.parents(node) for node in dag.nodes}


def _essential_of(dag: CausalGraph, targets: Sequence[Iterable[str]]) -> EssentialGraph:
    return interventional_essential_graph(dag, targets) if targets else cpdag(dag)


def _semi_directed_reachable(
    graph: EssentialGraph, start: str, blocked: frozenset[str]
) -> set[str]:
    """Nodes reachable from ``start`` without ever travelling against an arrow."""
    seen: set[str] = set()
    stack = [start]
    while stack:
        node = stack.pop()
        for nxt in graph.children(node) | graph.neighbours(node):
            if nxt in blocked or nxt in seen:
                continue
            seen.add(nxt)
            stack.append(nxt)
    return seen


def _is_clique(graph: EssentialGraph, nodes: Iterable[str]) -> bool:
    members = sorted(nodes)
    return all(graph.is_adjacent(a, b) for a, b in itertools.combinations(members, 2))


def _apply(
    graph: EssentialGraph,
    directed_added: Iterable[Edge],
    removed: Iterable[Edge],
    targets: Sequence[Iterable[str]],
) -> EssentialGraph | None:
    """Rebuild the class after an operator, or ``None`` if the result is not a valid class."""
    drop = {frozenset(e) for e in removed}
    directed = [e for e in graph.directed if frozenset(e) not in drop]
    undirected = [e for e in graph.undirected if frozenset(e) not in drop]
    for edge in directed_added:
        directed.append(edge)
        undirected = [e for e in undirected if frozenset(e) != frozenset(edge)]
    try:
        candidate = EssentialGraph(
            nodes=graph.nodes, directed=tuple(directed), undirected=tuple(undirected)
        )
        extension = consistent_extension(candidate)
    except (GraphError, ValueError):
        return None
    return _essential_of(extension, targets)


def _forward_moves(graph: EssentialGraph) -> list[tuple[list[Edge], list[Edge], str]]:
    """Insert operators: add ``x -> y`` and orient a subset of ``y``'s free neighbours into it."""
    moves: list[tuple[list[Edge], list[Edge], str]] = []
    for x, y in itertools.permutations(graph.nodes, 2):
        if graph.is_adjacent(x, y):
            continue
        adjacent_to_x = graph.adjacent(x)
        common = graph.neighbours(y) & adjacent_to_x
        free = sorted(graph.neighbours(y) - adjacent_to_x - {x})[:MAX_SUBSET]
        for size in range(len(free) + 1):
            for chosen in itertools.combinations(free, size):
                condition = frozenset(common) | frozenset(chosen)
                if not _is_clique(graph, condition):
                    continue
                if x in _semi_directed_reachable(graph, y, condition):
                    continue
                added = [(x, y), *[(t, y) for t in chosen]]
                label = f"insert {x} -> {y}" + (f" with {list(chosen)}" if chosen else "")
                moves.append((added, [], label))
    return moves


def _backward_moves(graph: EssentialGraph) -> list[tuple[list[Edge], list[Edge], str]]:
    """Delete operators: remove an edge and orient a subset of the shared neighbours away."""
    moves: list[tuple[list[Edge], list[Edge], str]] = []
    candidates = [(a, b) for a, b in graph.directed] + [(a, b) for a, b in graph.undirected]
    for x, y in candidates:
        common = sorted(graph.neighbours(y) & graph.adjacent(x))[:MAX_SUBSET]
        for size in range(len(common) + 1):
            for chosen in itertools.combinations(common, size):
                keep = frozenset(common) - frozenset(chosen)
                if not _is_clique(graph, keep):
                    continue
                added = [(y, h) for h in chosen]
                if frozenset((x, y)) in {frozenset(e) for e in graph.undirected}:
                    added += [(x, h) for h in chosen]
                label = f"delete {x} - {y}" + (f" orienting {list(chosen)}" if chosen else "")
                moves.append((added, [(x, y)], label))
    return moves


def _turning_moves(graph: EssentialGraph) -> list[tuple[list[Edge], list[Edge], str]]:
    """Turn operators: reverse a directed edge, or commit an undirected one either way.

    The phase that makes the difference once interventions are in play. An
    interventional essential graph orients a cut edge as soon as the edge is
    added, so a forward step that guesses the direction wrongly is locked in;
    forward and backward moves alone cannot undo it, because every single-edge
    change from there is worse. Turning is what escapes that, and without it
    the search reliably stops one reversal short of the truth.
    """
    moves: list[tuple[list[Edge], list[Edge], str]] = []
    for x, y in graph.directed:
        moves.append(([(y, x)], [(x, y)], f"turn {x} -> {y} into {y} -> {x}"))
    for x, y in graph.undirected:
        moves.append(([(x, y)], [(x, y)], f"commit {x} - {y} as {x} -> {y}"))
        moves.append(([(y, x)], [(x, y)], f"commit {x} - {y} as {y} -> {x}"))
    return moves


def _greedy(
    start: EssentialGraph,
    score: GaussianBIC,
    targets: Sequence[Iterable[str]],
    generator: Callable[[EssentialGraph], list[tuple[list[Edge], list[Edge], str]]],
    phase: str,
    steps: list[str],
) -> tuple[EssentialGraph, float]:
    current = start
    best_score = score.total(_parents_of(consistent_extension(current)))
    while True:
        best_move: tuple[float, EssentialGraph, str] | None = None
        for added, removed, label in generator(current):
            candidate = _apply(current, added, removed, targets)
            if candidate is None or candidate == current:
                continue
            value = score.total(_parents_of(consistent_extension(candidate)))
            if value > best_score + 1e-10 and (best_move is None or value > best_move[0]):
                best_move = (value, candidate, label)
        if best_move is None:
            return current, best_score
        best_score, current, label = best_move
        steps.append(f"{phase}: {label}")


def _search(
    score: GaussianBIC,
    targets: Sequence[Iterable[str]],
    *,
    start: EssentialGraph | None = None,
    name: str = "",
    max_rounds: int = 8,
) -> DiscoveryResult:
    nodes = tuple(score.data.names)
    current = start or EssentialGraph(nodes=nodes)
    steps: list[str] = []
    value = score.total(_parents_of(consistent_extension(current)))
    phases: list[str] = []
    for _ in range(max_rounds):
        before = value
        for label, generator in (
            ("forward", _forward_moves),
            ("backward", _backward_moves),
            ("turning", _turning_moves),
        ):
            current, value = _greedy(current, score, targets, generator, label, steps)
            if label not in phases:
                phases.append(label)
        if value <= before + 1e-10:
            break
    limits = []
    if any(len(current.neighbours(n)) > MAX_SUBSET for n in current.nodes):
        limits.append(f"neighbour subsets enumerated up to size {MAX_SUBSET}")
    return DiscoveryResult(
        essential=current.model_copy(
            update={"targets": tuple(tuple(sorted(t)) for t in targets), "name": name}
        ),
        score=value,
        n_observations=score.data.n_rows,
        targets=tuple(tuple(sorted(t)) for t in targets),
        phases=tuple(phases),
        steps=tuple(steps),
        penalty=score.penalty,
        limits_hit=tuple(limits),
        detail={"variables": str(len(nodes)), "rounds": str(len(steps))},
    )


def ges(score: GaussianBIC, *, name: str = "") -> DiscoveryResult:
    """Greedy equivalence search on observational data: forward, then backward.

    Returns the CPDAG the score prefers — an equivalence class, because that
    is what observational data identifies. Rows taken under an intervention
    are still used for every variable that was not the one randomized; to have
    the interventions *orient* edges, call ``gies``.
    """
    return _search(score, (), name=name)


def gies(
    score: GaussianBIC, targets: Sequence[Iterable[str]] | None = None, *, name: str = ""
) -> DiscoveryResult:
    """Greedy interventional equivalence search: the same, over I-essential graphs.

    ``targets`` defaults to the intervention targets present in the data. The
    result is an interventional essential graph, in which an edge cut by some
    target is oriented — the edges the experiments bought.
    """
    family = list(targets) if targets is not None else [list(t) for t in score.data.targets]
    return _search(score, family, name=name)
