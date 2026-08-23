"""FCI: discovery that does not assume you measured everything.

``ges`` and ``gies`` assume **causal sufficiency** — that no unmeasured variable
drives two measured ones. That is the assumption ADMGs exist to drop, and
dropping it everywhere else in axiom is why ``identify_effect`` can return a
hedge instead of a false estimand. Discovery was the one place still making it.

FCI (Spirtes, Glymour & Scheines; orientation completed by Zhang 2008) drops it.
Its output is a **PAG** — a partial ancestral graph — whose edges carry two
marks, one at each end, from three possibilities:

* an **arrowhead** ``>`` means *not an ancestor*: whatever else is true,
  ``Y`` is not a cause of ``X``;
* a **tail** ``-`` means *is an ancestor*;
* a **circle** ``o`` means the data does not determine which.

So ``X <-> Y`` says a latent common cause: neither causes the other. ``X o-> Y``
says ``Y`` does not cause ``X``, but whether ``X`` causes ``Y`` or something
hidden drives both is open. That vocabulary is the point — it lets the output
say "confounded" and "I don't know" as different things, which a CPDAG cannot.

What is implemented, and what is not
------------------------------------
The adjacency search, the Possible-D-SEP refinement that separates FCI from PC,
v-structure orientation, and Zhang's rules R1-R3. **R4 (the discriminating-path
rule) and R5-R10 are not implemented.** Leaving a rule out leaves circles where
a complete implementation would put marks: the result is *sound but not
maximally informative* — every mark it does place is right, and it may place
fewer than it could. Under-orientation is the safe direction to be incomplete
in, and ``PAG.limits_hit`` says so on every result.

Testing discovery with an oracle
---------------------------------
``oracle_independence`` answers conditional-independence queries from a known
graph by d-separation instead of from data. Running FCI against it separates
the algorithm from the statistics: with a perfect test the output must be the
true PAG, and any disagreement is a bug rather than a sample. The data-driven
tests then check that a real test with enough rows reaches the same place.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterable, Sequence
from typing import Literal

from pydantic import model_validator

from axiom.core import NonEmptyStr, Spec
from axiom.discover.independence import PartialCorrelation
from axiom.discover.score import Dataset
from axiom.identify.graph import CausalGraph, GraphError

__all__ = [
    "PAG",
    "Mark",
    "PagEdge",
    "fci",
    "fci_from_data",
    "oracle_independence",
]

Mark = Literal["circle", "arrow", "tail"]
SYMBOL: dict[Mark, tuple[str, str]] = {
    # (how it draws on the left end, on the right end)
    "circle": ("o", "o"),
    "arrow": ("<", ">"),
    "tail": ("-", "-"),
}

Independent = Callable[[str, str, Sequence[str]], bool]
"""Answers "is X independent of Y given Z?" — from data, or from a known graph."""


class PagEdge(Spec):
    """One edge of a PAG: two variables and the mark at each end."""

    a: NonEmptyStr
    b: NonEmptyStr
    mark_a: Mark = "circle"
    mark_b: Mark = "circle"

    @model_validator(mode="after")
    def _sorted(self) -> PagEdge:
        """Store the ends sorted, carrying each mark with its own end.

        Normalizing rather than refusing means ``x <-o y`` and ``y o-> x`` are
        the same object and hash the same, which is what makes an edge usable
        as a key.
        """
        if self.a == self.b:
            raise ValueError(f"a PAG edge joins two variables, not {self.a!r} to itself")
        if self.a > self.b:
            a, b, mark_a, mark_b = self.b, self.a, self.mark_b, self.mark_a
            object.__setattr__(self, "a", a)
            object.__setattr__(self, "b", b)
            object.__setattr__(self, "mark_a", mark_a)
            object.__setattr__(self, "mark_b", mark_b)
        return self

    def to_text(self) -> str:
        return f"{self.a} {SYMBOL[self.mark_a][0]}-{SYMBOL[self.mark_b][1]} {self.b}"

    @property
    def is_bidirected(self) -> bool:
        """Both ends arrowheads: neither is an ancestor of the other, so something hidden is."""
        return self.mark_a == "arrow" and self.mark_b == "arrow"

    @property
    def is_directed(self) -> bool:
        """A tail at one end and an arrowhead at the other: a definite causal direction."""
        return {self.mark_a, self.mark_b} == {"tail", "arrow"}

    def mark_at(self, node: str) -> Mark:
        if node == self.a:
            return self.mark_a
        if node == self.b:
            return self.mark_b
        raise KeyError(f"{node!r} is not an end of {self.to_text()}")

    def other(self, node: str) -> str:
        return self.b if node == self.a else self.a


class PAG(Spec):
    """A partial ancestral graph: what the data determines when latents are possible.

    ``bidirected`` edges are the finding a CPDAG cannot express — a latent
    common cause. ``definite_causes`` are the pairs where a direction is
    settled. Circles are honest ignorance.
    """

    nodes: tuple[NonEmptyStr, ...] = ()
    edges: tuple[PagEdge, ...] = ()
    limits_hit: tuple[str, ...] = ()
    detail: dict[str, str] = {}

    @model_validator(mode="after")
    def _well_formed(self) -> PAG:
        mentioned = {n for e in self.edges for n in (e.a, e.b)}
        nodes = tuple(sorted(mentioned | set(self.nodes)))
        if nodes != self.nodes:
            object.__setattr__(self, "nodes", nodes)
        seen = {(e.a, e.b) for e in self.edges}
        if len(seen) != len(self.edges):
            raise GraphError("a PAG holds at most one edge per pair")
        return self

    def edge(self, a: str, b: str) -> PagEdge | None:
        key = (a, b) if a <= b else (b, a)
        for candidate in self.edges:
            if (candidate.a, candidate.b) == key:
                return candidate
        return None

    def adjacent(self, node: str) -> frozenset[str]:
        return frozenset(e.other(node) for e in self.edges if node in (e.a, e.b))

    def is_adjacent(self, a: str, b: str) -> bool:
        return self.edge(a, b) is not None

    @property
    def bidirected(self) -> tuple[PagEdge, ...]:
        """Edges with an arrowhead at both ends: a latent common cause, definitely."""
        return tuple(e for e in self.edges if e.is_bidirected)

    @property
    def definite_causes(self) -> tuple[tuple[str, str], ...]:
        """``(cause, effect)`` for every edge whose direction the data settled."""
        out = []
        for e in self.edges:
            if not e.is_directed:
                continue
            out.append((e.a, e.b) if e.mark_a == "tail" else (e.b, e.a))
        return tuple(sorted(out))

    @property
    def undetermined(self) -> tuple[PagEdge, ...]:
        """Edges with a circle at either end — where the data stopped short."""
        return tuple(e for e in self.edges if "circle" in (e.mark_a, e.mark_b))

    @property
    def skeleton(self) -> tuple[tuple[str, str], ...]:
        return tuple(sorted((e.a, e.b) for e in self.edges))

    def to_text(self) -> str:
        return ", ".join(e.to_text() for e in self.edges)

    def __str__(self) -> str:
        return self.to_text() or f"PAG(nodes={list(self.nodes)})"

    def summary(self) -> str:
        return (
            f"{len(self.edges)} edges: {len(self.definite_causes)} with a settled direction, "
            f"{len(self.bidirected)} confounded, {len(self.undetermined)} still open"
        )


# -- the independence oracle ---------------------------------------------------------------


def oracle_independence(graph: CausalGraph) -> Independent:
    """A perfect conditional-independence test, answered by d-separation in a known graph.

    For testing and for teaching: run a discovery algorithm against this and
    any disagreement with the truth is the algorithm's fault, not the sample's.
    Latent variables stay in ``graph`` and simply never appear in the queries.
    """

    def independent(x: str, y: str, given: Sequence[str]) -> bool:
        return graph.d_separated(x, y, list(given))

    return independent


# -- the algorithm --------------------------------------------------------------------------


def _adjacency_search(
    variables: Sequence[str],
    independent: Independent,
    max_conditioning: int,
) -> tuple[dict[tuple[str, str], list[Mark]], dict[tuple[str, str], tuple[str, ...]], bool]:
    """Remove an edge as soon as some conditioning set makes its ends independent."""
    # Keys are always sorted pairs, so a lookup by either end finds the same edge.
    marks: dict[tuple[str, str], list[Mark]] = {
        _key(a, b): ["circle", "circle"] for a, b in itertools.combinations(variables, 2)
    }
    sepsets: dict[tuple[str, str], tuple[str, ...]] = {}
    truncated = False

    def neighbours(node: str) -> list[str]:
        return sorted(other for other in variables if other != node and _key(node, other) in marks)

    size = 0
    while True:
        acted = False
        for a, b in list(marks):
            for x, y in ((a, b), (b, a)):
                pool = [n for n in neighbours(x) if n != y]
                if len(pool) < size:
                    continue
                acted = True
                for chosen in itertools.combinations(pool, size):
                    if independent(x, y, list(chosen)):
                        marks.pop(_key(a, b), None)
                        sepsets[_key(a, b)] = tuple(chosen)
                        break
                if _key(a, b) not in marks:
                    break
        if not acted:
            break
        size += 1
        if size > max_conditioning:
            truncated = True
            break
    return marks, sepsets, truncated


def _key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def _possible_d_sep(
    variables: Sequence[str], marks: dict[tuple[str, str], list[Mark]], start: str
) -> set[str]:
    """Nodes reachable from ``start`` by a path of colliders-or-triangles.

    FCI's extra stage, and the reason it is not just PC with different marks:
    under latent confounding the separating set may live outside the
    neighbourhood the first pass searched.
    """

    def adjacent(node: str) -> set[str]:
        return {(b if a == node else a) for a, b in marks if node in (a, b)}

    found: set[str] = set()
    seen: set[tuple[str, str]] = set()
    stack = [(start, other) for other in adjacent(start)]
    while stack:
        previous, node = stack.pop()
        if (previous, node) in seen:
            continue
        seen.add((previous, node))
        if node != start:
            found.add(node)
        for nxt in adjacent(node):
            if nxt == previous:
                continue
            collider = (
                _mark(marks, node, previous) == "arrow" and _mark(marks, node, nxt) == "arrow"
            )
            triangle = _key(previous, nxt) in marks
            if collider or triangle:
                stack.append((node, nxt))
    return found - {start}


def _mark(marks: dict[tuple[str, str], list[Mark]], at: str, other: str) -> Mark | None:
    """The mark of the edge ``at``-``other`` on the ``at`` end."""
    key = _key(at, other)
    pair = marks.get(key)
    if pair is None:
        return None
    return pair[0] if key[0] == at else pair[1]


def _set_mark(marks: dict[tuple[str, str], list[Mark]], at: str, other: str, value: Mark) -> bool:
    key = _key(at, other)
    pair = marks.get(key)
    if pair is None:
        return False
    index = 0 if key[0] == at else 1
    if pair[index] == value:
        return False
    pair[index] = value
    return True


def _orient_colliders(
    variables: Sequence[str],
    marks: dict[tuple[str, str], list[Mark]],
    sepsets: dict[tuple[str, str], tuple[str, ...]],
) -> None:
    """An unshielded triple whose middle is not in the separating set is a collider."""
    for middle in variables:
        neighbours = sorted((b if a == middle else a) for a, b in marks if middle in (a, b))
        for x, y in itertools.combinations(neighbours, 2):
            if _key(x, y) in marks:
                continue  # shielded
            separating = sepsets.get(_key(x, y))
            if separating is None or middle in separating:
                continue
            _set_mark(marks, middle, x, "arrow")
            _set_mark(marks, middle, y, "arrow")


def _apply_rules(variables: Sequence[str], marks: dict[tuple[str, str], list[Mark]]) -> None:
    """Zhang's rules R1-R3, to a fixed point."""
    changed = True
    while changed:
        changed = False
        for a, b in itertools.permutations(variables, 2):
            if _key(a, b) not in marks:
                continue
            # R1: a *-> b o-* c with a, c non-adjacent  =>  b -> c.
            # The circle that moves is the one at *b*, the middle of the triple:
            # b is not a collider there, so its end of the b-c edge becomes a tail.
            if _mark(marks, b, a) == "arrow":
                for c in variables:
                    if c in (a, b) or _key(b, c) not in marks or _key(a, c) in marks:
                        continue
                    if _mark(marks, b, c) == "circle":
                        changed |= _set_mark(marks, b, c, "tail")
                        changed |= _set_mark(marks, c, b, "arrow")
            # R2: a -> b *-> c or a *-> b -> c, with a *-o c  =>  arrowhead at c
            for c in variables:
                if c in (a, b) or _key(a, c) not in marks:
                    continue
                if _mark(marks, c, a) != "circle":
                    continue
                if _key(a, b) not in marks or _key(b, c) not in marks:
                    continue
                first = _mark(marks, b, a) == "arrow" and _mark(marks, a, b) == "tail"
                second = _mark(marks, c, b) == "arrow" and _mark(marks, b, c) == "tail"
                if (first and _mark(marks, c, b) == "arrow") or (
                    second and _mark(marks, b, a) == "arrow"
                ):
                    changed |= _set_mark(marks, c, a, "arrow")
        # R3: a *-> b <-* c, a *-o d o-* c, a and c non-adjacent, d *-o b  =>  arrowhead at b
        for b in variables:
            for a, c in itertools.combinations(sorted(variables), 2):
                if b in (a, c) or _key(a, c) in marks:
                    continue
                if _key(a, b) not in marks or _key(c, b) not in marks:
                    continue
                if _mark(marks, b, a) != "arrow" or _mark(marks, b, c) != "arrow":
                    continue
                for d in variables:
                    if d in (a, b, c) or _key(a, d) not in marks or _key(c, d) not in marks:
                        continue
                    if _key(d, b) not in marks:
                        continue
                    if (
                        _mark(marks, d, a) == "circle"
                        and _mark(marks, d, c) == "circle"
                        and _mark(marks, b, d) == "circle"
                    ):
                        changed |= _set_mark(marks, b, d, "arrow")


def fci(
    variables: Sequence[str],
    independent: Independent,
    *,
    max_conditioning: int = 3,
    name: str = "",
) -> PAG:
    """Discover a PAG without assuming causal sufficiency.

    ``independent`` answers conditional-independence queries — from data
    (``fci_from_data``) or from a known graph (``oracle_independence``).

    The stages are the adjacency search, the Possible-D-SEP refinement, collider
    orientation, and Zhang's rules R1-R3. What that leaves out is recorded on
    the result: an unimplemented rule leaves circles where marks could go, which
    understates what the data determines and never overstates it.
    """
    names = list(dict.fromkeys(variables))
    if len(names) < 2:
        raise ValueError("a PAG needs at least two variables")
    marks, sepsets, truncated = _adjacency_search(names, independent, max_conditioning)
    _orient_colliders(names, marks, sepsets)

    # Possible-D-SEP: look again, now conditioning on sets the first pass could not see.
    for a, b in list(marks):
        for x, y in ((a, b), (b, a)):
            if _key(a, b) not in marks:
                break
            pool = sorted(_possible_d_sep(names, marks, x) - {y})
            if len(pool) > max_conditioning + 4:
                pool = pool[: max_conditioning + 4]
                truncated = True
            for size in range(min(max_conditioning, len(pool)) + 1):
                found = False
                for chosen in itertools.combinations(pool, size):
                    if independent(x, y, list(chosen)):
                        marks.pop(_key(a, b), None)
                        sepsets[_key(a, b)] = tuple(chosen)
                        found = True
                        break
                if found:
                    break

    for pair in marks.values():
        pair[0] = pair[1] = "circle"
    _orient_colliders(names, marks, sepsets)
    _apply_rules(names, marks)

    limits = ["Zhang's rules R4 and R5-R10 are not implemented, so some marks stay circles"]
    if truncated:
        limits.append(f"conditioning sets were searched up to size {max_conditioning}")
    return PAG(
        nodes=tuple(sorted(names)),
        edges=tuple(
            PagEdge(a=a, b=b, mark_a=pair[0], mark_b=pair[1])
            for (a, b), pair in sorted(marks.items())
        ),
        limits_hit=tuple(limits),
        detail={"variables": str(len(names)), "name": name},
    )


def fci_from_data(
    data: Dataset,
    *,
    alpha: float = 0.05,
    max_conditioning: int = 3,
    variables: Iterable[str] | None = None,
    name: str = "",
) -> PAG:
    """Run FCI with a partial-correlation test at level ``alpha``.

    The test assumes linear-Gaussian dependence; a nonlinear association that
    happens to be uncorrelated reads as independence and removes an edge that
    should stay.
    """
    tester = PartialCorrelation(data)
    names = list(variables) if variables is not None else list(data.names)

    def independent(x: str, y: str, given: Sequence[str]) -> bool:
        return tester.independent(x, y, given, alpha=alpha)

    result = fci(names, independent, max_conditioning=max_conditioning, name=name)
    return result.model_copy(
        update={"detail": {**result.detail, "alpha": f"{alpha:g}", "rows": str(data.n_rows)}}
    )
