"""Front-door and instrumental-variable admissibility on a ``CausalGraph``.

Both criteria are purely graphical and are decided with the graph's own
d-separation (``CausalGraph.d_separated``) on the mutilated graphs
``G_{\\underline{X}}`` (``remove_edges_out_of``); nothing here reimplements
path blocking. Measurement is a separate question: ``frontdoor_admissible``
and ``instrument_admissible`` answer "does the graph license this route",
and the enumerators (``frontdoor_sets``, ``instruments``,
``conditional_instruments``) restrict candidates to measured nodes unless
told otherwise. The verdict module decides what an unmeasured mediator or
instrument downgrades. Likewise ``graph.feedback`` (a summary graph hiding
treatment–outcome feedback over time) and ``graph.selection`` (S-nodes of a
selection diagram) are ignored here: both are the concern of the verdict and
transport modules, which read them before trusting any route this module
reports.

Front-door (Pearl 2009, Definition 3.3.3 and Theorem 3.3.4). A set ``M``
satisfies the front-door criterion relative to ``(x, y)`` if

(i)   ``M`` intercepts every directed path from ``x`` to ``y``;
(ii)  there is no unblocked back-door path from ``x`` to ``M``;
(iii) every back-door path from ``M`` to ``y`` is blocked by ``x``.

Then ``P(y | do(x)) = sum_m P(m | x) sum_x' P(y | x', m) P(x')``.

Instrument (Pearl 2009 §7.4.5; Brito & Pearl 2002 for the conditional form).
``z`` is an instrument for the total effect of ``x`` on ``y`` given ``W`` if

(a) ``z`` is d-connected to ``x`` given ``W`` in ``G`` (relevance);
(b) ``z`` is d-separated from ``y`` given ``W`` in ``G_{\\underline{x}}``
    (exclusion and exogeneity, relative to the total effect);
(c) neither ``z`` nor any node of ``W`` is a descendant of ``x``, and no node
    of ``W`` is a descendant of ``y``.

Condition (c) is not redundant with (b): removing the edges out of ``x`` also
severs the dependence of ``x``'s own descendants on ``x``, so without (c) a
child of ``x`` would pass (b) trivially while conditioning on it opens the
collider at ``x`` in the real graph. The "no node of ``W`` is a descendant
of ``x``" clause is a conservative *sufficient* condition: it rejects every
``W`` that could open such a collider, at the price of also rejecting some
descendants of ``x`` that would be harmless to condition on (Brito & Pearl
2002 state the weaker requirement that ``W`` not be descendants of ``y``).
A candidate rejected here may still be a valid conditional instrument under
a finer test; a candidate accepted here is one.
"""

from __future__ import annotations

from collections.abc import Iterable
from itertools import combinations

from pydantic import field_validator, model_validator

from axiom.core import Spec, Unsupported
from axiom.identify.graph import CausalGraph, GraphError

__all__ = [
    "FrontDoorRoute",
    "InstrumentRoute",
    "conditional_instruments",
    "frontdoor_admissible",
    "frontdoor_sets",
    "instrument_admissible",
    "instruments",
]


# -- shared checks -------------------------------------------------------------------


def _node_set(nodes: str | Iterable[str]) -> frozenset[str]:
    return frozenset((nodes,)) if isinstance(nodes, str) else frozenset(nodes)


def _require_known(graph: CausalGraph, nodes: Iterable[str]) -> None:
    unknown = sorted(set(nodes) - set(graph.nodes))
    if unknown:
        raise GraphError(f"unknown nodes {unknown}; nodes are {list(graph.nodes)}")


def _require_pair(graph: CausalGraph, x: str, y: str) -> None:
    _require_known(graph, (x, y))
    if x == y:
        raise GraphError(f"treatment and outcome must differ; both are {x!r}")


def _pool(graph: CausalGraph, *, measured_only: bool) -> frozenset[str]:
    return graph.measured if measured_only else frozenset(graph.nodes)


def _without_edges_out_of(graph: CausalGraph, nodes: Iterable[str]) -> CausalGraph:
    """``G_{\\underline{M}}`` for a set ``M``: edges out of every node of ``M`` removed."""
    out = graph
    for node in sorted(nodes):
        out = out.remove_edges_out_of(node)
    return out


# -- front-door --------------------------------------------------------------------------


def frontdoor_admissible(graph: CausalGraph, x: str, y: str, m: Iterable[str]) -> bool:
    """Whether ``m`` satisfies the front-door criterion relative to ``(x, y)``.

    Implements Pearl (2009) Definition 3.3.3 through three d-separation
    queries, with ``M`` the set ``m``:

    (i)   ``y`` is not a descendant of ``x`` in ``G_{\\underline{M}}`` — every
          directed path from ``x`` to ``y`` passes through ``M``;
    (ii)  ``x`` and ``M`` are d-separated by the empty set in
          ``G_{\\underline{x}}`` — the only paths left from ``x`` start with an
          arrow into ``x``, so this is exactly "no unblocked back-door path";
    (iii) for every ``m`` in ``M``, ``m`` and ``y`` are d-separated given
          ``{x} ∪ (M \\ {m})`` in ``G_{\\underline{M}}``. In that graph a
          mediator can only be a collider on a path, so conditioning on the
          other mediators can open paths but never close them, and under (ii)
          this is equivalent to the set statement ``(M ⊥ y | x)`` in
          ``G_{\\underline{M}}`` used by do-calculus rule 2.

    The criterion is only applied when ``x`` has at least one directed path to
    ``y``. If it has none the effect is structurally null, the front-door
    formula degenerates to ``P(y)``, and no set is reported admissible; an
    empty ``m`` is therefore never admissible. Mediators need not be measured
    here; ``frontdoor_sets`` applies that filter. Raises ``GraphError`` for
    unknown nodes, ``x == y``, or a mediator equal to ``x`` or ``y``.
    """
    _require_pair(graph, x, y)
    mediators = _node_set(m)
    _require_known(graph, mediators)
    if mediators & {x, y}:
        raise GraphError(
            f"mediators {sorted(mediators & {x, y})} coincide with treatment or outcome"
        )
    if not mediators or y not in graph.descendants(x):
        return False
    g_under_m = _without_edges_out_of(graph, mediators)
    if y in g_under_m.descendants(x):
        return False  # (i) a directed path avoids M
    if not graph.remove_edges_out_of(x).d_separated(x, mediators):
        return False  # (ii) an open back-door path from x into M
    return all(  # (iii)
        g_under_m.d_separated(node, y, {x} | (mediators - {node})) for node in mediators
    )


def frontdoor_sets(
    graph: CausalGraph,
    x: str,
    y: str,
    *,
    measured_only: bool = True,
    max_candidates: int = 12,
) -> tuple[frozenset[str], ...] | Unsupported:
    """Every front-door admissible mediator set, smallest first, then lexicographic.

    Candidates are the nodes other than ``x`` and ``y`` (measured ones only
    by default). Sets are not restricted to nodes lying on a directed path:
    an off-path node can be the one that blocks a mediator's back-door path,
    as in ``X -> M -> Y, W -> M, W -> Y, X <-> Y`` where ``{M}`` fails and
    ``{M, W}`` is admissible. Every non-empty subset is tested, so more than
    ``max_candidates`` candidates returns ``Unsupported`` rather than an
    exponential search; raise the bound deliberately if the graph warrants it.
    """
    _require_pair(graph, x, y)
    if y not in graph.descendants(x):
        return ()  # structurally null effect: no set is admissible, whatever the pool size
    candidates = sorted(_pool(graph, measured_only=measured_only) - {x, y})
    if len(candidates) > max_candidates:
        return Unsupported(
            reason=(
                f"{len(candidates)} candidate mediators exceed max_candidates={max_candidates}; "
                "the front-door search is exhaustive over subsets"
            ),
            detail={"candidates": str(len(candidates)), "max_candidates": str(max_candidates)},
        )
    return tuple(
        frozenset(combo)
        for size in range(1, len(candidates) + 1)
        for combo in combinations(candidates, size)
        if frontdoor_admissible(graph, x, y, combo)
    )


# -- instrumental variables ----------------------------------------------------------------


def instrument_admissible(
    graph: CausalGraph, x: str, y: str, z: str, conditioning: Iterable[str] = ()
) -> bool:
    """Whether ``z`` is an instrument for the total effect of ``x`` on ``y`` given ``W``.

    With ``W = conditioning`` (empty for the classical, unconditional
    instrument) the three conditions are, in the order checked:

    (c) no node of ``W ∪ {z}`` is a descendant of ``x``, and no node of ``W``
        is a descendant of ``y`` (Brito & Pearl 2002 require ``W`` to be
        non-descendants of ``y``; the ``x`` clause is what makes (b) honest,
        see the module docstring);
    (a) ``z`` and ``x`` are d-connected given ``W`` in ``G`` — relevance;
    (b) ``z`` and ``y`` are d-separated given ``W`` in ``G_{\\underline{x}}``
        (Pearl 2009 §7.4.5) — every path from ``z`` to ``y`` that does not
        run through an edge out of ``x`` is blocked by ``W``, which covers
        both the exclusion restriction and exogeneity of the instrument.

    The test is for the *total* effect, hence all edges out of ``x`` are
    removed rather than only ``x -> y``. Purely graphical: whether ``z`` and
    ``W`` are measured is the enumerators' concern. Raises ``GraphError`` for
    unknown nodes, ``x == y``, ``z`` in ``{x, y}``, or ``W`` meeting
    ``{x, y, z}``.
    """
    _require_pair(graph, x, y)
    w = _node_set(conditioning)
    _require_known(graph, (z, *w))
    if z in (x, y):
        raise GraphError(f"instrument {z!r} coincides with treatment or outcome")
    if w & {x, y, z}:
        raise GraphError(f"conditioning set {sorted(w)} meets treatment, outcome, or instrument")
    descendants_x = graph.descendants(x)
    if z in descendants_x or w & descendants_x or w & graph.descendants(y):
        return False  # (c)
    if graph.d_separated(z, x, w):
        return False  # (a) irrelevant
    return graph.remove_edges_out_of(x).d_separated(z, y, w)  # (b)


def instruments(
    graph: CausalGraph, x: str, y: str, *, measured_only: bool = True
) -> tuple[str, ...]:
    """All unconditional instruments (``W = ∅``) for ``x -> y``, sorted by name."""
    _require_pair(graph, x, y)
    candidates = sorted(_pool(graph, measured_only=measured_only) - {x, y})
    return tuple(z for z in candidates if instrument_admissible(graph, x, y, z))


def conditional_instruments(
    graph: CausalGraph,
    x: str,
    y: str,
    *,
    measured_only: bool = True,
    max_candidates: int = 12,
    max_conditioning: int = 2,
) -> tuple[tuple[str, frozenset[str]], ...] | Unsupported:
    """All ``(z, W)`` pairs with ``|W| <= max_conditioning`` that satisfy the criterion.

    ``z`` and the members of ``W`` are drawn from the nodes other than ``x``
    and ``y`` that are descendants of neither ``x`` nor ``y`` (measured only
    by default), since condition (c) rejects everything else. Pairs with
    ``W = ∅`` are included, so every
    entry of ``instruments`` appears here with an empty set. Sorted by
    instrument, then by size of ``W``, then lexicographically. Non-minimal
    ``W`` are reported too: a superset of an admissible ``W`` is admissible
    whenever it still satisfies (a)–(c), and the caller may prefer it for
    precision. When the pool of conditioning candidates for an instrument
    exceeds ``max_candidates`` the search returns ``Unsupported``.
    """
    _require_pair(graph, x, y)
    pool = (
        _pool(graph, measured_only=measured_only)
        - {x, y}
        - graph.descendants(x)
        - graph.descendants(y)
    )
    if len(pool) - 1 > max_candidates:
        return Unsupported(
            reason=(
                f"{len(pool) - 1} candidate conditioning nodes per instrument exceed "
                f"max_candidates={max_candidates}"
            ),
            detail={"candidates": str(len(pool) - 1), "max_candidates": str(max_candidates)},
        )
    out: list[tuple[str, frozenset[str]]] = []
    for z in sorted(pool):
        others = sorted(pool - {z})
        for size in range(0, min(max_conditioning, len(others)) + 1):
            for combo in combinations(others, size):
                if instrument_admissible(graph, x, y, z, combo):
                    out.append((z, frozenset(combo)))
    return tuple(out)


# -- route records -------------------------------------------------------------------------


def _sorted_names(v: object, field: str) -> tuple[str, ...]:
    """Normalise a validator input to a sorted tuple of names, or raise ``ValueError``."""
    if v is None or isinstance(v, bytes | dict) or not isinstance(v, str | Iterable):
        raise ValueError(f"{field} must be a node name or an iterable of node names, got {v!r}")
    items = [v] if isinstance(v, str) else list(v)
    if not all(isinstance(n, str) and n.strip() for n in items):
        raise ValueError(f"{field} must contain only non-empty node names, got {v!r}")
    return tuple(sorted(set(items)))


class FrontDoorRoute(Spec):
    """A front-door identification route: ``P(y | do(x))`` through ``mediators``.

    A plain record of the verdict; ``mediators`` is sorted and non-empty and
    is disjoint from ``treatment`` and ``outcome``. Admissibility is decided
    by ``frontdoor_admissible`` against a graph, not here.
    """

    mediators: tuple[str, ...]
    treatment: str
    outcome: str

    @field_validator("mediators", mode="before")
    @classmethod
    def _norm_mediators(cls, v: object) -> tuple[str, ...]:
        out = _sorted_names(v, "mediators")
        if not out:
            raise ValueError("a front-door route needs at least one mediator")
        return out

    @model_validator(mode="after")
    def _disjoint(self) -> FrontDoorRoute:
        if self.treatment == self.outcome:
            raise ValueError("treatment and outcome must differ")
        clash = sorted(set(self.mediators) & {self.treatment, self.outcome})
        if clash:
            raise ValueError(f"mediators {clash} coincide with treatment or outcome")
        return self


class InstrumentRoute(Spec):
    """An instrumental-variable route: ``instrument`` for ``treatment -> outcome``.

    A plain record of the verdict; ``conditioning`` is sorted (empty for an
    unconditional instrument) and disjoint from the three named nodes.
    Admissibility is decided by ``instrument_admissible`` against a graph.
    """

    instrument: str
    conditioning: tuple[str, ...] = ()
    treatment: str
    outcome: str

    @field_validator("conditioning", mode="before")
    @classmethod
    def _norm_conditioning(cls, v: object) -> tuple[str, ...]:
        return _sorted_names(v, "conditioning")

    @model_validator(mode="after")
    def _disjoint(self) -> InstrumentRoute:
        if self.treatment == self.outcome:
            raise ValueError("treatment and outcome must differ")
        if self.instrument in (self.treatment, self.outcome):
            raise ValueError("instrument coincides with treatment or outcome")
        clash = sorted(set(self.conditioning) & {self.instrument, self.treatment, self.outcome})
        if clash:
            raise ValueError(f"conditioning {clash} meets instrument, treatment, or outcome")
        return self
