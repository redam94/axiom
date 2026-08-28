"""Refuting the graph itself — the one input nothing else checks.

Every identification result in axiom is conditional on the diagram, and says
so: ``identify_effect`` returns an ``Assumption`` named ``graph_is_correct``
in state ``unverified``. ``diagnose.refute`` then works hard on the *estimate*
— placebo treatments, permutations, subsets, added noise — while the graph
that licensed the estimate goes unexamined.

But a graph is not merely an assumption. It is a *falsifiable* one: it implies
conditional independencies, and the data either shows them or does not. Testing
them is the cheapest, sharpest check available, and it is the missing step in
the loop:

    draw -> **refute the structure** -> repair -> identify -> estimate ->
    refute the estimate -> decide

Three properties this module insists on
---------------------------------------
**It can only refute, never confirm.** A test that fails to reject is not
evidence of independence — it may just be underpowered, or the dependence may
be nonlinear and invisible to a correlation. So a graph that survives comes
back ``downgraded`` with ``graph_is_correct`` still ``unverified``, never
``identified``. A graph that fails comes back ``blocked`` with that assumption
marked ``violated``.

**Effect size decides, not p alone.** With few rows nothing rejects and a graph
passes by being untested; with a million rows everything rejects, because every
graph is an idealization and idealizations are detectably false at scale. Both
``alpha`` and ``effect_threshold`` are parameters, and the report carries the
partial correlation of every check so a reader can see which it was.

**A refutation names its repair.** Each implication tested is the absence of
one edge, so a failure points at exactly the edge whose absence the data
denies. ``implicated`` lists them worst-first: not "your graph is wrong" but
"these four edges are missing, this one most of all". That is what makes
refutation a *step* rather than a verdict, and it is the bridge to discovery —
the same edges a search would add, but anchored on the graph you drew.
"""

from __future__ import annotations

import itertools
from typing import Literal

from pydantic import Field, model_validator

from axiom.core import Assumption, Multiplicity, NonEmptyStr, Spec, Unsupported, Verdict, adjust
from axiom.discover.independence import IndependenceResult, PartialCorrelation
from axiom.discover.score import Dataset
from axiom.identify import CausalGraph

__all__ = [
    "Correction",
    "ImpliedIndependence",
    "IndependenceCheck",
    "StructureRefutation",
    "adjust",
    "implied_independencies",
    "refute_structure",
]

Correction = Multiplicity
"""This subpackage's name for ``core.Multiplicity``; the arithmetic is ``core.adjust``."""

MAX_CONDITIONING = 4
"""Largest separating set searched for when the obvious one does not separate."""


class ImpliedIndependence(Spec):
    """``x`` is independent of ``y`` given ``given`` — a claim the graph makes.

    Each one corresponds to the *absence* of an edge between ``x`` and ``y``,
    which is why a failed test names a repair.
    """

    x: NonEmptyStr
    y: NonEmptyStr
    given: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _distinct(self) -> ImpliedIndependence:
        if self.x == self.y:
            raise ValueError("an implication relates two different variables")
        return self

    @property
    def edge(self) -> tuple[str, str]:
        """The adjacency whose absence this claim asserts."""
        return (self.x, self.y) if self.x <= self.y else (self.y, self.x)

    def describe(self) -> str:
        given = f" given {', '.join(self.given)}" if self.given else " marginally"
        return f"{self.x} is independent of {self.y}{given}"


class IndependenceCheck(Spec):
    """One implication, tested: the evidence against it and the verdict on it."""

    implication: ImpliedIndependence
    result: IndependenceResult
    adjusted_p: float = Field(ge=0.0, le=1.0)
    refuted: bool

    @property
    def effect(self) -> float:
        return self.result.effect

    def describe(self) -> str:
        mark = "REFUTED" if self.refuted else "held"
        return (
            f"{mark}: {self.implication.describe()} — r = {self.result.correlation:+.3f}, "
            f"p = {self.adjusted_p:.3g}"
        )


class StructureRefutation(Spec):
    """What the data says about the graph's own claims.

    ``implicated`` is the useful output when something fails: the edges whose
    absence the data denies, worst violation first. ``verdict`` is ``blocked``
    with ``graph_is_correct`` marked ``violated`` when anything was refuted,
    and ``downgraded`` with it still ``unverified`` otherwise — because
    surviving a test is not passing one.
    """

    graph_hash: str = ""
    graph_name: str = ""
    checks: tuple[IndependenceCheck, ...] = ()
    alpha: float = Field(gt=0, lt=1)
    effect_threshold: float = Field(ge=0, le=1)
    correction: Correction = "holm"
    n_tested: int = Field(ge=0)
    n_refuted: int = Field(ge=0)
    implicated: tuple[tuple[str, str, float], ...] = ()
    untested: tuple[tuple[str, str], ...] = ()
    verdict: Verdict
    detail: dict[str, str] = {}

    @model_validator(mode="after")
    def _counts(self) -> StructureRefutation:
        if self.n_tested != len(self.checks):
            raise ValueError("n_tested counts the checks")
        if self.n_refuted != sum(c.refuted for c in self.checks):
            raise ValueError("n_refuted counts the refuted checks")
        return self

    @property
    def refuted(self) -> bool:
        return self.n_refuted > 0

    @property
    def worst(self) -> IndependenceCheck | None:
        """The check with the largest violation, refuted or not."""
        return max(self.checks, key=lambda c: c.effect) if self.checks else None

    def summary(self) -> str:
        if not self.checks:
            return "the graph implies nothing testable here"
        if not self.refuted:
            biggest = self.worst
            assert biggest is not None
            return (
                f"{self.n_tested} implications tested, none refuted "
                f"(largest dependence left over: r = {biggest.result.correlation:+.3f}). "
                "Surviving is not passing"
            )
        first = self.implicated[0]
        return (
            f"{self.n_refuted} of {self.n_tested} implications refuted; the graph denies "
            f"an edge between {first[0]} and {first[1]} that the data shows at r = {first[2]:.3f}"
        )


# -- what a graph claims ------------------------------------------------------------------


def implied_independencies(
    graph: CausalGraph, *, max_conditioning: int = MAX_CONDITIONING
) -> tuple[tuple[ImpliedIndependence | None, tuple[str, str]], ...]:
    """One testable claim per non-adjacent pair, with a set the graph says separates them.

    For a DAG the local Markov property supplies it directly: a variable is
    independent of its non-descendants given its parents, so ``pa(y)`` separates
    ``y`` from any non-adjacent non-descendant ``x``. Where that does not
    separate — bidirected edges make it possible — subsets of the pair's
    ancestors are searched up to ``max_conditioning``.

    A pair for which nothing separating was found comes back with ``None``,
    and the caller reports it as untested rather than as passed.
    """

    def adjacent(node: str) -> frozenset[str]:
        return graph.parents(node) | graph.children(node) | graph.siblings(node)

    out: list[tuple[ImpliedIndependence | None, tuple[str, str]]] = []
    for x, y in itertools.combinations(sorted(graph.nodes), 2):
        if y in adjacent(x):
            continue
        separating = _separating_set(graph, x, y, max_conditioning=max_conditioning)
        pair = (x, y)
        if separating is None:
            out.append((None, pair))
            continue
        out.append((ImpliedIndependence(x=x, y=y, given=separating), pair))
    return tuple(out)


def _separating_set(
    graph: CausalGraph, x: str, y: str, *, max_conditioning: int
) -> tuple[str, ...] | None:
    """A set the graph says d-separates ``x`` and ``y``, preferring the parent sets."""
    for first, second in ((x, y), (y, x)):
        candidate = tuple(sorted(graph.parents(second) - {first}))
        if graph.d_separated(first, second, candidate):
            return candidate
    pool = sorted((graph.ancestors({x, y}, include_self=True) - {x, y}) & graph.measured)
    for size in range(min(max_conditioning, len(pool)) + 1):
        for chosen in itertools.combinations(pool, size):
            if graph.d_separated(x, y, chosen):
                return tuple(chosen)
    return None


# -- multiplicity ---------------------------------------------------------------------------


# ``adjust`` moved to ``core.multiplicity`` so ``design`` could reach it too, and is
# imported above because this module has always been where callers found it.


# -- the check ------------------------------------------------------------------------------


def _assumption(state: Literal["violated", "unverified"]) -> Assumption:
    base = Assumption(
        name="graph_is_correct",
        facet="identification",
        statement=(
            "the diagram is right: every arrow absent is an assumed absence of a direct cause"
        ),
        challenged_by="a conditional independence the graph implies and the data rejects",
    )
    return base.violated() if state == "violated" else base


def refute_structure(
    graph: CausalGraph,
    data: Dataset,
    *,
    alpha: float = 0.05,
    effect_threshold: float = 0.0,
    correction: Correction = "holm",
    max_conditioning: int = MAX_CONDITIONING,
) -> StructureRefutation | Unsupported:
    """Test every independence the graph implies, and report the edges any failure implicates.

    An implication is *refuted* when its adjusted p-value is below ``alpha``
    **and** its partial correlation is at least ``effect_threshold``. The
    second condition is what keeps the answer meaningful at large sample size,
    where everything is significant; leave it at zero for a pure significance
    test and read ``worst`` to see what it would have taken.

    Unmeasured nodes are excluded — the data cannot speak about them — so a
    graph whose testable content lies entirely behind latents comes back with
    nothing tested, and says so rather than passing.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}")
    measured = sorted(graph.measured)
    missing = sorted(set(measured) - set(data.names))
    if missing:
        return Unsupported(
            reason=f"the data has no column for {missing}, which the graph says is measured",
            missing=tuple(missing),
        )
    tester = PartialCorrelation(data)
    claims = implied_independencies(graph, max_conditioning=max_conditioning)

    testable: list[ImpliedIndependence] = []
    untested: list[tuple[str, str]] = []
    results: list[IndependenceResult] = []
    for implication, pair in claims:
        if implication is None:
            untested.append(pair)
            continue
        if implication.x in graph.unmeasured or implication.y in graph.unmeasured:
            untested.append(pair)
            continue
        if set(implication.given) & set(graph.unmeasured):
            untested.append(pair)
            continue
        outcome = tester.test(implication.x, implication.y, implication.given)
        if isinstance(outcome, Unsupported):
            untested.append(pair)
            continue
        testable.append(implication)
        results.append(outcome)

    adjusted = adjust([r.p_value for r in results], correction)
    checks = tuple(
        IndependenceCheck(
            implication=implication,
            result=result,
            adjusted_p=p,
            refuted=bool(p < alpha and result.effect >= effect_threshold),
        )
        for implication, result, p in zip(testable, results, adjusted, strict=True)
    )
    implicated = tuple(
        sorted(
            ((c.implication.edge[0], c.implication.edge[1], c.effect) for c in checks if c.refuted),
            key=lambda item: item[2],
            reverse=True,
        )
    )
    n_refuted = sum(c.refuted for c in checks)
    if n_refuted:
        verdict = Verdict(
            status="blocked",
            reason=(
                f"the data rejects {n_refuted} of {len(checks)} independencies this graph "
                f"implies; the edges whose absence it denies are "
                f"{[f'{a}-{b}' for a, b, _ in implicated]}, worst first"
            ),
            assumptions=(_assumption("violated"),),
            route="implied_independence",
        )
    else:
        verdict = Verdict(
            status="downgraded",
            reason=(
                f"{len(checks)} implied independencies were tested and none was rejected. "
                "That is a failure to refute, not a confirmation: a test with little power, "
                "or a dependence that is not linear, looks exactly like this"
            ),
            assumptions=(_assumption("unverified"),),
            route="implied_independence",
        )
    return StructureRefutation(
        graph_hash=graph.content_hash(),
        graph_name=graph.name,
        checks=checks,
        alpha=alpha,
        effect_threshold=effect_threshold,
        correction=correction,
        n_tested=len(checks),
        n_refuted=n_refuted,
        implicated=implicated,
        untested=tuple(sorted(untested)),
        verdict=verdict,
        detail={
            "rows": str(data.n_rows),
            "variables": str(len(measured)),
            "largest_effect": f"{max((c.effect for c in checks), default=0.0):.4f}",
        },
    )
