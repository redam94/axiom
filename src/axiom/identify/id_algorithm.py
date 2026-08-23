"""ID and IDC: sound *and complete* identification under latent confounding.

``identify()`` searches a menu — back-door, front-door, instrument — and
reports the first route that works. That is a useful answer and an incomplete
one: an effect can be identifiable while no named route applies, and a search
that finds nothing has not proved anything. Under latent confounding, drawn
honestly as bidirected edges, the incompleteness is the common case rather
than a corner.

The ID algorithm (Tian & Pearl 2002; Shpitser & Pearl 2006; Huang & Valtorta
2006) settles it. Given an ADMG it decides whether ``P(y | do(x))`` is
identifiable, returns the estimand as a ``Formula`` when it is, and when it is
not returns a **hedge** — a pair of nested C-components that constitutes a
*proof* of non-identifiability. Sound and complete: no route left unexplored,
and "not identifiable" means not identifiable, not "we did not find one".

That distinction is the point of this module. "The search found nothing" and
"no estimand exists" are different findings with different consequences: the
first says look harder, the second says go and run an experiment.

Latent variables come in two spellings and both work. A bidirected edge is
already a projected latent; a node named in ``unmeasured`` is projected out
here by ``latent_projection`` before the algorithm runs, so the graph you
drew and the graph the algorithm sees agree by construction.

IDC (Shpitser & Pearl 2008) extends this to conditional effects
``P(y | do(x), z)`` by moving what can be moved from the conditioning set into
the intervention set and calling ID on the rest.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from pydantic import Field, model_validator

from axiom.core import Assumption, NonEmptyStr, Spec, Verdict
from axiom.identify.formula import (
    Density,
    Formula,
    marginal,
    product,
    ratio,
    to_text,
)
from axiom.identify.graph import CausalGraph, GraphError

__all__ = [
    "Hedge",
    "IdentifiedEffect",
    "districts",
    "identify_conditional_effect",
    "identify_effect",
    "induced_subgraph",
    "latent_projection",
]


# -- graph surgery ---------------------------------------------------------------------


def induced_subgraph(graph: CausalGraph, nodes: Iterable[str]) -> CausalGraph:
    """The subgraph on ``nodes``, keeping every edge with both ends inside it."""
    keep = set(nodes)
    unknown = sorted(keep - set(graph.nodes))
    if unknown:
        raise GraphError(f"unknown nodes: {unknown}")
    return CausalGraph(
        nodes=tuple(sorted(keep)),
        edges=tuple((a, b) for a, b in graph.edges if a in keep and b in keep),
        bidirected=tuple((a, b) for a, b in graph.bidirected if a in keep and b in keep),
        unmeasured=tuple(n for n in graph.unmeasured if n in keep),
        name=graph.name,
    )


def _without_edges_into(graph: CausalGraph, nodes: Iterable[str]) -> CausalGraph:
    """``G_{bar X}``: every edge pointing into any of ``nodes`` removed."""
    targets = set(nodes)
    return CausalGraph(
        nodes=graph.nodes,
        edges=tuple((a, b) for a, b in graph.edges if b not in targets),
        bidirected=tuple((a, b) for a, b in graph.bidirected if not ({a, b} & targets)),
        unmeasured=graph.unmeasured,
        name=graph.name,
    )


def _without_edges_out_of(graph: CausalGraph, nodes: Iterable[str]) -> CausalGraph:
    """``G_{underline X}``: every directed edge leaving any of ``nodes`` removed."""
    sources = set(nodes)
    return CausalGraph(
        nodes=graph.nodes,
        edges=tuple((a, b) for a, b in graph.edges if a not in sources),
        bidirected=graph.bidirected,
        unmeasured=graph.unmeasured,
        name=graph.name,
    )


def districts(graph: CausalGraph) -> tuple[tuple[str, ...], ...]:
    """The C-components: connected components of the bidirected part (Tian & Pearl 2002).

    Two variables share a district when a latent cause reaches both, possibly
    through a chain of them. The districts are what the joint factorizes over
    once latents are marginalized out, and they are what the ID algorithm
    recurses on.
    """
    parent: dict[str, str] = {n: n for n in graph.nodes}

    def find(node: str) -> str:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for a, b in graph.bidirected:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    groups: dict[str, list[str]] = {}
    for node in graph.nodes:
        groups.setdefault(find(node), []).append(node)
    return tuple(sorted(tuple(sorted(members)) for members in groups.values()))


def latent_projection(graph: CausalGraph, keep: Iterable[str] | None = None) -> CausalGraph:
    """Project the unmeasured nodes out, leaving an ADMG over the ones kept.

    The standard construction (Verma 1993; Richardson et al. 2023): a directed
    edge ``a -> b`` survives when a directed path runs from ``a`` to ``b``
    through unmeasured nodes only, and a bidirected edge ``a <-> b`` appears
    when some unmeasured node reaches both that way. Existing bidirected edges
    are latents already projected once, so they are expanded and re-projected
    rather than special-cased — the two spellings of "there is something I did
    not measure" cannot disagree.
    """
    kept = set(keep) if keep is not None else set(graph.measured)
    unknown = sorted(kept - set(graph.nodes))
    if unknown:
        raise GraphError(f"unknown nodes: {unknown}")

    # expand every bidirected edge into an explicit hidden common cause
    edges = list(graph.edges)
    hidden = [n for n in graph.nodes if n not in kept]
    for i, (a, b) in enumerate(graph.bidirected):
        node = f"__u{i}"
        hidden.append(node)
        edges.append((node, a))
        edges.append((node, b))
    latent = set(hidden)
    children: dict[str, set[str]] = {}
    for a, b in edges:
        children.setdefault(a, set()).add(b)

    def reach_through_latents(start: str) -> set[str]:
        """Kept nodes reachable from ``start`` by a directed path through latents only."""
        found: set[str] = set()
        stack = list(children.get(start, ()))
        seen: set[str] = set()
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            if node in latent:
                stack.extend(children.get(node, ()))
            else:
                found.add(node)
        return found

    directed = {(a, b) for a in kept for b in reach_through_latents(a)}
    bidirected: set[tuple[str, str]] = set()
    for node in latent:
        touched = sorted(reach_through_latents(node))
        for i, a in enumerate(touched):
            for b in touched[i + 1 :]:
                bidirected.add((a, b))
    return CausalGraph(
        nodes=tuple(sorted(kept)),
        edges=tuple(sorted(directed)),
        bidirected=tuple(sorted(bidirected)),
        name=graph.name,
    )


# -- the failure object ----------------------------------------------------------------


class Hedge(Spec):
    """The witness that an effect is *not* identifiable — a proof, not a failed search.

    A hedge is a pair of C-components ``F subset F'`` in a subgraph, both
    containing part of the treatment's ancestry, which together show that no
    estimand in the observational distribution equals the interventional one
    (Shpitser & Pearl 2006, Thm. 4). ``root`` is ``F'`` and ``subset`` is
    ``F``; ``variables`` is the vertex set they live in.

    It is reported rather than reduced to ``False`` because the two sets say
    *where* the obstruction is, and therefore what would remove it: measure
    something that breaks the district, or intervene.
    """

    root: tuple[NonEmptyStr, ...] = Field(min_length=1)
    subset: tuple[NonEmptyStr, ...] = Field(min_length=1)
    variables: tuple[NonEmptyStr, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _nested(self) -> Hedge:
        if not set(self.subset) <= set(self.root):
            raise ValueError("a hedge's subset must lie inside its root")
        return self

    def describe(self) -> str:
        return (
            f"the C-component {list(self.root)} and its sub-component {list(self.subset)} "
            "form a hedge: no function of the observational distribution equals this effect"
        )


class _HedgeFound(Exception):
    """Raised inside the recursion; converted to a verdict at the boundary."""

    def __init__(self, hedge: Hedge) -> None:
        self.hedge = hedge
        super().__init__(hedge.describe())


class IdentifiedEffect(Spec):
    """What ID returned: a formula, or a hedge, and a verdict either way.

    ``formula`` is an expression in the *observational* distribution — no
    ``do`` left in it — that equals ``P(outcome | do(treatment))`` under the
    graph. ``hedge`` is set exactly when there is no such expression.
    """

    treatment: tuple[NonEmptyStr, ...] = Field(min_length=1)
    outcome: tuple[NonEmptyStr, ...] = Field(min_length=1)
    conditioned: tuple[NonEmptyStr, ...] = ()
    graph_hash: str = ""
    formula: Formula | None = None
    hedge: Hedge | None = None
    verdict: Verdict
    detail: dict[str, str] = {}

    @model_validator(mode="after")
    def _one_or_the_other(self) -> IdentifiedEffect:
        if (self.formula is None) == (self.hedge is None):
            raise ValueError("an ID result carries either a formula or a hedge, never both")
        return self

    @property
    def identified(self) -> bool:
        return self.formula is not None

    def to_text(self) -> str:
        """The estimand as a line of text, or the hedge that says there is none."""
        if self.formula is None:
            assert self.hedge is not None
            return self.hedge.describe()
        return to_text(self.formula)


# -- the algorithm ---------------------------------------------------------------------


def _conditional_of(
    expression: Formula, node: str, preceding: Sequence[str], universe: set[str]
) -> Formula:
    """``P(node | preceding)`` derived from ``expression``, a distribution over ``universe``.

    When the expression is the raw joint this is just a density; otherwise it
    is the ratio of two marginals of it, which is what makes ``Ratio`` a
    necessary node type rather than a convenience.
    """
    before = tuple(sorted(preceding))
    if (
        isinstance(expression, Density)
        and set(expression.outcomes) == universe
        and not expression.given
    ):
        return Density(outcomes=(node,), given=before)
    numerator = marginal(universe - set(before) - {node}, expression)
    denominator = marginal(universe - set(before), expression)
    if numerator == denominator:  # pragma: no cover - only when node is not in the universe
        return numerator
    return ratio(numerator, denominator)


def _identify(
    outcome: set[str], treatment: set[str], expression: Formula, graph: CausalGraph
) -> Formula:
    """The ID algorithm (Shpitser & Pearl 2006, Fig. 3); raises ``_HedgeFound``."""
    universe = set(graph.nodes)

    # 1. nothing intervened on: marginalize
    if not treatment:
        return marginal(universe - outcome, expression)

    # 2. drop everything that is not an ancestor of the outcome
    ancestors = set(graph.ancestors(outcome, include_self=True))
    if universe - ancestors:
        return _identify(
            outcome,
            treatment & ancestors,
            marginal(universe - ancestors, expression),
            induced_subgraph(graph, ancestors),
        )

    # 3. anything that is not an ancestor of the outcome once X is cut off can be
    #    intervened on for free
    cut = _without_edges_into(graph, treatment)
    extra = (universe - treatment) - set(cut.ancestors(outcome, include_self=True))
    if extra:
        return _identify(outcome, treatment | extra, expression, graph)

    # 4. the remaining graph splits into districts: identify each and multiply
    remainder = induced_subgraph(graph, universe - treatment)
    components = districts(remainder)
    if len(components) > 1:
        factors = [
            _identify(set(component), universe - set(component), expression, graph)
            for component in components
        ]
        return marginal(universe - (outcome | treatment), product(*factors))

    # 5. one district left
    component = set(components[0])
    whole = districts(graph)
    if len(whole) == 1 and set(whole[0]) == universe:
        raise _HedgeFound(
            Hedge(
                root=tuple(sorted(universe)),
                subset=tuple(sorted(component)),
                variables=tuple(sorted(universe)),
            )
        )
    order = list(graph.topological_order())
    if any(set(c) == component for c in whole):
        factors = [
            _conditional_of(expression, node, order[: order.index(node)], universe)
            for node in order
            if node in component
        ]
        return marginal(component - outcome, product(*factors))
    bigger = next(set(c) for c in whole if component < set(c))
    factors = [
        _conditional_of(expression, node, order[: order.index(node)], universe)
        for node in order
        if node in bigger
    ]
    return _identify(
        outcome, treatment & bigger, product(*factors), induced_subgraph(graph, bigger)
    )


def _assumptions(graph: CausalGraph) -> tuple[Assumption, ...]:
    return (
        Assumption(
            name="graph_is_correct",
            facet="identification",
            statement=(
                "the diagram is right: every arrow present is a possible direct cause and "
                "every arrow absent is an assumed absence of one"
            ),
            challenged_by="an omitted edge, or an omitted latent common cause",
        ),
        Assumption(
            name="positivity",
            facet="identification",
            statement=(
                "every combination of the variables the estimand conditions on occurs with "
                "positive probability"
            ),
            challenged_by="a covariate pattern with no observations at some treatment level",
        ),
    )


def identify_effect(
    graph: CausalGraph,
    treatment: str | Iterable[str],
    outcome: str | Iterable[str],
) -> IdentifiedEffect:
    """``P(outcome | do(treatment))`` by the ID algorithm: the estimand, or the hedge.

    Sound and complete for acyclic graphs with latent confounding. Unmeasured
    nodes are projected out first, so a graph drawn with explicit latents and
    the same graph drawn with bidirected edges give the same answer.

    The verdict is ``downgraded`` rather than ``identified`` when an estimand
    exists, because the graph itself and positivity are assumptions the
    algorithm cannot check — it identifies *given* the diagram.
    """
    xs = {treatment} if isinstance(treatment, str) else set(treatment)
    ys = {outcome} if isinstance(outcome, str) else set(outcome)
    if not xs or not ys:
        raise ValueError("identification needs at least one treatment and one outcome")
    overlap = sorted(xs & ys)
    if overlap:
        raise GraphError(f"a variable cannot be both treatment and outcome: {overlap}")
    for node in xs | ys:
        graph._require(node)
    hidden = sorted((xs | ys) & set(graph.unmeasured))
    if hidden:
        raise GraphError(f"treatment and outcome must be measured; {hidden} are not")

    admg = latent_projection(graph)
    joint = Density(outcomes=tuple(sorted(admg.nodes)))
    try:
        formula = _identify(ys, xs, joint, admg)
    except _HedgeFound as found:
        return IdentifiedEffect(
            treatment=tuple(sorted(xs)),
            outcome=tuple(sorted(ys)),
            graph_hash=graph.content_hash(),
            hedge=found.hedge,
            verdict=Verdict(
                status="blocked",
                reason=(
                    f"{found.hedge.describe()}. This is a proof, not a failed search: no "
                    "adjustment set, no instrument and no front-door path exists either. "
                    "Measure a variable that breaks the district, or intervene"
                ),
                route="id_algorithm",
            ),
            detail={"districts": str(len(districts(admg)))},
        )
    return IdentifiedEffect(
        treatment=tuple(sorted(xs)),
        outcome=tuple(sorted(ys)),
        graph_hash=graph.content_hash(),
        formula=formula,
        verdict=Verdict(
            status="downgraded",
            reason="",
            assumptions=_assumptions(admg),
            route="id_algorithm",
        ),
        detail={"districts": str(len(districts(admg)))},
    )


def identify_conditional_effect(
    graph: CausalGraph,
    treatment: str | Iterable[str],
    outcome: str | Iterable[str],
    conditioned: Iterable[str],
) -> IdentifiedEffect:
    """``P(outcome | do(treatment), conditioned)`` by IDC (Shpitser & Pearl 2008).

    A conditioning variable the outcome is independent of — given the other
    conditioning variables and the intervention, in the graph with edges into
    the treatment and out of that variable removed — can be *moved into the
    ``do``* without changing the quantity. IDC moves everything movable and
    then calls ID on what is left, dividing by its own marginal to condition.

    ``conditioned`` on the result records the query that was asked, not the
    rewritten one: the estimand answers the original question either way, and
    a reader should not have to reverse-engineer which variables the algorithm
    chose to reinterpret.
    """
    xs = {treatment} if isinstance(treatment, str) else set(treatment)
    ys = {outcome} if isinstance(outcome, str) else set(outcome)
    zs = set(conditioned)
    for node in xs | ys | zs:
        graph._require(node)
    clash = sorted(zs & (xs | ys))
    if clash:
        raise GraphError(f"a conditioning variable cannot also be treatment or outcome: {clash}")
    if not zs:
        return identify_effect(graph, xs, ys)

    admg = latent_projection(graph)
    intervened = set(xs)
    remaining = set(zs)
    moved = True
    while moved and remaining:
        moved = False
        for node in sorted(remaining):
            surgical = _without_edges_out_of(_without_edges_into(admg, intervened), [node])
            if surgical.d_separated(ys, {node}, (intervened | remaining) - {node}):
                intervened.add(node)
                remaining.discard(node)
                moved = True
                break

    joint = Density(outcomes=tuple(sorted(admg.nodes)))
    try:
        formula = _identify(ys | remaining, intervened, joint, admg)
    except _HedgeFound as found:
        return IdentifiedEffect(
            treatment=tuple(sorted(xs)),
            outcome=tuple(sorted(ys)),
            conditioned=tuple(sorted(zs)),
            graph_hash=graph.content_hash(),
            hedge=found.hedge,
            verdict=Verdict(
                status="blocked",
                reason=(
                    f"{found.hedge.describe()}. IDC needs the joint effect on the outcome and "
                    "the conditioning variables it could not move into the intervention, and "
                    "that effect is not identifiable"
                ),
                route="idc_algorithm",
            ),
            detail={"moved_into_do": ", ".join(sorted(intervened - xs))},
        )
    conditional = formula if not remaining else ratio(formula, marginal(ys, formula))
    return IdentifiedEffect(
        treatment=tuple(sorted(xs)),
        outcome=tuple(sorted(ys)),
        conditioned=tuple(sorted(zs)),
        graph_hash=graph.content_hash(),
        formula=conditional,
        verdict=Verdict(
            status="downgraded",
            reason="",
            assumptions=_assumptions(admg),
            route="idc_algorithm",
        ),
        detail={"moved_into_do": ", ".join(sorted(intervened - xs))},
    )
