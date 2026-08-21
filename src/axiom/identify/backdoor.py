"""Back-door adjustment sets and node roles for one treatment–outcome pair.

Everything here is graphical: it reads a ``CausalGraph`` and answers, for a
single treatment ``x`` and outcome ``y``, *which sets of covariates license
covariate adjustment* and *what part each node plays*. No data is touched.

**Criterion.** ``backdoor_admissible`` implements Pearl's back-door criterion
(Pearl 2009, *Causality*, 2nd ed., Def. 3.3.1): ``Z`` is admissible when no
node of ``Z`` is a descendant of ``x`` and ``Z`` d-separates ``x`` from ``y``
in :math:`G_{\\underline{x}}`, the graph with every edge out of ``x`` removed.
Admissibility licenses the adjustment formula
:math:`P(y \\mid do(x)) = \\sum_z P(y \\mid x, z)\\,P(z)` (Thm. 3.3.2).

This is the back-door criterion, *not* the complete adjustment criterion of
Shpitser, VanderWeele & Robins (2010). The difference does not matter for a
single treatment: if ``Z`` satisfies the adjustment criterion for one ``x``
and one ``y``, then ``Z`` minus the descendants of ``x`` satisfies the
back-door criterion. (Sketch: a non-collider ``w`` of a path in
:math:`G_{\\underline{x}}` that lies in ``Z`` and descends from ``x`` has an
out-edge along the path; following it reaches ``y`` — so ``w`` is on a proper
causal path and forbidden — or a collider that must be an ancestor of some
node of ``Z`` outside ``De(x)``, which is impossible for a descendant of
``x``.) So the two criteria agree on whether *any* admissible set exists,
measured or not, and the sets the back-door criterion misses are exactly
those padded with harmless descendants of ``x``. That is sufficient for 1.0,
where ``identify`` chooses a route (back-door, front-door, instrument) and
a verdict status; the generalization to multiple treatments is a later phase.

**The canonical set.** Let ``R`` be the candidate pool (every node except
``x``, ``y`` and the descendants of ``x``; measured nodes only by default)
and :math:`Z_0 = R \\cap An_{G_{\\underline{x}}}(\\{x, y\\})` the candidates
that are ancestors of ``x`` or ``y`` in :math:`G_{\\underline{x}}`. Then
(van der Zander, Liśkiewicz & Textor 2014, Lemma 3, for ``I = ∅``):

* some subset of ``R`` is admissible iff :math:`Z_0` is — so existence is
  decided in polynomial time by one d-separation test
  (``admissible_set_exists``, ``canonical_adjustment_set``); and
* every inclusion-minimal admissible subset of ``R`` is a subset of
  :math:`Z_0` — so ``minimal_adjustment_sets`` enumerates subsets of
  :math:`Z_0` only, and nodes outside it (isolated nodes, descendants of
  ``y``, upstream causes of nothing relevant) do not count towards its
  ``max_candidates`` cap.

The moral-graph argument: if some ``Z ⊆ R`` d-separates ``x`` and ``y`` in
:math:`G = G_{\\underline{x}}`, then ``Z`` separates them in the moral graph
of the ancestral set of ``{x, y} ∪ Z``. The moral graph of the ancestral
set of ``{x, y}`` is a subgraph of it, so any path there from ``x`` to ``y``
avoiding :math:`Z_0` would avoid ``Z`` as well (``Z ∩ An({x, y}) ⊆ Z_0``) — a
contradiction. Hence :math:`Z \\cap An(\\{x, y\\}) \\subseteq Z_0` separates
whenever ``Z`` does; a minimal ``Z`` therefore equals that intersection and
lies inside :math:`Z_0`.

Enumerating *all* admissible sets (``adjustment_sets``) is inherently
exponential in ``|R|`` and is capped by ``max_candidates``; past the cap the
functions return ``Unsupported`` rather than a partial list.

**Measurement.** Functions that take ``measured_only`` raise ``GraphError``
when ``measured_only=True`` and ``x`` or ``y`` is itself in
``graph.unmeasured``: a measured adjustment set for an unobserved treatment
or outcome is a malformed query, not an empty answer.

**Roles** are descriptive and are never used to decide admissibility — that
is always d-separation. Their definitions are at ``roles``.

The graph's ``selection`` and ``feedback`` fields are ignored here: transport
is ``identify.transport``'s job, and the verdict layer is where a summary
graph that hides feedback downgrades a static adjustment.
"""

from __future__ import annotations

from collections.abc import Iterable
from itertools import combinations
from typing import Literal

from axiom.core.result import Unsupported
from axiom.core.spec import Spec
from axiom.identify.graph import CausalGraph, GraphError

__all__ = [
    "ROLE_PRECEDENCE",
    "AdjustmentSets",
    "Role",
    "RoleAssignment",
    "adjustment_sets",
    "admissible_set_exists",
    "assign_roles",
    "backdoor_admissible",
    "canonical_adjustment_set",
    "minimal_adjustment_sets",
    "requires_unmeasured",
    "roles",
]

AdjustmentSets = tuple[frozenset[str], ...]
"""Admissible sets, sorted by size and then lexicographically by member names."""

Role = Literal[
    "treatment",
    "outcome",
    "confounder",
    "mediator",
    "collider",
    "instrument",
    "proxy",
    "descendant_of_outcome",
    "neutral",
]

ROLE_PRECEDENCE: tuple[Role, ...] = (
    "treatment",
    "outcome",
    "mediator",
    "confounder",
    "instrument",
    "collider",
    "descendant_of_outcome",
    "proxy",
    "neutral",
)
"""When a node meets several role definitions it receives the first one listed."""


# -- validation ---------------------------------------------------------------------


def _check_pair(graph: CausalGraph, x: str, y: str, *, measured_only: bool = False) -> None:
    for n in (x, y):
        if n not in graph.nodes:
            raise GraphError(f"unknown node {n!r}; nodes are {list(graph.nodes)}")
    if x == y:
        raise GraphError(f"treatment and outcome must differ; both are {x!r}")
    if measured_only:
        hidden = [n for n in (x, y) if n in graph.unmeasured]
        if hidden:
            raise GraphError(
                f"treatment/outcome is unmeasured: {hidden}; a measured adjustment set for "
                "an unobserved treatment or outcome is not a meaningful query "
                "(pass measured_only=False to ask the purely graphical question)"
            )


def _check_conditioning(graph: CausalGraph, x: str, y: str, zs: frozenset[str]) -> None:
    unknown = sorted(zs - set(graph.nodes))
    if unknown:
        raise GraphError(f"adjustment set names unknown nodes: {unknown}")
    if x in zs or y in zs:
        raise GraphError(
            f"an adjustment set may not contain the treatment {x!r} or the outcome {y!r}"
        )


def _candidates(graph: CausalGraph, x: str, y: str, *, measured_only: bool) -> frozenset[str]:
    """Nodes eligible for a back-door set: not ``x``, ``y``, or a descendant of ``x``."""
    pool = graph.measured if measured_only else frozenset(graph.nodes)
    return pool - graph.descendants(x, include_self=True) - {y}


# -- the criterion ------------------------------------------------------------------


def backdoor_admissible(graph: CausalGraph, x: str, y: str, z: Iterable[str]) -> bool:
    """Pearl's back-door criterion for ``z`` relative to ``(x, y)``.

    ``z`` is admissible iff (i) no node of ``z`` is a descendant of ``x`` and
    (ii) ``z`` d-separates ``x`` and ``y`` in :math:`G_{\\underline{x}}`
    (Pearl 2009, Def. 3.3.1). Bidirected edges are latent common causes and
    are respected by the d-separation test. Whether the nodes of ``z`` are
    measured is *not* checked here; see ``adjustment_sets(measured_only=...)``
    and ``requires_unmeasured``.

    Raises ``GraphError`` if ``x``, ``y``, or any node of ``z`` is unknown,
    if ``x == y``, or if ``z`` contains ``x`` or ``y`` (conditioning on the
    treatment or outcome is a malformed query, not an inadmissible set).
    """
    zs = frozenset(z)
    _check_pair(graph, x, y)
    _check_conditioning(graph, x, y, zs)
    if zs & graph.descendants(x):
        return False
    return graph.remove_edges_out_of(x).d_separated(x, y, zs)


def _canonical_within(
    graph: CausalGraph, x: str, y: str, candidates: frozenset[str]
) -> frozenset[str] | None:
    """The canonical subset :math:`Z_0 = R \\cap An_{G_{\\underline{x}}}(\\{x, y\\})` if admissible.

    Returns ``None`` when :math:`Z_0` is not admissible, which by Lemma 3 of
    van der Zander, Liśkiewicz & Textor (2014) means *no* subset of
    ``candidates`` is (the argument is in the module docstring).
    ``candidates`` must already exclude ``x``, ``y``, and the descendants of
    ``x``.
    """
    g = graph.remove_edges_out_of(x)
    canonical = candidates & g.ancestors((x, y))
    return canonical if g.d_separated(x, y, canonical) else None


def canonical_adjustment_set(
    graph: CausalGraph, x: str, y: str, *, measured_only: bool = True
) -> frozenset[str] | None:
    """The canonical back-door set, or ``None`` when no admissible set exists.

    With ``R`` the candidates (every node but ``x``, ``y`` and the descendants
    of ``x``; measured nodes only when ``measured_only``), the canonical set
    is :math:`Z_0 = R \\cap An_{G_{\\underline{x}}}(\\{x, y\\})`. It is
    admissible iff some subset of ``R`` is, and every inclusion-minimal
    admissible subset of ``R`` lies inside it (van der Zander, Liśkiewicz &
    Textor 2014, Lemma 3). Computed in polynomial time: one ancestor sweep
    and one d-separation test. ``frozenset()`` is a valid answer (no
    adjustment needed); ``None`` means no admissible set exists among the
    candidates, which is a definite negative, not a failure.

    :math:`Z_0` is usually not minimal — it carries every relevant ancestor —
    so use ``minimal_adjustment_sets`` for the smallest sets and this for a
    cheap, always-available admissible set.
    """
    _check_pair(graph, x, y, measured_only=measured_only)
    return _canonical_within(graph, x, y, _candidates(graph, x, y, measured_only=measured_only))


def admissible_set_exists(
    graph: CausalGraph, x: str, y: str, *, measured_only: bool = True
) -> bool:
    """Whether any back-door admissible set exists, in polynomial time.

    Candidates are every node except ``x``, ``y``, and the descendants of
    ``x`` — restricted to measured nodes when ``measured_only``. Decided by
    testing the canonical set rather than enumerating; see
    ``canonical_adjustment_set``.
    """
    _check_pair(graph, x, y, measured_only=measured_only)
    candidates = _candidates(graph, x, y, measured_only=measured_only)
    return _canonical_within(graph, x, y, candidates) is not None


def _sort_key(s: frozenset[str]) -> tuple[int, tuple[str, ...]]:
    return (len(s), tuple(sorted(s)))


def _too_many(candidates: frozenset[str], max_candidates: int, what: str) -> Unsupported:
    return Unsupported(
        reason=(
            f"{len(candidates)} {what} exceed max_candidates={max_candidates}; "
            f"enumeration needs 2^{len(candidates)} d-separation tests. An admissible set "
            "does exist — raise max_candidates, mark nodes unmeasured, or use "
            "canonical_adjustment_set / admissible_set_exists."
        ),
        detail={
            "candidates": ",".join(sorted(candidates)),
            "n_candidates": str(len(candidates)),
            "max_candidates": str(max_candidates),
        },
    )


def adjustment_sets(
    graph: CausalGraph,
    x: str,
    y: str,
    *,
    measured_only: bool = True,
    max_candidates: int = 16,
) -> AdjustmentSets | Unsupported:
    """Every back-door admissible set among the candidates, deterministically ordered.

    Candidates are all nodes except ``x``, ``y``, and the descendants of ``x``
    (restricted to measured nodes when ``measured_only``). The result is
    sorted by size, then lexicographically by sorted member names, so equal
    graphs give equal tuples regardless of edge order.

    Enumeration visits :math:`2^k` subsets of ``k`` candidates — *all*
    candidates, since a non-ancestor of ``{x, y}`` can pad an admissible set
    harmlessly. When ``k > max_candidates`` the function returns
    ``Unsupported`` (a typed failure, not an exception) — unless the
    polynomial existence test already shows that *no* admissible set
    exists, in which case the empty tuple is a complete and correct answer
    and is returned. For the inclusion-minimal sets, whose enumeration is
    confined to the canonical set, see ``minimal_adjustment_sets``.
    """
    _check_pair(graph, x, y, measured_only=measured_only)
    candidates = _candidates(graph, x, y, measured_only=measured_only)
    if _canonical_within(graph, x, y, candidates) is None:
        return ()
    if len(candidates) > max_candidates:
        return _too_many(candidates, max_candidates, "candidate nodes")
    g = graph.remove_edges_out_of(x)
    ordered = sorted(candidates)
    found = [
        frozenset(combo)
        for k in range(len(ordered) + 1)
        for combo in combinations(ordered, k)
        if g.d_separated(x, y, combo)
    ]
    return tuple(sorted(found, key=_sort_key))


def minimal_adjustment_sets(
    graph: CausalGraph,
    x: str,
    y: str,
    *,
    measured_only: bool = True,
    max_candidates: int = 16,
) -> AdjustmentSets | Unsupported:
    """Inclusion-minimal back-door admissible sets, ordered as ``adjustment_sets``.

    A set is minimal when no proper subset of it is admissible. The back-door
    property is not monotone — a superset of an admissible set may open a
    collider — so minimality is checked by enumeration, but only over the
    canonical set :math:`Z_0 = R \\cap An_{G_{\\underline{x}}}(\\{x, y\\})`:
    every minimal admissible subset of the candidates ``R`` is a subset of
    :math:`Z_0` (van der Zander, Liśkiewicz & Textor 2014, Lemma 3; module
    docstring). ``max_candidates`` therefore bounds :math:`|Z_0|`, not
    ``|R|``; isolated nodes and other non-ancestors of ``{x, y}`` never push
    this function to ``Unsupported``. Subsets are visited smallest first and
    a superset of a minimal set found earlier is skipped without a test.
    """
    _check_pair(graph, x, y, measured_only=measured_only)
    candidates = _candidates(graph, x, y, measured_only=measured_only)
    canonical = _canonical_within(graph, x, y, candidates)
    if canonical is None:
        return ()
    if len(canonical) > max_candidates:
        return _too_many(canonical, max_candidates, "canonical-set nodes")
    g = graph.remove_edges_out_of(x)
    ordered = sorted(canonical)
    minimal: list[frozenset[str]] = []
    for k in range(len(ordered) + 1):
        for combo in combinations(ordered, k):
            s = frozenset(combo)
            if any(t < s for t in minimal):
                continue  # a proper subset is already known admissible
            if g.d_separated(x, y, s):
                minimal.append(s)
    return tuple(sorted(minimal, key=_sort_key))


def requires_unmeasured(graph: CausalGraph, x: str, y: str) -> bool:
    """True iff adjustment is possible only through a node the panel does not observe.

    Exactly: some admissible set exists among all nodes, and none exists among
    the measured nodes alone. This is the signal the verdict turns into
    ``status="downgraded"`` — identified on paper, not from these data.
    ``False`` both when a measured set exists and when no set exists at all.
    Raises ``GraphError`` when ``x`` or ``y`` is itself unmeasured, since the
    measured question is then malformed.
    """
    _check_pair(graph, x, y, measured_only=True)
    any_set = _canonical_within(graph, x, y, _candidates(graph, x, y, measured_only=False))
    if any_set is None:
        return False
    measured = _canonical_within(graph, x, y, _candidates(graph, x, y, measured_only=True))
    return measured is None


# -- roles --------------------------------------------------------------------------


def _latent_parents(graph: CausalGraph) -> dict[str, frozenset[str]]:
    """Latent common cause per bidirected edge, keyed by a private name, mapped to its children."""
    return {f"<{a}~{b}>": frozenset((a, b)) for a, b in graph.bidirected}


def roles(graph: CausalGraph, x: str, y: str) -> dict[str, Role]:
    """Assign each node a descriptive role relative to ``(x, y)``.

    Let :math:`G_{\\underline{x}}` be the graph without edges out of ``x``;
    ``An``/``De`` are proper ancestors/descendants in the full graph, and
    :math:`An_{\\underline{x}}` ancestors in :math:`G_{\\underline{x}}`
    (paths into ``y`` that avoid ``x``). Bidirected edges count as latent
    common causes with the two endpoints as children. "Open to ``v``" means
    d-connected to ``v`` given the empty set in :math:`G_{\\underline{x}}`.
    Definitions, in precedence order (a node gets the first that applies):

    * ``treatment`` / ``outcome`` — ``x`` / ``y``.
    * ``mediator`` — on a directed path from ``x`` to ``y``:
      :math:`n \\in De(x) \\cap An(y)`.
    * ``confounder`` — a common cause: :math:`n \\in An(x) \\cap
      An_{\\underline{x}}(y)`. Conditioning on it blocks at least one
      back-door path.
    * ``instrument`` — the unconditional instrument of Pearl 2009 §7.4.5,
      exactly as ``identify.frontdoor.instrument_admissible`` with an empty
      conditioning set: :math:`n \\notin De(x)`, ``n`` is d-connected to
      ``x`` in ``G`` (relevance), and ``n`` is d-separated from ``y`` in
      :math:`G_{\\underline{x}}` (exclusion and exogeneity). Relevance by
      d-connection rather than ancestry admits ``Z <-> X`` — ``Z`` is a
      proxy of a latent cause of ``x`` — and the d-separation test is what
      rejects ``Z -> X, Z <-> Y`` and ``Z <- W -> Y``. The set of
      ``instrument`` nodes equals ``frontdoor.instruments(graph, x, y,
      measured_only=False)``.
    * ``collider`` — a common effect sitting between the two sides:
      :math:`n \\notin An(x) \\cup An_{\\underline{x}}(y)` and ``n`` has two
      distinct parents ``a ≠ b`` (latent parents included) with ``a`` on
      the treatment side and ``b`` on the outcome side. A node ``a`` is on
      the treatment side when ``a = x`` or ``a`` is open to ``x``; a latent
      parent is, when one of its children is in :math:`An(x) \\cup \\{x\\}`.
      Likewise ``b = y``, ``b`` open to ``y``, or a latent with a child in
      :math:`An_{\\underline{x}}(y) \\cup \\{y\\}`. Because ``n`` is an
      ancestor of neither side, the two open walks avoid ``n``, so
      :math:`x \\sim a \\rightarrow n \\leftarrow b \\sim y` is a walk that
      conditioning on ``n`` activates: every collider satisfies
      ``not d_separated(x, y, {n})`` in :math:`G_{\\underline{x}}` with the
      edge ``x -> n`` kept when ``n`` is a child of ``x`` (M-bias and
      collider stratification; Greenland, Pearl & Robins 1999). Sides are
      reached through bidirected edges too
      (``X <-> A, A -> C, B -> C, B -> Y``) and through forks
      (``X -> C, B -> C, W -> B, W -> Y``). A node that is an ancestor of
      ``x``, or of ``y`` avoiding ``x``, is never a collider even when it is
      also a common effect: it is a non-collider on some path and is reported
      as ``confounder``, ``mediator``, or ``neutral`` (``A -> N, B -> N,
      N -> X, B -> Y`` — ``N`` blocks ``X <- N <- B -> Y`` and is neutral).
    * ``descendant_of_outcome`` — :math:`n \\in De(y)`.
    * ``proxy`` — a measured non-descendant of ``x`` with a parent (latent
      parents included) that is an unmeasured confounder. It stands in for
      that confounder in the narrative; conditioning on it does *not* block
      the confounder's path, which stays open through the unmeasured node.
    * ``neutral`` — everything else.

    Roles are explanatory: a node's role never decides admissibility, and a
    ``neutral`` node may still belong to an admissible set (e.g. ``Z`` in
    ``Z -> X, W -> Z, W -> Y`` with ``W`` measured). Descendants of colliders
    and of mediators are reported ``neutral`` although conditioning on them
    is also harmful; they are excluded from candidates by ``De(x)`` or by
    the d-separation test, not by their role. ``selection`` and ``feedback``
    do not affect roles.
    """
    _check_pair(graph, x, y)
    g_under = graph.remove_edges_out_of(x)
    anc_x = graph.ancestors(x)
    anc_y_under = g_under.ancestors(y)
    anc_y = graph.ancestors(y)
    de_x = graph.descendants(x)
    de_y = graph.descendants(y)
    anc_x_star = anc_x | {x}
    anc_y_under_star = anc_y_under | {y}
    open_to_x = frozenset(n for n in graph.nodes if n == x or not g_under.d_separated(n, x))
    open_to_y = frozenset(n for n in graph.nodes if n == y or not g_under.d_separated(n, y))

    latents = _latent_parents(graph)
    unmeasured_confounders: set[str] = {
        n for n in graph.unmeasured if n in anc_x and n in anc_y_under
    }
    for name, kids in latents.items():
        if kids & anc_x_star and kids & anc_y_under_star:
            unmeasured_confounders.add(name)

    def parents_aug(n: str) -> frozenset[str]:
        return graph.parents(n) | frozenset(name for name, kids in latents.items() if n in kids)

    def side_x(p: str) -> bool:
        return p in open_to_x if p in graph.nodes else bool(latents[p] & anc_x_star)

    def side_y(p: str) -> bool:
        return p in open_to_y if p in graph.nodes else bool(latents[p] & anc_y_under_star)

    def is_collider(n: str) -> bool:
        if n in anc_x or n in anc_y_under:
            return False
        ps = parents_aug(n)
        return any(side_x(a) and side_y(b) for a in ps for b in ps if a != b)

    out: dict[str, Role] = {}
    for n in graph.nodes:
        role: Role
        if n == x:
            role = "treatment"
        elif n == y:
            role = "outcome"
        elif n in de_x and n in anc_y:
            role = "mediator"
        elif n in anc_x and n in anc_y_under:
            role = "confounder"
        elif n not in de_x and not graph.d_separated(n, x) and n not in open_to_y:
            role = "instrument"
        elif is_collider(n):
            role = "collider"
        elif n in de_y:
            role = "descendant_of_outcome"
        elif graph.is_measured(n) and n not in de_x and parents_aug(n) & unmeasured_confounders:
            role = "proxy"
        else:
            role = "neutral"
        out[n] = role
    return out


class RoleAssignment(Spec):
    """The roles of every node of one graph relative to one treatment–outcome pair.

    ``graph_hash`` is the ``content_hash`` of the ``CausalGraph`` the roles
    were computed on, so a verdict or narrative that carries this object can
    be checked against the graph it claims to describe.
    """

    graph_hash: str
    treatment: str
    outcome: str
    roles: dict[str, Role]

    def with_role(self, role: Role) -> tuple[str, ...]:
        """Nodes carrying ``role``, sorted."""
        return tuple(sorted(n for n, r in self.roles.items() if r == role))


def assign_roles(graph: CausalGraph, x: str, y: str) -> RoleAssignment:
    """``roles`` packaged as a ``Spec`` with the graph's content hash."""
    return RoleAssignment(
        graph_hash=graph.content_hash(),
        treatment=x,
        outcome=y,
        roles=dict(sorted(roles(graph, x, y).items())),
    )
