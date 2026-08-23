"""Cluster-DAGs: reasoning at the granularity you actually know.

You rarely know the micro-level DAG. You often do know how *groups* of
variables relate: that nothing in the demand block causes anything in the cost
block, that the treatment block precedes the outcome block. A cluster-DAG
(Anand, Ribeiro, Tian & Bareinboim, AAAI 2023) is that partial knowledge
written down as a graph over clusters rather than over variables.

The semantics is entirely about *absence*. An edge ``A -> B`` says that any
cross-cluster edge between the two points that way; the **missing** edge
between ``A`` and ``C`` says no variable in ``A`` is adjacent to any variable
in ``C``. Within a cluster, anything goes — that is the knowledge you are
admitting you do not have.

What makes this more than a drawing convention is that the answers transfer.
d-separation on the cluster graph is sound: a separation the C-DAG asserts
holds in *every* compatible micro-level DAG. So is identification —
``identify_cluster_effect`` runs the ID algorithm on the cluster graph, and an
effect identified there is identified in every compatible DAG, by the same
estimand. Both properties are checked here by brute force over every
compatible DAG of a small model rather than cited
(``tests/unit/test_identify_cluster.py``).

The discipline the framework imposes is worth as much as the inference. A
cluster-level question is the only kind a cluster-level graph can answer:
asking for the effect of one variable inside a cluster is refused, because the
C-DAG genuinely does not contain that information. An aggregate that behaves
like a variable is a claim, and this is where it gets stated rather than
assumed.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from pydantic import Field, model_validator

from axiom.core import NonEmptyStr, Spec
from axiom.identify.graph import CausalGraph, Edge, GraphError
from axiom.identify.id_algorithm import IdentifiedEffect, identify_effect

__all__ = [
    "ClusterDAG",
    "compatible",
    "identify_cluster_effect",
]


class ClusterDAG(Spec):
    """A DAG over clusters of variables, with the micro-level structure left open.

    ``clusters`` maps each cluster's name to its members; the members
    partition the variables, so every variable belongs to exactly one. Edges
    and bidirected edges are between *cluster names*.

    A cluster name may coincide with its single member, which is how a
    variable you do know about individually sits alongside blocks you do not.
    """

    clusters: dict[str, tuple[NonEmptyStr, ...]] = Field(min_length=1)
    edges: tuple[Edge, ...] = ()
    bidirected: tuple[Edge, ...] = ()
    unmeasured: tuple[str, ...] = ()
    name: str = ""

    @model_validator(mode="after")
    def _partition(self) -> ClusterDAG:
        seen: dict[str, str] = {}
        for cluster, members in self.clusters.items():
            if not members:
                raise ValueError(f"cluster {cluster!r} is empty; a cluster has at least one member")
            for member in members:
                if member in seen:
                    raise ValueError(
                        f"variable {member!r} is in both {seen[member]!r} and {cluster!r}; "
                        "clusters partition the variables"
                    )
                seen[member] = cluster
        for a, b in (*self.edges, *self.bidirected):
            for endpoint in (a, b):
                if endpoint not in self.clusters:
                    raise GraphError(
                        f"edge endpoint {endpoint!r} is not a cluster; "
                        f"clusters are {sorted(self.clusters)}"
                    )
        unknown = sorted(set(self.unmeasured) - set(self.clusters))
        if unknown:
            raise GraphError(f"unmeasured names unknown clusters: {unknown}")
        self.to_graph()  # acyclicity is CausalGraph's invariant; borrow it
        return self

    # -- views --------------------------------------------------------------------

    def to_graph(self) -> CausalGraph:
        """The cluster graph itself: an ordinary ``CausalGraph`` over cluster names."""
        return CausalGraph(
            nodes=tuple(sorted(self.clusters)),
            edges=self.edges,
            bidirected=self.bidirected,
            unmeasured=self.unmeasured,
            name=self.name,
        )

    @property
    def variables(self) -> tuple[str, ...]:
        return tuple(sorted(m for members in self.clusters.values() for m in members))

    def cluster_of(self, variable: str) -> str:
        for cluster, members in self.clusters.items():
            if variable in members:
                return cluster
        raise KeyError(f"no variable {variable!r}; variables are {list(self.variables)}")

    def members(self, cluster: str) -> tuple[str, ...]:
        if cluster not in self.clusters:
            raise GraphError(f"unknown cluster {cluster!r}; have {sorted(self.clusters)}")
        return self.clusters[cluster]

    @property
    def singletons(self) -> tuple[str, ...]:
        """Clusters with one member — the variables you do know individually."""
        return tuple(sorted(c for c, m in self.clusters.items() if len(m) == 1))

    def d_separated(
        self, x: str | Iterable[str], y: str | Iterable[str], z: Iterable[str] = ()
    ) -> bool:
        """d-separation over clusters. Sound: it holds in every compatible DAG."""
        return self.to_graph().d_separated(x, y, z)

    # -- construction from a known DAG ---------------------------------------------

    @classmethod
    def from_dag(
        cls,
        graph: CausalGraph,
        clusters: Mapping[str, Iterable[str]],
        *,
        name: str = "",
    ) -> ClusterDAG:
        """Coarsen a DAG you *do* know into the C-DAG a reader would be told.

        Refuses a partition that is not admissible: if two clusters have edges
        running both ways between them, no single cluster edge describes that,
        and the partition is the wrong one rather than the graph being wrong.
        """
        members = {str(c): tuple(sorted(str(v) for v in vs)) for c, vs in clusters.items()}
        placed = {v: c for c, vs in members.items() for v in vs}
        missing = sorted(set(graph.nodes) - set(placed))
        if missing:
            raise ValueError(f"the partition does not cover {missing}")
        extra = sorted(set(placed) - set(graph.nodes))
        if extra:
            raise ValueError(f"the partition names variables the graph does not have: {extra}")

        directed: set[Edge] = set()
        for a, b in graph.edges:
            if placed[a] != placed[b]:
                directed.add((placed[a], placed[b]))
        both_ways = sorted({(a, b) for a, b in directed if (b, a) in directed})
        if both_ways:
            raise ValueError(
                f"clusters {both_ways[0]} have edges running both ways between them; "
                "no cluster edge describes that, so the partition is not admissible"
            )
        bidirected = {
            (min(placed[a], placed[b]), max(placed[a], placed[b]))
            for a, b in graph.bidirected
            if placed[a] != placed[b]
        }
        unmeasured = sorted({placed[v] for v in graph.unmeasured})
        return cls(
            clusters=members,
            edges=tuple(sorted(directed)),
            bidirected=tuple(sorted(bidirected)),
            unmeasured=tuple(unmeasured),
            name=name or graph.name,
        )


def compatible(cdag: ClusterDAG, graph: CausalGraph) -> bool:
    """Whether a micro-level DAG is compatible with the cluster-level claim.

    Compatibility is about what the C-DAG *rules out*: a cross-cluster edge
    must run the way its cluster edge does, and two clusters with no edge
    between them must have no adjacency between any of their members. Edges
    inside a cluster are unconstrained — that is the knowledge the C-DAG does
    not claim to have.
    """
    placed = {v: c for c, vs in cdag.clusters.items() for v in vs}
    if set(placed) != set(graph.nodes):
        return False
    allowed = set(cdag.edges)
    allowed_bi = set(cdag.bidirected)
    for a, b in graph.edges:
        first, second = placed[a], placed[b]
        if first == second:
            continue
        if (first, second) not in allowed:
            return False
    for a, b in graph.bidirected:
        first, second = placed[a], placed[b]
        if first == second:
            continue
        if (min(first, second), max(first, second)) not in allowed_bi:
            return False
    return True


def identify_cluster_effect(
    cdag: ClusterDAG,
    treatment: str | Iterable[str],
    outcome: str | Iterable[str],
) -> IdentifiedEffect:
    """``P(outcome cluster | do(treatment cluster))`` by running ID on the cluster graph.

    Sound and complete for cluster-level queries (Anand et al. 2023): the
    effect is identifiable from the C-DAG exactly when it is identifiable in
    every compatible DAG, and the estimand returned is valid for all of them.

    The query has to be about clusters. A variable inside a multi-member
    cluster is refused by name, because the C-DAG does not contain the
    within-cluster structure that a variable-level answer would need — and
    quietly answering the coarse question instead of the one asked is the
    failure this refusal exists to prevent.
    """
    xs = {treatment} if isinstance(treatment, str) else set(treatment)
    ys = {outcome} if isinstance(outcome, str) else set(outcome)
    for name in sorted(xs | ys):
        if name in cdag.clusters:
            continue
        try:
            owner = cdag.cluster_of(name)
        except KeyError:
            raise GraphError(
                f"unknown cluster {name!r}; clusters are {sorted(cdag.clusters)}"
            ) from None
        raise GraphError(
            f"{name!r} is a variable inside the cluster {owner!r}, which has "
            f"{len(cdag.members(owner))} members. A cluster-DAG holds no within-cluster "
            f"structure, so ask about {owner!r} or split the cluster"
        )
    return identify_effect(cdag.to_graph(), xs, ys)
