"""Cluster-DAGs, checked against every compatible micro-level DAG rather than cited."""

from __future__ import annotations

import itertools

import pytest

from axiom.identify import CausalGraph, ClusterDAG, compatible, identify_cluster_effect
from axiom.identify.graph import GraphError
from axiom.identify.id_algorithm import identify_effect

CLUSTERS = {"A": ("a1", "a2"), "B": ("b1",), "C": ("c1",)}
VARIABLES = ["a1", "a2", "b1", "c1"]
OPTIONS = [(), ("fwd",), ("rev",), ("bi",), ("fwd", "bi"), ("rev", "bi")]


def _admgs(variables: list[str], with_confounding: bool = False):
    """Every ADMG over the variables, by enumerating each pair's possible edges."""
    options = OPTIONS if with_confounding else [(), ("fwd",), ("rev",)]
    pairs = list(itertools.combinations(variables, 2))
    for choice in itertools.product(options, repeat=len(pairs)):
        edges: list[tuple[str, str]] = []
        bidirected: list[tuple[str, str]] = []
        for (u, v), picks in zip(pairs, choice, strict=True):
            for pick in picks:
                if pick == "fwd":
                    edges.append((u, v))
                elif pick == "rev":
                    edges.append((v, u))
                else:
                    bidirected.append((u, v))
        try:
            yield CausalGraph(
                nodes=tuple(variables), edges=tuple(edges), bidirected=tuple(bidirected)
            )
        except GraphError:
            continue  # cyclic


# -- construction ----------------------------------------------------------------------


def test_the_partition_must_cover_every_variable_exactly_once() -> None:
    with pytest.raises(ValueError, match="clusters partition"):
        ClusterDAG(clusters={"A": ("x",), "B": ("x",)})
    with pytest.raises(ValueError, match="at least one member"):
        ClusterDAG(clusters={"A": ()})


def test_an_edge_must_join_clusters_not_variables() -> None:
    with pytest.raises(GraphError, match="not a cluster"):
        ClusterDAG(clusters={"A": ("a1",), "B": ("b1",)}, edges=(("a1", "B"),))


def test_the_cluster_graph_must_be_acyclic() -> None:
    with pytest.raises(GraphError, match="cycle"):
        ClusterDAG(clusters={"A": ("a",), "B": ("b",)}, edges=(("A", "B"), ("B", "A")))


def test_a_cdag_reports_its_shape() -> None:
    cdag = ClusterDAG(clusters=CLUSTERS, edges=(("A", "B"), ("B", "C")))
    assert cdag.variables == ("a1", "a2", "b1", "c1")
    assert cdag.cluster_of("a2") == "A"
    assert cdag.members("A") == ("a1", "a2")
    assert cdag.singletons == ("B", "C")
    assert cdag.to_graph().nodes == ("A", "B", "C")
    with pytest.raises(GraphError, match="unknown cluster"):
        cdag.members("Q")
    with pytest.raises(KeyError, match="no variable"):
        cdag.cluster_of("zzz")


def test_coarsening_a_known_dag_gives_the_cdag_a_reader_would_be_told() -> None:
    graph = CausalGraph.from_edges("a1 -> a2, a2 -> b1, b1 -> c1, a1 -> c1")
    cdag = ClusterDAG.from_dag(graph, CLUSTERS)
    assert set(cdag.edges) == {("A", "B"), ("B", "C"), ("A", "C")}
    assert compatible(cdag, graph)


def test_a_partition_with_edges_running_both_ways_is_refused() -> None:
    graph = CausalGraph.from_edges("a1 -> b1, b1 -> a2")
    with pytest.raises(ValueError, match="not admissible"):
        ClusterDAG.from_dag(graph, {"A": ("a1", "a2"), "B": ("b1",)})


def test_the_partition_must_match_the_graph() -> None:
    graph = CausalGraph.from_edges("a1 -> b1")
    with pytest.raises(ValueError, match="does not cover"):
        ClusterDAG.from_dag(graph, {"A": ("a1",)})
    with pytest.raises(ValueError, match="does not have"):
        ClusterDAG.from_dag(graph, {"A": ("a1",), "B": ("b1",), "C": ("zzz",)})


# -- compatibility ---------------------------------------------------------------------


def test_compatibility_constrains_across_clusters_and_not_within() -> None:
    cdag = ClusterDAG(clusters=CLUSTERS, edges=(("A", "B"),))
    # within a cluster, either orientation is fine -- that is the unknown part
    assert compatible(cdag, CausalGraph.from_edges("a1 -> a2", nodes=VARIABLES))
    assert compatible(cdag, CausalGraph.from_edges("a2 -> a1", nodes=VARIABLES))
    # across clusters, the direction is fixed
    assert compatible(cdag, CausalGraph.from_edges("a1 -> b1", nodes=VARIABLES))
    assert not compatible(cdag, CausalGraph.from_edges("b1 -> a1", nodes=VARIABLES))
    # a missing cluster edge means no adjacency at all
    assert not compatible(cdag, CausalGraph.from_edges("b1 -> c1", nodes=VARIABLES))


# -- the theorems ----------------------------------------------------------------------


def test_cluster_separation_holds_in_every_compatible_dag() -> None:
    """Soundness and completeness of d-separation on the cluster graph."""
    cdag = ClusterDAG(clusters=CLUSTERS, edges=(("A", "B"), ("B", "C")))
    pool = [g for g in _admgs(VARIABLES) if compatible(cdag, g)]
    assert len(pool) > 10
    checked = 0
    for x, y in itertools.combinations(sorted(CLUSTERS), 2):
        rest = [c for c in sorted(CLUSTERS) if c not in (x, y)]
        for size in range(len(rest) + 1):
            for z in itertools.combinations(rest, size):
                micro_z = [m for c in z for m in CLUSTERS[c]]
                in_every = all(
                    g.d_separated(list(CLUSTERS[x]), list(CLUSTERS[y]), micro_z) for g in pool
                )
                assert cdag.d_separated(x, y, z) == in_every, (x, y, z)
                checked += 1
    assert checked >= 6


def test_cluster_identification_matches_every_compatible_dag() -> None:
    """An effect is identified from the C-DAG exactly when it is in every compatible DAG.

    The confounded case is the one that could have gone either way: the
    cluster-level bow arc says *not* identified, and only some of the
    compatible ADMGs identify it — so "identified in all" is false too.
    """
    variables = ["a1", "a2", "b1"]
    clusters = {"A": ("a1", "a2"), "B": ("b1",)}
    for edges, bidirected in (
        ((("A", "B"),), ()),
        ((("A", "B"),), (("A", "B"),)),
        ((), (("A", "B"),)),
    ):
        cdag = ClusterDAG(clusters=clusters, edges=edges, bidirected=bidirected)
        pool = [g for g in _admgs(variables, with_confounding=True) if compatible(cdag, g)]
        assert pool
        for x, y in itertools.permutations(sorted(clusters), 2):
            cluster_answer = identify_cluster_effect(cdag, x, y).identified
            in_every = all(
                identify_effect(g, list(clusters[x]), list(clusters[y])).identified for g in pool
            )
            assert cluster_answer == in_every, (edges, bidirected, x, y)


def test_the_cluster_level_bow_arc_is_the_case_that_discriminates() -> None:
    variables = ["a1", "a2", "b1"]
    clusters = {"A": ("a1", "a2"), "B": ("b1",)}
    cdag = ClusterDAG(clusters=clusters, edges=(("A", "B"),), bidirected=(("A", "B"),))
    pool = [g for g in _admgs(variables, with_confounding=True) if compatible(cdag, g)]
    identified = [identify_effect(g, ["a1", "a2"], ["b1"]).identified for g in pool]
    assert not identify_cluster_effect(cdag, "A", "B").identified
    assert any(identified), "some compatible DAGs do identify it"
    assert not all(identified), "and some do not, which is why the C-DAG must say no"


# -- what a cluster graph will not answer ----------------------------------------------


def test_a_variable_level_question_is_refused_by_name() -> None:
    cdag = ClusterDAG(clusters=CLUSTERS, edges=(("A", "B"),))
    with pytest.raises(GraphError, match="inside the cluster 'A'"):
        identify_cluster_effect(cdag, "a1", "B")
    with pytest.raises(GraphError, match="unknown cluster"):
        identify_cluster_effect(cdag, "Q", "B")


def test_a_singleton_cluster_is_askable_because_it_is_the_variable() -> None:
    cdag = ClusterDAG(clusters=CLUSTERS, edges=(("A", "B"), ("B", "C")))
    result = identify_cluster_effect(cdag, "B", "C")
    assert result.identified
