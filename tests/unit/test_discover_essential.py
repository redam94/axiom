"""Essential graphs, checked against the definition of Markov equivalence by enumeration."""

from __future__ import annotations

import itertools

import pytest

from axiom.discover import (
    EssentialGraph,
    consistent_extension,
    consistent_extensions,
    cpdag,
    interventional_essential_graph,
    markov_equivalent,
    meek_closure,
    orientation_gain,
    v_structures,
)
from axiom.identify import CausalGraph
from axiom.identify.graph import GraphError

NODES = ["a", "b", "c", "d"]


def _all_dags(nodes: list[str]):
    pairs = list(itertools.combinations(nodes, 2))
    for choice in itertools.product([None, "f", "r"], repeat=len(pairs)):
        edges: list[tuple[str, str]] = []
        for (u, v), pick in zip(pairs, choice, strict=True):
            if pick == "f":
                edges.append((u, v))
            elif pick == "r":
                edges.append((v, u))
        try:
            yield CausalGraph(nodes=tuple(nodes), edges=tuple(edges))
        except GraphError:
            continue


def _queries(nodes: list[str]):
    for x, y in itertools.combinations(nodes, 2):
        rest = [n for n in nodes if n not in (x, y)]
        for size in range(len(rest) + 1):
            for z in itertools.combinations(rest, size):
                yield x, y, z


def _signature(graph: CausalGraph) -> tuple[bool, ...]:
    """Every d-separation the graph entails — the *definition* of its Markov class."""
    return tuple(graph.d_separated(x, y, z) for x, y, z in _queries(NODES))


ALL = list(_all_dags(NODES))
CLASSES: dict[tuple[bool, ...], list[CausalGraph]] = {}
for _g in ALL:
    CLASSES.setdefault(_signature(_g), []).append(_g)


# -- against the definition -------------------------------------------------------------


def test_there_are_543_dags_and_185_classes_on_four_nodes() -> None:
    """A published count, and a check that the enumeration is the enumeration."""
    assert len(ALL) == 543
    assert len(CLASSES) == 185


def test_the_cpdag_is_the_union_of_its_markov_equivalence_class() -> None:
    """Meek's rules are a closure algorithm; the class is a definition. They agree.

    For every DAG on four nodes: an edge is directed in the CPDAG exactly when
    every DAG entailing the same independencies orients it that way, and
    undirected exactly when they disagree.
    """
    for graph in ALL:
        klass = CLASSES[_signature(graph)]
        directed_everywhere = {
            (a, b) for a, b in graph.edges if all((a, b) in set(h.edges) for h in klass)
        }
        skeleton = {tuple(sorted((a, b))) for a, b in graph.edges}
        undirected = skeleton - {tuple(sorted(e)) for e in directed_everywhere}
        got = cpdag(graph)
        assert set(got.directed) == directed_everywhere, graph.to_text()
        assert set(got.undirected) == undirected, graph.to_text()


def test_the_classical_characterization_matches_entailed_independencies() -> None:
    """Same skeleton and v-structures iff same independencies (Verma & Pearl 1990)."""
    sample = ALL[:150]
    for first in sample:
        klass = CLASSES[_signature(first)]
        for second in sample:
            assert markov_equivalent(first, second) == (second in klass)


def test_enumerating_the_class_from_the_cpdag_recovers_it() -> None:
    for graph in ALL[:200]:
        klass = CLASSES[_signature(graph)]
        got = consistent_extensions(cpdag(graph))
        assert {h.to_text() for h in got} == {h.to_text() for h in klass}


def test_a_consistent_extension_is_in_the_class() -> None:
    for graph in ALL[:200]:
        essential = cpdag(graph)
        assert markov_equivalent(consistent_extension(essential), graph)


# -- what the objects say ---------------------------------------------------------------


def test_v_structures_need_the_parents_to_be_non_adjacent() -> None:
    assert v_structures(CausalGraph.from_edges("a -> c, b -> c")) == {("a", "c", "b")}
    assert v_structures(CausalGraph.from_edges("a -> c, b -> c, a -> b")) == frozenset()


def test_a_chain_orients_nothing_and_a_collider_orients_everything() -> None:
    assert cpdag(CausalGraph.from_edges("a -> b, b -> c")).to_text() == "a - b, b - c"
    assert cpdag(CausalGraph.from_edges("a -> c, b -> c")).to_text() == "a -> c, b -> c"


def test_an_essential_graph_reports_its_shape() -> None:
    essential = cpdag(CausalGraph.from_edges("a -> c, b -> c, c -> d"))
    assert essential.oriented + essential.undecided == len(essential.skeleton)
    assert essential.parents("c") == frozenset({"a", "b"})
    assert not essential.is_dag or essential.undecided == 0
    with pytest.raises(GraphError, match="undirected"):
        cpdag(CausalGraph.from_edges("a -> b, b -> c")).to_graph()


def test_a_fully_directed_essential_graph_is_a_dag() -> None:
    essential = cpdag(CausalGraph.from_edges("a -> c, b -> c"))
    assert essential.is_dag
    assert essential.to_graph().edges == (("a", "c"), ("b", "c"))


def test_an_essential_graph_refuses_contradictory_edges() -> None:
    with pytest.raises(GraphError, match="not both"):
        EssentialGraph(directed=(("a", "b"),), undirected=(("a", "b"),))
    with pytest.raises(GraphError, match="both ways"):
        EssentialGraph(directed=(("a", "b"), ("b", "a")))
    with pytest.raises(GraphError, match="self-edge"):
        EssentialGraph(directed=(("a", "a"),))


def test_meek_closure_propagates_and_is_idempotent() -> None:
    start = EssentialGraph(nodes=("a", "b", "c"), directed=(("a", "b"),), undirected=(("b", "c"),))
    closed = meek_closure(start)
    assert ("b", "c") in closed.directed  # rule 1: a -> b, b - c, a and c non-adjacent
    assert meek_closure(closed) == closed


def test_a_latent_confounded_graph_has_no_cpdag() -> None:
    with pytest.raises(GraphError, match="bidirected"):
        cpdag(CausalGraph.from_edges("a -> b, a <-> b"))


# -- what an experiment buys -------------------------------------------------------------


def test_intervening_orients_the_edges_it_cuts_and_what_they_force() -> None:
    chain = CausalGraph.from_edges("a -> b, b -> c")
    assert cpdag(chain).to_text() == "a - b, b - c"
    after = interventional_essential_graph(chain, [["a"]])
    assert after.to_text() == "a -> b, b -> c"  # a -> b is cut; Meek gives b -> c


def test_intervening_on_both_ends_of_an_edge_reveals_nothing_about_it() -> None:
    """Randomizing both endpoints tells you nothing: neither is caused by the other now."""
    chain = CausalGraph.from_edges("a -> b, b -> c")
    both = interventional_essential_graph(chain, [["a", "b"]])
    assert ("a", "b") not in both.directed
    assert tuple(sorted(both.undirected)) == (("a", "b"),)


def test_the_orientation_gain_prices_an_experiment_before_it_is_run() -> None:
    graph = CausalGraph.from_edges("a -> b, b -> c, a -> d, d -> c")
    assert orientation_gain(graph, [["a"]]) == (("a", "b"), ("a", "d"))
    assert orientation_gain(graph, [["b"]]) == (("a", "b"),)
    # b -> c is already oriented by the collider, so an experiment adds nothing there
    assert ("b", "c") in cpdag(graph).directed


def test_an_experiment_that_teaches_nothing_says_so() -> None:
    collider = CausalGraph.from_edges("a -> c, b -> c")
    assert cpdag(collider).is_dag
    assert orientation_gain(collider, [["a"]]) == ()


def test_a_target_naming_an_unknown_node_is_refused() -> None:
    with pytest.raises(GraphError, match="unknown nodes"):
        interventional_essential_graph(CausalGraph.from_edges("a -> b"), [["zzz"]])


def test_enumerating_a_huge_class_refuses_rather_than_truncates() -> None:
    big = EssentialGraph(
        nodes=tuple("abcdefghij"),
        undirected=tuple(itertools.combinations("abcdefghij", 2)),
    )
    with pytest.raises(ValueError, match="past the limit"):
        consistent_extensions(big, limit=16)
