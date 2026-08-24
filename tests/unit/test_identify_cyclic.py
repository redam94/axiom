"""Sigma-separation, twice: the direct walk and the acyclification must agree."""

from __future__ import annotations

import itertools
import random

import pytest

from axiom.core import D, Param, dimensionless
from axiom.dynamics import Variable, parse_system, unrolled_edges
from axiom.identify import (
    CausalGraph,
    MixedGraph,
    acyclify,
    sigma_separated,
    unrolled_graph,
    unrolled_mixed_graph,
)
from axiom.identify.graph import GraphError

NONE = dimensionless()
NAMES = ("a", "b", "c", "d", "e")


def _random_mixed(rng: random.Random, directed: float, bidirected: float) -> MixedGraph:
    edges = [p for p in itertools.permutations(NAMES, 2) if rng.random() < directed]
    bi = [p for p in itertools.combinations(NAMES, 2) if rng.random() < bidirected]
    return MixedGraph(nodes=NAMES, edges=tuple(edges), bidirected=tuple(bi))


def _queries(nodes: tuple[str, ...]):
    for x, y in itertools.combinations(nodes, 2):
        rest = [n for n in nodes if n not in (x, y)]
        for size in range(len(rest) + 1):
            for z in itertools.combinations(rest, size):
                yield x, y, z


# -- the two implementations ---------------------------------------------------------


@pytest.mark.parametrize("density", [(0.18, 0.08), (0.35, 0.15)])
def test_the_direct_walk_and_the_acyclification_agree(density: tuple[float, float]) -> None:
    """Mooij & Claassen's Proposition 2, held to over random graphs."""
    rng = random.Random(4)
    cyclic = 0
    for _ in range(60):
        graph = _random_mixed(rng, *density)
        acyclic_form = acyclify(graph)
        cyclic += not graph.is_acyclic
        for x, y, z in _queries(NAMES):
            assert sigma_separated(graph, x, y, z) == acyclic_form.d_separated(
                x, y, z
            ), f"{graph.to_text()} | {x} {y} {z}"
    assert cyclic > 10, "the sample should contain cyclic graphs"


def test_on_an_acyclic_graph_sigma_is_d() -> None:
    rng = random.Random(7)
    checked = 0
    for _ in range(80):
        graph = _random_mixed(rng, 0.15, 0.06)
        if not graph.is_acyclic:
            continue
        plain = CausalGraph(nodes=graph.nodes, edges=graph.edges, bidirected=graph.bidirected)
        for x, y, z in _queries(NAMES):
            assert sigma_separated(graph, x, y, z) == plain.d_separated(x, y, z)
            checked += 1
    assert checked > 1000


# -- what the criterion actually says -------------------------------------------------


def test_conditioning_inside_a_loop_does_not_block_it() -> None:
    """The whole difference from d-separation, in one graph.

    ``x -> a``, ``a <-> b`` as a two-cycle, ``b -> y``: conditioning on ``a``
    would block ``x -> a -> b -> y`` under d-separation, but ``a`` points at
    ``b`` inside its own component, so it cannot block.
    """
    loop = MixedGraph.from_edges("x -> a, a -> b, b -> a, b -> y")
    assert loop.cyclic_components == (("a", "b"),)
    assert not sigma_separated(loop, "x", "y", ["a"])
    acyclic_reading = CausalGraph.from_edges("x -> a, a -> b, b -> y")
    assert acyclic_reading.d_separated("x", "y", ["a"])


def test_a_non_collider_pointing_out_of_its_loop_still_blocks() -> None:
    loop = MixedGraph.from_edges("x -> a, a -> b, b -> a, b -> y")
    # b points at y, which is outside the loop, so conditioning on b blocks
    assert sigma_separated(loop, "x", "y", ["b"])


def test_the_acyclification_makes_a_loop_a_bidirected_clique() -> None:
    loop = MixedGraph.from_edges("x -> a, a -> b, b -> a, b -> y")
    acyclic_form = acyclify(loop)
    assert ("a", "b") in acyclic_form.bidirected
    assert ("a", "b") not in acyclic_form.edges
    assert ("x", "a") in acyclic_form.edges
    assert ("x", "b") in acyclic_form.edges  # a parent of the loop reaches every member
    assert acyclic_form.topological_order()


def test_a_mixed_graph_reports_its_loops() -> None:
    graph = MixedGraph.from_edges("a -> b, b -> a, c -> d")
    assert not graph.is_acyclic
    assert graph.cyclic_components == (("a", "b"),)
    assert graph.component_of("c") == frozenset({"c"})
    assert graph.parents("b") == frozenset({"a"})
    assert graph.children("a") == frozenset({"b"})
    assert graph.measured == frozenset(graph.nodes)


def test_an_acyclic_graph_converts_both_ways() -> None:
    graph = CausalGraph.from_edges("z -> x, x -> y, u <-> x", unmeasured=["u"])
    mixed = MixedGraph.from_causal_graph(graph)
    assert mixed.is_acyclic
    assert mixed.unmeasured == ("u",)
    assert mixed.d_separated("z", "y", ["x"]) == graph.d_separated("z", "y", ["x"])
    assert "acyclified" in mixed.acyclify().name or mixed.acyclify().name == ""


def test_unknown_nodes_are_named() -> None:
    graph = MixedGraph.from_edges("a -> b")
    with pytest.raises(GraphError, match="unknown node"):
        sigma_separated(graph, "a", "zzz")
    with pytest.raises(GraphError, match="self-edge"):
        MixedGraph(edges=(("a", "a"),))


# -- the simultaneous system this was built for ----------------------------------------


def market_system():
    return parse_system(
        """
        quantity = a - b * price + c * income
        price    = d + e * quantity + cost
        """,
        variables=(
            Variable(name="quantity", dimension=D.outcome),
            Variable(name="price", dimension=D.currency),
            Variable(name="income", dimension=D.currency, role="exogenous"),
            Variable(name="cost", dimension=D.currency, role="exogenous"),
        ),
        parameters=(
            Param(name="a", dimension=D.outcome),
            Param(name="b", dimension=D.outcome / D.currency),
            Param(name="c", dimension=D.outcome / D.currency),
            Param(name="d", dimension=D.currency),
            Param(name="e", dimension=D.currency / D.outcome),
        ),
        name="market",
    )


def test_the_unrolled_mixed_graph_keeps_the_cycle() -> None:
    mixed = unrolled_mixed_graph(market_system(), periods=2)
    assert not mixed.is_acyclic
    assert set(mixed.cyclic_components) == {
        ("price.t0", "quantity.t0"),
        ("price.t1", "quantity.t1"),
    }


def test_the_unrolled_graph_is_the_acyclification_of_it() -> None:
    system = market_system()
    assert unrolled_graph(system, 3) == acyclify(unrolled_mixed_graph(system, 3))


def test_the_acyclification_agrees_with_the_reduced_form_dynamics_computes() -> None:
    """Two independent derivations of the same directed edges."""
    system = market_system()
    for periods in (1, 2, 3):
        graph = unrolled_graph(system, periods)
        assert set(graph.edges) == set(unrolled_edges(system, periods, reduced=True))


def test_simultaneous_variables_are_dependent_given_their_shared_parents() -> None:
    """The bidirected edge is what makes this right, and it is easy to leave out.

    Quantity and price are determined together. Conditioning on everything
    feeding the block does not make them independent — they still share the
    block's disturbances — and a reduced-form graph with shared parents but no
    bidirected edge would wrongly say it does.
    """
    graph = unrolled_graph(market_system(), periods=1)
    assert ("price.t0", "quantity.t0") in graph.bidirected
    assert not graph.d_separated("quantity.t0", "price.t0", ["cost.t0", "income.t0"])
    mixed = unrolled_mixed_graph(market_system(), periods=1)
    assert not mixed.sigma_separated("quantity.t0", "price.t0", ["cost.t0", "income.t0"])
