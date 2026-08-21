from __future__ import annotations

import pytest

from axiom.core import Spec
from axiom.identify.graph import CausalGraph, GraphError, parse_edges


def test_parse_and_normalize() -> None:
    d, b = parse_edges("Z -> X; X -> Y\nY <- W, A <-> B, B <-> A")
    assert d == (("Z", "X"), ("X", "Y"), ("W", "Y")) and b == (("A", "B"), ("B", "A"))
    g = CausalGraph.from_edges("Y <- W, A <-> B, B <-> A, Z -> X", nodes=["Q"])
    assert g.edges == (("W", "Y"), ("Z", "X")) and g.bidirected == (("A", "B"),)
    assert g.nodes == ("A", "B", "Q", "W", "X", "Y", "Z")
    with pytest.raises(GraphError, match="cannot parse"):
        parse_edges("A => B")
    with pytest.raises(GraphError, match="self-edge"):
        CausalGraph(edges=[("A", "A")])
    with pytest.raises(GraphError, match="unknown nodes"):
        CausalGraph.from_edges("A -> B", unmeasured=["C"])


def test_cycle_detection_names_the_cycle() -> None:
    with pytest.raises(GraphError, match="A -> B -> C -> A"):
        CausalGraph.from_edges("A -> B, B -> C, C -> A")
    CausalGraph.from_edges("A -> B, B -> C, A -> C")  # a DAG with a triangle is fine


def test_adjacency_and_order() -> None:
    g = CausalGraph.from_edges("Z -> X, Z -> Y, X -> M, M -> Y, X <-> W, W -> Y", unmeasured=["W"])
    assert g.parents("Y") == {"Z", "M", "W"} and g.children("X") == {"M"}
    assert g.siblings("X") == {"W"} and g.siblings("Z") == frozenset()
    assert g.ancestors("Y") == {"Z", "X", "M", "W"} and g.ancestors("Y", include_self=True) >= {"Y"}
    assert g.descendants("X") == {"M", "Y"} and g.descendants(["X", "Z"]) == {"X", "M", "Y"}
    assert g.is_ancestor("Z", "Y") and not g.is_ancestor("Y", "Z")
    order = g.topological_order()
    assert order.index("Z") < order.index("X") < order.index("M") < order.index("Y")
    assert g.measured == {"Z", "X", "M", "Y"} and not g.is_measured("W")
    assert g.directed_paths("X", "Y") == (("X", "M", "Y"),)
    assert g.directed_paths("Z", "Y") == (("Z", "X", "M", "Y"), ("Z", "Y"))
    with pytest.raises(GraphError, match="unknown node"):
        g.parents("nope")


def test_d_separation_primitives() -> None:
    chain = CausalGraph.from_edges("A -> B, B -> C")
    assert not chain.d_separated("A", "C") and chain.d_separated("A", "C", ["B"])
    fork = CausalGraph.from_edges("B -> A, B -> C")
    assert not fork.d_separated("A", "C") and fork.d_separated("A", "C", ["B"])
    collider = CausalGraph.from_edges("A -> C, B -> C, C -> D")
    assert collider.d_separated("A", "B")
    assert not collider.d_separated("A", "B", ["C"])
    assert not collider.d_separated("A", "B", ["D"])  # descendant of a collider opens it
    assert collider.d_separated(["A"], ["B"], [])
    assert not collider.d_separated("A", "A")


def test_d_separation_with_latents_and_surgery() -> None:
    g = CausalGraph.from_edges("Z -> X, Z -> Y, X -> M, M -> Y, X <-> W, W -> Y", unmeasured=["W"])
    assert not g.d_separated("X", "Y", ["Z", "M"])  # X <-> W -> Y stays open
    assert g.d_separated("Z", "W") and not g.d_separated("Z", "W", ["X"])
    under = g.remove_edges_out_of("X")
    assert (
        under.edges == (("M", "Y"), ("W", "Y"), ("Z", "X"), ("Z", "Y")) and under.nodes == g.nodes
    )
    assert not under.d_separated("X", "Y", ["Z"])  # latent path remains
    bar = g.remove_edges_into("X")
    assert bar.bidirected == () and bar.parents("X") == frozenset()
    m_bias = CausalGraph.from_edges(
        "U1 -> X, U1 -> M, U2 -> M, U2 -> Y, X -> Y", unmeasured=["U1", "U2"]
    )
    cut = m_bias.remove_edges_out_of("X")
    assert cut.d_separated("X", "Y") and not cut.d_separated("X", "Y", ["M"])


def test_conditioning_on_endpoint_and_helpers() -> None:
    g = CausalGraph.from_edges("A -> B")
    assert g.d_separated("A", "B", ["A"])
    s = g.with_selection("B").with_unmeasured("A")
    assert s.selection == ("B",) and s.unmeasured == ("A",)
    assert str(g) == "A -> B" and str(CausalGraph(nodes=["A"])) == "CausalGraph(nodes=['A'])"


def test_spec_identity() -> None:
    a = CausalGraph.from_edges("Z -> X, X -> Y, Z -> Y", feedback=True, name="g")
    b = CausalGraph.from_edges("Z -> Y, X -> Y, Z -> X", feedback=True, name="g")
    assert a == b and a.content_hash() == b.content_hash()
    assert Spec.from_json(a.to_json()) == a
    assert a != a.model_copy(update={"feedback": False})
