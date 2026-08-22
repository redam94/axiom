"""FCI: every mark it places must be true of the graph that generated the data."""

from __future__ import annotations

import random

import numpy as np
import pytest

from axiom.discover import (
    PAG,
    Dataset,
    PagEdge,
    cpdag,
    fci,
    fci_from_data,
    oracle_independence,
)
from axiom.identify import CausalGraph
from axiom.identify.graph import GraphError


def _unsound_marks(pag: PAG, truth: CausalGraph) -> list[str]:
    """An arrowhead claims *not an ancestor*; a tail claims *is an ancestor*. Check both."""
    wrong: list[str] = []
    for edge in pag.edges:
        for at, other in ((edge.a, edge.b), (edge.b, edge.a)):
            mark = edge.mark_at(at)
            if mark == "arrow" and at in truth.ancestors(other):
                wrong.append(f"{edge.to_text()}: {at} is an ancestor of {other}")
            if mark == "tail" and at not in truth.ancestors(other):
                wrong.append(f"{edge.to_text()}: {at} is not an ancestor of {other}")
    return wrong


# -- the vocabulary ---------------------------------------------------------------------


def test_an_edge_renders_its_marks() -> None:
    assert PagEdge(a="x", b="y", mark_a="arrow", mark_b="arrow").to_text() == "x <-> y"
    assert PagEdge(a="x", b="y", mark_a="tail", mark_b="arrow").to_text() == "x --> y"
    assert PagEdge(a="x", b="y").to_text() == "x o-o y"
    assert PagEdge(a="x", b="y", mark_a="arrow", mark_b="arrow").is_bidirected
    assert PagEdge(a="x", b="y", mark_a="tail", mark_b="arrow").is_directed
    # The ends are normalized rather than refused, so the same edge written
    # either way is the same object -- and each mark travels with its own end.
    flipped = PagEdge(a="y", b="x", mark_a="arrow", mark_b="tail")
    assert (flipped.a, flipped.b) == ("x", "y")
    assert (flipped.mark_a, flipped.mark_b) == ("tail", "arrow")
    assert flipped == PagEdge(a="x", b="y", mark_a="tail", mark_b="arrow")
    with pytest.raises(ValueError, match="not 'x' to itself"):
        PagEdge(a="x", b="x")


def test_an_edge_knows_its_ends() -> None:
    edge = PagEdge(a="x", b="y", mark_a="tail", mark_b="arrow")
    assert edge.mark_at("x") == "tail"
    assert edge.other("x") == "y"
    with pytest.raises(KeyError, match="not an end"):
        edge.mark_at("q")


def test_a_pag_reports_what_it_settled() -> None:
    pag = PAG(
        edges=(
            PagEdge(a="a", b="b", mark_a="tail", mark_b="arrow"),
            PagEdge(a="b", b="c", mark_a="arrow", mark_b="arrow"),
            PagEdge(a="c", b="d"),
        )
    )
    assert pag.definite_causes == (("a", "b"),)
    assert [e.to_text() for e in pag.bidirected] == ["b <-> c"]
    assert [e.to_text() for e in pag.undetermined] == ["c o-o d"]
    assert pag.is_adjacent("a", "b") and not pag.is_adjacent("a", "d")
    assert pag.adjacent("b") == frozenset({"a", "c"})
    assert "settled" in pag.summary()


def test_a_pag_holds_one_edge_per_pair() -> None:
    with pytest.raises(GraphError, match="one edge per pair"):
        PAG(edges=(PagEdge(a="a", b="b"), PagEdge(a="a", b="b", mark_a="arrow")))


# -- against a perfect test --------------------------------------------------------------


def test_a_chain_leaves_every_mark_open() -> None:
    truth = CausalGraph.from_edges("a -> b, b -> c")
    pag = fci(["a", "b", "c"], oracle_independence(truth))
    assert pag.skeleton == (("a", "b"), ("b", "c"))
    assert len(pag.undetermined) == 2, "observation cannot orient a chain"
    assert not _unsound_marks(pag, truth)


def test_a_collider_gets_both_arrowheads() -> None:
    truth = CausalGraph.from_edges("a -> c, b -> c")
    pag = fci(["a", "b", "c"], oracle_independence(truth))
    assert all(edge.mark_at("c") == "arrow" for edge in pag.edges)
    assert not _unsound_marks(pag, truth)


def test_a_latent_common_cause_is_found_and_named() -> None:
    """The finding a CPDAG cannot express: neither variable causes the other.

    Two instruments make it derivable — ``z1`` is independent of ``y`` and
    ``z2`` of ``x``, which forces arrowheads at both ends of the ``x``-``y``
    edge. That mark *is* the statement "something you did not measure drives
    both".
    """
    truth = CausalGraph.from_edges("z1 -> x, z2 -> y, u -> x, u -> y", unmeasured=["u"])
    pag = fci(["z1", "z2", "x", "y"], oracle_independence(truth))
    confounded = {frozenset({e.a, e.b}) for e in pag.bidirected}
    assert frozenset({"x", "y"}) in confounded
    assert not _unsound_marks(pag, truth)


def test_every_mark_is_sound_over_random_graphs_with_a_latent() -> None:
    """The property that matters: FCI may under-orient, but never mis-orient."""
    names = ["v1", "v2", "v3", "v4", "L"]
    rng = random.Random(3)
    checked = 0
    for _ in range(120):
        order = names[:]
        rng.shuffle(order)
        edges = [(u, v) for i, u in enumerate(order) for v in order[i + 1 :] if rng.random() < 0.45]
        if not edges:
            continue
        truth = CausalGraph(nodes=tuple(names), edges=tuple(edges), unmeasured=("L",))
        pag = fci([n for n in sorted(names) if n != "L"], oracle_independence(truth))
        assert not _unsound_marks(pag, truth), truth.to_text()
        checked += 1
    assert checked > 100


def test_the_result_says_which_rules_were_left_out() -> None:
    truth = CausalGraph.from_edges("a -> b, b -> c")
    pag = fci(["a", "b", "c"], oracle_independence(truth))
    assert any("R4" in limit for limit in pag.limits_hit)


def test_fci_needs_at_least_two_variables() -> None:
    truth = CausalGraph.from_edges("a -> b")
    with pytest.raises(ValueError, match="at least two"):
        fci(["a"], oracle_independence(truth))


# -- from data ----------------------------------------------------------------------------


def _linear(edges: dict[tuple[str, str], float], names: list[str], n: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    values: dict[str, np.ndarray] = {}
    for node in names:
        column = rng.normal(size=n)
        for (parent, child), weight in edges.items():
            if child == node:
                column = column + weight * values[parent]
        values[node] = column
    return values


def test_fci_on_data_finds_the_chain_and_leaves_it_open() -> None:
    values = _linear({("a", "b"): 1.3, ("b", "c"): -0.9}, ["a", "b", "c"], 4000, 0)
    data = Dataset.observational(
        np.column_stack([values[n] for n in ("a", "b", "c")]), ["a", "b", "c"]
    )
    pag = fci_from_data(data, alpha=0.01)
    assert pag.skeleton == (("a", "b"), ("b", "c"))
    assert len(pag.undetermined) == 2
    assert pag.detail["alpha"] == "0.01"


def test_fci_on_data_recovers_the_collider_the_cpdag_would() -> None:
    values = _linear({("a", "c"): 1.1, ("b", "c"): 0.9}, ["a", "b", "c"], 4000, 1)
    data = Dataset.observational(
        np.column_stack([values[n] for n in ("a", "b", "c")]), ["a", "b", "c"]
    )
    pag = fci_from_data(data, alpha=0.01)
    assert all(edge.mark_at("c") == "arrow" for edge in pag.edges)
    truth = cpdag(CausalGraph.from_edges("a -> c, b -> c"))
    assert {tuple(sorted(e)) for e in truth.skeleton} == set(pag.skeleton)


def test_fci_on_data_sees_the_confounding_ges_would_have_to_ignore() -> None:
    values = _linear(
        {("z1", "x"): 1.2, ("z2", "y"): 1.1, ("u", "x"): 1.4, ("u", "y"): 1.3},
        ["z1", "z2", "u", "x", "y"],
        6000,
        2,
    )
    observed = ["z1", "z2", "x", "y"]
    data = Dataset.observational(np.column_stack([values[n] for n in observed]), observed)
    pag = fci_from_data(data, alpha=0.01)
    confounded = {frozenset({e.a, e.b}) for e in pag.bidirected}
    assert frozenset({"x", "y"}) in confounded
