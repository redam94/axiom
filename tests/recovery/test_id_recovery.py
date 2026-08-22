"""Recovery for identification: does the ID estimand reproduce the world's true effect?

An identification algorithm that returns a formula is making a claim that can
be checked without any statistics: build a discrete world, compute its true
``P(y | do(x))`` by intervening on the generating process, then evaluate the
returned formula on the world's *observational* joint. If identification is
sound the two agree exactly — no sampling, no estimation, no tolerance beyond
floating point.

This is the test that separates "the algorithm ran" from "the algorithm is
right", and it is why ``identify.formula`` can be evaluated at all.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from axiom.identify import CausalGraph, identify_effect
from axiom.identify.formula import JointTable, evaluate

pytestmark = pytest.mark.recovery


class DiscreteWorld:
    """A discrete structural causal model: every node a random conditional table."""

    def __init__(
        self,
        graph: CausalGraph,
        levels: int = 2,
        seed: int = 0,
        tables: dict[str, np.ndarray] | None = None,
    ) -> None:
        self.graph = graph
        self.order = list(graph.topological_order())
        self.levels = levels
        rng = np.random.default_rng(seed)
        given = tables or {}
        self.tables: dict[str, tuple[list[str], np.ndarray]] = {}
        for node in self.order:
            parents = sorted(graph.parents(node))
            if node in given:
                self.tables[node] = (parents, np.asarray(given[node], dtype=float))
                continue
            rows = levels ** len(parents)
            table = rng.dirichlet(np.full(levels, 0.8), size=rows)
            self.tables[node] = (parents, table.reshape([levels] * len(parents) + [levels]))

    def joint(self, do: dict[str, int] | None = None) -> np.ndarray:
        """The joint over every node, optionally with some nodes held fixed."""
        do = do or {}
        shape = tuple(self.levels for _ in self.order)
        axis = {name: i for i, name in enumerate(self.order)}
        out = np.ones(shape)
        for node in self.order:
            parents, table = self.tables[node]
            block = np.zeros(shape)
            for key in itertools.product(range(self.levels), repeat=len(self.order)):
                if node in do:
                    block[key] = 1.0 if key[axis[node]] == do[node] else 0.0
                else:
                    lookup = tuple(key[axis[p]] for p in parents) + (key[axis[node]],)
                    block[key] = table[lookup]
            out = out * block
        return out

    def observational(self, observed: list[str]) -> JointTable:
        table = self.joint()
        keep = [self.order.index(n) for n in observed]
        summed = tuple(i for i in range(len(self.order)) if i not in keep)
        marginal = np.sum(table, axis=summed)
        order = np.argsort(np.argsort(keep))
        marginal = np.transpose(marginal, order)
        return JointTable(names=tuple(observed), table=marginal / marginal.sum())

    def true_effect(self, treatment: str, outcome: str) -> np.ndarray:
        """``P(outcome | do(treatment))`` indexed ``[treatment level, outcome level]``."""
        out = np.zeros((self.levels, self.levels))
        for value in range(self.levels):
            table = self.joint(do={treatment: value})
            keep = self.order.index(outcome)
            summed = tuple(i for i in range(len(self.order)) if i != keep)
            out[value] = np.sum(table, axis=summed)
        return out


def _agrees(
    edges: str, unmeasured: tuple[str, ...], treatment: str, outcome: str, seed: int
) -> None:
    graph = CausalGraph.from_edges(edges, unmeasured=unmeasured)
    result = identify_effect(graph, treatment, outcome)
    assert result.identified, result.to_text()
    assert result.formula is not None

    world = DiscreteWorld(graph, levels=2, seed=seed)
    observed = [n for n in world.order if n not in unmeasured]
    joint = world.observational(observed)
    got = np.asarray(evaluate(result.formula, joint))
    truth = world.true_effect(treatment, outcome)
    for x, y in itertools.product(range(2), repeat=2):
        index = [0] * len(observed)
        index[joint.axis(treatment)] = x
        index[joint.axis(outcome)] = y
        assert got[tuple(index)] == pytest.approx(
            truth[x, y], abs=1e-12
        ), f"do({treatment}={x}), {outcome}={y}: {result.to_text()}"


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_back_door_estimand_is_the_truth(seed: int) -> None:
    _agrees("Z -> X, Z -> Y, X -> Y", (), "X", "Y", seed)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_front_door_estimand_is_the_truth(seed: int) -> None:
    _agrees("X -> M, M -> Y, U -> X, U -> Y", ("U",), "X", "Y", seed)


@pytest.mark.parametrize("seed", [0, 1])
def test_the_napkin_estimand_is_the_truth(seed: int) -> None:
    """No back-door set exists here; the estimand ID returns is still exactly right."""
    _agrees(
        "W1 -> W2, W2 -> X, X -> Y, U1 -> W1, U1 -> X, U2 -> W1, U2 -> Y",
        ("U1", "U2"),
        "X",
        "Y",
        seed,
    )


@pytest.mark.parametrize("seed", [0, 1])
def test_the_verma_estimand_is_the_truth(seed: int) -> None:
    _agrees("A -> B, B -> C, C -> D, U -> B, U -> D", ("U",), "A", "D", seed)


@pytest.mark.parametrize("seed", [0, 1])
def test_the_estimand_no_named_route_would_have_found_is_the_truth(seed: int) -> None:
    """The graph where back-door, front-door and instrument all fail and ID does not."""
    _agrees("M -> Y, W -> M, W -> X, W -> Y, X -> M, M <-> W, X <-> Y", (), "X", "Y", seed)


def test_a_hedge_really_is_unidentifiable() -> None:
    """Two worlds, the same observational distribution, different effects.

    The bow arc's non-identifiability is not a limitation of the algorithm. Let
    ``U`` be a fair coin and ``X = U`` in both worlds; let ``Y = X`` in one and
    ``Y = U`` in the other. Since ``X = U``, the two worlds produce *identical*
    observational distributions over ``(X, Y)`` — and different answers to
    ``P(Y=1 | do(X=1))``: one and a half respectively. No function of what is
    observed can return both, which is exactly what the hedge asserts.
    """
    graph = CausalGraph.from_edges("X -> Y, U -> X, U -> Y", unmeasured=("U",))
    result = identify_effect(graph, "X", "Y")
    assert not result.identified
    assert result.hedge is not None

    coin = np.array([0.5, 0.5])
    copy_first = np.array([[1.0, 0.0], [0.0, 1.0]])
    # Y's parents are sorted -- ("U", "X") -- so the table is indexed [u][x][y]
    y_copies_x = np.array([[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]])
    y_copies_u = np.array([[[1.0, 0.0], [1.0, 0.0]], [[0.0, 1.0], [0.0, 1.0]]])

    first = DiscreteWorld(graph, tables={"U": coin, "X": copy_first, "Y": y_copies_x})
    second = DiscreteWorld(graph, tables={"U": coin, "X": copy_first, "Y": y_copies_u})

    observed = ["X", "Y"]
    assert np.allclose(
        first.observational(observed).table, second.observational(observed).table
    ), "the two worlds must be observationally indistinguishable"

    assert first.true_effect("X", "Y")[1, 1] == pytest.approx(1.0)
    assert second.true_effect("X", "Y")[1, 1] == pytest.approx(0.5)
