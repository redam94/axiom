"""Recovery for structure learning: does the search find the class the world is in?

The honest target is not the DAG — observational data cannot identify one — but
the *equivalence class*. So the test is: simulate from a known DAG, and check
that GES returns that DAG's CPDAG, no more and no less. Interventional data
should then narrow the class exactly as far as the theory says it can, and no
further.
"""

from __future__ import annotations

import numpy as np
import pytest

from axiom.discover import (
    Dataset,
    GaussianBIC,
    cpdag,
    ges,
    gies,
    interventional_essential_graph,
    orientation_gain,
)
from axiom.identify import CausalGraph

pytestmark = pytest.mark.recovery

TRUTH = CausalGraph.from_edges("a -> b, b -> c, a -> d, d -> c")
COEFFICIENTS = {("a", "b"): 1.4, ("b", "c"): -0.9, ("a", "d"): 0.8, ("d", "c"): 1.1}


def simulate(n: int, do: tuple[str, ...] = (), seed: int = 0) -> Dataset:
    """Linear-Gaussian data from TRUTH; a randomized variable ignores its parents."""
    rng = np.random.default_rng(seed)
    order = list(TRUTH.topological_order())
    values: dict[str, np.ndarray] = {}
    for node in order:
        if node in do:
            values[node] = rng.normal(0.0, 1.5, n)
            continue
        column = rng.normal(0.0, 1.0, n)
        for parent in sorted(TRUTH.parents(node)):
            column = column + COEFFICIENTS[(parent, node)] * values[parent]
        values[node] = column
    return Dataset(np.column_stack([values[n] for n in order]), tuple(order), (frozenset(do),) * n)


def test_ges_recovers_the_true_cpdag_from_observation() -> None:
    result = ges(GaussianBIC(simulate(4000, seed=1)))
    truth = cpdag(TRUTH)
    assert set(result.essential.directed) == set(truth.directed)
    assert set(result.essential.undirected) == set(truth.undirected)
    assert result.undecided == 2, "observation cannot orient a -> b or a -> d"


def test_observation_cannot_do_better_than_the_class() -> None:
    """More observational data narrows nothing further: the class is the ceiling."""
    small = ges(GaussianBIC(simulate(1000, seed=5)))
    large = ges(GaussianBIC(simulate(20000, seed=6)))
    assert small.essential.undirected == large.essential.undirected


def test_one_intervention_orients_exactly_what_the_theory_says_it_would() -> None:
    predicted = orientation_gain(TRUTH, [["b"]])
    assert predicted == (("a", "b"),)

    pooled = Dataset.stack(simulate(4000, seed=1), simulate(2000, do=("b",), seed=2))
    result = gies(GaussianBIC(pooled))
    expected = interventional_essential_graph(TRUTH, [["b"]])
    assert set(result.essential.directed) == set(expected.directed)
    assert set(result.essential.undirected) == set(expected.undirected)


def test_enough_interventions_recover_the_dag_itself() -> None:
    pooled = Dataset.stack(
        simulate(4000, seed=1),
        simulate(2000, do=("b",), seed=2),
        simulate(2000, do=("a",), seed=3),
        simulate(2000, do=("d",), seed=4),
    )
    result = gies(GaussianBIC(pooled))
    assert result.essential.is_dag
    assert set(result.essential.directed) == set(TRUTH.edges)


def test_the_turning_phase_is_what_makes_that_work() -> None:
    """Without turning the search stops at the complete graph, far below the truth.

    An interventional essential graph orients a cut edge as soon as the edge
    is inserted, so a forward step that guesses wrong is locked in and no
    insert or delete escapes it. This pins the size of the gap, so a future
    change that quietly drops the turning phase fails here.
    """
    pooled = Dataset.stack(
        simulate(4000, seed=1),
        simulate(2000, do=("b",), seed=2),
        simulate(2000, do=("a",), seed=3),
        simulate(2000, do=("d",), seed=4),
    )
    score = GaussianBIC(pooled)
    truth_score = score.total({n: TRUTH.parents(n) for n in TRUTH.nodes})
    complete = CausalGraph.from_edges("a -> b, a -> c, a -> d, b -> c, b -> d, c -> d")
    complete_score = score.total({n: complete.parents(n) for n in complete.nodes})
    assert truth_score - complete_score > 1000, "the trap is a real one, not a rounding error"

    result = gies(score)
    assert "turning" in result.phases
    assert any(step.startswith("turning") for step in result.steps)
    assert result.score == pytest.approx(truth_score)


def test_the_search_reports_what_it_did() -> None:
    result = ges(GaussianBIC(simulate(2000, seed=7)))
    assert result.phases == ("forward", "backward", "turning")
    assert result.steps
    assert result.penalty == 1.0
    assert "oriented" in result.summary()


def test_the_answer_is_stable_across_a_sensible_penalty_range() -> None:
    data = simulate(2000, seed=8)
    truth = cpdag(TRUTH)
    for penalty in (0.5, 1.0, 2.0, 5.0):
        result = ges(GaussianBIC(data, penalty=penalty))
        assert set(result.essential.directed) == set(truth.directed), penalty
        assert set(result.essential.undirected) == set(truth.undirected), penalty


def test_an_overwhelming_penalty_empties_the_graph() -> None:
    result = ges(GaussianBIC(simulate(2000, seed=8), penalty=400.0))
    assert result.essential.skeleton == ()


def test_the_penalty_is_not_monotone_through_a_greedy_search() -> None:
    """A caveat with teeth: more penalty can give *more* edges, not fewer.

    At ``penalty=20`` on this world the search lands in a local optimum with
    five edges, where ``penalty=5`` finds the true four-edge class. Greedy
    search follows a path, and changing the score changes the path, not just
    the destination. Pinned here so the caveat in the docs stays true and so
    nobody reads the knob as a monotone dial.
    """
    data = simulate(2000, seed=8)
    lenient = ges(GaussianBIC(data, penalty=5.0))
    stricter = ges(GaussianBIC(data, penalty=20.0))
    assert len(stricter.essential.skeleton) > len(lenient.essential.skeleton)


def test_interventional_rows_still_inform_the_variables_they_did_not_randomize() -> None:
    """Randomizing `a` destroys information about a's parents and no one else's."""
    score = GaussianBIC(simulate(3000, do=("a",), seed=9))
    with_parent = score.local("b", {"a"})
    without = score.local("b", set())
    assert with_parent > without, "a -> b is still learnable from rows where a was randomized"
    assert score.local("a", {"b"}) == pytest.approx(score.local("a", {"b"}))
