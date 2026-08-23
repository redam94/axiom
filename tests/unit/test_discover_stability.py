"""Bootstrap stability: which parts of a discovered graph survive resampling."""

from __future__ import annotations

import numpy as np
import pytest

from axiom.discover import Dataset, EdgeSupport, StabilityReport, edge_stability

NAMES = ["a", "b", "c", "d"]


def world(n: int = 400, seed: int = 0) -> Dataset:
    """a -> b -> c <- d <- a: two colliders into c, two edges out of a."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    b = 1.4 * a + rng.normal(size=n)
    d = 0.8 * a + rng.normal(size=n)
    c = -0.9 * b + 1.1 * d + rng.normal(size=n)
    return Dataset.observational(np.column_stack([a, b, c, d]), NAMES)


def test_the_true_edges_are_stable_and_the_rest_are_not() -> None:
    report = edge_stability(world(), n_bootstrap=30, seed=1)
    assert isinstance(report, StabilityReport)
    stable = {frozenset({e.a, e.b}) for e in report.stable(0.8)}
    assert stable == {
        frozenset({"a", "b"}),
        frozenset({"a", "d"}),
        frozenset({"b", "c"}),
        frozenset({"c", "d"}),
    }


def test_a_stable_edge_with_an_unsettled_direction_is_its_own_finding() -> None:
    """The distinction that changes what you do next.

    ``a - b`` is present in every resample and oriented in none: observational
    data has said everything it can, and only an intervention settles it. More
    rows would not help, which is exactly what ``contested`` does not capture
    and ``undecided_direction`` does.
    """
    report = edge_stability(world(), n_bootstrap=30, seed=1)
    undecided = {frozenset({e.a, e.b}) for e in report.undecided_direction(0.8)}
    assert undecided == {frozenset({"a", "b"}), frozenset({"a", "d"})}
    for edge in report.edges:
        if {edge.a, edge.b} == {"b", "c"}:
            assert edge.oriented > 0.8, "the collider edges do get oriented"


def test_the_report_summarizes_and_splits_the_edges() -> None:
    report = edge_stability(world(), n_bootstrap=20, seed=2)
    assert report.n_bootstrap == 20
    assert report.n_rows == 400
    assert report.detail["search"] == "ges"
    assert "stable edges" in report.summary()
    assert set(report.stable(0.9)).isdisjoint(report.contested(0.2, 0.9))
    assert all(e.describe() for e in report.edges)


def test_the_orientation_shares_sum_to_the_adjacency() -> None:
    report = edge_stability(world(), n_bootstrap=15, seed=3)
    for edge in report.edges:
        assert edge.forward + edge.backward + edge.undirected == pytest.approx(edge.adjacent)
    with pytest.raises(ValueError, match="orientations sum"):
        EdgeSupport(a="a", b="b", adjacent=1.0, forward=0.5, backward=0.0, undirected=0.0)
    # Writing the pair the other way round swaps the two directions with it.
    flipped = EdgeSupport(a="b", b="a", adjacent=0.6, forward=0.1, backward=0.5, undirected=0.0)
    assert (flipped.a, flipped.b) == ("a", "b")
    assert (flipped.forward, flipped.backward) == (0.5, 0.1)


def test_interventional_rows_keep_their_regime_through_the_resample() -> None:
    rng = np.random.default_rng(4)
    n = 300
    a = rng.normal(size=n)
    b = 1.4 * a + rng.normal(size=n)
    d = 0.8 * a + rng.normal(size=n)
    c = -0.9 * b + 1.1 * d + rng.normal(size=n)
    observational = Dataset.observational(np.column_stack([a, b, c, d]), NAMES)
    randomized_a = rng.normal(0.0, 1.5, n)
    b2 = 1.4 * randomized_a + rng.normal(size=n)
    d2 = 0.8 * randomized_a + rng.normal(size=n)
    c2 = -0.9 * b2 + 1.1 * d2 + rng.normal(size=n)
    intervened = Dataset(
        np.column_stack([randomized_a, b2, c2, d2]), tuple(NAMES), (frozenset({"a"}),) * n
    )
    report = edge_stability(Dataset.stack(observational, intervened), n_bootstrap=15, seed=5)
    assert report.detail["search"] == "gies"
    assert report.targets == (("a",),)
    forward = {frozenset({e.a, e.b}): e.oriented for e in report.edges}
    assert forward[frozenset({"a", "b"})] > 0.5, "the experiment orients what observation could not"


def test_one_resample_is_allowed_and_zero_is_not() -> None:
    assert edge_stability(world(n=200), n_bootstrap=1, seed=6).n_bootstrap == 1
    with pytest.raises(ValueError, match="at least one"):
        edge_stability(world(n=200), n_bootstrap=0)
