"""Refuting a graph with its own implied independencies."""

from __future__ import annotations

import numpy as np
import pytest

from axiom.core import Unsupported
from axiom.diagnose import (
    ImpliedIndependence,
    StructureRefutation,
    implied_independencies,
    refute_structure,
)
from axiom.diagnose.structure import adjust
from axiom.discover import Dataset, IndependenceResult, PartialCorrelation
from axiom.identify import CausalGraph

TRUTH = CausalGraph.from_edges("a -> b, b -> c, a -> d, d -> c")


def world(n: int = 3000, seed: int = 0, extra: float = 0.0) -> Dataset:
    """Data from TRUTH; ``extra`` adds an edge a -> c the graph does not have."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    b = 1.4 * a + rng.normal(size=n)
    d = 0.8 * a + rng.normal(size=n)
    c = -0.9 * b + 1.1 * d + extra * a + rng.normal(size=n)
    return Dataset.observational(np.column_stack([a, b, c, d]), ["a", "b", "c", "d"])


# -- what a graph claims ---------------------------------------------------------------


def test_a_graph_claims_one_independence_per_missing_edge() -> None:
    claims = implied_independencies(TRUTH)
    pairs = {pair for _, pair in claims}
    assert pairs == {("a", "c"), ("b", "d")}
    by_pair = {pair: implication for implication, pair in claims}
    assert by_pair[("b", "d")].given == ("a",)
    assert set(by_pair[("a", "c")].given) == {"b", "d"}


def test_an_implication_names_the_edge_it_denies() -> None:
    implication = ImpliedIndependence(x="c", y="a", given=("b",))
    assert implication.edge == ("a", "c")
    assert "independent" in implication.describe()
    with pytest.raises(ValueError, match="two different variables"):
        ImpliedIndependence(x="a", y="a")


def test_a_complete_graph_claims_nothing() -> None:
    complete = CausalGraph.from_edges("a -> b, a -> c, b -> c")
    assert implied_independencies(complete) == ()


# -- the check -------------------------------------------------------------------------


def test_the_true_graph_survives_but_is_not_confirmed() -> None:
    report = refute_structure(TRUTH, world())
    assert isinstance(report, StructureRefutation)
    assert not report.refuted
    assert report.verdict.status == "downgraded"
    assert {a.name: a.state for a in report.verdict.assumptions} == {
        "graph_is_correct": "unverified"
    }
    assert "not a confirmation" in report.verdict.reason
    assert "Surviving is not passing" in report.summary()


def test_a_missing_edge_is_refuted_and_named() -> None:
    wrong = CausalGraph.from_edges("a -> b, b -> c, a -> d")  # d -> c dropped
    report = refute_structure(wrong, world())
    assert report.refuted
    assert report.verdict.status == "blocked"
    assert {a.name: a.state for a in report.verdict.assumptions} == {"graph_is_correct": "violated"}
    assert report.implicated[0][:2] == ("c", "d")
    assert report.implicated[0][2] > 0.5


def test_an_edge_the_graph_omits_shows_up_as_the_biggest_violation() -> None:
    """The data has a -> c; the graph does not. The refutation points straight at it."""
    report = refute_structure(TRUTH, world(extra=1.5))
    assert report.refuted
    assert report.implicated[0][:2] == ("a", "c")


def test_the_effect_threshold_is_what_keeps_large_samples_meaningful() -> None:
    """A tiny extra edge is significant at n = 40000 and unimportant at any n."""
    data = world(n=40000, extra=0.05)
    significant = refute_structure(TRUTH, data, effect_threshold=0.0)
    practical = refute_structure(TRUTH, data, effect_threshold=0.1)
    assert significant.refuted, "at this sample size the tiny edge is detectable"
    assert not practical.refuted, "and at this effect size it is not worth acting on"
    assert practical.worst is not None
    assert practical.worst.effect < 0.1


def test_the_correction_makes_a_refutation_harder_not_easier() -> None:
    data = world(extra=0.08)
    raw = refute_structure(TRUTH, data, correction="none")
    holm = refute_structure(TRUTH, data, correction="holm")
    fdr = refute_structure(TRUTH, data, correction="benjamini_hochberg")
    assert raw.n_refuted >= fdr.n_refuted >= holm.n_refuted


def test_multiplicity_corrections_behave() -> None:
    assert adjust([], "holm") == ()
    assert adjust([0.01, 0.5], "none") == (0.01, 0.5)
    assert adjust([0.01, 0.5], "holm") == (0.02, 0.5)
    assert adjust([0.01, 0.04], "benjamini_hochberg") == (0.02, 0.04)
    assert all(0.0 <= p <= 1.0 for p in adjust([0.9, 0.95], "holm"))


def test_a_graph_naming_columns_the_data_lacks_is_unsupported() -> None:
    graph = CausalGraph.from_edges("a -> zzz")
    out = refute_structure(graph, world())
    assert isinstance(out, Unsupported)
    assert out.missing == ("zzz",)


def test_implications_that_touch_a_latent_are_untested_not_passed() -> None:
    graph = CausalGraph.from_edges("a -> b, u -> b, u -> c", unmeasured=["u"])
    data = Dataset.observational(np.zeros((50, 4)) + np.arange(4), ["a", "b", "c", "u"])
    report = refute_structure(graph, data)
    assert isinstance(report, StructureRefutation)
    assert any("u" in pair for pair in report.untested)


def test_alpha_must_be_a_probability() -> None:
    with pytest.raises(ValueError, match="alpha"):
        refute_structure(TRUTH, world(), alpha=1.5)


# -- the test underneath ----------------------------------------------------------------


def test_the_partial_correlation_recovers_a_known_conditional_independence() -> None:
    tester = PartialCorrelation(world())
    dependent = tester.test("a", "c")
    conditioned = tester.test("a", "c", ["b", "d"])
    assert isinstance(dependent, IndependenceResult)
    assert isinstance(conditioned, IndependenceResult)
    # The two paths a->b->c and a->d->c carry opposite signs and partly cancel,
    # which is why the marginal dependence is 0.21 rather than something larger --
    # a mild version of exactly the near-unfaithfulness these methods assume away.
    assert dependent.effect > 0.15, "a and c are dependent marginally"
    assert conditioned.effect < 0.05, "and independent given b and d"
    assert conditioned.p_value > 0.01
    assert "r =" in conditioned.describe()


def test_a_test_with_too_few_rows_is_unsupported_rather_than_confident() -> None:
    tiny = Dataset.observational(np.random.default_rng(0).normal(size=(4, 3)), ["a", "b", "c"])
    out = PartialCorrelation(tiny).test("a", "b", ["c"])
    assert isinstance(out, Unsupported)
    assert "rows" in out.reason


def test_a_constant_column_has_no_correlation_to_report() -> None:
    values = np.random.default_rng(0).normal(size=(50, 2))
    flat = np.column_stack([values, np.ones(50)])
    out = PartialCorrelation(Dataset.observational(flat, ["a", "b", "k"])).test("a", "k")
    assert isinstance(out, Unsupported)
    assert "does not vary" in out.reason


def test_rows_where_a_variable_was_randomized_are_left_out_of_its_tests() -> None:
    rng = np.random.default_rng(0)
    n = 400
    a = rng.normal(size=n)
    b = 1.5 * a + rng.normal(size=n)
    observational = Dataset.observational(np.column_stack([a, b]), ["a", "b"])
    randomized = Dataset(
        np.column_stack([rng.normal(size=n), rng.normal(size=n)]),
        ("a", "b"),
        (frozenset({"a"}),) * n,
    )
    pooled = Dataset.stack(observational, randomized)
    tester = PartialCorrelation(pooled)
    result = tester.test("a", "b")
    assert isinstance(result, IndependenceResult)
    assert result.n == n, "only the rows where neither was randomized can answer"
    assert result.effect > 0.5


def test_a_variable_is_not_tested_against_itself() -> None:
    with pytest.raises(ValueError, match="not tested against itself"):
        IndependenceResult(x="a", y="a", correlation=0.0, p_value=1.0, n=10)
    with pytest.raises(ValueError, match="cannot condition on"):
        IndependenceResult(x="a", y="b", given=("a",), correlation=0.0, p_value=1.0, n=10)
