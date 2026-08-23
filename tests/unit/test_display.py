"""Showing a result: the same content whether or not rich is installed."""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

from axiom.core import Assumption, Interval, LedgerLine, Summary, Unsupported, Unverified, Verdict
from axiom.display import (
    REGISTRY,
    STATUS_MARK,
    Card,
    Row,
    available,
    card_for,
    generic_card,
    render,
    renders,
    show,
)
from axiom.identify import CausalGraph, identify, ols


@pytest.fixture
def frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 300
    age = rng.normal(size=n)
    dose = 0.6 * age + rng.normal(size=n)
    pressure = -1.2 * dose + 0.8 * age + rng.normal(size=n)
    return pd.DataFrame({"age": age, "dose": dose, "pressure": pressure})


# -- the card ----------------------------------------------------------------------------


def test_a_card_skips_empty_values() -> None:
    """A blank line says nothing, and a card of blank lines says nothing loudly."""
    card = Card(title="T").add("kept", "value").add("dropped", "").add("also", None)
    assert [r.label for r in card.rows] == ["kept"]


def test_the_plain_form_is_the_same_string_every_time() -> None:
    card = Card(title="T", status="good", rows=[Row("a", "1")], note="why")
    once, twice = render(card), render(card)
    assert once == twice
    assert "[ok]" in once and "a" in once and "why" in once


def test_every_status_has_a_mark_and_neutral_has_none() -> None:
    assert set(STATUS_MARK) == {"good", "assumed", "bad", "neutral"}
    assert STATUS_MARK["neutral"] == ""
    assert "T" == render(Card(title="T")).splitlines()[0], "neutral prints no mark"


# -- what the renderers say --------------------------------------------------------------


def test_an_interval_keeps_its_definition_and_mass() -> None:
    """A point estimate that lost its interval is the failure this prevents."""
    text = render(Interval(lower=-16.7, upper=-8.1, definition="eti", mass=0.9))
    assert "-16.7 to -8.1" in text
    assert "90%" in text
    assert "ETI" in text


def test_a_verdict_colours_by_what_it_concluded() -> None:
    # a downgraded verdict must name what downgraded it -- axiom's own rule, and
    # the reason this case cannot be written as a one-liner like the others
    standing = Assumption(name="ignorable", facet="population", statement="s")
    assert card_for(Verdict(status="identified", reason="r")).status == "good"
    assert (
        card_for(Verdict(status="downgraded", reason="r", assumptions=(standing,))).status
        == "assumed"
    )
    assert card_for(Verdict(status="blocked", reason="r")).status == "bad"


def test_an_unchecked_assumption_is_assumed_not_good() -> None:
    """The honest colour for a condition nobody checked."""
    unchecked = Assumption(name="a", facet="population", statement="s", state="unverified")
    checked = Assumption(name="a", facet="population", statement="s", state="satisfied")
    broken = Assumption(name="a", facet="population", statement="s", state="violated")
    assert card_for(unchecked).status == "assumed"
    assert card_for(checked).status == "good"
    assert card_for(broken).status == "bad"


def test_a_failure_says_what_is_missing() -> None:
    text = render(Unsupported(reason="plotly is not installed", missing=("plotly",)))
    assert "[!]" in text and "plotly" in text
    assert card_for(Unverified(reason="did not converge")).status == "assumed"


def test_an_identification_verdict_names_the_route_and_the_adjustment_set() -> None:
    graph = CausalGraph.from_edges("age -> dose, age -> pressure, dose -> pressure")
    text = render(identify(graph, "dose", "pressure"))
    assert "identified" in text
    assert "backdoor" in text
    assert "age" in text


def test_an_estimate_leads_with_its_interval(frame: pd.DataFrame) -> None:
    estimate = ols(frame, "pressure", "dose", ["age"])
    card = card_for(estimate)
    text = render(card)
    assert any(r.emphasis for r in card.rows), "the number a reader came for is not emphasised"
    assert "90% interval" in text and "WALD" in text
    assert "adjusted for" in text and "age" in text


def test_a_ledger_line_carries_its_assumption() -> None:
    assumption = Assumption(name="ignorable", facet="population", statement="s")
    text = render(LedgerLine(kind="assumption", statement="stated", assumption=assumption))
    assert "stated" in text and "ignorable" in text


def test_a_summary_shows_its_interval() -> None:
    band = Interval(lower=1.0, upper=3.0, definition="eti", mass=0.9)
    text = render(Summary(mean=2.0, median=2.0, sd=0.5, interval=band, n=1000))
    assert "1.00 to 3.00" in text and "90%" in text


def test_a_summary_prints_no_more_digits_than_its_spread_supports() -> None:
    """The point summaries stop where the sd stops; the rest was arithmetic."""
    band = Interval(lower=10.1234, upper=14.5678, definition="hdi", mass=0.9)
    text = render(Summary(mean=12.3456789, median=12.2987, sd=1.23456, interval=band, n=4000))
    assert "12.3" in text and "12.3456" not in text
    assert "1.2" in text and "1.23456" not in text


# -- the fallback, which is what makes this cover everything -----------------------------


def test_a_type_with_no_renderer_still_renders_from_its_fields() -> None:
    """Sixty-one result types; a handful of renderers. The rest must not be worse off."""
    from axiom.core import Treatment

    treatment = Treatment(name="fertilizer", unit="kg")
    assert type(treatment) not in REGISTRY
    text = render(treatment)
    assert "Treatment" in text
    assert "fertilizer" in text


def test_the_generic_card_shortens_long_collections() -> None:
    card = generic_card(CausalGraph.from_edges(", ".join(f"n{i} -> y" for i in range(9))))
    nodes = next(r for r in card.rows if r.label == "nodes")
    assert "…" in nodes.value or len(nodes.value) < 80


def test_a_plain_object_is_not_a_crash() -> None:
    assert "value" in render(object()).lower() or "object" in render(object())


# -- registering your own ----------------------------------------------------------------


def test_a_new_type_can_register_a_renderer() -> None:
    class Readout:
        pass

    @renders(Readout)
    def _readout(obj: Readout) -> Card:
        return Card(title="Readout", status="good", rows=[Row("v", "1")])

    try:
        assert card_for(Readout()).title == "Readout"
        assert "[ok]" in render(Readout())
    finally:
        REGISTRY.pop(Readout, None)


def test_a_subclass_uses_its_parents_renderer() -> None:
    class Narrower(Interval):
        pass

    card = card_for(Narrower(lower=0.0, upper=1.0, definition="eti", mass=0.9))
    assert card.title == "Interval", "the mro was not walked"


# -- output ------------------------------------------------------------------------------


def test_show_writes_the_plain_form_when_asked() -> None:
    buffer = io.StringIO()
    show(Interval(lower=1.0, upper=2.0, definition="eti", mass=0.9), file=buffer, plain=True)
    written = buffer.getvalue()
    assert "1.00 to 2.00" in written
    assert "\x1b[" not in written, "plain output must carry no escape codes"


def test_show_works_whether_or_not_rich_is_installed() -> None:
    buffer = io.StringIO()
    show(Verdict(status="identified", reason="because"), file=buffer)
    assert "identified" in buffer.getvalue()


@pytest.mark.skipif(not available(), reason="rich is not installed")
def test_the_rich_form_carries_the_same_content() -> None:
    """Colour is a typeface choice; losing a row would be a content change."""
    band = Interval(lower=-16.7, upper=-8.1, definition="eti", mass=0.9)
    buffer = io.StringIO()
    show(band, file=buffer)
    rich_text = buffer.getvalue()
    for fragment in ("-16.7", "-8.1", "90%", "ETI"):
        assert fragment in rich_text, f"{fragment} was lost in the rich rendering"


def test_enable_is_false_outside_a_notebook() -> None:
    from axiom.display import enable

    assert enable() is False, "there is no IPython here to register with"


# -- the three that rendered worse than print until someone looked -----------------------


def test_a_posterior_card_is_not_just_its_title() -> None:
    """``Posterior`` is not a ``Spec``, so the generic renderer had no fields to
    walk and produced a card with a title and nothing else — strictly worse than
    the ``print`` it was meant to replace."""
    from axiom.core import Posterior

    rng = np.random.default_rng(0)
    post = Posterior(
        {"mu": rng.normal(size=(4, 500, 2)), "sigma": rng.normal(size=(4, 500))},
        coords={"treatment": ["a", "b"]},
        provenance={"seed": 42},
    )
    text = render(post)
    assert len(text.splitlines()) > 1, "a bare title is the bug this test exists for"
    for fragment in ("mu", "sigma", "4", "2000", "seed 42"):
        assert fragment in text, f"{fragment} is missing from the card"


def test_a_posterior_lists_its_parameters_in_a_stable_order() -> None:
    """``names()`` is a frozenset; rendering it unsorted reorders per run."""
    from axiom.core import Posterior

    rng = np.random.default_rng(1)
    draws = {name: rng.normal(size=(2, 50)) for name in ("zeta", "alpha", "mu")}
    assert "alpha, mu, zeta" in render(Posterior(draws))


def test_a_posterior_says_so_when_no_seed_was_recorded() -> None:
    from axiom.core import Posterior

    assert "no seed recorded" in render(Posterior({"a": np.zeros((2, 10))}))


def test_a_panel_card_carries_the_shape_and_where_the_gap_is() -> None:
    from axiom.core import Outcome
    from axiom.data import Panel, RoleMap

    frame = pd.DataFrame({"unit": ["a", "a", "b"], "period": [1, 2, 1], "y": [0.1, 0.2, 0.3]})
    panel = Panel(frame, RoleMap(unit="unit", time="period", outcome=("y", Outcome(name="y"))))
    text = render(panel)
    assert len(text.splitlines()) > 1
    assert "missing" in text, "an unbalanced panel should say it is unbalanced"
    assert "b=1" in text, "the gap should say which unit it is in"


def test_a_panel_names_its_treatments() -> None:
    """`roles.treatments` is a dict, so iterating yields names and not pairs.
    Reading it as pairs raised, and a panel with no treatments — which is what
    the first test used — never enters the loop that does it."""
    from axiom.core import D, Outcome, Treatment
    from axiom.data import Panel, RoleMap

    frame = pd.DataFrame(
        {"unit": ["a", "a"], "period": [1, 2], "y": [0.1, 0.2], "dose": [1.0, 2.0]}
    )
    panel = Panel(
        frame,
        RoleMap(
            unit="unit",
            time="period",
            outcome=("y", Outcome(name="y", dimension=D.outcome)),
            treatments={"dose": Treatment(name="dose", dimension=D.currency, unit="USD")},
        ),
    )
    assert "dose" in render(panel)


def test_a_transfer_plan_leads_with_the_cost_not_the_hashes() -> None:
    """The generic card led with two 64-character content hashes and truncated
    the three fields a reader wants. And *no correction but one assumption* is
    the common case, so it has to read as something rather than an empty tuple."""
    from axiom.core import D, Intervention, Outcome, Population, TimeWindow, Treatment
    from axiom.estimands import Estimand, Level, Quantity

    shared = dict(
        quantity=Quantity(kind="contrast"),
        treatment=Treatment(name="fertilizer", dimension=D.currency, unit="USD"),
        intervention=Intervention(doses={"fertilizer": 100.0}, version="granular"),
        reference=Intervention(doses={"fertilizer": 0.0}, version="granular"),
        outcome=Outcome(name="yield_total", dimension=D.outcome, unit="kg"),
        window=TimeWindow(start=0, stop=8),
        level=Level(unit="cluster"),
        dimension=D.outcome,
    )
    here = Estimand(name="trial", population=Population(name="north"), **shared)
    there = Estimand(name="region", population=Population(name="whole_region"), **shared)

    text = render(here.transfer_to(there))
    assert "population" in text, "the facet that differs is the finding"
    assert "s_admissibility" in text, "the assumption is the whole cost of this move"
    assert "corrections             0" in text, "no corrections must read as zero, not blank"
    assert "8e2c2ccb" not in text, "a content hash is not what a reader came for"


def test_a_causal_graph_card_keeps_the_arrows() -> None:
    """``edges`` are bare 2-tuples, so the generic flattener rendered
    ``X -> Y, Z -> X`` as ``X, Y, Z, X`` — every arrow gone, and the result read
    as a node list."""
    from axiom.identify import CausalGraph

    graph = CausalGraph.from_edges("Z -> X, Z -> Y, X -> Y, A <-> B", unmeasured=["A"], name="toy")
    text = render(graph)
    assert "->" in text, "a graph card without an arrow is a node list"
    assert "Z -> X" in text
    assert "A <-> B" in text, "the bidirected edge is the one that decides identifiability"
    assert "toy" in text
