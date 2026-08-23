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
    assert "1 to 3" in text and "90%" in text


# -- the fallback, which is what makes this cover everything -----------------------------


def test_a_type_with_no_renderer_still_renders_from_its_fields() -> None:
    """Sixty-one result types; a dozen renderers. The rest must not be worse off."""
    graph = CausalGraph.from_edges("a -> b", name="g")
    assert type(graph) not in REGISTRY
    text = render(graph)
    assert "CausalGraph" in text
    assert "nodes" in text


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
    assert "1 to 2" in written
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
