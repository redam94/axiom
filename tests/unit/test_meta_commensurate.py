from __future__ import annotations

import pytest

from axiom.core import (
    D,
    Intervention,
    LatentSelection,
    Outcome,
    Population,
    TimeWindow,
    Treatment,
    Unsupported,
)
from axiom.estimands import Estimand, Level, Quantity
from axiom.meta import (
    Commensurability,
    Corpus,
    Incompatibility,
    Pooled,
    PoolSpec,
    StudyRecord,
    commensurable,
    pool,
)

_LETTER = LatentSelection(kind="complier", instrument="letter", exposure="attended", share=0.61)
_CALL = LatentSelection(kind="complier", instrument="phone_call", exposure="attended", share=0.4)


def _estimand(population: Population, *, window: tuple[int, int] = (0, 8)) -> Estimand:
    return Estimand(
        name="lift",
        quantity=Quantity(kind="contrast"),
        treatment=Treatment(name="course", dimension=D.currency, unit="USD"),
        intervention=Intervention(doses={"course": 1.0}),
        reference=Intervention(doses={"course": 0.0}),
        outcome=Outcome(name="score", dimension=D.outcome, unit="pt", aggregation="mean"),
        population=population,
        window=TimeWindow(start=window[0], stop=window[1]),
        level=Level(unit="individual"),
        dimension=D.outcome,
    )


_EVERYBODY = Population(name="enrolled")
_COMPLIERS = Population(name="enrolled", latent=_LETTER)


def _record(study: str, party: str, *, estimate: float = 0.5, family: str = "f") -> StudyRecord:
    return StudyRecord(
        study=study,
        contributor=party,
        quantity="elasticity",
        family=family,
        estimate=estimate,
        se=0.1,
        read="experiment",
    )


def _corpus() -> Corpus:
    return Corpus(
        records=(
            _record("A-1", "acme"),
            _record("B-1", "northwind", estimate=0.6),
            _record("C-1", "globex", estimate=0.4),
        ),
        name="book",
    )


def _all_itt() -> dict[str, Estimand]:
    return {
        "A-1": _estimand(_EVERYBODY),
        "B-1": _estimand(_EVERYBODY),
        "C-1": _estimand(_EVERYBODY),
    }


# -- the check ------------------------------------------------------------------------------


def test_identical_estimands_are_identified() -> None:
    result = commensurable(_corpus(), _all_itt())
    assert result.poolable
    assert result.verdict().status == "identified"
    assert result.verdict().route == "commensurability"
    assert result.reference == "A-1" and result.checked == ("A-1", "B-1", "C-1")
    assert result.assumptions == ()


def test_different_populations_are_a_downgrade_the_pool_names() -> None:
    """Between-population heterogeneity is what a random-effects pool is for."""
    estimands = _all_itt() | {"B-1": _estimand(Population(name="south"))}
    result = commensurable(_corpus(), estimands)
    assert result.poolable
    verdict = result.verdict()
    assert verdict.status == "downgraded"
    assert [a.name for a in verdict.assumptions] == ["s_admissibility"]
    assert "population" in verdict.reason


def test_a_latent_subpopulation_blocks_a_pool_that_transfer_to_only_downgrades() -> None:
    """The headline: pooling is not transferring, so the rule here is stricter."""
    estimands = _all_itt() | {"B-1": _estimand(_COMPLIERS)}
    plan = estimands["B-1"].transfer_to(estimands["A-1"])
    assert plan.status == "downgraded"  # the transfer is licensed…

    result = commensurable(_corpus(), estimands)
    assert not result.poolable  # …and the pool is not
    verdict = result.verdict()
    assert verdict.status == "blocked"
    (entry,) = result.entries
    assert isinstance(entry, Incompatibility)
    assert entry.study == "B-1" and entry.facet == "population"
    assert entry.status == "downgraded"
    assert "a number over no population at all" in entry.reason


def test_two_different_latent_strata_block_as_the_transfer_does() -> None:
    estimands = {
        "A-1": _estimand(_COMPLIERS),
        "B-1": _estimand(Population(name="enrolled", latent=_CALL)),
        "C-1": _estimand(_COMPLIERS),
    }
    result = commensurable(_corpus(), estimands)
    assert [e.study for e in result.entries] == ["B-1"]
    assert result.verdict().status == "blocked"


def test_the_same_latent_stratum_pools_with_itself() -> None:
    resized = _LETTER.model_copy(update={"share": 0.55})
    estimands = {
        "A-1": _estimand(_COMPLIERS),
        "B-1": _estimand(Population(name="enrolled", latent=resized)),
        "C-1": _estimand(_COMPLIERS),
    }
    result = commensurable(_corpus(), estimands)
    assert result.poolable
    assert result.verdict().status == "downgraded"  # the share differs, so s_admissibility


def test_a_blocked_transfer_blocks_and_names_its_facet() -> None:
    estimands = _all_itt() | {
        "B-1": _estimand(_EVERYBODY).model_copy(update={"quantity": Quantity(kind="ratio")})
    }
    result = commensurable(_corpus(), estimands)
    (entry,) = result.entries
    assert entry.status == "blocked"
    assert entry.facet in {"quantity", "dimension"}


# -- what it will not pretend ---------------------------------------------------------------


def test_a_record_with_no_estimand_is_unchecked_not_passed() -> None:
    result = commensurable(_corpus(), {"A-1": _estimand(_EVERYBODY), "C-1": _estimand(_EVERYBODY)})
    assert result.checked == ("A-1", "C-1")
    assert result.unchecked == ("B-1",)
    assert result.verdict().status == "identified"
    assert "1 unchecked" in result.ledger_line().statement
    assert result.ledger_line().detail["unchecked"] == "B-1"


def test_a_corpus_with_no_estimands_at_all_is_an_error_not_a_pass() -> None:
    with pytest.raises(ValueError, match="no estimand supplied for any"):
        commensurable(_corpus(), {})
    with pytest.raises(ValueError, match="no records"):
        commensurable(_corpus(), _all_itt(), family="nope")


def test_the_reference_can_be_chosen_and_must_be_known() -> None:
    result = commensurable(_corpus(), _all_itt(), reference="C-1")
    assert result.reference == "C-1"
    with pytest.raises(KeyError, match="reference study"):
        commensurable(_corpus(), _all_itt(), reference="B-9")


def test_the_result_round_trips() -> None:
    result = commensurable(_corpus(), _all_itt() | {"B-1": _estimand(_COMPLIERS)})
    assert Commensurability.from_json(result.to_json()) == result


# -- the pool -------------------------------------------------------------------------------


def test_the_pool_refuses_a_corpus_that_is_averaging_two_quantities() -> None:
    estimands = _all_itt() | {"B-1": _estimand(_COMPLIERS)}
    result = pool(
        PoolSpec(family="f"), _corpus(), backend="laplace", draws=400, estimands=estimands
    )
    assert isinstance(result, Unsupported)
    assert "not pooling one quantity" in result.reason
    assert result.missing == ("commensurable estimands",)


def test_the_pool_runs_and_records_the_check_when_it_passes() -> None:
    result = pool(
        PoolSpec(family="f"), _corpus(), backend="laplace", draws=400, estimands=_all_itt()
    )
    assert isinstance(result, Pooled)
    assert "3 record(s) checked against 'A-1'" in result.result.detail["commensurability"]
    assert "identified" in result.result.detail["commensurability"]


def test_a_pool_without_estimands_says_so_rather_than_nothing() -> None:
    """Optional provenance is absent provenance; the detail records which it is."""
    result = pool(PoolSpec(family="f"), _corpus(), backend="laplace", draws=400)
    assert isinstance(result, Pooled)
    assert result.result.detail["commensurability"].startswith("unchecked:")
    assert "no estimands supplied" in result.result.detail["commensurability"]
