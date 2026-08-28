from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from axiom.core import D, Intervention, Outcome, Population, TimeWindow, Treatment
from axiom.estimands import Estimand, Level, Quantity
from axiom.io import Catalog, DefinitionRegistry, Program
from axiom.meta import (
    Corpus,
    Pooled,
    PoolPriors,
    PoolSpec,
    StudyRecord,
    ThreeLevel,
    commensurable,
    pool,
    prior_from_pool,
    random_effects,
    three_level,
    variance_shares,
)

MU, TAU_PARTY, TAU_STUDY = 0.5, 0.4, 0.15


def _nested_data(
    *, n_parties: int = 4, n_studies: int = 4, seed: int = 0
) -> tuple[list[float], list[float], list[int], list[int]]:
    rng = np.random.default_rng(seed)
    y, se, study, party = [], [], [], []
    sid = 0
    for p in range(n_parties):
        alpha = MU + TAU_PARTY * rng.standard_normal()
        for _ in range(n_studies):
            theta = alpha + TAU_STUDY * rng.standard_normal()
            s = float(rng.uniform(0.1, 0.2))
            y.append(float(theta + s * rng.standard_normal()))
            se.append(s)
            study.append(sid)
            party.append(p)
            sid += 1
    return y, se, study, party


# -- the sampler-free three-level fit -------------------------------------------------------


def test_it_recovers_both_variance_components_without_a_sampler() -> None:
    fit = three_level(*_nested_data())
    assert isinstance(fit, ThreeLevel)
    assert fit.estimate == pytest.approx(MU, abs=0.2)
    assert fit.tau_study == pytest.approx(TAU_STUDY, abs=0.1)
    assert fit.tau_party == pytest.approx(TAU_PARTY, abs=0.15)
    assert fit.k == 16 and fit.n_parties == 4
    assert fit.converged


def test_the_family_mean_is_wider_than_a_two_level_pool_says() -> None:
    """The headline of note 0029, reproduced with nothing installed but the core four."""
    y, se, study, party = _nested_data()
    flat = random_effects(y, se, tau_method="reml")
    nested = three_level(y, se, study, party)
    assert nested.interval.width > 1.8 * flat.interval.width
    assert nested.se > flat.se
    assert nested.tau_study < flat.tau  # the two-level tau conflates the levels


def test_it_agrees_with_what_the_sampler_gets() -> None:
    """Same data, same answer as PoolSpec(effect_key='nested') with tau_party fixed."""
    y, se, study, party = _nested_data(n_parties=3)
    fit = three_level(y, se, study, party)
    records = tuple(
        StudyRecord(
            study=f"s{i}",
            contributor=f"p{party[i]}",
            quantity="elasticity",
            family="f",
            estimate=y[i],
            se=se[i],
            read="experiment",
        )
        for i in range(len(y))
    )
    bayes = pool(
        PoolSpec(
            family="f",
            effect_key="nested",
            priors=PoolPriors(mu_scale=5.0, tau_party_fixed=fit.tau_party),
        ),
        Corpus(records=records, name="c"),
        backend="laplace",
        draws=4000,
        seed=1,
    )
    assert isinstance(bayes, Pooled)
    assert bayes.result.mu.mean == pytest.approx(fit.estimate, abs=0.15)
    assert bayes.result.tau.mean == pytest.approx(fit.tau_study, abs=0.1)


def test_the_variance_shares_apportion_the_spread() -> None:
    y, se, study, party = _nested_data()
    fit = three_level(y, se, study, party)
    shares = fit.shares(float(np.mean(np.asarray(se) ** 2)))
    assert sum(shares) == pytest.approx(1.0)
    assert shares[2] > shares[1]  # most of the spread is between parties here
    assert variance_shares(1.0, 1.0, 2.0) == pytest.approx((0.25, 0.25, 0.5))
    with pytest.raises(ValueError, match="total variance is zero"):
        variance_shares(0.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="non-negative"):
        variance_shares(-1.0, 0.0, 0.0)


def test_it_refuses_a_design_that_cannot_separate_the_levels() -> None:
    y, se, study, party = _nested_data(n_parties=1)
    with pytest.raises(ValueError, match="at least two parties"):
        three_level(y, se, study, party)
    y, se, study, party = _nested_data(n_parties=3, n_studies=1)
    with pytest.raises(ValueError, match="at least two studies"):
        three_level(y, se, study, party)
    with pytest.raises(ValueError, match="same length"):
        three_level([1.0, 2.0], [0.1], [0, 1], [0, 1])
    with pytest.raises(ValueError, match="standard errors must be positive"):
        three_level([1.0, 2.0, 3.0, 4.0], [0.1, 0.0, 0.1, 0.1], [0, 1, 2, 3], [0, 0, 1, 1])


def test_the_fit_round_trips_and_ledgers() -> None:
    fit = three_level(*_nested_data())
    assert ThreeLevel.from_json(fit.to_json()) == fit
    line = fit.ledger_line()
    assert line.kind == "three_level_pool"
    assert "tau within a party" in line.statement
    assert "16 records over 4 parties" in fit.summary()


# -- the prior handoff that knows which party it is for -------------------------------------


def _nested_pool() -> Pooled:
    y, se, study, party = _nested_data(n_parties=3)
    records = tuple(
        StudyRecord(
            study=f"s{i}",
            contributor=f"p{party[i]}",
            quantity="elasticity",
            family="f",
            estimate=y[i],
            se=se[i],
            read="experiment",
        )
        for i in range(len(y))
    )
    result = pool(
        PoolSpec(
            family="f",
            effect_key="nested",
            priors=PoolPriors(mu_scale=5.0, tau_party_fixed=TAU_PARTY),
        ),
        Corpus(records=records, name="c"),
        backend="laplace",
        draws=4000,
        seed=1,
    )
    assert isinstance(result, Pooled)
    return result


def test_a_new_party_gets_a_wider_prior_than_an_old_one() -> None:
    """The whole reason the party level was worth fitting."""
    result = _nested_pool().result
    fresh, _ = prior_from_pool(result, target="new_party")
    known, _ = prior_from_pool(result, target="same_party", party="p0")
    assert fresh.hyper["sigma"] > 2.0 * known.hyper["sigma"]
    assert known.hyper["mu"] == pytest.approx(
        next(a.mean for a in result.alphas if a.name.endswith("[p0]"))
    )
    assert fresh.hyper["mu"] == pytest.approx(result.mu.mean)


def test_the_party_targets_are_refused_by_a_two_level_pool() -> None:
    """A prior that says 'for this party' from a pool that never knew about parties."""
    y, se, study, party = _nested_data(n_parties=3)
    records = tuple(
        StudyRecord(
            study=f"s{i}",
            contributor=f"p{party[i]}",
            quantity="elasticity",
            family="f",
            estimate=y[i],
            se=se[i],
            read="experiment",
        )
        for i in range(len(y))
    )
    flat = pool(
        PoolSpec(family="f", effect_key="study"),
        Corpus(records=records, name="c"),
        backend="laplace",
        draws=2000,
        seed=1,
    )
    assert isinstance(flat, Pooled)
    for target in ("new_party", "same_party"):
        with pytest.raises(ValueError, match="needs a pool with a party level"):
            prior_from_pool(flat.result, target=target, party="p0")  # type: ignore[arg-type]


def test_same_party_needs_a_party_and_a_known_one() -> None:
    result = _nested_pool().result
    with pytest.raises(ValueError, match="needs the party it is for"):
        prior_from_pool(result, target="same_party")
    with pytest.raises(KeyError, match="no party 'nobody'"):
        prior_from_pool(result, target="same_party", party="nobody")
    with pytest.raises(ValueError, match="target must be"):
        prior_from_pool(result, target="sideways")  # type: ignore[arg-type]


def test_the_ledger_records_which_party_and_which_level() -> None:
    result = _nested_pool().result
    _, line = prior_from_pool(result, target="same_party", party="p1")
    assert line.detail["party"] == "p1"
    assert line.detail["target"] == "same_party"
    assert "n_parties" in line.detail
    assert "same_party" in line.statement


# -- the corpus that resolves its own definitions -------------------------------------------


def _estimand(population: Population) -> Estimand:
    return Estimand(
        name="lift",
        quantity=Quantity(kind="contrast"),
        treatment=Treatment(name="course", dimension=D.currency, unit="USD"),
        intervention=Intervention(doses={"course": 1.0}),
        reference=Intervention(doses={"course": 0.0}),
        outcome=Outcome(name="score", dimension=D.outcome, unit="pt", aggregation="mean"),
        population=population,
        window=TimeWindow(start=0, stop=8),
        level=Level(unit="individual"),
        dimension=D.outcome,
    )


def _definition_world() -> tuple[Corpus, dict, DefinitionRegistry, dict[str, Program]]:
    catalog = Catalog(Path(tempfile.mkdtemp()) / "cat")
    programs = {p: Program(party=p, program="growth") for p in ("acme", "northwind", "globex")}
    for program in programs.values():
        catalog.register(program)
    registry = DefinitionRegistry(catalog)
    records = tuple(
        StudyRecord(
            study=f"{p}-1",
            contributor=p,
            quantity="elasticity",
            family="f",
            estimate=0.5,
            se=0.1,
            read="experiment",
        )
        for p in programs
    )
    corpus = Corpus(records=records, name="book")
    estimands = {r.study: _estimand(Population(name="enrolled")) for r in records}
    return corpus, estimands, registry, programs


def test_parties_that_define_the_term_the_same_way_pool() -> None:
    corpus, estimands, registry, programs = _definition_world()
    paid = Outcome(name="conversion", dimension=D.outcome, unit="count", aggregation="sum")
    for program in programs.values():
        registry.register(program, "conversion", paid)
    result = commensurable(
        corpus, estimands, definitions=registry, programs=programs, term="conversion"
    )
    assert result.poolable and result.verdict().status == "identified"


def test_a_party_that_redefined_the_term_blocks_the_pool() -> None:
    """The wiring 0036 left open: the estimands agree and the definitions do not."""
    corpus, estimands, registry, programs = _definition_world()
    paid = Outcome(name="conversion", dimension=D.outcome, unit="count", aggregation="sum")
    rate = Outcome(name="conversion", dimension=D.outcome, unit="count", aggregation="mean")
    for name, program in programs.items():
        registry.register(program, "conversion", rate if name == "globex" else paid)
    without = commensurable(corpus, estimands)
    assert without.poolable  # the estimands are identical

    result = commensurable(
        corpus, estimands, definitions=registry, programs=programs, term="conversion"
    )
    assert not result.poolable
    (entry,) = result.entries
    assert entry.study == "globex-1" and entry.facet == "outcome"
    assert "are not one quantity" in entry.reason
    assert result.verdict().status == "blocked"


def test_the_three_definition_arguments_come_together() -> None:
    corpus, estimands, registry, programs = _definition_world()
    with pytest.raises(ValueError, match="supplied together or not at all"):
        commensurable(corpus, estimands, definitions=registry)
    with pytest.raises(ValueError, match="supplied together or not at all"):
        commensurable(corpus, estimands, term="conversion")


def test_a_party_with_no_definition_or_no_program_is_an_error() -> None:
    corpus, estimands, registry, programs = _definition_world()
    paid = Outcome(name="conversion", dimension=D.outcome, unit="count", aggregation="sum")
    registry.register(programs["acme"], "conversion", paid)
    with pytest.raises(KeyError, match="has no definition of"):
        commensurable(corpus, estimands, definitions=registry, programs=programs, term="conversion")
    with pytest.raises(KeyError, match="no Program supplied for contributor"):
        commensurable(
            corpus,
            estimands,
            definitions=registry,
            programs={"acme": programs["acme"]},
            term="conversion",
        )
