from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from axiom.design import CONCURRENT_EXPERIMENTS, Factorial, FactorialCell, factorial

_LEFT, _RIGHT = 2.0, 1.0


def _world(gamma: float = 0.0, *, n: int = 4000, noise: float = 2.0, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    price = (rng.random(n) < 0.5).astype(float)
    banner = (rng.random(n) < 0.5).astype(float)
    y = 10.0 + _LEFT * price + _RIGHT * banner + gamma * price * banner + rng.normal(0.0, noise, n)
    return pd.DataFrame({"y": y, "price": price, "banner": banner})


def test_the_four_cells_are_the_readable_output() -> None:
    result = factorial(_world(), "y", "price", "banner")
    assert isinstance(result, Factorial)
    assert len(result.cells) == 4 and result.n == 4000
    assert {c.label for c in result.cells} == {"00", "01", "10", "11"}
    base = result.cell(0, 0).mean
    assert result.cell(1, 0).mean - base == pytest.approx(_LEFT, abs=0.3)
    assert result.cell(0, 1).mean - base == pytest.approx(_RIGHT, abs=0.3)


def test_no_interaction_leaves_the_assumption_unverified_not_satisfied() -> None:
    """Failing to refute is not confirming, in a study powered for main effects."""
    result = factorial(_world(gamma=0.0), "y", "price", "banner")
    assert not result.interacts
    assert result.verdict().status == "unverified"
    assert result.assumption().state == "unverified"
    assert result.interaction.estimate == pytest.approx(0.0, abs=0.15)
    assert "could only have detected" in result.power_note
    assert result.power_note in result.verdict().reason


def test_a_real_interaction_blocks_and_says_what_it_does_to_the_mains() -> None:
    result = factorial(_world(gamma=1.5), "y", "price", "banner")
    assert result.interacts
    assert result.interaction.estimate == pytest.approx(1.5, abs=0.25)
    assert result.assumption().state == "violated"
    verdict = result.verdict()
    assert verdict.status == "blocked"
    assert "neither main effect is the effect of its own treatment" in verdict.reason
    assert "changes when the other experiment ends" in verdict.reason


def test_the_interaction_is_reported_against_the_larger_main_effect() -> None:
    small = factorial(_world(gamma=0.2), "y", "price", "banner")
    large = factorial(_world(gamma=1.5), "y", "price", "banner")
    assert small.relative < 0.2 < large.relative
    assert large.relative == pytest.approx(
        abs(large.interaction.estimate) / abs(large.main_left.estimate), rel=1e-6
    )


def test_it_is_the_assumption_collision_already_named() -> None:
    result = factorial(_world(gamma=1.5), "y", "price", "banner")
    assert result.assumption().name == CONCURRENT_EXPERIMENTS.name
    assert "factorial" in CONCURRENT_EXPERIMENTS.challenged_by
    assert result.assumption().detail["relative_to_main"]
    assert result.ledger_line().kind == "factorial"
    assert "they interact" in result.ledger_line().statement


def test_the_interaction_is_harder_to_detect_than_a_main_effect() -> None:
    """Roughly 2x the standard error, which is 4x the units for the same power."""
    result = factorial(_world(), "y", "price", "banner")
    ratio = result.interaction.se / max(result.main_left.se, result.main_right.se)
    assert 1.2 < ratio < 2.5


def test_covariates_reach_every_fit() -> None:
    frame = _world()
    rng = np.random.default_rng(1)
    frame = frame.assign(age=rng.normal(size=len(frame)))
    result = factorial(frame, "y", "price", "banner", covariates=("age",))
    assert "age" in result.main_left.covariates
    assert "age" in result.interaction.covariates


def test_an_empty_cell_is_refused_rather_than_reported() -> None:
    frame = _world()
    never_both = frame[~((frame["price"] == 1.0) & (frame["banner"] == 1.0))]
    with pytest.raises(ValueError, match="not separable from the main effects"):
        factorial(never_both, "y", "price", "banner")


def test_it_refuses_what_it_cannot_read() -> None:
    frame = _world()
    with pytest.raises(ValueError, match="three different columns"):
        factorial(frame, "y", "price", "price")
    with pytest.raises(ValueError, match="must be 0/1"):
        factorial(frame, "price", "y", "banner")
    with pytest.raises(KeyError, match="not in frame"):
        factorial(frame, "nope", "price", "banner")
    with pytest.raises(ValueError, match="alpha must be in"):
        factorial(frame, "y", "price", "banner", alpha=0.0)
    with pytest.raises(ValueError, match="already has a column named"):
        factorial(frame.assign(_price_x_banner=1.0), "y", "price", "banner")


def test_a_two_by_two_needs_all_four_cells() -> None:
    result = factorial(_world(), "y", "price", "banner")
    three = result.cells[:3] + (result.cells[0],)
    with pytest.raises(ValueError, match="needs all four cells"):
        Factorial.model_validate(result.model_copy(update={"cells": three}).to_dict())
    with pytest.raises(ValueError, match="finite mean and se"):
        FactorialCell(left=0, right=0, n=5, mean=float("nan"), se=1.0)


def test_the_result_round_trips() -> None:
    result = factorial(_world(gamma=1.5), "y", "price", "banner")
    assert Factorial.from_json(result.to_json()) == result
    assert "interaction" in result.summary() and "blocked" in result.summary()
