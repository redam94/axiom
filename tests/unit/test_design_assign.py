from __future__ import annotations

import numpy as np
import pytest

from axiom.design import (
    RERANDOMIZED,
    ArmAllocation,
    ArmAssignment,
    Assigned,
    arm_for,
    assign,
    bucket,
    standardized_differences,
)

_EQUAL = ArmAllocation.equal("control", "treated")
_TRI = ArmAllocation(arms=("control", "low", "high"), shares=(0.5, 0.25, 0.25))
_UNITS = tuple(f"u{i:05d}" for i in range(2000))


def _covariates(n: int = 2000, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {"x": rng.normal(size=n), "age": rng.uniform(20.0, 60.0, n)}


# -- the allocation -------------------------------------------------------------------------


def test_an_allocation_validates_its_shares() -> None:
    with pytest.raises(ValueError, match="at least two arms"):
        ArmAllocation(arms=("only",), shares=(1.0,))
    with pytest.raises(ValueError, match="they must match"):
        ArmAllocation(arms=("a", "b"), shares=(0.5, 0.25, 0.25))
    with pytest.raises(ValueError, match="distinct"):
        ArmAllocation(arms=("a", "a"), shares=(0.5, 0.5))
    with pytest.raises(ValueError, match="sum to 1"):
        ArmAllocation(arms=("a", "b"), shares=(0.5, 0.4))
    with pytest.raises(ValueError, match="finite and positive"):
        ArmAllocation(arms=("a", "b"), shares=(1.5, -0.5))


def test_equal_shares_and_the_reference_arm() -> None:
    assert _EQUAL.shares == (0.5, 0.5)
    assert _EQUAL.reference == "control" and _EQUAL.n_arms == 2
    assert _TRI.edges() == (0.5, 0.75, 1.0)
    assert _TRI.expected_counts(100) == (50.0, 25.0, 25.0)
    with pytest.raises(ValueError, match="non-negative"):
        _TRI.expected_counts(-1)


def test_the_block_size_is_the_smallest_whole_split() -> None:
    assert _EQUAL.block_size() == 2
    assert _TRI.block_size() == 4
    assert ArmAllocation(arms=("a", "b"), shares=(0.9, 0.1)).block_size() == 10
    with pytest.raises(ValueError, match="over the 200 limit"):
        ArmAllocation(arms=("a", "b"), shares=(0.997, 0.003)).block_size()


# -- the hash -------------------------------------------------------------------------------


def test_the_bucket_is_uniform_stable_and_salted() -> None:
    draws = np.asarray([bucket(u, salt="NW-14") for u in _UNITS])
    assert draws.min() >= 0.0 and draws.max() < 1.0
    assert abs(float(draws.mean()) - 0.5) < 0.02
    assert bucket("u00007", salt="NW-14") == bucket("u00007", salt="NW-14")
    assert bucket("u00007", salt="NW-14") != bucket("u00007", salt="other")
    assert bucket("u00007") != bucket("u00008")


def test_a_new_unit_lands_where_it_always_would_have() -> None:
    """The property the whole method exists for: no roster, no state, no ordering."""
    early = assign(_UNITS[:100], _EQUAL, salt="NW-14")
    late = assign(_UNITS, _EQUAL, salt="NW-14")
    for unit in _UNITS[:100]:
        assert early.arm(unit) == late.arm(unit) == arm_for(unit, _EQUAL, salt="NW-14")
    # a unit nobody had seen when the design was written
    assert arm_for("arrived-in-week-three", _EQUAL, salt="NW-14") in _EQUAL.arms


def test_the_salt_moves_units_between_arms() -> None:
    a = assign(_UNITS, _EQUAL, salt="one")
    b = assign(_UNITS, _EQUAL, salt="two")
    moved = np.count_nonzero(a.arm_of != b.arm_of)
    assert 0.4 * len(_UNITS) < moved < 0.6 * len(_UNITS)


def test_hash_shares_are_approximate_and_the_result_says_so() -> None:
    result = assign(_UNITS, _TRI, salt="s")
    assert set(result.counts()) == set(_TRI.arms)
    for arm, share in zip(_TRI.arms, _TRI.shares, strict=True):
        assert abs(result.spec.share_of(arm) - share) < 0.05
    assert "approximately" in result.spec.detail["realized_vs_target"]


# -- blocks ---------------------------------------------------------------------------------


def test_block_assignment_hits_the_target_split_exactly() -> None:
    result = assign(_UNITS, _TRI, method="block", seed=3)
    assert result.counts() == {"control": 1000, "low": 500, "high": 500}
    assert result.spec.block == 4


def test_blocks_are_exact_within_each_stratum() -> None:
    strata = ["north" if i % 2 else "south" for i in range(len(_UNITS))]
    result = assign(_UNITS, _EQUAL, method="block", seed=3, strata=strata)
    for stratum in ("north", "south"):
        members = [i for i, s in enumerate(strata) if s == stratum]
        treated = int(np.count_nonzero(result.arm_of[members] == 1))
        assert treated == len(members) // 2
    assert result.spec.n_strata == 2 and result.spec.strata_hash


def test_a_seedless_block_is_refused_and_a_seeded_hash_is_too() -> None:
    with pytest.raises(ValueError, match="needs a seed"):
        assign(_UNITS, _EQUAL, method="block")
    with pytest.raises(ValueError, match="takes no seed"):
        assign(_UNITS, _EQUAL, method="hash", seed=1)


# -- re-randomization -----------------------------------------------------------------------


def test_rerandomization_beats_the_threshold_and_records_the_cost() -> None:
    covariates = _covariates()
    plain = assign(_UNITS, _EQUAL, method="block", seed=5, covariates=covariates)
    tight = assign(
        _UNITS,
        _EQUAL,
        method="rerandomize",
        seed=5,
        covariates=covariates,
        threshold=0.01,
        max_draws=500,
    )
    assert tight.spec.balance_met
    assert tight.spec.worst_smd <= 0.01 < plain.spec.worst_smd or tight.spec.worst_smd <= 0.01
    assert tight.spec.draws_used > 1
    line = tight.spec.ledger_line()
    assert line.assumption == RERANDOMIZED
    assert line.assumption.state == "asserted"
    assert "conservative" in line.assumption.statement


def test_an_unreachable_threshold_keeps_the_best_draw_and_says_it_failed() -> None:
    result = assign(
        _UNITS[:40],
        _EQUAL,
        method="rerandomize",
        seed=5,
        covariates={"x": _covariates(40)["x"]},
        threshold=1e-9,
        max_draws=20,
    )
    assert not result.spec.balance_met
    assert result.spec.draws_used == 20
    assert "not met in 20 draws" in result.spec.ledger_line().statement
    assert "best draw was kept" in result.spec.ledger_line().statement


def test_rerandomization_needs_covariates_and_a_threshold() -> None:
    with pytest.raises(ValueError, match="needs covariates"):
        assign(_UNITS, _EQUAL, method="rerandomize", seed=1, threshold=0.1)
    with pytest.raises(ValueError, match="positive threshold"):
        assign(_UNITS, _EQUAL, method="rerandomize", seed=1, covariates=_covariates())


def test_only_rerandomization_may_report_an_unmet_threshold() -> None:
    good = assign(_UNITS, _EQUAL, salt="s").spec
    with pytest.raises(ValueError, match="only re-randomization"):
        good.model_copy(update={"balance_met": False}).model_validate(
            good.model_copy(update={"balance_met": False}).to_dict()
        )


# -- balance --------------------------------------------------------------------------------


def test_standardized_differences_report_arm_means_and_the_worst_gap() -> None:
    covariates = _covariates()
    result = assign(_UNITS, _EQUAL, salt="s", covariates=covariates)
    rows = {row.covariate: row for row in result.spec.balance}
    assert sorted(rows) == ["age", "x"]
    for row in rows.values():
        assert len(row.means) == 2
        assert row.smd == pytest.approx(abs(row.means[1] - row.means[0]) / row.sd)
    assert result.spec.worst_smd == max(r.smd for r in rows.values())


def test_a_constant_covariate_has_nothing_to_be_imbalanced_about() -> None:
    rows = standardized_differences(
        {"flat": np.ones(len(_UNITS))}, assign(_UNITS, _EQUAL, salt="s").arm_of, _EQUAL
    )
    assert rows[0].sd == 0.0 and rows[0].smd == 0.0


def test_a_covariate_of_the_wrong_length_is_refused() -> None:
    with pytest.raises(ValueError, match="values for"):
        standardized_differences({"x": np.zeros(3)}, np.zeros(5, dtype=np.int64), _EQUAL)
    with pytest.raises(ValueError, match="must be finite"):
        standardized_differences(
            {"x": np.array([1.0, np.nan])}, np.zeros(2, dtype=np.int64), _EQUAL
        )


# -- the audit ------------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["hash", "block", "rerandomize"])
def test_every_method_re_derives_to_the_same_arms(method: str) -> None:
    kwargs: dict[str, object] = {"method": method}
    if method != "hash":
        kwargs["seed"] = 11
    if method == "rerandomize":
        kwargs |= {"covariates": _covariates(), "threshold": 0.02, "max_draws": 200}
    result = assign(_UNITS, _EQUAL, **kwargs)  # type: ignore[arg-type]
    verdict = result.verify()
    assert verdict.status == "identified" and verdict.route == "assignment"


def test_a_tampered_assignment_is_blocked_and_names_the_unit() -> None:
    result = assign(_UNITS, _EQUAL, salt="s")
    moved = result.arm_of.copy()
    moved[7] = 1 - moved[7]
    tampered = Assigned(spec=result.spec, units=result.units, arm_of=moved)
    verdict = tampered.verify()
    assert verdict.status == "blocked"
    assert "u00007" in verdict.reason and "moved 1 of 2000" in verdict.reason


def test_the_roster_hash_distinguishes_rosters() -> None:
    a = assign(_UNITS, _EQUAL, salt="s")
    b = assign(_UNITS[:-1], _EQUAL, salt="s")
    assert a.spec.roster_hash != b.spec.roster_hash
    assert assign(_UNITS, _EQUAL, salt="s").spec.roster_hash == a.spec.roster_hash


def test_the_rule_round_trips_without_the_roster() -> None:
    spec = assign(_UNITS, _EQUAL, salt="s", covariates=_covariates()).spec
    assert ArmAssignment.from_json(spec.to_json()) == spec
    assert "u00000" not in spec.to_json()  # the roster is referenced, not carried
    assert spec.n_units == 2000 and len(spec.roster_hash) == 64


# -- reading the result ---------------------------------------------------------------------


def test_the_carrier_answers_the_questions_an_operator_asks() -> None:
    result = assign(_UNITS, _TRI, method="block", seed=1)
    assert len(result) == 2000
    assert sum(len(result.units_in(arm)) for arm in _TRI.arms) == 2000
    assert result.arm(result.units_in("high")[0]) == "high"
    assert "block" in repr(result) and "control=1000" in repr(result)
    with pytest.raises(KeyError, match="no unit"):
        result.arm("nobody")
    with pytest.raises(KeyError, match="no arm"):
        result.units_in("placebo")
    with pytest.raises(KeyError, match="no arm"):
        result.spec.share_of("placebo")


def test_duplicate_and_empty_rosters_are_refused() -> None:
    with pytest.raises(ValueError, match="at least one unit"):
        assign([], _EQUAL)
    with pytest.raises(ValueError, match="distinct"):
        assign(["a", "b", "a"], _EQUAL)
    with pytest.raises(ValueError, match="strata labels"):
        assign(["a", "b"], _EQUAL, method="block", seed=1, strata=["one"])
