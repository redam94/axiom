from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from axiom.design import (
    ARBITRARY_DEPENDENCE,
    ProgramReport,
    Readout,
    evalue,
    expected_false_go,
    from_p_value,
    program_decisions,
    tune,
)

_ALPHA = 0.05
_RHO = tune(_ALPHA, 1.0)


def _book(n_real: int = 8, effect: float = 4.5, seed: int = 0) -> list[Readout]:
    """Eleven parties, three treatments each, one primary and four guardrails."""
    rng = np.random.default_rng(seed)
    out: list[Readout] = []
    primaries = 0
    for party in range(11):
        for treatment in range(3):
            for metric in ("primary", "latency", "errors", "churn", "cost"):
                true = effect if (metric == "primary" and primaries < n_real) else 0.0
                z = float(rng.normal(true, 1.0))
                out.append(
                    Readout(
                        experiment=f"E{party}-{treatment}",
                        party=f"p{party}",
                        metric=metric,
                        p_value=float(2 * stats.norm.sf(abs(z))),
                        evalue=evalue(abs(z), 1.0, rho=_RHO),
                    )
                )
                if metric == "primary":
                    primaries += 1
    return out


# -- the arithmetic nobody does -------------------------------------------------------------


def test_the_expected_count_is_the_whole_point() -> None:
    assert expected_false_go(165, 0.05) == pytest.approx(8.25)
    assert expected_false_go(0, 0.05) == 0.0
    with pytest.raises(ValueError, match="non-negative"):
        expected_false_go(-1, 0.05)
    with pytest.raises(ValueError, match="alpha must be in"):
        expected_false_go(10, 1.0)


def test_the_report_carries_the_uncorrected_count_under_every_method() -> None:
    book = _book()
    for method in ("none", "holm", "benjamini_hochberg", "e_bh"):
        report = program_decisions(book, alpha=_ALPHA, method=method)  # type: ignore[arg-type]
        assert report.n_readouts == 165
        assert len(report.go_uncorrected) == 15
        assert report.expected_false_go_uncorrected == pytest.approx(8.25)


# -- the four routes ------------------------------------------------------------------------


def test_the_methods_trade_power_for_what_they_assume() -> None:
    """Eight treatments really work; the ladder is the whole decision."""
    book = _book(n_real=8)
    counts = {
        method: len(program_decisions(book, alpha=_ALPHA, method=method).go)  # type: ignore[arg-type]
        for method in ("none", "holm", "benjamini_hochberg", "e_bh")
    }
    assert counts["none"] == 15  # seven of them false
    assert counts["holm"] == 7
    assert counts["benjamini_hochberg"] == 8
    assert counts["e_bh"] == 5  # arbitrary dependence costs power, and it is worth naming


def test_every_corrected_go_here_is_a_real_one() -> None:
    book = _book(n_real=8)
    for method in ("holm", "benjamini_hochberg", "e_bh"):
        report = program_decisions(book, alpha=_ALPHA, method=method)  # type: ignore[arg-type]
        assert all(d.metric == "primary" for d in report.decisions if d.go)


def test_none_is_a_named_choice_and_says_what_it_controls() -> None:
    report = program_decisions(_book(), alpha=_ALPHA, method="none")
    assert report.method == "none"
    assert "nothing across the programme" in report.bound
    assert "8.25 go-decisions" in report.bound
    assert report.assumption is None
    assert set(report.go) == set(report.go_uncorrected)


def test_the_bound_says_which_dependence_it_needs() -> None:
    book = _book()
    holm = program_decisions(book, alpha=_ALPHA, method="holm")
    bh = program_decisions(book, alpha=_ALPHA, method="benjamini_hochberg")
    ebh = program_decisions(book, alpha=_ALPHA, method="e_bh")
    assert "under any dependence" in holm.bound and holm.assumption is None
    assert "positive regression dependence" in bh.bound
    assert bh.assumption is not None and bh.assumption.name == "positive_dependence"
    assert "arbitrary dependence" in ebh.bound
    assert ebh.assumption == ARBITRARY_DEPENDENCE
    assert ARBITRARY_DEPENDENCE.state == "satisfied"


def test_e_bh_thresholds_are_the_same_for_every_readout() -> None:
    """The e-BH cut is one number, ``n / (alpha·k*)``, and it is recorded per decision."""
    report = program_decisions(_book(), alpha=_ALPHA, method="e_bh")
    thresholds = {d.threshold for d in report.decisions}
    assert len(thresholds) == 1
    threshold = thresholds.pop()
    assert all(d.go == (d.evidence >= threshold) for d in report.decisions)


def test_e_bh_rejects_nothing_when_no_k_qualifies() -> None:
    weak = [Readout(experiment=f"E{i}", evalue=1.0, p_value=0.5) for i in range(20)]
    report = program_decisions(weak, alpha=_ALPHA, method="e_bh")
    assert report.go == ()
    assert all(d.threshold == float("inf") for d in report.decisions)


# -- the readouts ---------------------------------------------------------------------------


def test_a_readout_needs_something_to_decide_on() -> None:
    with pytest.raises(ValueError, match="neither a p-value nor an e-value"):
        Readout(experiment="E1")
    assert Readout(experiment="E1", p_value=0.01).key == "E1:primary"
    assert Readout(experiment="E1", party="acme", metric="churn", evalue=3.0).key == (
        "acme/E1:churn"
    )


def test_a_method_that_needs_what_a_readout_lacks_says_so() -> None:
    no_evalue = [Readout(experiment="E1", p_value=0.01), Readout(experiment="E2", p_value=0.2)]
    with pytest.raises(ValueError, match="e_bh needs an e-value"):
        program_decisions(no_evalue, method="e_bh")
    no_p = [Readout(experiment="E1", evalue=30.0), Readout(experiment="E2", evalue=1.0)]
    with pytest.raises(ValueError, match="needs a p-value"):
        program_decisions(no_p, method="holm")
    # dropping a decision is not a way of deciding it
    assert len(program_decisions(no_evalue, method="holm").decisions) == 2


def test_a_programme_needs_readouts_and_a_level() -> None:
    with pytest.raises(ValueError, match="at least one readout"):
        program_decisions([])
    with pytest.raises(ValueError, match="alpha must be in"):
        program_decisions([Readout(experiment="E1", p_value=0.1)], alpha=0.0)


def test_readout_keys_must_be_distinct() -> None:
    twice = [Readout(experiment="E1", p_value=0.01), Readout(experiment="E1", p_value=0.02)]
    with pytest.raises(ValueError, match="keys must be distinct"):
        program_decisions(twice, method="holm")


# -- the calibrator -------------------------------------------------------------------------


def test_the_calibrator_is_an_evalue_and_one_over_p_is_not() -> None:
    """E[p^(-1/2) - 1] = 1 under the null; E[1/p] diverges."""
    rng = np.random.default_rng(0)
    uniform = rng.random(200_000)
    calibrated = np.asarray([from_p_value(float(p)) for p in uniform])
    assert calibrated.mean() == pytest.approx(1.0, abs=0.05)
    assert from_p_value(1.0) == 0.0
    assert from_p_value(0.01) == pytest.approx(9.0)


def test_the_calibrator_is_not_the_same_object_as_a_mixture_evalue() -> None:
    """Not bigger or smaller — different. It crosses over, which is the tell."""

    def pair(z: float) -> tuple[float, float]:
        return from_p_value(float(2 * stats.norm.sf(z))), evalue(z, 1.0, rho=_RHO)

    weak_c, weak_n = pair(2.0)
    strong_c, strong_n = pair(6.0)
    assert weak_c > weak_n  # the mixture is paying for anytime validity
    assert weak_c / weak_n == pytest.approx(1.88, abs=0.05)
    assert strong_c < strong_n  # and the calibrator's tail is only polynomial
    assert strong_c / strong_n < 0.01


def test_the_calibrator_refuses_what_is_not_a_p_value() -> None:
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError, match=r"p_value must be finite and in \(0, 1\]"):
            from_p_value(bad)


# -- the readout -----------------------------------------------------------------------------


def test_the_report_summarizes_and_ledgers_and_round_trips() -> None:
    report = program_decisions(_book(), alpha=_ALPHA, method="e_bh", period="2026-Q3")
    assert "2026-Q3: 165 readouts" in report.summary()
    assert "a null book expects 8.25" in report.summary()
    line = report.ledger_line()
    assert line.kind == "program_error_control"
    assert line.detail["expected_false_go_uncorrected"] == "8.25"
    assert line.assumption == ARBITRARY_DEPENDENCE
    assert ProgramReport.from_json(report.to_json()) == report
