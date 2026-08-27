"""Error control across a book of experiments, not within one.

Every other module here controls the error rate of *an* experiment.
`design.sequential` spends alpha across the looks of one study;
`design.anytime` makes one study's interval valid at any time;
`diagnose.structure` corrects across the implications of one graph. Nothing
has ever looked at the quarter.

A house running experiments for eleven parties, three treatments each, with four
guardrail metrics per readout, takes on the order of a hundred and fifty
go/no-go decisions in a period. At the conventional 5 %, with nothing working at
all, **seven and a half of them are go** (note 0027, §D27.6.5). Every one of
those numbers is individually defensible and individually documented, and the
programme is wrong seven times a quarter.

The four routes, and when each is the right one:

| method | controls | needs |
|---|---|---|
| ``none`` | nothing; the count above is what you get | — |
| ``holm`` | family-wise error: the chance of **any** false go | any dependence |
| ``benjamini_hochberg`` | false discovery rate: the expected share of | independence or |
| | the goes that are false | positive dependence |
| ``e_bh`` | the same false discovery rate | **arbitrary** dependence |

The last row is why `design.anytime` exposes its e-value. A book of experiments
that share markets, seasons and a macroeconomy is dependent in ways nobody can
characterise, which is exactly the condition Benjamini-Hochberg on p-values does
not cover and e-BH (Wang and Ramdas 2022) does. Where a readout was monitored
with a confidence sequence its e-value is already there; where it was not, one
can be built from a p-value at some cost, and this module does not pretend
otherwise (see :func:`from_p_value`).

**What this is not.** Correcting a programme is a *decision* about what the
programme is willing to be wrong about, not a statistical fact. Nothing here
picks the method. What it does is make the uncorrected count visible beside the
corrected one, so that choosing `none` is a choice somebody made rather than a
default nobody noticed.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

from pydantic import Field, model_validator

from axiom.core import (
    Assumption,
    LedgerLine,
    Multiplicity,
    NonEmptyStr,
    Spec,
    adjust,
)

__all__ = [
    "ARBITRARY_DEPENDENCE",
    "Decision",
    "ProgramMethod",
    "ProgramReport",
    "Readout",
    "expected_false_go",
    "from_p_value",
    "program_decisions",
]

ProgramMethod = Literal["none", "holm", "benjamini_hochberg", "e_bh"]

ARBITRARY_DEPENDENCE = Assumption(
    name="arbitrary_dependence",
    facet="quantity",
    statement=(
        "the readouts in this programme may be dependent in any way at all — shared markets, "
        "a shared season, a shared macroeconomy — and the error control does not assume "
        "otherwise"
    ),
    challenged_by=(
        "nothing: this is the weakest of the dependence conditions and is the one e-BH "
        "requires. A programme that uses benjamini_hochberg on p-values instead is asserting "
        "positive regression dependence, which a book of experiments does not obviously have"
    ),
    state="satisfied",
)


def expected_false_go(n_readouts: int, alpha: float) -> float:
    """``n · alpha``: how many uncorrected go-decisions a null programme takes.

    The arithmetic nobody does. It is not a bound and not a worst case — it is
    the *expected* count when every treatment in the book does nothing, at the
    level each readout was individually judged at.
    """
    if n_readouts < 0:
        raise ValueError(f"n_readouts must be non-negative, got {n_readouts}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    return n_readouts * alpha


def from_p_value(p_value: float) -> float:
    """A calibrated e-value from a p-value, for a readout that has no e-value of its own.

    ``e = p^(-1/2) − 1``. Under the null ``p`` is uniform, so ``E[p^(-1/2)] = 2``
    and the expectation of ``e`` is exactly 1 — which is what makes it an
    e-value. The obvious ``1/p`` is **not** one: its expectation diverges.

    **It is not the same object as ``design.anytime.evalue`` and the difference
    is not a matter of size.** A calibrated e-value inherits the validity of the
    p-value it came from: one look, fixed sample. The mixture e-value is valid
    at every look. Comparing their magnitudes is therefore the wrong comparison,
    and doing it anyway is instructive — against a two-sided z the calibrator is
    *larger* for weak evidence (1.9x at ``z = 2``, because the mixture is paying
    for anytime validity) and collapses for strong evidence (0.30x at ``z = 4``,
    0.007x at ``z = 6``, because its tail is polynomial where the mixture's is
    essentially Gaussian).

    Converting is a fallback. A programme mixing calibrated and native e-values
    is running e-BH over objects with different guarantees, which is valid — an
    e-value is an e-value — and means the resulting decisions are only as
    anytime-valid as their weakest input.
    """
    if not (math.isfinite(p_value) and 0.0 < p_value <= 1.0):
        raise ValueError(f"p_value must be finite and in (0, 1], got {p_value}")
    return float(p_value**-0.5) - 1.0


class Readout(Spec):
    """One go/no-go decision the programme is about to take.

    A readout carries a ``p_value``, an ``evalue``, or both. ``metric`` is
    ``"primary"`` by default and names a guardrail otherwise — guardrails are
    part of the count, because a programme that checks four of them per
    experiment is taking five decisions, not one.
    """

    experiment: NonEmptyStr
    party: str = ""
    metric: NonEmptyStr = "primary"
    p_value: float | None = Field(default=None, ge=0.0, le=1.0)
    evalue: float | None = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def _has_evidence(self) -> Readout:
        if self.p_value is None and self.evalue is None:
            raise ValueError(
                f"readout {self.key!r} carries neither a p-value nor an e-value; "
                "there is nothing to decide on"
            )
        for name in ("p_value", "evalue"):
            value = getattr(self, name)
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value}")
        return self

    @property
    def key(self) -> str:
        """``experiment:metric``, prefixed by the party when there is one."""
        stem = f"{self.experiment}:{self.metric}"
        return f"{self.party}/{stem}" if self.party else stem


class Decision(Spec):
    """What the programme decided about one readout, and what it decided on.

    ``evidence`` is the adjusted p-value under the p-value methods and the
    e-value under ``e_bh``; ``threshold`` is what it was compared against. The
    two together are the whole decision, and they are on different scales by
    method, which is why both are recorded rather than only the verdict.
    """

    key: NonEmptyStr
    experiment: NonEmptyStr
    party: str = ""
    metric: NonEmptyStr = "primary"
    evidence: float
    threshold: float
    go: bool
    go_uncorrected: bool


class ProgramReport(Spec):
    """A period's decisions under one error-control method, beside what it cost.

    ``expected_false_go_uncorrected`` is ``n · alpha`` — what the programme
    would expect to get wrong with no correction at all — and is reported
    whatever method was used, because it is the number that makes the choice of
    method a choice.
    """

    period: str = ""
    method: ProgramMethod
    alpha: float = Field(gt=0, lt=1)
    decisions: tuple[Decision, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _distinct(self) -> ProgramReport:
        keys = [d.key for d in self.decisions]
        if len(set(keys)) != len(keys):
            repeated = sorted({k for k in keys if keys.count(k) > 1})
            raise ValueError(f"readout keys must be distinct; repeated: {repeated}")
        return self

    @property
    def n_readouts(self) -> int:
        return len(self.decisions)

    @property
    def go(self) -> tuple[str, ...]:
        return tuple(d.key for d in self.decisions if d.go)

    @property
    def go_uncorrected(self) -> tuple[str, ...]:
        return tuple(d.key for d in self.decisions if d.go_uncorrected)

    @property
    def expected_false_go_uncorrected(self) -> float:
        """``n · alpha``: the goes a null programme takes with no correction."""
        return expected_false_go(self.n_readouts, self.alpha)

    @property
    def bound(self) -> str:
        """What the chosen method actually promises, in words."""
        if self.method == "none":
            return (
                f"nothing across the programme; each readout is judged at {self.alpha:g} "
                f"on its own, so a null book expects "
                f"{self.expected_false_go_uncorrected:.3g} go-decisions"
            )
        if self.method == "holm":
            return (
                f"family-wise: the chance of *any* false go across all {self.n_readouts} "
                f"readouts is at most {self.alpha:g}, under any dependence"
            )
        if self.method == "benjamini_hochberg":
            return (
                f"false discovery rate: at most {self.alpha:g} of the {len(self.go)} "
                "go-decisions are expected to be false, assuming independence or positive "
                "regression dependence"
            )
        return (
            f"false discovery rate: at most {self.alpha:g} of the {len(self.go)} go-decisions "
            "are expected to be false, under arbitrary dependence"
        )

    @property
    def assumption(self) -> Assumption | None:
        """``ARBITRARY_DEPENDENCE`` under ``e_bh``; the weaker claim under the others."""
        if self.method == "e_bh":
            return ARBITRARY_DEPENDENCE
        if self.method == "benjamini_hochberg":
            return Assumption(
                name="positive_dependence",
                facet="quantity",
                statement=(
                    "the readouts are independent or positively regression dependent, which "
                    "is what Benjamini-Hochberg on p-values requires"
                ),
                challenged_by=(
                    "a book of experiments sharing markets, seasons and a macroeconomy; "
                    "e_bh needs no such condition"
                ),
            )
        return None

    def ledger_line(self) -> LedgerLine:
        return LedgerLine(
            kind="program_error_control",
            statement=(
                f"{self.n_readouts} readout(s) at alpha {self.alpha:g} under {self.method!r}: "
                f"{len(self.go)} go against {len(self.go_uncorrected)} uncorrected; "
                f"{self.bound}"
            ),
            assumption=self.assumption,
            detail={
                "period": self.period,
                "method": self.method,
                "alpha": f"{self.alpha:g}",
                "n": str(self.n_readouts),
                "go": str(len(self.go)),
                "go_uncorrected": str(len(self.go_uncorrected)),
                "expected_false_go_uncorrected": f"{self.expected_false_go_uncorrected:.6g}",
            },
        )

    def summary(self) -> str:
        lines = [
            f"{self.period or 'programme'}: {self.n_readouts} readouts at alpha {self.alpha:g}",
            f"  method            {self.method}",
            f"  go                {len(self.go)}",
            f"  go uncorrected    {len(self.go_uncorrected)}"
            f"  (a null book expects {self.expected_false_go_uncorrected:.3g})",
            f"  what that buys    {self.bound}",
        ]
        return "\n".join(lines)


def _e_bh(evalues: Sequence[float], alpha: float) -> tuple[float, ...]:
    """The e-BH threshold each e-value was compared against, in input order.

    Sort descending, find the largest ``k`` with ``e_(k) >= n / (alpha·k)``, and
    reject the top ``k``. Reporting the threshold per readout rather than a bare
    boolean keeps the decision inspectable: it is ``n / (alpha·k*)`` for every
    readout, and ``inf`` when no ``k`` qualifies.
    """
    n = len(evalues)
    order = sorted(range(n), key=lambda i: -evalues[i])
    best = 0
    for k in range(1, n + 1):
        if evalues[order[k - 1]] >= n / (alpha * k):
            best = k
    threshold = math.inf if best == 0 else n / (alpha * best)
    return tuple([threshold] * n)


def program_decisions(
    readouts: Sequence[Readout],
    *,
    alpha: float = 0.05,
    method: ProgramMethod = "e_bh",
    period: str = "",
) -> ProgramReport:
    """Decide a whole period's readouts together, and record what that controls.

    ``e_bh`` (the default) needs an ``evalue`` on every readout; the p-value
    methods need a ``p_value`` on every readout. A readout missing what the
    method needs is a ``ValueError`` naming it rather than a silent no-go —
    dropping a decision is not a way of deciding it.

    ``go_uncorrected`` is recorded on every decision under every method, so the
    report can always say what the programme would have done with no correction
    at all.
    """
    if not readouts:
        raise ValueError("a programme needs at least one readout")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    if method == "e_bh":
        missing = [r.key for r in readouts if r.evalue is None]
        if missing:
            raise ValueError(
                f"e_bh needs an e-value on every readout; missing on {missing[:5]}. "
                "Monitor with design.anytime, or calibrate with design.from_p_value "
                "and accept the loss"
            )
        evidence = [float(r.evalue) for r in readouts if r.evalue is not None]
        thresholds = _e_bh(evidence, alpha)
        go = [e >= t for e, t in zip(evidence, thresholds, strict=True)]
    else:
        missing = [r.key for r in readouts if r.p_value is None]
        if missing:
            raise ValueError(
                f"{method!r} needs a p-value on every readout; missing on {missing[:5]}"
            )
        raw = [float(r.p_value) for r in readouts if r.p_value is not None]
        adjusted: Multiplicity = method
        evidence = list(adjust(raw, adjusted))
        thresholds = tuple([alpha] * len(readouts))
        go = [e < t for e, t in zip(evidence, thresholds, strict=True)]

    decisions = []
    for readout, value, threshold, decided in zip(readouts, evidence, thresholds, go, strict=True):
        if readout.p_value is not None:
            uncorrected = readout.p_value < alpha
        else:
            uncorrected = float(readout.evalue or 0.0) >= 1.0 / alpha
        decisions.append(
            Decision(
                key=readout.key,
                experiment=readout.experiment,
                party=readout.party,
                metric=readout.metric,
                evidence=value,
                threshold=threshold,
                go=decided,
                go_uncorrected=uncorrected,
            )
        )
    return ProgramReport(period=period, method=method, alpha=alpha, decisions=tuple(decisions))
