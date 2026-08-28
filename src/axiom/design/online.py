"""Error control for a book whose experiments finish one at a time.

``design.program`` decides a period's readouts **together**: it sorts them,
finds a threshold, and returns. That is the right shape for a quarterly review
and the wrong shape for what actually happens, which is that experiments finish
on Tuesdays and somebody wants an answer on Tuesday. Waiting until March to
learn whether January's test worked is not a correction anybody will accept, and
deciding each one at 5 % as it lands is the failure ``design.program`` exists to
name (note 0035, §D35.7).

**LORD** (Javanmard and Montanari; Ramdas, Zrnic, Wainwright and Jordan 2017)
decides each readout when it arrives and still controls the false discovery
rate over the whole stream. It works by giving each test a level of its own out
of a budget that starts at ``w0`` and is **replenished by ``alpha`` every time
something is rejected**:

    alpha_t = gamma_t·w0 + (alpha − w0)·gamma_{t−tau_1} + alpha·Σ_{j≥2} gamma_{t−tau_j}

where ``tau_j`` are the indices of past rejections and ``gamma`` is any
non-negative sequence summing to at most one. Nothing here needs to know how
long the stream is.

The behaviour that falls out is the one worth understanding before using it: a
run of null results makes the levels shrink, because the budget is being spent
on tests that return nothing; a rejection pays the budget back and the levels
recover. A programme that discovers things can afford to keep testing. A
programme that does not, cannot — which is uncomfortable and correct.

**The trade against ``design.program``.** e-BH controls FDR under *arbitrary*
dependence and needs the whole batch. LORD decides one at a time and needs the
p-values to be independent. A house that can wait should batch; a house that
cannot should know which of the two conditions it is now relying on.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from pydantic import Field, model_validator
from scipy import special

from axiom.core import Assumption, LedgerLine, NonEmptyStr, Spec
from axiom.design.program import Readout, expected_false_go

__all__ = [
    "INDEPENDENT_READOUTS",
    "OnlineDecision",
    "OnlineProgram",
    "gamma_sequence",
    "online_decisions",
]

_DECAY = 1.6

INDEPENDENT_READOUTS = Assumption(
    name="independent_readouts",
    facet="quantity",
    statement=(
        "the readouts arriving in this stream are independent, which is what an online "
        "false-discovery-rate procedure needs and a batch e-value procedure does not"
    ),
    challenged_by=(
        "experiments sharing markets, seasons or a macroeconomy; where that is the case, "
        "design.program's e_bh route controls the same rate under arbitrary dependence at the "
        "cost of waiting for the batch"
    ),
)


def gamma_sequence(n: int, *, decay: float = _DECAY) -> tuple[float, ...]:
    """The first ``n`` terms of ``gamma_j = j^-decay / zeta(decay)``.

    Any non-negative sequence summing to at most one gives a valid procedure;
    this one sums to exactly one over the infinite stream, so no horizon has to
    be chosen and the truncation here is only for reading. ``decay`` trades
    early power against late: a larger value spends more of the budget on the
    first few readouts.
    """
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    if not (math.isfinite(decay) and decay > 1.0):
        raise ValueError(f"decay must be finite and greater than 1, got {decay}")
    total = float(special.zeta(decay, 1))
    return tuple(float((j + 1) ** -decay / total) for j in range(n))


class OnlineDecision(Spec):
    """One readout decided when it arrived, and the level it was decided at.

    ``level`` is this readout's own alpha, which depends on everything that
    came before it and nothing that came after. ``wealth`` is what the budget
    had accumulated by then, for reading the shrink-and-recover behaviour off a
    table.
    """

    index: int = Field(ge=1)
    key: NonEmptyStr
    experiment: NonEmptyStr
    metric: NonEmptyStr = "primary"
    p_value: float = Field(ge=0, le=1)
    level: float = Field(ge=0, le=1)
    reject: bool
    wealth: float = Field(ge=0)

    @model_validator(mode="after")
    def _consistent(self) -> OnlineDecision:
        if self.reject != (self.p_value <= self.level):
            raise ValueError("reject must equal p_value <= level")
        return self


class OnlineProgram(Spec):
    """A stream of readouts decided one at a time, and what that controls.

    ``alpha`` is the false discovery rate over the whole stream, however long it
    turns out to be. ``w0`` is the starting budget and must not exceed
    ``alpha``; a larger ``w0`` spends more on the earliest readouts, which is
    the right choice for a programme that expects its first few tests to work.
    """

    stream: str = ""
    alpha: float = Field(gt=0, lt=1)
    w0: float = Field(gt=0, lt=1)
    decay: float = Field(gt=1)
    decisions: tuple[OnlineDecision, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _valid(self) -> OnlineProgram:
        if self.w0 > self.alpha:
            raise ValueError(f"w0 {self.w0} exceeds alpha {self.alpha}")
        keys = [d.key for d in self.decisions]
        if len(set(keys)) != len(keys):
            repeated = sorted({k for k in keys if keys.count(k) > 1})
            raise ValueError(f"readout keys must be distinct; repeated: {repeated}")
        if [d.index for d in self.decisions] != list(range(1, len(self.decisions) + 1)):
            raise ValueError("decisions must be indexed 1..n in arrival order")
        return self

    @property
    def n(self) -> int:
        return len(self.decisions)

    @property
    def rejected(self) -> tuple[str, ...]:
        return tuple(d.key for d in self.decisions if d.reject)

    @property
    def rejected_uncorrected(self) -> tuple[str, ...]:
        """What deciding each readout at ``alpha`` as it landed would have given."""
        return tuple(d.key for d in self.decisions if d.p_value < self.alpha)

    @property
    def expected_false_uncorrected(self) -> float:
        return expected_false_go(self.n, self.alpha)

    @property
    def levels(self) -> tuple[float, ...]:
        return tuple(d.level for d in self.decisions)

    @property
    def bound(self) -> str:
        return (
            f"false discovery rate: at most {self.alpha:g} of the {len(self.rejected)} "
            "rejections over the whole stream are expected to be false, however long the "
            "stream runs, assuming the readouts are independent"
        )

    def ledger_line(self) -> LedgerLine:
        return LedgerLine(
            kind="online_error_control",
            statement=(
                f"{self.n} readout(s) decided on arrival at alpha {self.alpha:g} (LORD, "
                f"w0 {self.w0:g}, decay {self.decay:g}): {len(self.rejected)} rejected against "
                f"{len(self.rejected_uncorrected)} uncorrected; {self.bound}"
            ),
            assumption=INDEPENDENT_READOUTS,
            detail={
                "stream": self.stream,
                "alpha": f"{self.alpha:g}",
                "w0": f"{self.w0:g}",
                "decay": f"{self.decay:g}",
                "n": str(self.n),
                "rejected": str(len(self.rejected)),
                "first_level": f"{self.levels[0]:.6g}",
                "last_level": f"{self.levels[-1]:.6g}",
            },
        )

    def summary(self) -> str:
        return "\n".join(
            [
                f"{self.stream or 'stream'}: {self.n} readouts decided on arrival",
                f"  alpha             {self.alpha:g} over the whole stream",
                f"  rejected          {len(self.rejected)}",
                f"  uncorrected       {len(self.rejected_uncorrected)}"
                f"  (a null stream expects {self.expected_false_uncorrected:.3g})",
                f"  level             {self.levels[0]:.4g} at the first readout, "
                f"{self.levels[-1]:.4g} at the last",
                f"  what that buys    {self.bound}",
            ]
        )


def online_decisions(
    readouts: Sequence[Readout],
    *,
    alpha: float = 0.05,
    w0: float | None = None,
    decay: float = _DECAY,
    stream: str = "",
) -> OnlineProgram:
    """Decide each readout as it arrives, controlling the false discovery rate over the stream.

    ``readouts`` are in **arrival order** and that order is part of the
    procedure: the same readouts in a different order give different decisions,
    because each level depends on what came before it. Every readout needs a
    ``p_value``; ``design.program`` is the e-value route and this is not it.

    ``w0`` defaults to ``alpha / 2``, the usual compromise between spending on
    the first readouts and keeping budget for later ones.
    """
    if not readouts:
        raise ValueError("an online programme needs at least one readout")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    wealth0 = alpha / 2.0 if w0 is None else w0
    if not 0.0 < wealth0 <= alpha:
        raise ValueError(f"w0 must be in (0, alpha], got {wealth0} against alpha {alpha}")
    missing = [r.key for r in readouts if r.p_value is None]
    if missing:
        raise ValueError(
            f"an online programme needs a p-value on every readout; missing on {missing[:5]}. "
            "design.program is the e-value route"
        )

    gamma = gamma_sequence(len(readouts), decay=decay)
    rejections: list[int] = []
    decisions: list[OnlineDecision] = []
    for t, readout in enumerate(readouts, start=1):
        level = gamma[t - 1] * wealth0
        for order, tau in enumerate(rejections, start=1):
            gap = t - tau
            if gap >= 1:
                weight = (alpha - wealth0) if order == 1 else alpha
                level += weight * gamma[gap - 1]
        level = min(level, 1.0)
        p_value = float(readout.p_value or 0.0)
        reject = p_value <= level
        decisions.append(
            OnlineDecision(
                index=t,
                key=readout.key,
                experiment=readout.experiment,
                metric=readout.metric,
                p_value=p_value,
                level=level,
                reject=reject,
                wealth=level,
            )
        )
        if reject:
            rejections.append(t)
    return OnlineProgram(
        stream=stream, alpha=alpha, w0=wealth0, decay=decay, decisions=tuple(decisions)
    )
