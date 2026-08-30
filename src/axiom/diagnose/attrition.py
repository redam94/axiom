"""Units that were assigned and never reported, and why that is not the same as non-exposure.

``diagnose.delivery`` checks that the arms were *reached* at the same rate.
This module checks that they were *observed* at the same rate, and the two are
different problems with different fixes:

* a unit assigned to treatment that was never exposed is a **compliance**
  problem. It has an outcome, the outcome is real, and
  ``identify.compliance`` reports the two estimands that fall out of it.
* a unit assigned to treatment whose outcome is **missing** is a *selection*
  problem. There is no number, and the units without one are not a random
  subset of the units with one — they dropped out, churned, or died, and
  whatever made them do that is plausibly related to the outcome.

The first can be estimated around. The second cannot, and the honest answers
are two: a check that says how bad the imbalance is
(:func:`attrition`), and bounds that hold whatever the missing outcomes were
(``identify.lee_bounds``).

**Overall attrition is not the problem; differential attrition is.** A study
that loses 30 % of both arms has lost power. A study that loses 30 % of one arm
and 10 % of the other has lost the comparison, because the survivors of the two
arms are no longer the same population — and the size of that gap is a lower
bound on how wrong the naive contrast can be.

Alpha is :data:`~axiom.diagnose.delivery.SRM_ALPHA` for the same reason it is
there: a check that runs on every experiment and cries wolf weekly is a check
nobody reads.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import numpy.typing as npt
from pydantic import Field, model_validator
from scipy import stats

from axiom.core import Assumption, LedgerLine, NonEmptyStr, Spec, Verdict
from axiom.diagnose.delivery import SRM_ALPHA

__all__ = [
    "NO_DIFFERENTIAL_ATTRITION",
    "Attrition",
    "AttritionRow",
    "attrition",
]

NO_DIFFERENTIAL_ATTRITION = Assumption(
    name="no_differential_attrition",
    facet="population",
    statement=(
        "the units whose outcome is missing are missing for reasons unrelated to the arm they "
        "were assigned to, so the units observed in each arm are the same population"
    ),
    challenged_by=(
        "an observation rate that differs by arm — which is checkable — and, when it does, "
        "bounds that hold whatever the missing outcomes were (identify.lee_bounds)"
    ),
)


class AttritionRow(Spec):
    """One arm: how many units it was assigned and how many reported an outcome."""

    arm: NonEmptyStr
    assigned: int = Field(ge=0)
    observed: int = Field(ge=0)

    @model_validator(mode="after")
    def _consistent(self) -> AttritionRow:
        if self.observed > self.assigned:
            raise ValueError(
                f"arm {self.arm!r}: {self.observed} observed of {self.assigned} assigned; "
                "a unit cannot report without being assigned"
            )
        return self

    @property
    def rate(self) -> float:
        """The share of assigned units that reported; ``nan`` for an empty arm."""
        return self.observed / self.assigned if self.assigned else float("nan")

    @property
    def lost(self) -> int:
        return self.assigned - self.observed


class Attrition(Spec):
    """Observation rates by arm, and the test that they are one rate.

    ``overall`` is the share of all assigned units that reported — a power
    problem. ``differential`` is the largest gap between two arms' rates — a
    comparability problem, and the one ``differential_attrition`` flags.
    """

    arms: tuple[AttritionRow, ...] = Field(min_length=2)
    differential: float = Field(ge=0)
    chi_square: float = Field(ge=0)
    df: int = Field(ge=1)
    p_value: float = Field(ge=0, le=1)
    alpha: float = Field(gt=0, lt=1)
    differential_attrition: bool

    @model_validator(mode="after")
    def _consistent(self) -> Attrition:
        if self.differential_attrition != (self.p_value < self.alpha):
            raise ValueError("differential_attrition must equal p_value < alpha")
        return self

    @property
    def overall(self) -> float:
        assigned = sum(a.assigned for a in self.arms)
        return sum(a.observed for a in self.arms) / assigned if assigned else float("nan")

    @property
    def worst_arm(self) -> str:
        """The arm that lost the most, proportionally."""
        live = [a for a in self.arms if a.assigned > 0]
        return min(live, key=lambda a: a.rate).arm

    def assumption(self) -> Assumption:
        """``NO_DIFFERENTIAL_ATTRITION``, ``violated`` when the rates differ by arm."""
        detail = {
            "overall": f"{self.overall:.6g}",
            "differential": f"{self.differential:.6g}",
            "worst_arm": self.worst_arm,
        }
        base = NO_DIFFERENTIAL_ATTRITION.model_copy(update={"detail": detail})
        return base.violated() if self.differential_attrition else base

    def verdict(self) -> Verdict:
        """``identified`` when the rates agree, ``blocked`` when they do not.

        Never ``downgraded``: no assumption over observables licenses a
        contrast between two arms whose survivors are different populations.
        What licenses a number there is a *bound*, and that is a different
        estimand (``identify.lee_bounds``), not a downgraded version of this
        one.
        """
        if not self.differential_attrition:
            return Verdict(status="identified", route="attrition")
        return Verdict(
            status="blocked",
            reason=(
                f"observation rates differ by arm (gap {self.differential:.3f}, worst "
                f"{self.worst_arm}, p = {self.p_value:.3g}); the survivors of the two arms are "
                "not the same population, so their contrast is not the effect. Use "
                "identify.lee_bounds for a number that holds whatever the missing outcomes were"
            ),
            route="attrition",
        )

    def ledger_line(self) -> LedgerLine:
        rates = ", ".join(f"{a.arm} {a.rate:.3f}" for a in self.arms)
        verdict = "DIFFERENTIAL" if self.differential_attrition else "even"
        return LedgerLine(
            kind="attrition",
            statement=(
                f"attrition: {verdict} ({rates}); {self.overall:.3f} of assigned units "
                f"reported overall, largest gap {self.differential:.3f}, p = "
                f"{self.p_value:.3g} against alpha {self.alpha:g}"
            ),
            assumption=self.assumption(),
            detail={
                "overall": f"{self.overall:.6g}",
                "differential": f"{self.differential:.6g}",
                "p_value": f"{self.p_value:.12g}",
                "worst_arm": self.worst_arm,
            },
        )

    def summary(self) -> str:
        lines = [f"attrition over {len(self.arms)} arms, {self.overall:.1%} observed overall"]
        for row in self.arms:
            lines.append(
                f"  {row.arm:<12} {row.observed}/{row.assigned} reported "
                f"({row.rate:.1%}, {row.lost} lost)"
            )
        lines.append(f"  gap {self.differential:.3f}, p = {self.p_value:.3g}")
        lines.append(f"  {self.verdict().status}")
        return "\n".join(lines)


def _counts(values: npt.ArrayLike) -> npt.NDArray[np.float64]:
    return np.asarray(values, dtype=np.float64)


def attrition(
    assigned: Mapping[str, int],
    observed: Mapping[str, int],
    *,
    alpha: float = SRM_ALPHA,
) -> Attrition:
    """Observation rates by arm, and a test that the arms were observed at one rate.

    The same shape as ``diagnose.delivery`` and a different question: that one
    asks whether the treatment reached the units, this one asks whether the
    outcome reached the analyst. A unit can be exposed and unobserved, or
    observed and unexposed, and the two failures need different answers.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if set(assigned) != set(observed):
        raise ValueError(
            f"assigned names {sorted(assigned)} and observed names {sorted(observed)}; "
            "they must be the same arms"
        )
    if len(assigned) < 2:
        raise ValueError("an attrition check needs at least two arms")
    rows = tuple(
        AttritionRow(arm=arm, assigned=int(assigned[arm]), observed=int(observed[arm]))
        for arm in sorted(assigned)
    )
    live = [r for r in rows if r.assigned > 0]
    if len(live) < 2:
        raise ValueError("an attrition check needs at least two arms with units in them")
    table = _counts([[r.observed, r.lost] for r in live])
    rates = [r.rate for r in live]
    differential = float(max(rates) - min(rates))
    if table[:, 0].sum() == 0 or table[:, 1].sum() == 0:
        # Everybody reported, or nobody did: the table is degenerate and there is no gap.
        # float(), not int: scipy types chi2_contingency's dof as a float, and
        # this branch has to bind `df` to the same type the other one does. It
        # is narrowed back to an int on the way into the result either way.
        chi_square, p_value, df = 0.0, 1.0, float(len(live) - 1)
    else:
        chi_square, p_value, df, _ = stats.chi2_contingency(table, correction=False)
    if not math.isfinite(float(chi_square)):  # pragma: no cover - guarded by the branch above
        chi_square, p_value = 0.0, 1.0
    return Attrition(
        arms=rows,
        differential=differential,
        chi_square=float(chi_square),
        df=int(df),
        p_value=float(p_value),
        alpha=alpha,
        differential_attrition=float(p_value) < alpha,
    )
