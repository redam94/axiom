"""What is still identified when some outcomes are missing: Lee's trimming bounds.

``diagnose.attrition`` says the arms were observed at different rates and
refuses the contrast. That is the right refusal and it is not an answer. This
module is the answer that survives: **bounds that hold whatever the missing
outcomes were**.

The construction (Lee 2009). Assignment ``D`` moves selection ``S`` — whether a
unit reported an outcome at all — and the outcome ``Y`` is seen only where
``S = 1``. Under **monotonicity in selection** (assignment can only push
selection one way; nobody reports when assigned to control and goes silent when
assigned to treatment), the selected units of the arm with the *higher*
selection rate are a mixture of two groups in known proportions:

* **always-responders**, who would have reported either way — a share
  ``p_low / p_high`` of that arm's selected units;
* **marginal responders**, who reported only because of their assignment — the
  remaining ``q = 1 − p_low / p_high``.

The other arm's selected units are always-responders and nothing else. So
trimming the top ``q`` of the high arm's outcome distribution and comparing
gives the lowest the always-responder effect can be; trimming the bottom ``q``
gives the highest. Between those two numbers the data cannot distinguish, and
outside them nothing is consistent with what was observed.

**The population is a latent one, and this module says so.** Always-responders
are a principal stratum: sized by the two selection rates, never listed, no
covariate distinguishing a member. ``LeeBounds.population`` is a
``core.Population`` carrying the ``LatentSelection`` for them, so an estimand
built on it inherits everything
``docs/notes/0032-a-population-you-can-size-and-cannot-list.md`` decided —
including that ``meta.pool`` will refuse to average it with an effect over
everybody.

**Sampling uncertainty is not in these numbers.** What ``lee_bounds`` returns is
the *identified set*: where the parameter can be given infinite data. A
confidence interval for a partially identified parameter is a different and
more delicate object (Imbens and Manski 2004), and returning the identified set
labelled as one is better than returning something that reads like an interval
and is not.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import Field, model_validator

from axiom.core import (
    Assumption,
    LatentSelection,
    LedgerLine,
    NonEmptyStr,
    Population,
    Spec,
    Unverified,
    Verdict,
)

__all__ = [
    "SELECTION_MONOTONICITY",
    "LeeBounds",
    "lee_bounds",
]

Array = npt.NDArray[np.float64]

SELECTION_MONOTONICITY = Assumption(
    name="selection_monotonicity",
    facet="population",
    statement=(
        "assignment moves selection in one direction only: no unit reports an outcome when "
        "assigned to one arm and goes silent when assigned to the other, so the arm with the "
        "higher selection rate contains every always-responder plus a known share of marginal "
        "ones"
    ),
    challenged_by=(
        "a covariate stratum in which the selection rates run the other way, which is "
        "monotonicity failing within it; the assumption itself is not testable, because the "
        "units that would falsify it are exactly the ones never observed in one arm"
    ),
)


def _binary(frame: pd.DataFrame, column: str) -> Array:
    if column not in frame.columns:
        raise KeyError(f"columns not in frame: {[column]}")
    values = np.asarray(frame[column].to_numpy(dtype=np.float64))
    if not np.all(np.isfinite(values)):
        raise ValueError(f"column {column!r} must be finite")
    if not set(np.unique(values).tolist()) <= {0.0, 1.0}:
        raise ValueError(f"column {column!r} must be 0/1, got {sorted(set(values.tolist()))[:5]}")
    return values


class LeeBounds(Spec):
    """The identified set for the effect on the units that would have reported either way.

    ``lower`` and ``upper`` bracket the always-responder effect; ``trimmed`` is
    the share ``q`` of the over-selected arm's reports that had to be cut, which
    is the cost of the imbalance and the width of the bounds is roughly
    proportional to it. ``trimmed_arm`` names which arm was cut.

    ``point_identified`` is true when the two arms selected at the same rate:
    there is nothing to trim, the bounds collapse, and the always-responders are
    everybody who reported.
    """

    outcome: NonEmptyStr
    assignment: NonEmptyStr
    selection: NonEmptyStr
    lower: float
    upper: float
    trimmed: float = Field(ge=0, le=1)
    trimmed_arm: NonEmptyStr
    rate_treated: float = Field(ge=0, le=1)
    rate_control: float = Field(ge=0, le=1)
    n: int = Field(ge=1)
    n_observed: int = Field(ge=1)
    naive: float

    @model_validator(mode="after")
    def _ordered(self) -> LeeBounds:
        if not (math.isfinite(self.lower) and math.isfinite(self.upper)):
            raise ValueError(f"bounds must be finite, got [{self.lower}, {self.upper}]")
        if self.lower > self.upper:
            raise ValueError(f"lower {self.lower} exceeds upper {self.upper}")
        if self.n_observed > self.n:
            raise ValueError("more units reported than were assigned")
        return self

    @property
    def width(self) -> float:
        """How much the missing outcomes could move the answer."""
        return self.upper - self.lower

    @property
    def point_identified(self) -> bool:
        """The arms selected at the same rate, so nothing had to be trimmed."""
        return self.trimmed == 0.0

    @property
    def excludes_zero(self) -> bool:
        """Whether *every* value in the identified set has the same sign."""
        return self.lower > 0.0 or self.upper < 0.0

    @property
    def population(self) -> Population:
        """The always-responders, as a ``Population`` an ``Estimand`` can name.

        Sized — ``share`` is ``p_low / p_high``, the fraction of the trimmed
        arm's reports that would have reported either way — and never listed.
        """
        share = 1.0 - self.trimmed
        return Population(
            name="always_responders",
            description=(
                f"units that would have reported {self.outcome!r} under either arm of "
                f"{self.assignment!r}"
            ),
            latent=LatentSelection(
                kind="always_responder",
                instrument=self.assignment,
                exposure=self.selection,
                share=share,
                description="identified by Lee trimming; sized by the two selection rates",
            ),
        )

    def verdict(self) -> Verdict:
        """``downgraded`` under ``SELECTION_MONOTONICITY`` — never ``identified``.

        The bounds are a real answer and they are not a point. Even when they
        collapse (equal selection rates) the quantity is still the
        always-responder effect under an assumption the data cannot check, so
        the strongest honest status is a downgrade.
        """
        return Verdict(
            status="downgraded",
            reason=(
                f"the effect on the always-responders lies in [{self.lower:.4g}, "
                f"{self.upper:.4g}] (width {self.width:.4g}) after trimming "
                f"{self.trimmed:.1%} of the {self.trimmed_arm} arm's reports; the naive "
                f"contrast over reported units is {self.naive:.4g}"
            ),
            assumptions=(SELECTION_MONOTONICITY,),
            route="lee_bounds",
        )

    def ledger_line(self) -> LedgerLine:
        return LedgerLine(
            kind="lee_bounds",
            statement=(
                f"selection on {self.selection!r} differs by arm "
                f"({self.rate_treated:.3f} treated against {self.rate_control:.3f} control); "
                f"the always-responder effect on {self.outcome!r} is in "
                f"[{self.lower:.4g}, {self.upper:.4g}], against a naive {self.naive:.4g}"
            ),
            assumption=SELECTION_MONOTONICITY,
            detail={
                "outcome": self.outcome,
                "assignment": self.assignment,
                "selection": self.selection,
                "trimmed": f"{self.trimmed:.6g}",
                "trimmed_arm": self.trimmed_arm,
                "width": f"{self.width:.6g}",
                "n_observed": str(self.n_observed),
            },
        )

    def summary(self) -> str:
        return "\n".join(
            [
                f"always-responder effect on {self.outcome!r}: "
                f"[{self.lower:.4g}, {self.upper:.4g}]",
                f"  naive contrast over reported units  {self.naive:+.4g}",
                f"  selection {self.rate_treated:.1%} treated / "
                f"{self.rate_control:.1%} control",
                f"  trimmed {self.trimmed:.1%} of the {self.trimmed_arm} arm "
                f"({self.n_observed} of {self.n} units reported)",
                f"  width {self.width:.4g}"
                + ("  — every value has one sign" if self.excludes_zero else ""),
            ]
        )


def lee_bounds(
    frame: pd.DataFrame,
    y: str,
    assignment: str,
    selection: str,
    *,
    treated_arm: float = 1.0,
) -> LeeBounds | Unverified:
    """Trimming bounds on the always-responder effect, under selection monotonicity.

    ``assignment`` and ``selection`` are 0/1 columns; ``y`` need only be finite
    where ``selection == 1`` and is ignored elsewhere, which is the point — the
    missing outcomes are missing.

    Returns ``Unverified`` rather than a number when the bounds cannot be
    formed: an arm with no reports at all, or a frame with fewer than two units
    reporting in an arm. Trimming needs a distribution to trim.
    """
    if assignment == selection or y in (assignment, selection):
        raise ValueError("outcome, assignment and selection must be three different columns")
    assigned = _binary(frame, assignment)
    selected = _binary(frame, selection)
    if y not in frame.columns:
        raise KeyError(f"columns not in frame: {[y]}")
    outcome = np.asarray(frame[y].to_numpy(dtype=np.float64))
    if treated_arm not in (0.0, 1.0):
        raise ValueError(f"treated_arm must be 0 or 1, got {treated_arm}")

    treated = assigned == treated_arm
    control = ~treated
    n_treated, n_control = int(treated.sum()), int(control.sum())
    if n_treated == 0 or n_control == 0:
        return Unverified(reason="both arms must contain units", detail={"treated": str(n_treated)})
    rate_treated = float(selected[treated].mean())
    rate_control = float(selected[control].mean())
    if rate_treated == 0.0 or rate_control == 0.0:
        return Unverified(
            reason=(
                "an arm reported no outcomes at all "
                f"({rate_treated:.3f} treated, {rate_control:.3f} control); there is no "
                "distribution to trim and no contrast to bound"
            )
        )

    treated_y = outcome[treated & (selected == 1.0)]
    control_y = outcome[control & (selected == 1.0)]
    if treated_y.size < 2 or control_y.size < 2:
        return Unverified(
            reason=(
                f"an arm has fewer than two reported outcomes ({treated_y.size} treated, "
                f"{control_y.size} control); trimming needs a distribution"
            )
        )
    if not (np.all(np.isfinite(treated_y)) and np.all(np.isfinite(control_y))):
        raise ValueError(f"reported outcomes in {y!r} must be finite")

    naive = float(treated_y.mean() - control_y.mean())
    if rate_treated >= rate_control:
        trimmed_arm, over, under = assignment, treated_y, control_y
        q = 1.0 - rate_control / rate_treated
        sign = 1.0
    else:
        trimmed_arm, over, under = f"not-{assignment}", control_y, treated_y
        q = 1.0 - rate_treated / rate_control
        sign = -1.0

    low_tail = (
        float(np.mean(over[over <= np.quantile(over, 1.0 - q)])) if q > 0 else float(over.mean())
    )
    high_tail = float(np.mean(over[over >= np.quantile(over, q)])) if q > 0 else float(over.mean())
    other = float(under.mean())
    first, second = sign * (low_tail - other), sign * (high_tail - other)
    lower, upper = (first, second) if first <= second else (second, first)

    return LeeBounds(
        outcome=y,
        assignment=assignment,
        selection=selection,
        lower=lower,
        upper=upper,
        trimmed=float(q),
        trimmed_arm=trimmed_arm,
        rate_treated=rate_treated,
        rate_control=rate_control,
        n=int(assigned.size),
        n_observed=int(selected.sum()),
        naive=naive,
    )
