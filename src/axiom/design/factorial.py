"""Whether two concurrent experiments were each other's background, or each other's effect.

``design.collision`` classifies an overlap as *concurrent* when two designs
share units and periods but move different levers, and says the estimates are
unbiased because independent randomizations are orthogonal in expectation. That
is true of the *main effects* and it rests on a condition
``CONCURRENT_EXPERIMENTS`` names and nothing has checked:

    each estimate is the effect of its own treatment averaged over whatever the
    other one was doing, **and the two treatments do not interact**

If they interact, neither main effect is "the effect of this treatment". Each is
an average over the other experiment's assignment, and the average changes when
the other experiment ends. A price test that lifts revenue 3 % while a banner
test is running, and 1 % after it stops, was never one number.

The check is the analysis the assumption's own ``challenged_by`` asks for: fit
both assignments **and their product** in one regression and look at the
product's coefficient. ``design.factorial`` is that fit, over
``identify.ols`` — the same estimator, not a second one.

**A passing check leaves the assumption unverified, not satisfied.** An
interaction indistinguishable from zero in an experiment powered for main
effects is very weak evidence: detecting an interaction of a given size needs
roughly four times the units that detecting a main effect of that size does.
``Factorial.power_note`` says so with the study's own numbers rather than
leaving a reader to assume a null result meant something.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import Field, model_validator

from axiom.core import Assumption, Interval, LedgerLine, NonEmptyStr, Spec, Verdict
from axiom.design.collision import CONCURRENT_EXPERIMENTS
from axiom.identify import LinearEstimate, ols

__all__ = [
    "Factorial",
    "FactorialCell",
    "factorial",
]

Array = npt.NDArray[np.float64]


class FactorialCell(Spec):
    """One of the four cells of a two-by-two: how many units and what they did."""

    left: int = Field(ge=0, le=1)
    right: int = Field(ge=0, le=1)
    n: int = Field(ge=0)
    mean: float
    se: float = Field(ge=0)

    @model_validator(mode="after")
    def _finite(self) -> FactorialCell:
        if self.n and not (math.isfinite(self.mean) and math.isfinite(self.se)):
            raise ValueError("a non-empty cell needs a finite mean and se")
        return self

    @property
    def label(self) -> str:
        return f"{self.left}{self.right}"


class Factorial(Spec):
    """Two assignments, their interaction, and what the interaction does to the mains.

    ``interaction`` is the coefficient on the product of the two assignment
    columns in ``y ~ left + right + left·right``. ``interacts`` is whether it is
    distinguishable from zero at ``alpha``; when it is, neither main effect is
    the effect of its own treatment on its own.
    """

    outcome: NonEmptyStr
    left: NonEmptyStr
    right: NonEmptyStr
    cells: tuple[FactorialCell, ...] = Field(min_length=4, max_length=4)
    main_left: LinearEstimate
    main_right: LinearEstimate
    interaction: LinearEstimate
    alpha: float = Field(gt=0, lt=1)
    mass: float = Field(default=0.95, gt=0, lt=1)
    interacts: bool

    @model_validator(mode="after")
    def _consistent(self) -> Factorial:
        labels = {c.label for c in self.cells}
        if labels != {"00", "01", "10", "11"}:
            raise ValueError(f"a two-by-two needs all four cells, got {sorted(labels)}")
        return self

    @property
    def n(self) -> int:
        return sum(c.n for c in self.cells)

    def cell(self, left: int, right: int) -> FactorialCell:
        for c in self.cells:
            if (c.left, c.right) == (left, right):
                return c
        raise KeyError(f"no cell ({left}, {right})")  # pragma: no cover - validated above

    @property
    def interaction_interval(self) -> Interval:
        return self.interaction.ci(self.mass)

    @property
    def relative(self) -> float:
        """The interaction against the larger main effect; ``nan`` when both are zero."""
        largest = max(abs(self.main_left.estimate), abs(self.main_right.estimate))
        return abs(self.interaction.estimate) / largest if largest > 0 else float("nan")

    @property
    def power_note(self) -> str:
        """Why a null interaction in a main-effects study is weak evidence, with numbers."""
        ratio = self.interaction.se / max(self.main_left.se, self.main_right.se)
        return (
            f"the interaction's standard error is {ratio:.1f}x the larger main effect's, so "
            f"this study could only have detected an interaction of about "
            f"{2 * self.interaction.se:.4g} or more"
        )

    def assumption(self) -> Assumption:
        """``CONCURRENT_EXPERIMENTS``, ``violated`` when the two treatments interact."""
        base = CONCURRENT_EXPERIMENTS.model_copy(
            update={
                "detail": {
                    "interaction": f"{self.interaction.estimate:.6g}",
                    "interaction_se": f"{self.interaction.se:.6g}",
                    "z": f"{self.interaction.z():.6g}",
                    "relative_to_main": f"{self.relative:.6g}",
                }
            }
        )
        return base.violated() if self.interacts else base

    def verdict(self) -> Verdict:
        """``blocked`` on a real interaction, ``unverified`` otherwise — never ``identified``.

        A null interaction does not license the main effects; it fails to
        refute them, in a study that was not powered to. ``unverified`` is the
        status for an assumption the machinery could not check, and a
        main-effects experiment cannot check this one.
        """
        if self.interacts:
            return Verdict(
                status="blocked",
                reason=(
                    f"{self.left!r} and {self.right!r} interact "
                    f"({self.interaction.estimate:+.4g} {self.interaction_interval.text()}, "
                    f"{self.relative:.0%} of the larger main effect); neither main effect is "
                    "the effect of its own treatment — each is an average over the other's "
                    "assignment and changes when the other experiment ends"
                ),
                route="factorial",
            )
        return Verdict(
            status="unverified",
            reason=(
                f"no interaction detected ({self.interaction.estimate:+.4g} "
                f"{self.interaction_interval.text()}); {self.power_note}"
            ),
            route="factorial",
        )

    def ledger_line(self) -> LedgerLine:
        return LedgerLine(
            kind="factorial",
            statement=(
                f"{self.left!r} x {self.right!r} on {self.outcome!r}: interaction "
                f"{self.interaction.estimate:+.4g} {self.interaction_interval.text()} against "
                f"main effects {self.main_left.estimate:+.4g} and "
                f"{self.main_right.estimate:+.4g} — "
                + ("they interact" if self.interacts else "no interaction detected")
            ),
            assumption=self.assumption(),
            detail={
                "outcome": self.outcome,
                "left": self.left,
                "right": self.right,
                "n": str(self.n),
                "relative_to_main": f"{self.relative:.6g}",
            },
        )

    def summary(self) -> str:
        rows = [f"{self.left!r} x {self.right!r} on {self.outcome!r} ({self.n} units)"]
        for left in (0, 1):
            cells = [self.cell(left, right) for right in (0, 1)]
            rows.append(
                f"  {self.left}={left}: "
                + "  ".join(f"{self.right}={c.right} {c.mean:+.4g} (n={c.n})" for c in cells)
            )
        rows += [
            f"  main {self.left:<16} {self.main_left.estimate:+.4g} "
            f"{self.main_left.ci(self.mass).text()}",
            f"  main {self.right:<16} {self.main_right.estimate:+.4g} "
            f"{self.main_right.ci(self.mass).text()}",
            f"  interaction{'':<15} {self.interaction.estimate:+.4g} "
            f"{self.interaction_interval.text()}",
            f"  {self.verdict().status}",
        ]
        return "\n".join(rows)


def _binary(frame: pd.DataFrame, column: str) -> Array:
    if column not in frame.columns:
        raise KeyError(f"columns not in frame: {[column]}")
    values = np.asarray(frame[column].to_numpy(dtype=np.float64))
    if not np.all(np.isfinite(values)):
        raise ValueError(f"column {column!r} must be finite")
    if not set(np.unique(values).tolist()) <= {0.0, 1.0}:
        raise ValueError(f"column {column!r} must be 0/1 for a factorial check")
    return values


def factorial(
    frame: pd.DataFrame,
    y: str,
    left: str,
    right: str,
    covariates: str | Sequence[str] = (),
    *,
    alpha: float = 0.05,
    mass: float = 0.95,
) -> Factorial:
    """Fit ``y ~ left + right + left·right`` and report what the product says.

    ``left`` and ``right`` are the two 0/1 assignment columns of two concurrent
    experiments. Every cell of the two-by-two must contain at least two units —
    a design where one combination never occurred cannot separate the
    interaction from the mains, and this raises rather than reporting a number
    that is a main effect wearing an interaction's name.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if not 0.0 < mass < 1.0:
        raise ValueError(f"mass must be in (0, 1), got {mass}")
    if len({y, left, right}) != 3:
        raise ValueError("outcome and the two assignments must be three different columns")
    a, b = _binary(frame, left), _binary(frame, right)
    if y not in frame.columns:
        raise KeyError(f"columns not in frame: {[y]}")
    outcome = np.asarray(frame[y].to_numpy(dtype=np.float64))
    if not np.all(np.isfinite(outcome)):
        raise ValueError(f"column {y!r} must be finite")

    product = f"_{left}_x_{right}"
    if product in frame.columns:
        raise ValueError(f"the frame already has a column named {product!r}")
    fitted = frame.assign(**{product: a * b})

    cells = []
    for lo in (0, 1):
        for ro in (0, 1):
            members = outcome[(a == lo) & (b == ro)]
            if members.size < 2:
                raise ValueError(
                    f"cell {left}={lo}, {right}={ro} has {members.size} unit(s); a factorial "
                    "check needs at least two in every cell, or the interaction is not "
                    "separable from the main effects"
                )
            cells.append(
                FactorialCell(
                    left=lo,
                    right=ro,
                    n=int(members.size),
                    mean=float(members.mean()),
                    se=float(members.std(ddof=1) / math.sqrt(members.size)),
                )
            )

    others = tuple(covariates) if not isinstance(covariates, str) else (covariates,)
    main_left = ols(fitted, y=y, x=left, covariates=[right, product, *others])
    main_right = ols(fitted, y=y, x=right, covariates=[left, product, *others])
    interaction = ols(fitted, y=y, x=product, covariates=[left, right, *others])
    return Factorial(
        outcome=y,
        left=left,
        right=right,
        cells=tuple(cells),
        main_left=main_left,
        main_right=main_right,
        interaction=interaction,
        alpha=alpha,
        mass=mass,
        interacts=abs(interaction.z()) > float(abs(_z_critical(alpha))),
    )


def _z_critical(alpha: float) -> float:
    from scipy import stats

    return float(stats.norm.isf(alpha / 2.0))
