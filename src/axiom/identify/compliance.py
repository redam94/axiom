"""Assigned is not received: the two estimands a partly-complied experiment produces.

``diagnose.delivery`` reports that the arms were reached at different rates. It
does not say what to do about it, and what to do about it is not "adjust the
estimate" — it is to notice that there are now **two different quantities** and
that a readout has to say which one it is (note 0027, §D27.6.7).

    intention to treat   the effect of being *assigned* to the arm,
                         over everybody who was assigned
    complier effect      the effect of being *exposed*, over the units whose
                         exposure was moved by the assignment

Both are causal, both are identified by the randomization, and they are not
estimates of one number. They differ on two of the eight facets in the charter's
table at once:

* ``intervention`` — assigned-to-treatment against received-treatment;
* ``population`` — everybody against the compliers, a subpopulation that cannot
  be listed, only sized.

Pooling one party's ITT with another party's complier effect is exactly the
silent facet difference the estimand machinery exists to prevent, one layer
below where it can see it. A house running experiments for several parties whose
operations deliver at 95 % and 60 % will produce both without anybody choosing
to, which is why this module reports them side by side and never one alone.

**What licenses the complier effect.** It is 2SLS of the outcome on exposure,
instrumented by assignment — ``identify.two_stage_least_squares``, not a second
implementation of the Wald ratio — so it inherits that function's first-stage F
and ``weak_instrument_check``. On top of the randomization it needs:

* **exclusion** — assignment moves the outcome only through exposure. Untestable
  in the data, and it is the assumption that fails when the assignment itself is
  visible to the unit.
* **monotonicity** — no unit is exposed *because* it was assigned to control.
  Also untestable, but it has a checkable implication: the estimated complier
  share, which is the difference in exposure rates, cannot be negative.
* **relevance** — the assignment moved exposure at all. A first stage near zero
  is what makes the ratio explode, and it is the one of the three the data can
  speak to directly.

**A dose is not a switch.** Everything above assumes exposure is 0/1: a unit
either attended or did not. Half the sessions attended, two of five weeks
exposed, sixty per cent of the budget delivered — these are the ordinary case in
the field and there is no "complier" in them, because there is no single event a
unit did or did not respond to. The instrumented estimate is then a *local
average derivative*: an effect **per unit of exposure**, averaged over units in
proportion to how much the assignment moved each one (Angrist, Graddy and Imbens
2000). ``response_to_dose`` is that report and ``UNIFORM_FIRST_STAGE`` is the
assumption it adds, which is monotonicity's continuous cousin and no more
testable.

**When compliance is perfect** the exposure column *is* the assignment column,
so there is no instrumental variation and 2SLS is undefined. That is not a
failure: the intention-to-treat estimate is already the effect of exposure, over
everybody, and the two estimands have collapsed into one.
``ComplianceReport.perfect_compliance`` says so, and the complier row is the
reason rather than a number computed from a degenerate first stage.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import Field, model_validator

from axiom.core.result import NonEmptyStr, Unverified
from axiom.core.spec import Spec
from axiom.core.verdict import Assumption, LedgerLine, Verdict
from axiom.identify.estimators import (
    LinearEstimate,
    ols,
    two_stage_least_squares,
    weak_instrument_check,
)

__all__ = [
    "EXCLUSION",
    "MONOTONICITY",
    "UNIFORM_FIRST_STAGE",
    "ComplianceReport",
    "ComplianceTable",
    "DerivativeReport",
    "FirstStage",
    "complier_effect",
    "compliance",
    "compliance_table",
    "first_stage",
    "intention_to_treat",
    "local_derivative",
    "response_to_dose",
]

Array = npt.NDArray[np.float64]

EXCLUSION = Assumption(
    name="exclusion_restriction",
    facet="intervention",
    statement=(
        "assignment moves the outcome only through exposure: a unit that would not have "
        "been exposed either way is unaffected by which arm it was assigned to"
    ),
    challenged_by=(
        "an effect of assignment on units known not to have been exposed, or an assignment "
        "the unit can see and respond to on its own"
    ),
)

MONOTONICITY = Assumption(
    name="monotonicity",
    facet="population",
    statement=(
        "no unit is exposed because it was assigned to control and unexposed because it was "
        "assigned to treatment; there are no defiers, so the exposure difference between arms "
        "is the complier share"
    ),
    challenged_by=(
        "an exposure rate that is higher in the control arm than in the treated arm, which "
        "is the implication of monotonicity the data can check"
    ),
)


def _binary(frame: pd.DataFrame, column: str) -> Array:
    if column not in frame.columns:
        raise KeyError(f"columns not in frame: {[column]}")
    values = np.asarray(frame[column].to_numpy(dtype=np.float64))
    if not np.all(np.isfinite(values)):
        raise ValueError(f"column {column!r} must be finite")
    unique = set(np.unique(values).tolist())
    if not unique <= {0.0, 1.0}:
        raise ValueError(
            f"column {column!r} must be 0/1 for a compliance analysis, got values "
            f"{sorted(unique)[:5]}"
        )
    return values


class ComplianceTable(Spec):
    """Assignment against exposure, and the shares of the population it sizes.

    Under monotonicity the three types partition the units and their shares are
    identified without ever naming a member of any of them:

    ==================  ==============================================
    ``complier_share``  exposed if assigned, not if not — the effect's
                        population, sized but not listable
    ``always_taker``    exposed either way: the control arm's rate
    ``never_taker``     exposed neither way: one minus the treated rate
    ==================  ==============================================

    ``monotonic`` is ``False`` when the control arm was exposed at a *higher*
    rate than the treated arm, which is monotonicity's checkable implication
    failing. ``one_sided`` marks the common design where the control arm cannot
    be exposed at all, in which case there are no always-takers and the complier
    effect is the effect on the treated.
    """

    assignment: NonEmptyStr
    exposure: NonEmptyStr
    n: int = Field(ge=1)
    n_assigned: int = Field(ge=0)
    n_control: int = Field(ge=0)
    exposed_assigned: int = Field(ge=0)
    exposed_control: int = Field(ge=0)

    @model_validator(mode="after")
    def _consistent(self) -> ComplianceTable:
        if self.n_assigned + self.n_control != self.n:
            raise ValueError("the two arms must account for every unit")
        if self.exposed_assigned > self.n_assigned or self.exposed_control > self.n_control:
            raise ValueError("a unit cannot be exposed without being in the arm it is counted in")
        if self.n_assigned == 0 or self.n_control == 0:
            raise ValueError("a compliance table needs units in both arms")
        return self

    @property
    def rate_assigned(self) -> float:
        """Exposure rate in the treated arm."""
        return self.exposed_assigned / self.n_assigned

    @property
    def rate_control(self) -> float:
        """Exposure rate in the control arm — the always-taker share."""
        return self.exposed_control / self.n_control

    @property
    def complier_share(self) -> float:
        """The first stage: how much assignment moved exposure."""
        return self.rate_assigned - self.rate_control

    @property
    def always_taker_share(self) -> float:
        return self.rate_control

    @property
    def never_taker_share(self) -> float:
        return 1.0 - self.rate_assigned

    @property
    def monotonic(self) -> bool:
        """Whether monotonicity's checkable implication holds."""
        return self.complier_share >= 0.0

    @property
    def one_sided(self) -> bool:
        """No unit in the control arm was exposed: no always-takers by design."""
        return self.exposed_control == 0

    @property
    def perfect(self) -> bool:
        """Everyone assigned was exposed and nobody else: the two estimands coincide."""
        return self.exposed_assigned == self.n_assigned and self.exposed_control == 0

    def monotonicity(self) -> Assumption:
        """``MONOTONICITY`` with the implication the data could check applied."""
        detail = {
            "complier_share": f"{self.complier_share:.6g}",
            "rate_assigned": f"{self.rate_assigned:.6g}",
            "rate_control": f"{self.rate_control:.6g}",
        }
        base = MONOTONICITY.model_copy(update={"detail": detail})
        return base.violated() if not self.monotonic else base

    def ledger_line(self) -> LedgerLine:
        return LedgerLine(
            kind="compliance",
            statement=(
                f"exposure {self.rate_assigned:.3f} in the assigned arm against "
                f"{self.rate_control:.3f} in control: complier share "
                f"{self.complier_share:.3f}, always-takers {self.always_taker_share:.3f}, "
                f"never-takers {self.never_taker_share:.3f}"
            ),
            assumption=self.monotonicity(),
            detail={
                "n": str(self.n),
                "assignment": self.assignment,
                "exposure": self.exposure,
                "one_sided": str(self.one_sided),
                "perfect": str(self.perfect),
            },
        )


def compliance_table(frame: pd.DataFrame, assignment: str, exposure: str) -> ComplianceTable:
    """Cross-tabulate a 0/1 assignment column against a 0/1 exposure column."""
    if assignment == exposure:
        raise ValueError(f"assignment and exposure are the same column, {assignment!r}")
    assigned = _binary(frame, assignment)
    exposed = _binary(frame, exposure)
    treated_arm = assigned == 1.0
    return ComplianceTable(
        assignment=assignment,
        exposure=exposure,
        n=int(assigned.size),
        n_assigned=int(np.count_nonzero(treated_arm)),
        n_control=int(np.count_nonzero(~treated_arm)),
        exposed_assigned=int(np.count_nonzero(exposed[treated_arm] == 1.0)),
        exposed_control=int(np.count_nonzero(exposed[~treated_arm] == 1.0)),
    )


def intention_to_treat(
    frame: pd.DataFrame,
    y: str,
    assignment: str,
    covariates: str | Sequence[str] = (),
) -> LinearEstimate:
    """The effect of *being assigned*, over everybody who was assigned.

    OLS of the outcome on the assignment column. It is identified by the
    randomization alone — no exclusion restriction, no monotonicity — and it is
    the quantity a decision about *offering* the treatment needs. What it is not
    is the effect of the treatment on the units that took it.
    """
    return ols(frame, y=y, x=assignment, covariates=covariates)


def complier_effect(
    frame: pd.DataFrame,
    y: str,
    exposure: str,
    assignment: str,
    covariates: str | Sequence[str] = (),
) -> LinearEstimate | Unverified:
    """The effect of *being exposed*, over the compliers.

    2SLS of the outcome on exposure, instrumented by assignment
    (``identify.two_stage_least_squares``, so the first-stage F and
    ``weak_instrument_check`` come with it). Returns ``Unverified`` rather than
    a ratio when the first stage cannot support one: no units in an arm, an
    exposure column that does not vary, or an assignment that did not move it.
    """
    table = compliance_table(frame, assignment, exposure)
    if table.perfect:
        return Unverified(
            reason=(
                "compliance is perfect: the exposure column is the assignment column, so there "
                "is no instrumental variation to exploit and 2SLS is undefined. The "
                "intention-to-treat estimate is already the effect of exposure, over everybody"
            ),
            detail={"complier_share": "1"},
        )
    if table.complier_share == 0.0:
        return Unverified(
            reason=(
                f"assignment did not move exposure ({table.rate_assigned:.3f} against "
                f"{table.rate_control:.3f}); there are no compliers and the complier effect "
                "is a ratio with a zero denominator"
            ),
            detail={"complier_share": "0"},
        )
    if not table.monotonic:
        return Unverified(
            reason=(
                f"exposure was higher in the control arm ({table.rate_control:.3f}) than in "
                f"the assigned arm ({table.rate_assigned:.3f}); monotonicity's checkable "
                "implication fails, so the exposure difference is not a complier share"
            ),
            detail={"complier_share": f"{table.complier_share:.6g}"},
        )
    try:
        return two_stage_least_squares(
            frame, y=y, x=exposure, instruments=assignment, covariates=covariates
        )
    except ValueError as e:  # a rank-deficient or perfectly-fitting stage, named not swallowed
        return Unverified(reason=f"two-stage least squares could not be formed: {e}")


class ComplianceReport(Spec):
    """Both estimands, the table that sizes the second one's population, and the ledger.

    ``itt`` is always present — it is identified by the randomization alone.
    ``complier`` is ``None`` when the first stage could not support a ratio, and
    ``unverified`` then says why — including the good reason, that compliance
    was perfect and the two estimands are one. ``share`` is the fraction of the units the
    complier effect speaks for; the two estimates stand in the relation
    ``itt ≈ complier · share`` whenever both exist, which is the arithmetic
    behind reading one as the other.
    """

    itt: LinearEstimate
    table: ComplianceTable
    complier: LinearEstimate | None = None
    unverified: Unverified | None = None
    instrument_strength: Assumption | None = None
    mass: float = Field(default=0.95, gt=0, lt=1)

    @model_validator(mode="after")
    def _consistent(self) -> ComplianceReport:
        if (self.complier is None) == (self.unverified is None):
            raise ValueError(
                "a report carries either a complier effect or the reason there is none"
            )
        if self.complier is not None and self.instrument_strength is None:
            raise ValueError("a complier effect carries its instrument-strength assumption")
        return self

    @property
    def share(self) -> float:
        """The share of units the complier effect speaks for."""
        return self.table.complier_share

    @property
    def perfect_compliance(self) -> bool:
        """Both estimands are the same quantity; the distinction is a no-op here."""
        return self.table.perfect

    def assumptions(self) -> tuple[Assumption, ...]:
        """What the complier effect needs beyond the randomization, in order."""
        if self.complier is None:
            return ()
        out = [EXCLUSION, self.table.monotonicity()]
        if self.instrument_strength is not None:
            out.append(self.instrument_strength)
        return tuple(out)

    def verdict(self) -> Verdict:
        """Whether the effect of exposure is licensed, in the identification vocabulary.

        ``identified`` **only** under perfect compliance, where being assigned
        and being exposed are the same event and the randomization licenses the
        effect alone. With any non-compliance the strongest honest answer is
        ``downgraded``: the exclusion restriction is not checkable in the data
        and never will be. ``blocked`` when there is no complier effect to
        license — no instrumental variation, or an assumption the data
        contradict.
        """
        if self.table.perfect:
            return Verdict(
                status="identified",
                reason=(
                    "compliance is perfect, so being assigned and being exposed are the same "
                    "event: the intention-to-treat estimate is the effect of exposure over "
                    "everybody, licensed by the randomization and nothing else"
                ),
                route="randomization",
            )
        if self.complier is None:
            reason = self.unverified.reason if self.unverified is not None else "no complier effect"
            return Verdict(status="blocked", reason=reason, route="complier_effect")
        violated = [a.name for a in self.assumptions() if a.state == "violated"]
        if violated:
            return Verdict(
                status="blocked",
                reason=f"the complier effect rests on assumptions the data contradict: {violated}",
                route="complier_effect",
            )
        return Verdict(
            status="downgraded",
            reason=(
                "the complier effect is licensed by assumptions the data cannot check "
                "(exclusion, monotonicity); it is the effect on the "
                f"{self.share:.1%} of units whose exposure the assignment moved"
            ),
            assumptions=self.assumptions(),
            route="2sls",
        )

    def ledger(self) -> tuple[LedgerLine, ...]:
        """The compliance table's line, then one line naming the two estimands."""
        itt_ci = self.itt.ci(self.mass)
        lines = [self.table.ledger_line()]
        if self.complier is None:
            statement = (
                f"intention to treat {self.itt.estimate:.4g} {itt_ci.text()}; no complier "
                f"effect: {self.unverified.reason if self.unverified is not None else ''}"
            )
        else:
            complier_ci = self.complier.ci(self.mass)
            statement = (
                f"intention to treat {self.itt.estimate:.4g} {itt_ci.text()} over all "
                f"{self.table.n} units; complier effect {self.complier.estimate:.4g} "
                f"{complier_ci.text()} over the {self.share:.1%} the assignment moved — "
                "two estimands, not two estimates of one"
            )
        lines.append(
            LedgerLine(
                kind="compliance_estimands",
                statement=statement,
                assumption=EXCLUSION if self.complier is not None else None,
                detail={
                    "itt": f"{self.itt.estimate:.12g}",
                    "itt_se": f"{self.itt.se:.12g}",
                    "complier": "" if self.complier is None else f"{self.complier.estimate:.12g}",
                    "complier_share": f"{self.share:.12g}",
                },
            )
        )
        return tuple(lines)

    def summary(self) -> str:
        """The two numbers and what separates them, for a notebook or a log."""
        rows = [
            f"intention to treat  {self.itt.estimate:+.4g} {self.itt.ci(self.mass).text()}",
            f"  population        all {self.table.n} assigned units",
        ]
        if self.complier is None:
            rows.append(
                f"complier effect     none — "
                f"{self.unverified.reason if self.unverified is not None else ''}"
            )
        else:
            rows += [
                f"complier effect     {self.complier.estimate:+.4g} "
                f"{self.complier.ci(self.mass).text()}",
                f"  population        the {self.share:.1%} whose exposure assignment moved",
                f"  licensed by       {', '.join(a.name for a in self.assumptions())}",
            ]
        if self.perfect_compliance:
            rows.append("compliance is perfect: the two rows are the same quantity")
        return "\n".join(rows)


def compliance(
    frame: pd.DataFrame,
    y: str,
    assignment: str,
    exposure: str,
    covariates: str | Sequence[str] = (),
    *,
    mass: float = 0.95,
) -> ComplianceReport:
    """Both estimands from one partly-complied experiment, side by side.

    The intention-to-treat effect is always reported; the complier effect is
    reported when the first stage supports it and its reason for absence when it
    does not. Neither is presented as *the* effect, because which one answers a
    decision depends on whether the decision is to offer the treatment or to
    apply it.
    """
    if not 0.0 < mass < 1.0:
        raise ValueError(f"mass must be in (0, 1), got {mass}")
    table = compliance_table(frame, assignment, exposure)
    itt = intention_to_treat(frame, y, assignment, covariates)
    result = complier_effect(frame, y, exposure, assignment, covariates)
    if isinstance(result, Unverified):
        return ComplianceReport(itt=itt, table=table, unverified=result, mass=mass)
    return ComplianceReport(
        itt=itt,
        table=table,
        complier=result,
        instrument_strength=weak_instrument_check(result),
        mass=mass,
    )


UNIFORM_FIRST_STAGE = Assumption(
    name="uniform_first_stage",
    facet="population",
    statement=(
        "assignment moves every unit's exposure in the same direction, so the instrumented "
        "estimate is a weighted average of per-unit derivative effects with non-negative "
        "weights rather than a mixture of effects with opposing signs"
    ),
    challenged_by=(
        "a covariate stratum whose mean exposure moves the other way, which is the continuous "
        "analogue of a defier and the only implication the data can speak to"
    ),
)


class FirstStage(Spec):
    """How far the assignment moved a continuous exposure, and how reliably.

    ``shift`` is the difference in mean exposure between the arms — the
    denominator of the instrumented estimate, and the thing that makes it
    explode when it is near zero. ``standardized_shift`` puts it in units of the
    exposure's own standard deviation, which is the number to read: a shift of
    0.02 sd is a weak instrument whatever the raw units say.
    """

    assignment: NonEmptyStr
    exposure: NonEmptyStr
    n_assigned: int = Field(ge=1)
    n_control: int = Field(ge=1)
    mean_assigned: float
    mean_control: float
    sd_exposure: float = Field(ge=0)

    @model_validator(mode="after")
    def _finite(self) -> FirstStage:
        for name in ("mean_assigned", "mean_control", "sd_exposure"):
            if not np.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        return self

    @property
    def shift(self) -> float:
        """Mean exposure under assignment minus mean exposure under control."""
        return self.mean_assigned - self.mean_control

    @property
    def standardized_shift(self) -> float:
        """``shift`` in units of the exposure's own sd; ``nan`` for a constant exposure."""
        return self.shift / self.sd_exposure if self.sd_exposure > 0 else float("nan")

    @property
    def n(self) -> int:
        return self.n_assigned + self.n_control

    def monotonicity(self) -> Assumption:
        """``UNIFORM_FIRST_STAGE`` with the observed shift recorded on it."""
        return UNIFORM_FIRST_STAGE.model_copy(
            update={
                "detail": {
                    "shift": f"{self.shift:.6g}",
                    "standardized_shift": f"{self.standardized_shift:.6g}",
                    "sd_exposure": f"{self.sd_exposure:.6g}",
                }
            }
        )

    def ledger_line(self) -> LedgerLine:
        return LedgerLine(
            kind="first_stage",
            statement=(
                f"assignment moved mean {self.exposure!r} from {self.mean_control:.4g} to "
                f"{self.mean_assigned:.4g} — a shift of {self.shift:+.4g} "
                f"({self.standardized_shift:+.3f} sd)"
            ),
            assumption=self.monotonicity(),
            detail={
                "assignment": self.assignment,
                "exposure": self.exposure,
                "n": str(self.n),
            },
        )


def first_stage(frame: pd.DataFrame, exposure: str, assignment: str) -> FirstStage:
    """Mean exposure by arm, for an exposure that is a dose rather than a switch.

    Works for a binary exposure too, where ``shift`` is the complier share and
    ``compliance_table`` is the richer report.
    """
    if assignment == exposure:
        raise ValueError(f"assignment and exposure are the same column, {assignment!r}")
    assigned = _binary(frame, assignment)
    if exposure not in frame.columns:
        raise KeyError(f"columns not in frame: {[exposure]}")
    dose = np.asarray(frame[exposure].to_numpy(dtype=np.float64))
    if not np.all(np.isfinite(dose)):
        raise ValueError(f"column {exposure!r} must be finite")
    treated = assigned == 1.0
    if not treated.any() or treated.all():
        raise ValueError("a first stage needs units in both arms")
    return FirstStage(
        assignment=assignment,
        exposure=exposure,
        n_assigned=int(treated.sum()),
        n_control=int((~treated).sum()),
        mean_assigned=float(dose[treated].mean()),
        mean_control=float(dose[~treated].mean()),
        sd_exposure=float(dose.std(ddof=1)) if dose.size > 1 else 0.0,
    )


def local_derivative(
    frame: pd.DataFrame,
    y: str,
    exposure: str,
    assignment: str,
    covariates: str | Sequence[str] = (),
) -> LinearEstimate | Unverified:
    """The effect **per unit of exposure**, over the units the assignment moved.

    2SLS of the outcome on a continuous exposure, instrumented by assignment —
    the same estimator ``complier_effect`` uses, named for what it estimates
    when the exposure is a dose. It is not an average treatment effect and not a
    complier effect: it is a weighted average of per-unit derivatives, weighted
    by how much each unit's exposure moved.

    Returns ``Unverified`` when the assignment did not move the exposure at all,
    or when the two-stage fit cannot be formed.
    """
    stage = first_stage(frame, exposure, assignment)
    if stage.shift == 0.0:
        return Unverified(
            reason=(
                f"assignment did not move mean {exposure!r} at all "
                f"({stage.mean_control:.4g} in both arms); the derivative is a ratio with a "
                "zero denominator"
            ),
            detail={"shift": "0"},
        )
    try:
        return two_stage_least_squares(
            frame, y=y, x=exposure, instruments=assignment, covariates=covariates
        )
    except ValueError as e:
        return Unverified(reason=f"two-stage least squares could not be formed: {e}")


class DerivativeReport(Spec):
    """The two estimands a partly-delivered *dose* produces, beside each other.

    The continuous counterpart of ``ComplianceReport``. ``itt`` is the effect of
    being assigned, over everybody, in outcome units. ``derivative`` is the
    effect per unit of exposure, over the units the assignment moved, and they
    stand in the relation ``itt ≈ derivative · shift`` — the same arithmetic as
    the binary case with the complier share replaced by the first-stage shift.
    """

    itt: LinearEstimate
    stage: FirstStage
    derivative: LinearEstimate | None = None
    unverified: Unverified | None = None
    instrument_strength: Assumption | None = None
    mass: float = Field(default=0.95, gt=0, lt=1)

    @model_validator(mode="after")
    def _consistent(self) -> DerivativeReport:
        if (self.derivative is None) == (self.unverified is None):
            raise ValueError("a report carries either a derivative or the reason there is none")
        if self.derivative is not None and self.instrument_strength is None:
            raise ValueError("a derivative carries its instrument-strength assumption")
        return self

    @property
    def shift(self) -> float:
        """How far the assignment moved mean exposure — the ratio's denominator."""
        return self.stage.shift

    def assumptions(self) -> tuple[Assumption, ...]:
        if self.derivative is None:
            return ()
        out = [EXCLUSION, self.stage.monotonicity()]
        if self.instrument_strength is not None:
            out.append(self.instrument_strength)
        return tuple(out)

    def verdict(self) -> Verdict:
        """``downgraded`` under the named assumptions, or ``blocked`` with no first stage.

        Never ``identified``: a per-unit-of-exposure effect over units weighted
        by how far they moved is not something a randomization licenses on its
        own, and the exclusion restriction is not checkable here either.
        """
        if self.derivative is None:
            reason = self.unverified.reason if self.unverified is not None else "no derivative"
            return Verdict(status="blocked", reason=reason, route="local_derivative")
        violated = [a.name for a in self.assumptions() if a.state == "violated"]
        if violated:
            return Verdict(
                status="blocked",
                reason=f"the derivative rests on assumptions the data contradict: {violated}",
                route="local_derivative",
            )
        return Verdict(
            status="downgraded",
            reason=(
                "the derivative is licensed by assumptions the data cannot check (exclusion, "
                "a uniform first stage); it is an effect per unit of "
                f"{self.stage.exposure!r}, weighted by how far the assignment moved each unit"
            ),
            assumptions=self.assumptions(),
            route="2sls",
        )

    def ledger(self) -> tuple[LedgerLine, ...]:
        lines = [self.stage.ledger_line()]
        itt_ci = self.itt.ci(self.mass)
        if self.derivative is None:
            statement = (
                f"intention to treat {self.itt.estimate:.4g} {itt_ci.text()}; no derivative: "
                f"{self.unverified.reason if self.unverified is not None else ''}"
            )
        else:
            statement = (
                f"intention to treat {self.itt.estimate:.4g} {itt_ci.text()} per assignment "
                f"over all {self.stage.n} units; {self.derivative.estimate:.4g} "
                f"{self.derivative.ci(self.mass).text()} per unit of "
                f"{self.stage.exposure!r} — two estimands on two scales, not two estimates "
                "of one"
            )
        lines.append(
            LedgerLine(
                kind="derivative_estimands",
                statement=statement,
                assumption=EXCLUSION if self.derivative is not None else None,
                detail={
                    "itt": f"{self.itt.estimate:.12g}",
                    "derivative": (
                        "" if self.derivative is None else f"{self.derivative.estimate:.12g}"
                    ),
                    "shift": f"{self.shift:.12g}",
                },
            )
        )
        return tuple(lines)

    def summary(self) -> str:
        rows = [
            f"intention to treat  {self.itt.estimate:+.4g} {self.itt.ci(self.mass).text()}",
            f"  per               being assigned, over all {self.stage.n} units",
            f"  first stage       {self.shift:+.4g} "
            f"({self.stage.standardized_shift:+.3f} sd) of {self.stage.exposure!r}",
        ]
        if self.derivative is None:
            rows.append(
                "derivative          none — "
                f"{self.unverified.reason if self.unverified is not None else ''}"
            )
        else:
            rows += [
                f"derivative          {self.derivative.estimate:+.4g} "
                f"{self.derivative.ci(self.mass).text()}",
                f"  per               one unit of {self.stage.exposure!r}, over the units "
                "the assignment moved",
                f"  licensed by       {', '.join(a.name for a in self.assumptions())}",
            ]
        return "\n".join(rows)


def response_to_dose(
    frame: pd.DataFrame,
    y: str,
    assignment: str,
    exposure: str,
    covariates: str | Sequence[str] = (),
    *,
    mass: float = 0.95,
) -> DerivativeReport:
    """Both estimands from an experiment whose exposure is a dose rather than a switch.

    The continuous counterpart of :func:`compliance`. Use that one when a unit
    either was or was not exposed and the three response types mean something;
    use this one when exposure is an amount, where "complier" does not name a
    set of units and the instrumented estimate is a per-unit-of-exposure
    derivative instead.
    """
    if not 0.0 < mass < 1.0:
        raise ValueError(f"mass must be in (0, 1), got {mass}")
    stage = first_stage(frame, exposure, assignment)
    itt = intention_to_treat(frame, y, assignment, covariates)
    result = local_derivative(frame, y, exposure, assignment, covariates)
    if isinstance(result, Unverified):
        return DerivativeReport(itt=itt, stage=stage, unverified=result, mass=mass)
    return DerivativeReport(
        itt=itt,
        stage=stage,
        derivative=result,
        instrument_strength=weak_instrument_check(result),
        mass=mass,
    )
