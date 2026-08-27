"""Checking that the experiment was delivered, not that the model was fitted well.

Everything else under ``diagnose`` asks whether a model deserves belief. This
module asks the question that comes first and gets skipped: **did the units
arrive in the arms the design put them in?** A model diagnosed to three decimal
places on a split that silently ran 51/49 instead of 50/50 is a well-diagnosed
wrong number, and across parties whose operations differ this is the cheapest
wrong readout there is (note 0027, §D27.6.1).

Three checks, each the challenge to an assumption ``design`` already names:

* :func:`sample_ratio` — the sample-ratio check. A chi-square goodness of fit
  of the realized arm counts against the allocation's shares. Challenges
  ``random_assignment``: a split that does not match the one the randomizer was
  asked for means something happened between the randomizer and the data, and
  whatever it was is unlikely to be independent of the outcome.
* :func:`delivery` — assigned against actually exposed, per arm. Challenges
  ``exposure_known``. Equal *rates* are what matters, not equal counts: an
  intention-to-treat contrast between arms that were exposed at 90 % and 60 %
  is not the contrast anybody wrote down.
* :func:`balance` — a one-way F test per covariate on the realized arms, with a
  multiplicity correction, beside the standardized differences
  ``design.standardized_differences`` computes. Also challenges
  ``random_assignment``, and is the weakest of the three: it looks at the
  covariates somebody thought to record.

**Alpha is 0.001, not 0.05.** These checks run on every experiment, so at 0.05
one experiment in twenty fails one of them by construction and the checks stop
being read. The convention in the industry that runs the most of them is
0.001 for the sample-ratio check, and it is the default here.

**A passing check does not satisfy the assumption.** ``random_assignment``
comes back ``unverified`` when the checks pass and ``violated`` when one fails.
Absence of evidence of a broken randomizer is not evidence of a working one,
and the ledger line says which of the two it is holding.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
import numpy.typing as npt
from pydantic import Field, model_validator
from scipy import stats

from axiom.core import Assumption, LedgerLine, NonEmptyStr, Spec, Verdict
from axiom.design import ASSUMPTIONS, ArmAllocation, Assigned, BalanceRow, standardized_differences
from axiom.diagnose.structure import Correction, adjust

__all__ = [
    "SRM_ALPHA",
    "ArmCount",
    "BalanceCheck",
    "BalanceTest",
    "Delivery",
    "DeliveryReport",
    "DeliveryRow",
    "SampleRatio",
    "arm_counts",
    "balance",
    "check_delivery",
    "delivery",
    "sample_ratio",
]

Array = npt.NDArray[np.float64]

SRM_ALPHA = 0.001
"""The conventional level for a check that runs on every experiment (Kohavi et al.).

At 0.05 one experiment in twenty trips a delivery check with nothing wrong, and
a check that cries wolf weekly is a check nobody reads."""

_RANDOM_ASSIGNMENT = ASSUMPTIONS["random_assignment"]
_EXPOSURE_KNOWN = ASSUMPTIONS["exposure_known"]


def _counts_of(
    counts: Mapping[str, int] | Assigned, allocation: ArmAllocation | None
) -> tuple[ArmAllocation, tuple[int, ...]]:
    if isinstance(counts, Assigned):
        return counts.spec.allocation, counts.spec.counts
    if allocation is None:
        raise ValueError("counts given as a mapping need the allocation they are checked against")
    missing = [arm for arm in allocation.arms if arm not in counts]
    if missing:
        raise ValueError(f"no count for arm(s) {missing}; the check needs one per arm")
    extra = [arm for arm in counts if arm not in allocation.arms]
    if extra:
        raise ValueError(
            f"count for unknown arm(s) {extra}; the allocation has {list(allocation.arms)}"
        )
    values = tuple(int(counts[arm]) for arm in allocation.arms)
    if any(v < 0 for v in values):
        raise ValueError(f"counts must be non-negative, got {dict(counts)}")
    return allocation, values


class ArmCount(Spec):
    """One arm's realized count against the count its share implies."""

    arm: NonEmptyStr
    observed: int = Field(ge=0)
    expected: float = Field(ge=0)
    share: float = Field(gt=0, lt=1)

    @property
    def deviation(self) -> float:
        """``observed - expected``, in units."""
        return self.observed - self.expected


class SampleRatio(Spec):
    """The sample-ratio check: are the arms the size the allocation asked for?

    ``p_value`` is the chi-square goodness-of-fit tail against the allocation's
    shares with ``n_arms - 1`` degrees of freedom. ``mismatch`` is
    ``p_value < alpha``, and ``alpha`` defaults to :data:`SRM_ALPHA` rather
    than to 0.05.
    """

    arms: tuple[ArmCount, ...]
    n: int = Field(ge=0)
    chi_square: float = Field(ge=0)
    df: int = Field(ge=1)
    p_value: float = Field(ge=0, le=1)
    alpha: float = Field(gt=0, lt=1)
    mismatch: bool

    @model_validator(mode="after")
    def _consistent(self) -> SampleRatio:
        if self.mismatch != (self.p_value < self.alpha):
            raise ValueError("mismatch must equal p_value < alpha")
        if sum(a.observed for a in self.arms) != self.n:
            raise ValueError("arm counts must sum to n")
        return self

    @property
    def worst_arm(self) -> str:
        """The arm furthest from its expected count, by absolute deviation."""
        return max(self.arms, key=lambda a: abs(a.deviation)).arm

    def realized_share(self, arm: str) -> float:
        """The share one arm actually got, against ``ArmCount.share`` for the target."""
        for row in self.arms:
            if row.arm == arm:
                return row.observed / self.n if self.n else float("nan")
        raise KeyError(f"no arm {arm!r}; have {[a.arm for a in self.arms]}")

    def assumption(self) -> Assumption:
        """``random_assignment``, ``violated`` on a mismatch and ``unverified`` otherwise."""
        return _RANDOM_ASSIGNMENT.violated() if self.mismatch else _RANDOM_ASSIGNMENT

    def ledger_line(self) -> LedgerLine:
        observed = ", ".join(f"{a.arm} {a.observed}" for a in self.arms)
        expected = ", ".join(f"{a.arm} {a.expected:.1f}" for a in self.arms)
        verdict = "MISMATCH" if self.mismatch else "no mismatch"
        return LedgerLine(
            kind="sample_ratio",
            statement=(
                f"sample ratio: {verdict} (chi-square {self.chi_square:.3f} on {self.df} df, "
                f"p = {self.p_value:.3g} against alpha {self.alpha:g}); observed {observed} "
                f"against expected {expected}"
            ),
            assumption=self.assumption(),
            detail={
                "n": str(self.n),
                "p_value": f"{self.p_value:.12g}",
                "alpha": f"{self.alpha:g}",
                "worst_arm": self.worst_arm,
            },
        )


def sample_ratio(
    counts: Mapping[str, int] | Assigned,
    allocation: ArmAllocation | None = None,
    *,
    alpha: float = SRM_ALPHA,
) -> SampleRatio:
    """Chi-square goodness of fit of the realized arm counts against the allocation.

    ``counts`` is either an ``Assigned`` (whose allocation is used) or a
    mapping of arm to realized count with the ``allocation`` it is checked
    against — the second form is the one that matters, because the counts that
    need checking come from the outcome table, not from the randomizer.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    alloc, observed = _counts_of(counts, allocation)
    n = int(sum(observed))
    expected = alloc.expected_counts(n)
    if n == 0:
        raise ValueError("the sample-ratio check needs at least one unit")
    chi_square = math.fsum(
        (o - e) ** 2 / e for o, e in zip(observed, expected, strict=True) if e > 0
    )
    df = alloc.n_arms - 1
    p_value = float(stats.chi2.sf(chi_square, df))
    return SampleRatio(
        arms=tuple(
            ArmCount(arm=arm, observed=o, expected=e, share=s)
            for arm, o, e, s in zip(alloc.arms, observed, expected, alloc.shares, strict=True)
        ),
        n=n,
        chi_square=float(chi_square),
        df=df,
        p_value=p_value,
        alpha=alpha,
        mismatch=p_value < alpha,
    )


class DeliveryRow(Spec):
    """One arm's exposure: how many units were assigned to it and how many it reached."""

    arm: NonEmptyStr
    assigned: int = Field(ge=0)
    exposed: int = Field(ge=0)

    @model_validator(mode="after")
    def _consistent(self) -> DeliveryRow:
        if self.exposed > self.assigned:
            raise ValueError(
                f"arm {self.arm!r}: {self.exposed} exposed of {self.assigned} assigned; "
                "a unit cannot be exposed without being assigned"
            )
        return self

    @property
    def rate(self) -> float:
        """Exposed over assigned; ``nan`` for an arm with no units."""
        return self.exposed / self.assigned if self.assigned else float("nan")


class Delivery(Spec):
    """Whether the arms were *reached* at the same rate, not whether they were equal size.

    ``differential`` is the largest gap between any two arms' exposure rates.
    ``p_value`` is the chi-square test of independence on the arm-by-exposed
    table: the tail probability of a gap this large if every arm were reached
    at one common rate.
    """

    arms: tuple[DeliveryRow, ...]
    differential: float = Field(ge=0)
    chi_square: float = Field(ge=0)
    df: int = Field(ge=1)
    p_value: float = Field(ge=0, le=1)
    alpha: float = Field(gt=0, lt=1)
    differential_delivery: bool

    @model_validator(mode="after")
    def _consistent(self) -> Delivery:
        if self.differential_delivery != (self.p_value < self.alpha):
            raise ValueError("differential_delivery must equal p_value < alpha")
        return self

    @property
    def overall_rate(self) -> float:
        assigned = sum(a.assigned for a in self.arms)
        return sum(a.exposed for a in self.arms) / assigned if assigned else float("nan")

    def assumption(self) -> Assumption:
        """``exposure_known``, ``violated`` when the arms were reached at different rates."""
        return _EXPOSURE_KNOWN.violated() if self.differential_delivery else _EXPOSURE_KNOWN

    def ledger_line(self) -> LedgerLine:
        rates = ", ".join(f"{a.arm} {a.rate:.3f}" for a in self.arms)
        verdict = "DIFFERENTIAL" if self.differential_delivery else "even"
        return LedgerLine(
            kind="delivery",
            statement=(
                f"delivery: {verdict} exposure across arms ({rates}); largest gap "
                f"{self.differential:.3f}, chi-square {self.chi_square:.3f} on {self.df} df, "
                f"p = {self.p_value:.3g} against alpha {self.alpha:g}"
            ),
            assumption=self.assumption(),
            detail={
                "overall_rate": f"{self.overall_rate:.6g}",
                "differential": f"{self.differential:.6g}",
                "p_value": f"{self.p_value:.12g}",
            },
        )


def delivery(
    assigned: Mapping[str, int],
    exposed: Mapping[str, int],
    *,
    alpha: float = SRM_ALPHA,
) -> Delivery:
    """Assigned against exposed, per arm, and a test that the rates are one rate.

    Both mappings are keyed by arm name and must name the same arms. An arm
    whose assigned count is zero contributes nothing to the test and comes back
    with a ``nan`` rate rather than a zero one.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if set(assigned) != set(exposed):
        raise ValueError(
            f"assigned names {sorted(assigned)} and exposed names {sorted(exposed)}; "
            "they must be the same arms"
        )
    if len(assigned) < 2:
        raise ValueError("a delivery check needs at least two arms")
    rows = tuple(
        DeliveryRow(arm=arm, assigned=int(assigned[arm]), exposed=int(exposed[arm]))
        for arm in sorted(assigned)
    )
    live = [r for r in rows if r.assigned > 0]
    if len(live) < 2:
        raise ValueError("a delivery check needs at least two arms with units in them")
    table = np.asarray([[r.exposed, r.assigned - r.exposed] for r in live], dtype=np.float64)
    rates = [r.rate for r in live]
    differential = float(max(rates) - min(rates))
    if table[:, 0].sum() == 0 or table[:, 1].sum() == 0:
        # Every unit exposed, or none: the table is degenerate and there is no gap to test.
        chi_square, p_value, df = 0.0, 1.0, len(live) - 1
    else:
        chi_square, p_value, df, _ = stats.chi2_contingency(table, correction=False)
    return Delivery(
        arms=rows,
        differential=differential,
        chi_square=float(chi_square),
        df=int(df),
        p_value=float(p_value),
        alpha=alpha,
        differential_delivery=float(p_value) < alpha,
    )


class BalanceTest(Spec):
    """One covariate's standardized difference and the test that goes with it."""

    covariate: NonEmptyStr
    smd: float = Field(ge=0)
    f_statistic: float = Field(ge=0)
    p_value: float = Field(ge=0, le=1)
    adjusted_p: float = Field(ge=0, le=1)
    imbalanced: bool


class BalanceCheck(Spec):
    """Every recorded covariate's balance across arms, corrected for testing many.

    ``correction`` defaults to Holm: the question is whether *any* covariate is
    out of balance, which is a family-wise question. ``skipped`` names
    covariates that were constant over the units — there is nothing for them to
    be imbalanced about, and they are reported rather than counted as passes.
    """

    tests: tuple[BalanceTest, ...]
    rows: tuple[BalanceRow, ...]
    correction: Correction
    alpha: float = Field(gt=0, lt=1)
    imbalanced: tuple[str, ...]
    skipped: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _consistent(self) -> BalanceCheck:
        expected = tuple(t.covariate for t in self.tests if t.imbalanced)
        if self.imbalanced != expected:
            raise ValueError("imbalanced must list exactly the covariates that failed")
        return self

    @property
    def worst_smd(self) -> float:
        return max((t.smd for t in self.tests), default=0.0)

    def assumption(self) -> Assumption:
        return _RANDOM_ASSIGNMENT.violated() if self.imbalanced else _RANDOM_ASSIGNMENT

    def ledger_line(self) -> LedgerLine:
        failed = ", ".join(self.imbalanced) if self.imbalanced else "none"
        return LedgerLine(
            kind="balance",
            statement=(
                f"balance: {len(self.tests)} covariate(s) tested with a {self.correction} "
                f"correction at alpha {self.alpha:g}; imbalanced: {failed}; largest "
                f"standardized difference {self.worst_smd:.3f}"
            ),
            assumption=self.assumption(),
            detail={
                "correction": self.correction,
                "skipped": ", ".join(self.skipped),
                "worst_smd": f"{self.worst_smd:.6g}",
            },
        )


def balance(
    covariates: Mapping[str, npt.ArrayLike],
    arm_of: npt.ArrayLike | Assigned,
    allocation: ArmAllocation | None = None,
    *,
    alpha: float = SRM_ALPHA,
    correction: Correction = "holm",
) -> BalanceCheck:
    """Standardized differences plus a one-way F test per covariate, corrected.

    The F test asks whether the arm means of one covariate differ by more than
    the within-arm spread predicts. It is the weakest of the three checks in
    this module, because it can only look at the covariates somebody recorded —
    a passing balance table says nothing about the ones nobody wrote down.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if isinstance(arm_of, Assigned):
        allocation, index = arm_of.spec.allocation, arm_of.arm_of
    else:
        if allocation is None:
            raise ValueError("an arm index needs the allocation naming its arms")
        index = np.asarray(arm_of, dtype=np.int64)
    if not covariates:
        raise ValueError("a balance check needs at least one covariate")

    rows = standardized_differences(covariates, index, allocation)
    by_name = {row.covariate: row for row in rows}
    names, p_values, statistics, skipped = [], [], [], []
    for name in sorted(covariates):
        x = np.asarray(covariates[name], dtype=np.float64)
        groups = [x[index == a] for a in range(allocation.n_arms) if np.any(index == a)]
        if by_name[name].sd == 0.0 or len(groups) < 2 or any(g.size < 2 for g in groups):
            skipped.append(name)
            continue
        f_statistic, p_value = stats.f_oneway(*groups)
        if not math.isfinite(float(f_statistic)):
            skipped.append(name)
            continue
        names.append(name)
        statistics.append(float(f_statistic))
        p_values.append(float(p_value))

    adjusted = adjust(p_values, correction)
    tests = tuple(
        BalanceTest(
            covariate=name,
            smd=by_name[name].smd,
            f_statistic=stat,
            p_value=p,
            adjusted_p=q,
            imbalanced=q < alpha,
        )
        for name, stat, p, q in zip(names, statistics, p_values, adjusted, strict=True)
    )
    return BalanceCheck(
        tests=tests,
        rows=rows,
        correction=correction,
        alpha=alpha,
        imbalanced=tuple(t.covariate for t in tests if t.imbalanced),
        skipped=tuple(skipped),
    )


class DeliveryReport(Spec):
    """The three checks together, and the one verdict a readout should carry.

    ``verdict`` is ``identified`` when every check that ran passed —
    ``route`` naming which ones did — and ``blocked`` when any failed, with
    every failure named in the reason. It is never ``downgraded``: a delivery
    failure is not something an assumption licenses you past.
    """

    ratio: SampleRatio
    exposure: Delivery | None = None
    covariates: BalanceCheck | None = None

    @property
    def failures(self) -> tuple[str, ...]:
        out = []
        if self.ratio.mismatch:
            out.append(f"sample ratio (p = {self.ratio.p_value:.3g}, worst {self.ratio.worst_arm})")
        if self.exposure is not None and self.exposure.differential_delivery:
            out.append(f"delivery (gap {self.exposure.differential:.3f})")
        if self.covariates is not None and self.covariates.imbalanced:
            out.append(f"balance ({', '.join(self.covariates.imbalanced)})")
        return tuple(out)

    @property
    def checks_run(self) -> tuple[str, ...]:
        out = ["sample_ratio"]
        if self.exposure is not None:
            out.append("delivery")
        if self.covariates is not None:
            out.append("balance")
        return tuple(out)

    def verdict(self) -> Verdict:
        failures = self.failures
        if failures:
            return Verdict(
                status="blocked",
                reason="the experiment was not delivered as assigned: " + "; ".join(failures),
                route="delivery",
            )
        return Verdict(status="identified", route="+".join(self.checks_run))

    def ledger(self) -> tuple[LedgerLine, ...]:
        """One line per check that ran, in the order they were run."""
        lines = [self.ratio.ledger_line()]
        if self.exposure is not None:
            lines.append(self.exposure.ledger_line())
        if self.covariates is not None:
            lines.append(self.covariates.ledger_line())
        return tuple(lines)

    def summary(self) -> str:
        verdict = self.verdict()
        head = f"delivery: {verdict.status} ({', '.join(self.checks_run)})"
        return head if not verdict.reason else f"{head}\n  {verdict.reason}"


def check_delivery(
    counts: Mapping[str, int] | Assigned,
    allocation: ArmAllocation | None = None,
    *,
    exposed: Mapping[str, int] | None = None,
    covariates: Mapping[str, npt.ArrayLike] | None = None,
    arm_of: npt.ArrayLike | Assigned | None = None,
    alpha: float = SRM_ALPHA,
    correction: Correction = "holm",
) -> DeliveryReport:
    """Every delivery check the arguments support, and one verdict over them.

    The sample-ratio check always runs. ``exposed`` adds the delivery check;
    ``covariates`` (with ``arm_of``, or with ``counts`` given as an
    ``Assigned``) adds the balance check. What is not supplied is not silently
    passed — ``DeliveryReport.checks_run`` names what actually ran, and the
    verdict's ``route`` carries it.
    """
    ratio = sample_ratio(counts, allocation, alpha=alpha)
    exposure = None
    if exposed is not None:
        assigned = {a.arm: a.observed for a in ratio.arms}
        exposure = delivery(assigned, exposed, alpha=alpha)
    covariate_check = None
    if covariates:
        index: npt.ArrayLike | Assigned
        if arm_of is not None:
            index = arm_of
        elif isinstance(counts, Assigned):
            index = counts
        else:
            raise ValueError(
                "covariates need arm_of (or counts given as an Assigned) to know which arm "
                "each unit was in"
            )
        covariate_check = balance(covariates, index, allocation, alpha=alpha, correction=correction)
    return DeliveryReport(ratio=ratio, exposure=exposure, covariates=covariate_check)


def arm_counts(arms: Sequence[str], allocation: ArmAllocation) -> dict[str, int]:
    """Count a column of realized arm labels into the mapping the checks take.

    An arm label the allocation does not name is an error, not a zero: a column
    with ``"treatment"`` where the allocation says ``"treated"`` is a data
    problem the check exists to surface.
    """
    counts = dict.fromkeys(allocation.arms, 0)
    unknown = sorted({a for a in arms if a not in counts})
    if unknown:
        raise ValueError(
            f"arm label(s) {unknown} are not in the allocation {list(allocation.arms)}"
        )
    for a in arms:
        counts[a] += 1
    return counts
