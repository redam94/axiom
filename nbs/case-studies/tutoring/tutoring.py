"""TUTOR-60: the synthetic world the tutoring case-study notebooks share.

One data-generating process, one costing, and one small plotting vocabulary, so
that five notebooks carry a single decision from "what are we even asking" to
"here is what we recommend and here is what would change it".

**The decision.** A state has 9.0 million dollars for one school year and 12,000
students reading two or more grade levels behind. It can buy two things for those
students: *structured small-group tutoring*, sold in weekly minutes, and
*caregiver messaging*, sold in weekly messages. It has to choose how much of each
to buy per student -- and, implicitly, how many students it can afford to reach.

**Why the doses are in dollars.** Both treatments are dosed here in *dollars per
eligible student per school year*, not in minutes and messages. The state's
decision variable is the per-student allocation; putting both treatments on one
scale is what makes the budget an actual constraint rather than a footnote, and
it is what lets ``axiom.surface.allocate`` answer the question that was asked.
The operational reading is one multiplication away and every figure carries it:
``MINUTE_COST`` dollars buys one weekly minute of tutoring for one student for a
year, ``MESSAGE_COST`` dollars buys one weekly message.

**The truth the notebooks are trying to recover.** Annualized reading gain for a
school allocating ``t`` dollars to tutoring and ``m`` dollars to messaging is

    school_effect + tutoring_gain(t) + messaging_gain(m) + noise

where ``tutoring_gain`` is a *sigmoid* in dose -- a tutoring block shorter than
about twenty minutes is not a session, so there is very little to be had at the
bottom -- minus a quadratic penalty for the instructional time the block displaces,
and ``messaging_gain`` saturates almost immediately and is capped by how many
messages a caregiver will read.

Two consequences drive the whole case study, and neither is visible in the effect
curve alone:

1. The dose that helps a student most is not the dose that helps the most
   students. Effect peaks near 90 weekly minutes; *gain per dollar* peaks near 45,
   because the budget is fixed and every additional minute is another student not
   served.
2. The cheap treatment should be bought to its cap before the expensive one gets
   anything, and after that the answer is remarkably flat -- which is a finding,
   not a failure, and is the kind of thing a recommendation has to say out loud.

Nothing here is imported by ``axiom``; it is notebook scaffolding, and every
statistical claim it supports is computed with ``axiom`` in the notebooks.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

Array = npt.NDArray[np.float64]

# -- the program -----------------------------------------------------------------------

#: Dollars per eligible student per year that buy one weekly minute of tutoring.
MINUTE_COST = 20.0
#: Dollars per eligible student per year that buy one weekly caregiver message.
MESSAGE_COST = 9.0
#: Caregivers stop reading past this; the vendor will not sell more.
MESSAGE_CAP_PER_WEEK = 5
#: The most tutoring any school in the trial was willing to timetable, in weekly minutes.
MINUTES_MAX = 150

TUTORING_MAX = MINUTES_MAX * MINUTE_COST  # 3000 USD / student-year
MESSAGING_MAX = MESSAGE_CAP_PER_WEEK * MESSAGE_COST  # 45 USD / student-year

DOSE_UNIT = "USD/student-year"
OUTCOME_UNIT = "points/year"

#: The statewide program the decision is about.
BUDGET = 9_000_000.0
N_ELIGIBLE = 12_000

#: What a point is worth. One scale point is about 0.045 student standard deviations, and
#: the department values a standard deviation of reading achievement at 31,000 USD of
#: discounted lifetime earnings. Every dollar figure in the case study is this number
#: times a number of points, and notebook 5 shows what happens when it is wrong.
VALUE_PER_POINT = 1_400.0

#: The trial. Every school in the state's lowest-performing tier that agreed to be
#: randomized -- which is what it takes: the effect being chased is a few points on a
#: scale whose school-to-school spread is five, and schools are the unit of assignment.
N_SCHOOLS = 120
BENCHMARKS = 6
#: Eligible students per school. 240 schools in the tier, 12,000 eligible statewide.
ELIGIBLE_PER_SCHOOL = (35, 65)


def minutes(tutoring: Array | float) -> Any:
    """Weekly tutoring minutes per student that a per-student dollar dose buys."""
    return np.asarray(tutoring, dtype=np.float64) / MINUTE_COST


def messages(messaging: Array | float) -> Any:
    """Weekly caregiver messages per student that a per-student dollar dose buys."""
    return np.asarray(messaging, dtype=np.float64) / MESSAGE_COST


@dataclass(frozen=True)
class Truth:
    """Every parameter of the data-generating process, in points/year and dollars."""

    # tutoring: sigmoid in dose, minus the instructional time the block displaces
    tutoring_max_gain: float = 7.0
    tutoring_half: float = 900.0  # 45 weekly minutes
    tutoring_shape: float = 2.2  # > 1: a short block is not a session
    displacement: float = 3.0  # points lost at the very top of the range
    # messaging: saturates almost at once, then the cap binds
    messaging_max_gain: float = 2.2
    messaging_half: float = 10.8  # 1.2 weekly messages
    # what the two treatments do together, over and above what each does alone
    interaction: float = 0.0
    # Variance, in four places, because which of them the analysis can remove is
    # the whole precision argument in notebook 2.
    #   school_sd     persistent: this school gains more every year, for reasons
    #                 nobody in the data can name. Cancels out of a change score.
    #   year_sd       this school, this year. Does not cancel.
    #   benchmark_sd  measurement, benchmark to benchmark. Averages away over six.
    #   student_sd    within a school; sets the intraclass correlation with
    #                 school_sd, and is what makes cluster assignment expensive.
    school_sd: float = 4.5
    year_sd: float = 1.8
    benchmark_sd: float = 3.0
    student_sd: float = 22.0
    # schools do not deliver every minute they were assigned
    fidelity_mean: float = 0.93
    fidelity_concentration: float = 60.0
    # the prior-year observational record
    resources_sd: float = 1.0
    resources_on_gain: float = 3.2
    resources_on_tutoring: float = 620.0
    poverty_on_tutoring: float = -450.0
    poverty_on_gain: float = -2.6
    poverty_resources_correlation: float = -0.55
    baseline_gain: float = 21.0


TRUTH = Truth()


# -- the response ----------------------------------------------------------------------


def tutoring_gain(tutoring: Array | float, truth: Truth = TRUTH) -> Any:
    """Annualized reading points from ``tutoring`` dollars per student.

    Sigmoid in dose with shape > 1, less a quadratic penalty for displaced
    instruction. The penalty is what makes the curve turn over near 90 weekly
    minutes -- no monotone response family can represent that, which is the
    modelling decision notebook 3 has to make.
    """
    d = np.maximum(np.asarray(tutoring, dtype=np.float64), 0.0)
    rise = d**truth.tutoring_shape / (
        truth.tutoring_half**truth.tutoring_shape + d**truth.tutoring_shape
    )
    return truth.tutoring_max_gain * rise - truth.displacement * (d / TUTORING_MAX) ** 2


def messaging_gain(messaging: Array | float, truth: Truth = TRUTH) -> Any:
    """Annualized reading points from ``messaging`` dollars per student."""
    d = np.maximum(np.asarray(messaging, dtype=np.float64), 0.0)
    return truth.messaging_max_gain * d / (truth.messaging_half + d)


def mean_gain(tutoring: Array | float, messaging: Array | float, truth: Truth = TRUTH) -> Any:
    """The whole response surface: what a school allocating this much would gain."""
    t = np.asarray(tutoring, dtype=np.float64)
    m = np.asarray(messaging, dtype=np.float64)
    both = truth.interaction * (t / TUTORING_MAX) * (m / MESSAGING_MAX)
    return tutoring_gain(t, truth) + messaging_gain(m, truth) + both


# -- the money -------------------------------------------------------------------------


def cost_per_student(tutoring: Array | float, messaging: Array | float) -> Any:
    """What one served student costs for a year. The doses *are* the cost."""
    return np.asarray(tutoring, dtype=np.float64) + np.asarray(messaging, dtype=np.float64)


def students_served(
    tutoring: Array | float,
    messaging: Array | float,
    *,
    budget: float = BUDGET,
    n_eligible: int = N_ELIGIBLE,
) -> Any:
    """How many of the eligible students the budget reaches at this per-student cost.

    Capped at the eligible cohort: past that point money has nowhere left to go,
    which is exactly why the objective is not linear in the dose.
    """
    cost = np.maximum(cost_per_student(tutoring, messaging), 1e-9)
    return np.minimum(np.asarray(n_eligible, dtype=np.float64), budget / cost)


def cohort_points(
    tutoring: Array | float,
    messaging: Array | float,
    *,
    budget: float = BUDGET,
    n_eligible: int = N_ELIGIBLE,
    gain: Array | float | None = None,
    truth: Truth = TRUTH,
) -> Any:
    """Total reading points the whole program delivers -- the decision's objective.

    ``gain`` lets a *posterior* response take the place of the truth, which is how
    every figure in notebook 5 gets its interval.
    """
    per_student = (
        mean_gain(tutoring, messaging, truth)
        if gain is None
        else np.asarray(gain, dtype=np.float64)
    )
    return students_served(tutoring, messaging, budget=budget, n_eligible=n_eligible) * per_student


# -- the prior-year observational record -----------------------------------------------


def prior_year(n_schools: int = 240, *, seed: int, truth: Truth = TRUTH) -> pd.DataFrame:
    """What the state already has: last year, nobody randomized anything.

    Schools chose their own tutoring intensity. Better-resourced schools chose more
    of it *and* would have gained more anyway, and the only proxy for resources in
    the administrative record is the share of students in poverty -- which is a
    proxy, not the thing. The naive slope is biased upward; adjusting for the proxy
    removes some of the bias and not all of it. Notebook 1 is about why no amount of
    care with these columns fixes that.
    """
    rng = np.random.default_rng(seed)
    resources = rng.normal(0.0, truth.resources_sd, size=n_schools)
    poverty = np.clip(
        0.62
        + truth.poverty_resources_correlation * 0.11 * resources
        + rng.normal(0.0, 0.11, size=n_schools),
        0.15,
        0.98,
    )
    tutoring = np.clip(
        700.0
        + truth.resources_on_tutoring * resources
        + truth.poverty_on_tutoring * (poverty - 0.62)
        + rng.normal(0.0, 260.0, size=n_schools),
        0.0,
        TUTORING_MAX,
    )
    gain = (
        truth.baseline_gain
        + tutoring_gain(tutoring, truth)
        + truth.resources_on_gain * resources
        + truth.poverty_on_gain * (poverty - 0.62)
        + rng.normal(0.0, truth.school_sd, size=n_schools)
    )
    return pd.DataFrame(
        {
            "school": [f"prior_{i:03d}" for i in range(n_schools)],
            "poverty_share": poverty,
            "tutoring": tutoring,
            "minutes_per_week": minutes(tutoring),
            "gain": gain,
            # Recorded here so a notebook can *show* the confounder it is not allowed
            # to use. Nothing that estimates anything is permitted to touch it.
            "resources_unobserved": resources,
        }
    )


# -- the design ------------------------------------------------------------------------

#: The three messaging levels the trial buys: none, a couple a week, the vendor cap.
#: Messaging is cheap, capped, and saturates almost at once, so the decision only ever
#: needs to know whether the cap beats nothing -- three levels answer that far better
#: than six. Notebook 2 is the argument.
MESSAGING_LEVELS: tuple[float, ...] = (0.0, 18.0, MESSAGING_MAX)

#: Tutoring can only be timetabled in five-minute blocks, and messaging is sold by the
#: message, so a design point that is not on this lattice cannot be run.
TUTORING_STEP = 5 * MINUTE_COST
MESSAGING_STEP = MESSAGE_COST

#: The knots notebook 2 designs against: eight of them, deliberately more than the
#: five the analysis expects to need. A design chosen against the model you hope is
#: right cannot tell you whether it is -- see notebook 2 section 4.
PLANNING_KNOTS: tuple[float, ...] = (200.0, 500.0, 850.0, 1200.0, 1600.0, 2050.0, 2500.0, 2800.0)
MESSAGING_KNOTS: tuple[float, ...] = (9.0, 18.0, 36.0)


def snap(points: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    """Round a design onto the lattice schools can actually deliver."""
    return [
        (
            float(np.round(t / TUTORING_STEP) * TUTORING_STEP),
            float(np.round(m / MESSAGING_STEP) * MESSAGING_STEP),
        )
        for t, m in points
    ]


def spec(kernels: Mapping[str, Any], *, name: str = "tutor60", **overrides: Any) -> Any:
    """A ``SurfaceSpec`` over the two treatments, with the case study's prior scales.

    ``intercept="shared"`` is not a shortcut. Every school holds one allocation for the
    whole year, so a per-school intercept is collinear with the treatment and a
    hierarchical one is nearly so; the between-school variance belongs in the residual,
    which is exactly the price notebook 2 charges cluster assignment for.
    """
    from axiom.surface import SurfaceSpec

    fields: dict[str, Any] = {
        "name": name,
        "treatments": treatments(),
        "outcome": outcome(),
        "kernels": dict(kernels),
        "intercept": "shared",
        "unit_column": "school",
        "time_column": "benchmark",
        "intercept_scale": 6.0,
        "noise_scale": 6.0,
    }
    return SurfaceSpec(**{**fields, **overrides})


def planning_spec() -> Any:
    """The linear-in-parameters surface notebook 2 evaluates candidate designs against."""
    from axiom.surface import SplineKernel

    return spec(
        {
            "tutoring": SplineKernel(
                reference_dose=TUTORING_MAX, amplitude_scale=8.0, knots=PLANNING_KNOTS
            ),
            "messaging": SplineKernel(
                reference_dose=MESSAGING_MAX, amplitude_scale=2.0, knots=MESSAGING_KNOTS
            ),
        },
        name="tutor60_planning",
    )


def lattice() -> Any:
    """Every allocation a school could actually be assigned: the exchange candidate set."""
    from axiom.surface import Design

    return Design(
        treatments=("tutoring", "messaging"),
        points=tuple(
            (float(t), float(m))
            for t in np.arange(0.0, TUTORING_MAX + 1.0, TUTORING_STEP)
            for m in MESSAGING_LEVELS
        ),
        kind="lattice",
    )


def chosen_design(*, n_schools: int = N_SCHOOLS, seed: int = 3) -> list[tuple[float, float]]:
    """The design notebook 2 selects: Fedorov D-optimal over the deliverable lattice.

    Optimal against :data:`PLANNING_KNOTS` -- a spline with more bends than the analysis
    expects to need -- so that the design does not quietly assume the answer. Notebook 2
    scores it against the alternatives, including the five-arm factorial that beats it by
    a fifth *if* the five-knot model is right and is singular if it is not.
    """
    from axiom.core import is_failure
    from axiom.surface import Surface, optimal_exchange

    surface = Surface(planning_spec())
    design = optimal_exchange(surface, lattice(), n_schools, [{}], criterion="d", seed=seed)
    if is_failure(design):
        raise RuntimeError(f"the trial has no feasible design: {design}")
    return [(float(a), float(b)) for a, b in np.asarray(design.points)]


# -- the trial -------------------------------------------------------------------------


def assign(
    points: Sequence[tuple[float, float]], *, seed: int, truth: Truth = TRUTH
) -> pd.DataFrame:
    """Randomize schools to the design points. One school, one allocation, all year.

    Schools are randomized to *allocations*, not to arms: the design is a set of
    points in the two-treatment dose region and notebook 2 chooses it. Each school
    gets an eligible-student count, a prior-year gain, and a delivery fidelity --
    the share of the assigned tutoring it actually manages to timetable.
    """
    rng = np.random.default_rng(seed)
    n = len(points)
    order = rng.permutation(n)
    eligible = rng.integers(ELIGIBLE_PER_SCHOOL[0], ELIGIBLE_PER_SCHOOL[1] + 1, size=n)
    fidelity = rng.beta(
        truth.fidelity_mean * truth.fidelity_concentration,
        (1.0 - truth.fidelity_mean) * truth.fidelity_concentration,
        size=n,
    )
    # The persistent part of a school -- it shows up in the prior year and again
    # in this one, which is exactly why the prior year is worth measuring.
    school_effect = rng.normal(0.0, truth.school_sd, size=n)
    prior_shock = rng.normal(0.0, truth.year_sd, size=n)
    rows = []
    for i, slot in enumerate(order):
        tutoring, messaging = points[int(slot)]
        rows.append(
            {
                "school": f"school_{i:02d}",
                "tutoring": float(tutoring),
                "messaging": float(messaging),
                "minutes_per_week": float(minutes(tutoring)),
                "messages_per_week": float(messages(messaging)),
                "cost_per_student": float(cost_per_student(tutoring, messaging)),
                "eligible": int(eligible[i]),
                "school_effect": float(school_effect[i]),
                "prior_year_gain": float(truth.baseline_gain + school_effect[i] + prior_shock[i]),
                "fidelity": float(fidelity[i]),
            }
        )
    return pd.DataFrame(rows)


def benchmark_frame(schools: pd.DataFrame, *, seed: int, truth: Truth = TRUTH) -> pd.DataFrame:
    """Six benchmark assessments per school, as annualized gain and as growth.

    Each benchmark window reports the gain since the previous one, scaled to a
    year, which is how growth measures are read and which keeps every number in
    the case study in one unit.

    Two columns, and the difference between them is the reason notebook 2 spends a
    section on it. ``gain`` is what the school gained. ``growth`` is that less the
    same school's prior-year gain, which cancels the persistent school effect and
    leaves a quantity roughly three times more precise -- worth more than doubling
    the number of schools, and free, because the prior year was measured anyway.
    """
    rng = np.random.default_rng(seed)
    year_shock = rng.normal(0.0, truth.year_sd, size=len(schools))
    rows = []
    for i, record in enumerate(schools.to_dict("records")):
        # Fidelity attenuates the tutoring actually delivered. It is a consequence
        # of the assignment, so nothing downstream is allowed to condition on it:
        # the estimand is what the *assigned* allocation achieves.
        delivered = float(record["tutoring"]) * float(record["fidelity"])
        mean = (
            truth.baseline_gain
            + mean_gain(delivered, float(record["messaging"]), truth)
            + float(record["school_effect"])
            + year_shock[i]
        )
        for period in range(BENCHMARKS):
            gain = float(mean + rng.normal(0.0, truth.benchmark_sd))
            rows.append(
                {
                    "school": record["school"],
                    "benchmark": period,
                    "tutoring": float(record["tutoring"]),
                    "messaging": float(record["messaging"]),
                    "gain": gain,
                    "growth": gain - float(record["prior_year_gain"]),
                }
            )
    return pd.DataFrame(rows)


def student_frame(schools: pd.DataFrame, *, seed: int, truth: Truth = TRUTH) -> pd.DataFrame:
    """One row per eligible student: the frame the ICC and the attendance rate come from.

    Students inside a school share the school effect, which is the whole reason
    the trial randomizes schools and pays for it in effective sample size.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for record in schools.to_dict("records"):
        n = int(record["eligible"])
        delivered = float(record["tutoring"]) * float(record["fidelity"])
        mean = truth.baseline_gain + mean_gain(delivered, float(record["messaging"]), truth)
        attended = rng.beta(9.0, 2.0, size=n) if record["tutoring"] > 0 else np.zeros(n)
        gains = mean + float(record["school_effect"]) + rng.normal(0.0, truth.student_sd, size=n)
        rows.append(
            pd.DataFrame(
                {
                    "school": record["school"],
                    "student": [f"{record['school']}_{j:03d}" for j in range(n)],
                    "tutoring": float(record["tutoring"]),
                    "messaging": float(record["messaging"]),
                    "attendance": attended,
                    "gain": gains,
                    "growth": gains - float(record["prior_year_gain"]),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


@dataclass(frozen=True)
class Program:
    """One realized trial: its schools, its benchmarks, its students, and the truth."""

    schools: pd.DataFrame
    benchmarks: pd.DataFrame
    students: pd.DataFrame
    truth: Truth
    design: tuple[tuple[float, float], ...]

    @property
    def n_schools(self) -> int:
        return len(self.schools)

    def school_means(self, column: str = "growth") -> pd.DataFrame:
        """One row per school: its allocation and its mean outcome over the year."""
        agg = self.benchmarks.groupby("school", as_index=False).agg(
            outcome=(column, "mean"),
            tutoring=("tutoring", "first"),
            messaging=("messaging", "first"),
        )
        return agg.merge(
            self.schools[["school", "eligible", "fidelity", "prior_year_gain", "cost_per_student"]],
            on="school",
        )

    def dose_arrays(self) -> dict[str, Array]:
        """``(n_schools, n_benchmarks)`` dose arrays -- the layout ``axiom.data`` wants."""
        wide = self.benchmarks.pivot(index="school", columns="benchmark")
        return {
            name: np.asarray(wide[name].to_numpy(), dtype=np.float64)
            for name in ("tutoring", "messaging")
        }

    def outcome_array(self, column: str = "growth") -> Array:
        wide = self.benchmarks.pivot(index="school", columns="benchmark", values=column)
        return np.asarray(wide.to_numpy(), dtype=np.float64)

    def unit_labels(self) -> list[str]:
        return sorted(self.benchmarks["school"].unique())

    def icc(self, column: str = "gain") -> float:
        """The observed intraclass correlation of student outcomes within a school.

        The number that decides how much a school-randomized trial costs: with
        students correlated inside a school, adding a student to a school already
        in the trial buys much less than adding a school. Note how much smaller it
        is for ``growth`` than for ``gain`` -- most of what students in a school
        share is the part the prior year already measured.
        """
        grouped = self.students.groupby("school")[column]
        between = float(grouped.mean().var(ddof=1))
        within = float(grouped.var(ddof=1).mean())
        return between / (between + within)


def run(
    design: Sequence[tuple[float, float]], *, seed: int = 20260867, truth: Truth = TRUTH
) -> Program:
    """Randomize, teach for a year, and measure. Notebooks 3-5 start from this."""
    schools = assign(list(design), seed=seed, truth=truth)
    return Program(
        schools=schools,
        benchmarks=benchmark_frame(schools, seed=seed + 1, truth=truth),
        students=student_frame(schools, seed=seed + 2, truth=truth),
        truth=truth,
        design=tuple((float(a), float(b)) for a, b in design),
    )


def school_panel(program: Program, *, column: str = "growth") -> Any:
    """The trial collapsed to **one row per school**: the analysis of record.

    Each school holds one allocation for the whole year, so its six benchmark readings
    are six looks at one number, not six independent observations. Averaging them first
    is what makes the residual an honest school-level residual -- notebook 3 section 4
    fits it both ways and the interval doubles.
    """
    from axiom.sim import panel_from_arrays

    means = program.school_means(column).sort_values("school")
    return panel_from_arrays(
        doses={
            "tutoring": means["tutoring"].to_numpy()[:, None],
            "messaging": means["messaging"].to_numpy()[:, None],
        },
        outcome=means["outcome"].to_numpy()[:, None],
        treatments=list(treatments()),
        outcome_entity=outcome(column),
        units=list(means["school"]),
        periods=[0],
        unit_column="school",
        time_column="benchmark",
    )


def panel(program: Program, *, column: str = "growth") -> Any:
    """The realized trial as an ``axiom.data.Panel``, one row per school-benchmark."""
    from axiom.sim import panel_from_arrays

    return panel_from_arrays(
        doses=program.dose_arrays(),
        outcome=program.outcome_array(column),
        treatments=list(treatments()),
        outcome_entity=outcome(column),
        units=program.unit_labels(),
        periods=list(range(BENCHMARKS)),
        unit_column="school",
        time_column="benchmark",
    )


def treatments() -> tuple[Any, ...]:
    """The two dosed treatments, as ``axiom.core`` entities."""
    from axiom.core import D, Treatment

    return (
        Treatment(
            name="tutoring",
            dimension=D.currency,
            unit=DOSE_UNIT,
            description="structured small-group tutoring, at 20 USD per weekly minute",
        ),
        Treatment(
            name="messaging",
            dimension=D.currency,
            unit=DOSE_UNIT,
            description="caregiver messaging, at 9 USD per weekly message, capped at 5",
        ),
    )


def outcome(column: str = "growth") -> Any:
    """The outcome entity: annualized reading gain, or growth over the prior year."""
    from axiom.core import D, Outcome

    described = {
        "gain": "annualized reading gain on the state benchmark, in scale points",
        "growth": "annualized reading gain less the same school's prior-year gain",
    }
    return Outcome(
        name=column,
        dimension=D.outcome,
        unit=OUTCOME_UNIT,
        description=described[column],
        aggregation="mean",
    )


def eligible_population(*, stratified: bool = True) -> Any:
    """Who the program is for: students two or more grade levels behind in reading.

    ``stratified=True`` is what the state would like to know -- the effect weighted across
    grade bands. The trial cannot deliver it: schools serve both bands and were randomized
    whole, so no unit has a grade band. ``estimands.realize`` refuses the stratified
    version rather than quietly averaging over a weighting it was not given, which is why
    notebook 3 asks for both.
    """
    from axiom.core import Population

    if not stratified:
        return Population(name="eligible_students_in_the_trial_schools")
    return Population(
        name="two_or_more_grades_behind",
        strata={"grade_band": {"3-5": 0.55, "6-8": 0.45}},
    )


def school_year() -> Any:
    """The window every estimand is over: one school year, read as one annualized figure.

    ``stop=1`` rather than ``stop=6``: the six benchmarks are six looks at one year, and
    the analysis of record collapses them before fitting (:func:`school_panel`). An
    estimand declared over six periods would not be evaluable against that fit -- which is
    the point of declaring the window rather than assuming it.
    """
    from axiom.core import TimeWindow

    return TimeWindow(start=0, stop=1, basis="cumulative")


def bounds() -> Any:
    """The feasible dose region: what the vendors will sell and schools will timetable."""
    from axiom.surface import Bounds

    return Bounds(
        treatments=("tutoring", "messaging"),
        low=(0.0, 0.0),
        high=(TUTORING_MAX, MESSAGING_MAX),
    )


# -- plotting --------------------------------------------------------------------------

# The shared notebook style: one registered plotly template and one validated palette
# for all eighty-four notebooks. Imported for its side effect (the template) and for the
# colours below, so this case study cannot drift into a look of its own.
import sys as _sys
from pathlib import Path as _Path

_NBS = str(_Path(__file__).resolve().parents[2])
if _NBS not in _sys.path:
    _sys.path.insert(0, _NBS)

from _style import AQUA, AXIS, BLUE, CRITICAL, GRID as _GRID, INK, MUTED, ORANGE, SUBTLE, SURFACE, VIOLET

# Two treatments are two categorical slots — the first two, which are the pair with the
# widest separation under every colour-vision simulation. Truth and the decision
# threshold are reference marks rather than series, so they take ink and the status red.
TUTORING_COLOR = BLUE
MESSAGING_COLOR = ORANGE
TRUTH_COLOR = MUTED
DECISION_COLOR = CRITICAL
ACCENT = VIOLET
GRID = _GRID


def figure(title: str, x: str, y: str, *, height: int = 380, **kwargs: object) -> Any:
    """A plotly figure with the case study's house layout already applied."""
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.update_layout(
        title=title,
        xaxis_title=x,
        yaxis_title=y,
        height=height,
        template="axiom",
        margin={"l": 65, "r": 30, "t": 60, "b": 50},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.0, "x": 0.0},
        **kwargs,
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
    return fig


def band(
    fig: Any,
    x: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
    color: str,
    *,
    name: str = "",
    opacity: float = 0.18,
) -> Any:
    """A filled interval band, drawn under whatever line it belongs to."""
    import plotly.graph_objects as go

    fig.add_trace(
        go.Scatter(
            x=list(x) + list(x)[::-1],
            y=list(upper) + list(lower)[::-1],
            fill="toself",
            fillcolor=rgba(color, opacity),
            line={"width": 0},
            hoverinfo="skip",
            showlegend=bool(name),
            name=name,
        )
    )
    return fig


def rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def minutes_axis(fig: Any, *, row: object = None, col: object = None) -> Any:
    """Label a dollar dose axis with the weekly minutes it buys."""
    ticks = [0.0, 600.0, 1200.0, 1800.0, 2400.0, 3000.0]
    fig.update_xaxes(
        tickvals=ticks,
        ticktext=[f"${t:,.0f}<br>{t / MINUTE_COST:.0f} min" for t in ticks],
    )
    return fig


def mean_with_error(frame: pd.DataFrame, by: Sequence[str], value: str) -> pd.DataFrame:
    """Group means with their standard errors and counts -- the plotting workhorse."""
    grouped = frame.groupby(list(by))[value]
    out = grouped.agg(["mean", "std", "count"]).reset_index()
    out["se"] = out["std"] / np.sqrt(out["count"].clip(lower=1))
    return out
