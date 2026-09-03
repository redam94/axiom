"""NIT-4: the synthetic world the income-guarantee case-study notebooks share.

Four experiments, one question, fourteen years. Between 1968 and 1982 the
United States ran four randomized income-maintenance experiments -- New Jersey,
the Rural experiment in Iowa and North Carolina, Gary, and Seattle-Denver -- to
learn what a **negative income tax** would do to how much people work. Congress
was considering a national plan: a guarantee of three quarters of the poverty
line for a family that earns nothing, taxed away at fifty cents on every dollar
earned. This module holds one data-generating process that all four experiments
sample from, so that five notebooks can carry a single decision from "what are
we asking" to "here is the number, its interval, and what it rests on".

**The decision.** Enact a national plan at guarantee ``G = 0.75`` of the poverty
line and tax rate ``t = 0.50``, if a family under it works fewer than
``HOURS_THRESHOLD`` hours a year less than it would have. Every hour lost has a
fiscal cost -- the transfer that replaces it plus the tax it no longer pays --
and ``VALUE_PER_HOUR`` times ``N_ELIGIBLE`` families is what one hour per family
is worth to the Treasury. Notebook 1 prices the uncertainty in that number.

**The dose is a plan, and a plan is two numbers.** A negative income tax pays a
family with earnings ``y`` a transfer of ``max(G * P - t * y, 0)`` where ``P`` is
the poverty line. Both ``G`` and ``t`` are dimensionless -- a fraction of a
poverty line and a fraction of a dollar -- and they are declared that way so that
the surface can be written over ``(G, t)`` without inventing a unit for either.
Their operational readings are the *break-even income* ``G * P / t`` and the
*transfer at zero earnings* ``G * P``, and every figure carries them.

**The truth the notebooks are trying to recover.** The change in annual hours
worked by the family head, relative to the year before enrolment, is

    site_scale * duration_factor * -(a_I * G + a_S * t + a_X * G * t + a_T * t^2)

plus family noise. ``a_I`` is the income effect (a bigger guarantee buys leisure),
``a_S`` the substitution effect (a higher tax rate makes work pay less), ``a_X``
their interaction and ``a_T`` a curvature in the tax rate the planning model does
not know about. ``site_scale`` is a party-level multiplier -- rural families
responded more, Gary's less -- and is exactly the variance component the nested
pool of notebook 4 exists to estimate. ``duration_factor`` grows with the years a
family was promised the plan; a five-year plan is a different window from a
three-year one and the estimand says so.

Four things go wrong, one per site, and each is a chapter of the case study:

* **New Jersey**: the state adopted AFDC-UP in 1969, a year into the experiment,
  so control families gained a welfare alternative and the counterfactual the
  plan was committed against moved. That is a deviation on the *reference*.
* **Gary**: families under-reported earnings to the experiment, so the tax rate
  actually applied to true earnings was lower than the assigned one. The dose
  delivered is not the dose assigned, and it is a dose rather than a switch.
* **Seattle-Denver**: a second treatment -- manpower counselling and training
  subsidies -- was randomized inside the same families in the same years, and a
  20-year cohort was enrolled and then cancelled. The experiment also reported a
  rise in marital dissolution that later reanalysis did not sustain; here the
  truth is no effect, and the draw is the one history dealt.
* **Everywhere**: control families and families on the least generous plans
  dropped out more, and the ones who left were the ones whose hours were
  rising. Attrition is a selection problem, not a compliance one.

Nothing here is imported by ``axiom``; it is notebook scaffolding, and every
statistical claim it supports is computed with ``axiom`` in the notebooks.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

Array = npt.NDArray[np.float64]

# -- the programme ---------------------------------------------------------------------

#: The 1970 poverty line for a family of four, in dollars of the day.
POVERTY_LINE = 4_000.0
#: Families under 150 % of the poverty line with a working-age head, nationally.
N_ELIGIBLE = 5_600_000
#: What one hour of work per eligible family per year is worth to the Treasury: the
#: transfer that replaces it plus the tax it no longer pays. A 1970 memorandum figure.
VALUE_PER_HOUR = 1.80
#: The plan Congress is considering: guarantee 0.75 of the poverty line, taxed at 50 %.
NATIONAL_PLAN = (0.75, 0.50)
#: Enact if a family under the national plan works fewer than this many hours a year
#: less than it would have. The threshold is a Treasury number, not a labour-economics one.
HOURS_THRESHOLD = 150.0
#: Interviewing, payments administration and data handling per family per year.
ADMIN_COST = 300.0

DOSE_UNIT_G = "fraction of poverty line"
DOSE_UNIT_T = "fraction of earnings"
OUTCOME_UNIT = "hours/year"

#: The plan lattice notebook 2 designs over. Every plan any site ran is on it.
GUARANTEES: tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.4)
TAX_RATES: tuple[float, ...] = (0.3, 0.5, 0.7, 0.8)


def plan_label(guarantee: float, tax_rate: float) -> str:
    """``G0.75-t0.50``: the arm label a plan is randomized under."""
    return f"G{guarantee:.2f}-t{tax_rate:.2f}"


def break_even(guarantee: float, tax_rate: float) -> float:
    """Earnings at which the transfer reaches zero, in dollars: ``G * P / t``."""
    return guarantee * POVERTY_LINE / tax_rate


# -- the truth -------------------------------------------------------------------------


@dataclass(frozen=True)
class Truth:
    """Every parameter of the data-generating process, in hours per year and dollars."""

    # the labour-supply response, hours per year per unit of each dose
    income_effect: float = 80.0  # per unit G
    substitution_effect: float = 110.0  # per unit t
    interaction: float = 40.0  # per unit G*t
    tax_curvature: float = 25.0  # per unit t^2; the planning model does not have it
    # a plan promised for longer moves hours more
    duration_gain_per_year: float = 0.15
    # baseline hours of a family head and the variance, in three places
    baseline_hours: float = 1_950.0
    family_sd: float = 350.0  # persistent: cancels out of a change score
    year_sd: float = 220.0  # this family, this year: does not cancel
    # how far a family's hours move with unobserved taste for work / mobility
    taste_on_hours: float = 300.0
    mobility_on_change: float = 70.0
    # attrition: a base rate, more for controls, less for generous plans, and the
    # movers leave
    attrition_base: float = 0.22
    attrition_control_extra: float = 0.10
    attrition_generosity: float = 0.08
    attrition_mobility: float = 0.9
    # Gary: the share of true earnings families reported to the experiment
    gary_reporting_mean: float = 0.72
    gary_reporting_concentration: float = 24.0
    # New Jersey: the hours control families gave up once AFDC-UP arrived
    afdc_up_control_shift: float = -25.0
    # Seattle-Denver: the training subsidy, and the interaction it does not have
    training_effect: float = -20.0
    training_interaction: float = 0.0
    # marital dissolution over the enrolment years: one rate, no plan effect
    dissolution_rate: float = 0.11
    # the observational record's confounding
    taste_on_tax: float = -0.03
    taste_on_guarantee: float = -0.06
    education_taste_correlation: float = 0.5


TRUTH = Truth()


def duration_factor(years: Array | float, truth: Truth = TRUTH) -> Any:
    """How much more a plan promised for ``years`` moves hours than a three-year one."""
    return 1.0 + truth.duration_gain_per_year * (np.asarray(years, dtype=np.float64) - 3.0)


def hours_change(
    guarantee: Array | float,
    tax_rate: Array | float,
    *,
    years: Array | float = 3.0,
    site_scale: float = 1.0,
    truth: Truth = TRUTH,
) -> Any:
    """Expected change in annual hours under a plan, against no plan. Negative is fewer."""
    g = np.asarray(guarantee, dtype=np.float64)
    t = np.asarray(tax_rate, dtype=np.float64)
    core = (
        truth.income_effect * g
        + truth.substitution_effect * t
        + truth.interaction * g * t
        + truth.tax_curvature * t**2
    )
    return -site_scale * duration_factor(years, truth) * core


# -- the money -------------------------------------------------------------------------

#: The earnings of eligible families, as a fraction of the poverty line: a fixed
#: quantile grid of a lognormal with median 0.9, truncated at the 150 % ceiling.
_EARNINGS_GRID = np.minimum(
    np.exp(np.log(0.9) + 0.35 * np.asarray(np.linspace(-2.4, 2.4, 97))), 1.5
)


def transfer(guarantee: float, tax_rate: float, earnings: Array | float) -> Any:
    """The payment a family with ``earnings`` (dollars) receives under a plan."""
    y = np.asarray(earnings, dtype=np.float64)
    return np.maximum(guarantee * POVERTY_LINE - tax_rate * y, 0.0)


def plan_cost(guarantee: float, tax_rate: float) -> float:
    """Expected cost of one family-year on a plan: the transfer plus administration.

    Averaged over the earnings the eligible population actually has. A control
    family costs only the administration. This is arithmetic the programme owns,
    and it is what makes the allocation in notebook 2 a budget problem rather
    than a sample-size problem.
    """
    if guarantee == 0.0 and tax_rate == 0.0:
        return ADMIN_COST
    return ADMIN_COST + float(np.mean(transfer(guarantee, tax_rate, POVERTY_LINE * _EARNINGS_GRID)))


# -- the sites -------------------------------------------------------------------------


@dataclass(frozen=True)
class Site:
    """One experiment: who ran it, how many families, which plans, for how long."""

    name: str
    party: str
    experiment: str
    started: int
    n_families: int
    plans: tuple[tuple[float, float], ...]
    control_share: float
    #: years of enrolment offered, with the share of treated families on each
    durations: tuple[tuple[int, float], ...] = ((3, 1.0),)
    site_scale: float = 1.0
    afdc_up: bool = False
    under_reporting: bool = False
    training: bool = False
    city: str = ""

    @property
    def arms(self) -> tuple[str, ...]:
        return ("control", *[plan_label(g, t) for g, t in self.plans])

    @property
    def shares(self) -> tuple[float, ...]:
        each = (1.0 - self.control_share) / len(self.plans)
        return (self.control_share, *[each] * len(self.plans))

    def plan_of(self, arm: str) -> tuple[float, float]:
        if arm == "control":
            return (0.0, 0.0)
        for g, t in self.plans:
            if plan_label(g, t) == arm:
                return (g, t)
        raise KeyError(f"{arm!r} is not an arm of {self.name}")


#: The four experiments, in the order they began. Family counts are the husband-wife
#: samples each one enrolled; the plans are the ones each one ran, snapped to the lattice.
SITES: tuple[Site, ...] = (
    Site(
        name="new_jersey",
        party="mathematica",
        experiment="NJ-1968",
        started=1968,
        n_families=1_216,
        plans=(
            (0.5, 0.3),
            (0.5, 0.5),
            (0.75, 0.3),
            (0.75, 0.5),
            (0.75, 0.7),
            (1.0, 0.5),
            (1.0, 0.7),
            (1.25, 0.5),
        ),
        control_share=0.40,
        site_scale=0.80,
        afdc_up=True,
        city="Trenton, Paterson, Passaic, Jersey City, Scranton",
    ),
    Site(
        name="rural",
        party="irp",
        experiment="RURAL-1970",
        started=1970,
        n_families=809,
        plans=((0.5, 0.5), (0.75, 0.3), (0.75, 0.5), (0.75, 0.7), (1.0, 0.5)),
        control_share=0.40,
        site_scale=1.25,
        city="Iowa and North Carolina",
    ),
    Site(
        name="gary",
        party="indiana",
        experiment="GARY-1971",
        started=1971,
        n_families=1_800,
        plans=((0.75, 0.4), (0.75, 0.6), (1.0, 0.4), (1.0, 0.6)),
        control_share=0.40,
        site_scale=0.85,
        under_reporting=True,
        city="Gary, Indiana",
    ),
    Site(
        name="seattle_denver",
        party="sri",
        experiment="SIME-DIME-1971",
        started=1971,
        n_families=4_800,
        plans=(
            (1.0, 0.5),
            (1.0, 0.7),
            (1.25, 0.5),
            (1.25, 0.7),
            (1.25, 0.8),
            (1.4, 0.5),
            (1.4, 0.7),
            (1.4, 0.8),
        ),
        control_share=0.40,
        durations=((3, 0.60), (5, 0.34), (20, 0.06)),
        site_scale=1.10,
        training=True,
        city="Seattle and Denver",
    ),
)

SITE_BY_NAME: dict[str, Site] = {s.name: s for s in SITES}

#: The plan the four sites have in common with the national proposal, or nearest to it.
PLAN_LABELS_ON_LATTICE: tuple[str, ...] = tuple(
    plan_label(g, t) for g in GUARANTEES for t in TAX_RATES
)


# -- families --------------------------------------------------------------------------


def roster(site: Site, *, seed: int, truth: Truth = TRUTH) -> pd.DataFrame:
    """The families a site enrolled, before anybody is assigned to anything.

    Every family carries its pre-enrolment hours and earnings (recorded, and
    what the assignment blocks on), a years-of-schooling proxy (recorded), and
    two things nobody records: a taste for work and a mobility that decides
    who will still be answering interviews three years on.
    """
    rng = np.random.default_rng(seed)
    n = site.n_families
    taste = rng.normal(0.0, 1.0, size=n)
    education = truth.education_taste_correlation * taste + np.sqrt(
        1.0 - truth.education_taste_correlation**2
    ) * rng.normal(0.0, 1.0, size=n)
    mobility = rng.normal(0.0, 1.0, size=n)
    family_effect = rng.normal(0.0, truth.family_sd, size=n)
    pre_hours = (
        truth.baseline_hours
        + truth.taste_on_hours * taste
        + family_effect
        + rng.normal(0.0, truth.year_sd, size=n)
    )
    earnings = np.clip(
        POVERTY_LINE * np.exp(np.log(0.9) + 0.35 * rng.normal(size=n) + 0.15 * taste),
        200.0,
        1.5 * POVERTY_LINE,
    )
    return pd.DataFrame(
        {
            "family": [f"{site.name}-{i:04d}" for i in range(n)],
            "site": site.name,
            "pre_hours": pre_hours,
            "earnings": earnings,
            "income_band": np.where(earnings < POVERTY_LINE, "below_poverty", "poverty_to_150"),
            "schooling": 10.5 + 2.0 * education,
            # Latent. Recorded here so a notebook can *show* what it is not allowed
            # to use; nothing that estimates anything is permitted to touch these.
            "taste_unobserved": taste,
            "mobility_unobserved": mobility,
            "family_effect_unobserved": family_effect,
        }
    )


def allocation(site: Site) -> Any:
    """The site's arms and shares, as the ``ArmAllocation`` ``design.assign`` takes."""
    from axiom.design import ArmAllocation

    return ArmAllocation(arms=site.arms, shares=site.shares)


def assign_site(site: Site, families: pd.DataFrame, *, seed: int) -> Any:
    """Randomize a site's families to its arms: permuted blocks within income band.

    The rule -- allocation, method, seed, stratum hash, roster hash -- is what
    gets filed with the experiment; the arms themselves are re-derivable from it.
    """
    from axiom.design import assign

    return assign(
        list(families["family"]),
        allocation(site),
        method="block",
        seed=seed,
        strata=list(families["income_band"]),
        covariates={
            "pre_hours": families["pre_hours"].to_numpy(),
            "earnings": families["earnings"].to_numpy(),
            "schooling": families["schooling"].to_numpy(),
        },
    )


def run_site(
    site: Site,
    families: pd.DataFrame,
    arm_of: Sequence[str],
    *,
    seed: int,
    dissolution_seed: int | None = None,
    truth: Truth = TRUTH,
) -> pd.DataFrame:
    """Enrol the families, pay them for their years, and interview whoever is left.

    One row per family. ``hours_change`` is the truth the experiment is
    trying to see -- mean annual hours over the enrolment years less the
    pre-enrolment year -- and ``hours_observed`` is what the analyst gets:
    the same number where the family was still answering, ``NaN`` where it
    was not. ``delivered_tax`` is the tax rate actually applied to the
    family's true earnings, which differs from ``tax_rate`` only where
    earnings were under-reported.
    """
    rng = np.random.default_rng(seed)
    n = len(families)
    arms = np.asarray(arm_of, dtype=object)
    plans = np.asarray([site.plan_of(str(a)) for a in arms], dtype=np.float64)
    guarantee, tax_rate = plans[:, 0], plans[:, 1]
    assigned = (arms != "control").astype(float)

    # enrolment duration, drawn per treated family from the site's offer
    years = np.full(n, 3.0)
    if len(site.durations) > 1:
        labels = np.asarray([d for d, _ in site.durations], dtype=np.float64)
        shares = np.asarray([s for _, s in site.durations], dtype=np.float64)
        years = np.where(assigned > 0, rng.choice(labels, size=n, p=shares / shares.sum()), 3.0)

    # the dose delivered: Gary's families reported less than they earned
    reporting = np.ones(n)
    if site.under_reporting:
        m, c = truth.gary_reporting_mean, truth.gary_reporting_concentration
        reporting = rng.beta(m * c, (1.0 - m) * c, size=n)
    delivered_tax = tax_rate * reporting

    # a second, independently randomized treatment inside Seattle-Denver
    training = np.zeros(n)
    if site.training:
        training = (rng.random(n) < 0.5).astype(float)

    mobility = families["mobility_unobserved"].to_numpy()
    response = hours_change(
        guarantee, delivered_tax, years=years, site_scale=site.site_scale, truth=truth
    )
    training_shift = truth.training_effect * training + truth.training_interaction * (
        training * assigned
    )
    control_shift = np.zeros(n)
    if site.afdc_up:
        control_shift = np.where(assigned > 0, 0.0, truth.afdc_up_control_shift)
    # a change score: the persistent family effect cancels, the year shocks do not
    noise = rng.normal(0.0, truth.year_sd, size=n) / np.sqrt(np.minimum(years, 5.0)) + rng.normal(
        0.0, truth.year_sd, size=n
    )
    change = response + training_shift + control_shift + truth.mobility_on_change * mobility + noise

    # attrition: controls and thin plans lose more, and the movers are the ones who go
    generosity = guarantee * (1.0 - tax_rate / 2.0)
    base = (
        truth.attrition_base
        + truth.attrition_control_extra * (1.0 - assigned)
        - truth.attrition_generosity * generosity
    )
    logit = np.log(base / (1.0 - base)) + truth.attrition_mobility * mobility
    p_drop = 1.0 / (1.0 + np.exp(-logit))
    observed = (rng.random(n) >= p_drop).astype(float)

    # Marital dissolution: one rate, no plan effect, drawn from its own generator so
    # that the draw is the same whatever else the world is asked for.
    rng_d = np.random.default_rng(seed if dissolution_seed is None else dissolution_seed)
    dissolved = (rng_d.random(n) < truth.dissolution_rate).astype(float)

    out = families[["family", "site", "pre_hours", "earnings", "income_band", "schooling"]].copy()
    out["arm"] = arms
    out["assigned"] = assigned
    out["guarantee"] = guarantee
    out["tax_rate"] = tax_rate
    out["delivered_tax"] = delivered_tax
    out["years"] = years
    out["training"] = training
    out["hours_change"] = change
    out["observed"] = observed
    out["hours_observed"] = np.where(observed > 0, change, np.nan)
    out["dissolved"] = dissolved
    out["relative_change"] = change / families["pre_hours"].to_numpy()
    return out


@dataclass(frozen=True)
class SiteRun:
    """One experiment as it happened: the site, its roster, the rule, and the rows."""

    site: Site
    families: pd.DataFrame
    assigned: Any
    frame: pd.DataFrame

    @property
    def observed(self) -> pd.DataFrame:
        """Only the families still answering at the end -- what the analyst sees."""
        return self.frame[self.frame["observed"] > 0].copy()

    def cohort(self, years: int) -> pd.DataFrame:
        """The observed treated families on plans of one duration, with the controls."""
        f = self.observed
        return f[(f["years"] == years) | (f["assigned"] == 0)].copy()


@dataclass(frozen=True)
class Programme:
    """The four experiments, run from one seed so every notebook sees the same world."""

    runs: dict[str, SiteRun] = field(default_factory=dict)
    truth: Truth = TRUTH

    def __getitem__(self, name: str) -> SiteRun:
        return self.runs[name]

    @property
    def sites(self) -> tuple[Site, ...]:
        return tuple(run.site for run in self.runs.values())

    def frames(self) -> pd.DataFrame:
        return pd.concat([run.frame for run in self.runs.values()], ignore_index=True)


#: The seed every notebook rebuilds the programme from.
SEED = 19680801
#: The seed the marital-dissolution draws come from. Chosen so that the null draw at
#: Seattle-Denver lands where history did: a difference an uncorrected readout calls a
#: finding, and a corrected programme does not. Every other site's draw is unremarkable.
DISSOLUTION_SEED = 35_000


def run_programme(*, seed: int = SEED, truth: Truth = TRUTH) -> Programme:
    """Enrol, randomize, pay and interview at all four sites. Notebooks 3-5 start here."""
    runs: dict[str, SiteRun] = {}
    for k, site in enumerate(SITES):
        families = roster(site, seed=seed + 10 * k, truth=truth)
        assigned = assign_site(site, families, seed=seed + 10 * k + 1)
        arms = [assigned.arm(f) for f in families["family"]]
        frame = run_site(
            site,
            families,
            arms,
            seed=seed + 10 * k + 2,
            dissolution_seed=DISSOLUTION_SEED + k,
            truth=truth,
        )
        runs[site.name] = SiteRun(site=site, families=families, assigned=assigned, frame=frame)
    return Programme(runs=runs, truth=truth)


# -- the observational record ----------------------------------------------------------


def prior_record(n_families: int = 2_000, *, seed: int, truth: Truth = TRUTH) -> pd.DataFrame:
    """What Congress already has: a cross-section where nobody randomized anything.

    Families face different effective tax rates and welfare guarantees because
    of where they live and what they earn. Families with a stronger taste for
    work earn more, face lower effective rates, and would work more anyway; the
    only proxy for that taste in the survey is schooling. The naive slope of
    hours on the tax rate is biased away from zero, and adjusting for the proxy
    removes some of it and not all.
    """
    rng = np.random.default_rng(seed)
    taste = rng.normal(0.0, 1.0, size=n_families)
    education = truth.education_taste_correlation * taste + np.sqrt(
        1.0 - truth.education_taste_correlation**2
    ) * rng.normal(0.0, 1.0, size=n_families)
    tax_rate = np.clip(
        0.42 + truth.taste_on_tax * taste + rng.normal(0.0, 0.15, n_families), 0.1, 0.8
    )
    guarantee = np.clip(
        0.55 + truth.taste_on_guarantee * taste + rng.normal(0.0, 0.15, n_families), 0.1, 1.2
    )
    hours = (
        truth.baseline_hours
        + truth.taste_on_hours * taste
        + hours_change(guarantee, tax_rate, truth=truth)
        + rng.normal(0.0, truth.family_sd, size=n_families)
    )
    return pd.DataFrame(
        {
            "family": [f"survey-{i:04d}" for i in range(n_families)],
            "tax_rate": tax_rate,
            "guarantee": guarantee,
            "schooling": 10.5 + 2.0 * education,
            "hours": hours,
            "taste_unobserved": taste,
        }
    )


# -- entities and surfaces -------------------------------------------------------------


def treatments() -> tuple[Any, ...]:
    """The two doses, as ``axiom.core`` entities. Both dimensionless, on purpose."""
    from axiom.core import Treatment, dimensionless

    return (
        Treatment(
            name="guarantee",
            dimension=dimensionless(),
            unit=DOSE_UNIT_G,
            description="the transfer at zero earnings, as a fraction of the poverty line",
        ),
        Treatment(
            name="tax_rate",
            dimension=dimensionless(),
            unit=DOSE_UNIT_T,
            description="the rate at which the transfer is withdrawn as earnings rise",
        ),
    )


def outcome() -> Any:
    """Annual hours of the family head, relative to the year before enrolment."""
    from axiom.core import D, Outcome

    return Outcome(
        name="hours_change",
        dimension=D.outcome,
        unit=OUTCOME_UNIT,
        aggregation="mean",
        description=(
            "mean annual hours worked over the enrolment years, less the pre-enrolment year"
        ),
    )


def hours_definition(version: str) -> Any:
    """What a site means by "hours worked": the spec a definition registry versions."""
    from axiom.core import D, Outcome

    described = {
        "self_report": "hours worked in the past quarter, as the family head reported at interview",
        "employer_verified": (
            "hours worked in the past quarter, reconciled against employer and "
            "unemployment-insurance records"
        ),
    }
    return Outcome(
        name="hours_worked",
        dimension=D.outcome,
        unit=OUTCOME_UNIT,
        aggregation="mean",
        description=described[version],
    )


def spec(
    kernels: Mapping[str, Any] | None = None,
    *,
    name: str = "nit4",
    unit_labels: Sequence[str] = (),
    intercept: str = "shared",
    **overrides: Any,
) -> Any:
    """A ``SurfaceSpec`` over the two doses with the case study's prior scales.

    Polynomial kernels because the response is *negative* and every saturating
    family in ``axiom.surface`` carries a positive amplitude; a polynomial's
    coefficients are signed. ``reference_dose`` is the top of each lattice so
    that ``u`` lands in ``[0, 1]`` and one prior scale means the same thing for
    both doses. The interaction is the product of the two saturations.
    """
    from axiom.surface import PolynomialKernel, SurfaceSpec

    default_kernels = {
        "guarantee": PolynomialKernel(
            reference_dose=max(GUARANTEES), amplitude_scale=200.0, degree=1
        ),
        "tax_rate": PolynomialKernel(
            reference_dose=max(TAX_RATES), amplitude_scale=200.0, degree=1
        ),
    }
    fields: dict[str, Any] = {
        "name": name,
        "treatments": treatments(),
        "outcome": outcome(),
        "kernels": dict(kernels) if kernels is not None else default_kernels,
        "interactions": (("guarantee", "tax_rate"),),
        "intercept": intercept,
        "unit_labels": tuple(unit_labels),
        "unit_column": "site",
        "time_column": "family_index",
        "intercept_scale": 100.0,
        "interaction_scale": 100.0,
        "noise_scale": 300.0,
    }
    return SurfaceSpec(**{**fields, **overrides})


def planning_spec() -> Any:
    """The linear-in-parameters surface notebook 2 designs against: two slopes and a product."""
    return spec(name="nit4_planning", unit_labels=("planning",), intercept="shared")


def analysis_spec(unit_labels: Sequence[str] = (), *, intercept: str = "shared") -> Any:
    """The surface notebook 5 fits: a quadratic in the tax rate the planning model lacked."""
    from axiom.surface import PolynomialKernel

    return spec(
        {
            "guarantee": PolynomialKernel(
                reference_dose=max(GUARANTEES), amplitude_scale=200.0, degree=1
            ),
            "tax_rate": PolynomialKernel(
                reference_dose=max(TAX_RATES), amplitude_scale=200.0, degree=2
            ),
        },
        name="nit4_analysis",
        unit_labels=tuple(sorted(unit_labels)),
        intercept=intercept,
    )


def lattice() -> Any:
    """Every plan a family could be assigned to, plus no plan: the candidate set."""
    from axiom.surface import Design

    points = [(0.0, 0.0)] + [(float(g), float(t)) for g in GUARANTEES for t in TAX_RATES]
    return Design(treatments=("guarantee", "tax_rate"), points=tuple(points), kind="lattice")


def design_data(points: Sequence[tuple[float, float]]) -> dict[str, Array]:
    """The data mapping the planning surface's ``forward`` reads for a set of plans."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    n = pts.shape[0]
    return {
        "guarantee": pts[:, 0][:, None],
        "tax_rate": pts[:, 1][:, None],
        "site": np.zeros((n, 1), dtype=np.int64),
        "family_index": np.zeros((n, 1)),
    }


def analysis_panel(
    programme: Programme,
    *,
    sites: Sequence[str] | None = None,
    block: int = 100,
    cohort_years: int = 3,
    seed: int = 7,
) -> Any:
    """The experiments as one balanced ``Panel``: unit = a block of families, time = index.

    A cross-section laid out as a panel, because the surface wants ``(unit,
    time)``. Each site's observed families -- controls and the families on
    plans of ``cohort_years`` -- are shuffled and cut into blocks of ``block``,
    so the panel is balanced and nearly every family is used; the block label
    carries the site name so a per-block intercept is a per-site one with
    extra freedom. Only the chosen cohort enters, so the window the surface
    fits is the estimand's window. ``sites`` restricts which experiments are
    pooled; the default is all of them, and notebook 5 says why it is not.
    """
    from axiom.sim import panel_from_arrays

    rng = np.random.default_rng(seed)
    names = sorted(programme.runs) if sites is None else sorted(sites)
    g_rows, t_rows, y_rows, labels = [], [], [], []
    for name in names:
        rows = programme[name].cohort(cohort_years)
        order = rng.permutation(len(rows))
        n_blocks = len(rows) // block
        for b in range(n_blocks):
            pick = rows.iloc[order[b * block : (b + 1) * block]]
            g_rows.append(pick["guarantee"].to_numpy())
            t_rows.append(pick["tax_rate"].to_numpy())
            y_rows.append(pick["hours_observed"].to_numpy())
            labels.append(f"{name}-{b:02d}")
    order = np.argsort(labels)
    return panel_from_arrays(
        doses={
            "guarantee": np.vstack(g_rows)[order],
            "tax_rate": np.vstack(t_rows)[order],
        },
        outcome=np.vstack(y_rows)[order],
        treatments=list(treatments()),
        outcome_entity=outcome(),
        units=[labels[i] for i in order],
        periods=list(range(block)),
        unit_column="site",
        time_column="family_index",
    )


def eligible_population(*, national: bool = False) -> Any:
    """The families the experiments enrolled, or the ones a national plan would reach."""
    from axiom.core import Population

    if national:
        return Population(
            name="us_families_below_150pct_poverty",
            description="every US family under 150 % of the poverty line with a working-age head",
            strata={"region": {"urban": 0.72, "rural": 0.28}},
        )
    return Population(
        name="enrolled_families_below_150pct_poverty",
        description=(
            "husband-wife families under 150 % of the poverty line in the experimental cities"
        ),
    )


def national_estimand(
    *,
    plan: tuple[float, float] = NATIONAL_PLAN,
    years: int = 3,
    version: str = "assigned",
    reference_version: str = "assigned",
    population: Any = None,
    name: str = "hours_under_national_plan",
) -> Any:
    """The contrast the decision needs: hours under a plan against no plan, per family.

    ``version`` and ``reference_version`` are how the plan and its absence were
    delivered; New Jersey's readout ends up answering a reference nobody
    declared, and the version is where that shows.
    """
    from axiom.core import D, Intervention
    from axiom.estimands import Estimand, Level, Quantity

    guarantee_entity, _ = treatments()
    g, t = plan
    return Estimand(
        name=name,
        quantity=Quantity(kind="contrast"),
        treatment=guarantee_entity,
        intervention=Intervention(doses={"guarantee": g, "tax_rate": t}, version=version),
        reference=Intervention(
            doses={"guarantee": 0.0, "tax_rate": 0.0}, version=reference_version
        ),
        outcome=outcome(),
        population=population if population is not None else eligible_population(),
        window=enrolment_window(years),
        level=Level(unit="individual", interference="none"),
        dimension=D.outcome,
        description=(
            f"change in annual hours for a family head assigned a plan with guarantee {g} "
            f"and tax rate {t} for {years} years, against no plan"
        ),
    )


def enrolment_window(years: int = 3) -> Any:
    """The years a family was promised the plan, as the estimand's window."""
    from axiom.core import TimeWindow

    return TimeWindow(start=0, stop=years, basis="per_period")


# -- the house style -------------------------------------------------------------------

# The shared notebook style: one registered plotly template and one validated palette
# for every notebook under nbs/. Imported for its side effect (the template) and for the
# colours below, so this case study cannot drift into a look of its own.
import sys as _sys  # noqa: E402
from pathlib import Path as _Path  # noqa: E402

_NBS = str(_Path(__file__).resolve().parents[2])
if _NBS not in _sys.path:
    _sys.path.insert(0, _NBS)

from _style import (  # noqa: E402
    AQUA,
    BLUE,
    CRITICAL,
    GOOD,
    GRID,
    INK,
    MAGENTA,
    MUTED,
    ORANGE,
    SERIES,
    SUBTLE,
    VIOLET,
    YELLOW,
)

PLAN_COLOR = BLUE
CONTROL_COLOR = MUTED
DECISION_COLOR = ORANGE
TRUTH_COLOR = MUTED
ACCENT = VIOLET
#: One colour per site, in the order the sites began.
SITE_COLORS: dict[str, str] = {
    "new_jersey": BLUE,
    "rural": ORANGE,
    "gary": AQUA,
    "seattle_denver": VIOLET,
}
__all_colours__ = (AQUA, CRITICAL, GOOD, GRID, INK, MAGENTA, SERIES, SUBTLE, YELLOW)


def site_title(name: str) -> str:
    """``seattle_denver`` -> ``Seattle-Denver``, for axis labels and tables."""
    return {
        "new_jersey": "New Jersey",
        "rural": "Rural",
        "gary": "Gary",
        "seattle_denver": "Seattle-Denver",
    }[name]
