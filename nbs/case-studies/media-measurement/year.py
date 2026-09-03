"""HOUSE-3: the year as the house planned, ran and filed it.

Notebook 2 plans the calendar and notebook 3 runs it, step by step and with
the argument for every step. Notebooks 4 and 5 start from the *filed* year —
the catalog, the runs, the measurements — and should not have to re-tell it.
This module is the replay: the same decisions, in the same order, producing
the same catalog, so that every notebook stands alone and the numbers in
notebook 5 are the numbers notebook 3 filed.

Nothing here decides anything. Every choice — which experiments run, at what
holdout share, how each departure from plan is recorded — was made in a
notebook and is repeated here with a pointer to where it was argued. The
statistics are ``axiom``'s (``design.assign``, ``design.estimate``,
``design.cluster_mde``, ``design.stopped_estimate`` through
``build.ExperimentBuilder.read_stopped``); this file only calls them in the
order the year happened.
"""

from __future__ import annotations

import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import house as H
import numpy as np
import numpy.typing as npt
import pandas as pd

from axiom.build import ExperimentBuilder
from axiom.calibrate import Measurement
from axiom.core import D, Intervention, Outcome, Population, TimeWindow, Unit
from axiom.design import (
    ArmAllocation,
    Assigned,
    ClusterDesign,
    DesignCandidate,
    LookSchedule,
    MethodEstimate,
    StoppingRule,
    assign,
    cluster_mde,
    estimate,
    monitor,
    obrien_fleming,
    panel_arrays,
)
from axiom.estimands import Estimand, Level, Quantity
from axiom.io import Catalog, ExperimentRun, Program

Array = npt.NDArray[np.float64]

# -- the house -------------------------------------------------------------------------

MARKET_UNIT = Unit(name="market", dimension=D.entity, kind="cluster")

#: A tenth-of-the-map holdout for the two retail parties, half the map for acme, whose
#: channels are cheap to switch off (notebook 2 §3).
ALLOCATIONS: dict[int, ArmAllocation] = {
    10: ArmAllocation(arms=("control", "treated"), shares=(0.75, 0.25)),
    20: ArmAllocation.equal("control", "treated"),
}

#: What each party says one conversion is worth, in kUSD, and where the number came from.
VALUE_PER_KPI: dict[str, tuple[float, str]] = {
    "northwind": (
        0.34,
        "northwind finance: 340 USD average order; the growth plan budgets on revenue",
    ),
    "acme": (
        0.16,
        "acme growth team: 160 USD twelve-month LTV per paid subscription, cohort model 2025",
    ),
    "globex": (0.12, "globex trade marketing: 120 USD contribution per case at shelf"),
}

#: The house's fee per experiment, kUSD. The invoice is the small part.
FIXED_COST = 40.0


def programs() -> dict[str, Program]:
    return {
        p: Program(
            party=p,
            program="incrementality",
            started="2026-01-05",
            description=f"{H.PARTIES[p].business}: incrementality of paid media",
        )
        for p in H.PARTIES
    }


def open_house(
    root: str | Path | None = None,
) -> tuple[Catalog, dict[str, Program], dict[str, ExperimentBuilder]]:
    """A fresh catalog with the three parties registered and a builder for each."""
    catalog = Catalog(Path(root) if root is not None else Path(tempfile.mkdtemp()) / "house")
    progs = programs()
    for p in progs.values():
        catalog.register(p)
    return catalog, progs, {name: ExperimentBuilder(catalog, p) for name, p in progs.items()}


# -- what a party means by a conversion --------------------------------------------------


def conversion(kind: str) -> Outcome:
    """The event a party's KPI counts, as a spec the registry can version."""
    return Outcome(name="conversion", dimension=D.outcome, unit=kind, aggregation="sum")


def register_definitions(
    builders: Mapping[str, ExperimentBuilder], *, through_week: int = 52
) -> None:
    """Northwind and globex count a paid event; acme counted a trial start until week 12
    and a paid subscription after it (notebook 1 §5). Globex registers late."""
    builders["northwind"].define("conversion", conversion("paid"), at="2026-01-05")
    builders["acme"].define("conversion", conversion("trial_start"), at="2026-01-05")
    if through_week >= 6:
        builders["globex"].define("conversion", conversion("paid"), at="2026-02-16")
    if through_week >= 12:
        builders["acme"].define(
            "conversion",
            conversion("paid"),
            at="2026-03-23",
            note="growth team moved the KPI to paid subscriptions",
        )


# -- the estimand every experiment is committed to ---------------------------------------


def lift_estimand(exp: H.Experiment, *, population: Population | None = None) -> Estimand:
    """The lift the channel delivers per market-week in the experiment's window: the KPI
    at observed spend against the KPI with the channel off (a holdout), or at the
    scaled spend against observed (a dose test). Positive when the channel works."""
    party = H.PARTIES[exp.party]
    treatment = H.channel_entity(exp.channel)
    if exp.kind == "dose":
        intervention = Intervention(
            doses={exp.channel: 1.0 + exp.uplift}, mode="scale", version="assigned"
        )
        reference = Intervention(doses={exp.channel: 1.0}, mode="scale", version="observed")
        what = f"{exp.channel} spend scaled by {1.0 + exp.uplift:.1f} against observed spend"
    else:
        intervention = Intervention(doses={exp.channel: 1.0}, mode="scale", version="observed")
        reference = Intervention(doses={exp.channel: 0.0}, mode="scale", version="holdout")
        what = f"observed {exp.channel} spend against the channel switched off"
    return Estimand(
        name=f"{exp.id.lower().replace('-', '_')}_lift",
        quantity=Quantity(kind="contrast"),
        treatment=treatment,
        intervention=intervention,
        reference=reference,
        outcome=H.kpi_entity(party),
        population=population or Population(name="all_markets"),
        window=TimeWindow(start=exp.start, stop=exp.stop, basis="per_period"),
        level=Level(unit="individual", interference="none"),
        dimension=D.outcome,
        description=f"conversions per market-week, weeks {exp.start}-{exp.stop}: {what}",
    )


# -- the design inputs, from last year's panel ------------------------------------------


@dataclass(frozen=True)
class DesignInputs:
    """What last year's panel says about the noise a market-week carries once its own
    level and the week's season are removed: the residual sd and how much of it
    persists within a market from week to week (the ICC weeks-within-market)."""

    party: str
    sd: float
    icc: float


def design_inputs(w0: H.World) -> DesignInputs:
    kpi = w0.kpi()
    resid = kpi - kpi.mean(axis=1, keepdims=True) - kpi.mean(axis=0, keepdims=True) + kpi.mean()
    n_units, n_periods = resid.shape
    m = H.PRE_WEEKS
    blocks = resid[:, : (n_periods // m) * m].reshape(n_units, -1, m)
    block_means = blocks.mean(axis=2)
    within = float(blocks.var(axis=2, ddof=1).mean())
    between = float(block_means.var(ddof=1) - within / m)
    icc = max(between, 0.0) / (max(between, 0.0) + within)
    return DesignInputs(w0.party.name, float(resid.std(ddof=1)), float(icc))


def cluster_design(exp: H.Experiment, inputs: DesignInputs) -> ClusterDesign:
    return ClusterDesign(
        unit=MARKET_UNIT,
        n_clusters=H.N_MARKETS,
        cluster_size=exp.weeks,
        icc=inputs.icc,
        allocation=exp.n_treated / H.N_MARKETS,
    )


def experiment_se(exp: H.Experiment, inputs: DesignInputs) -> float:
    """The standard error the design math expects for the lift per market-week."""
    return float(cluster_mde(cluster_design(exp, inputs), inputs.sd).se)


def candidate(
    exp: H.Experiment, inputs: DesignInputs, *, cost: float = FIXED_COST
) -> DesignCandidate:
    return DesignCandidate(
        name=exp.id,
        method="difference_in_differences",
        n_units=H.N_MARKETS,
        n_periods=exp.weeks,
        holdout_fraction=exp.n_treated / H.N_MARKETS,
        experiment_se=experiment_se(exp, inputs),
        cost=cost,
        n_clusters=H.N_MARKETS,
        cooldown_periods=4,
        detail={"party": exp.party, "channel": exp.channel, "kind": exp.kind},
    )


# -- assignment ------------------------------------------------------------------------


def arms(exp: H.Experiment, *, seed: int | None = None) -> Assigned:
    """Permuted blocks within size tier, seeded by the experiment's number (notebook 2 §4)."""
    return assign(
        H.MARKETS,
        ALLOCATIONS[exp.n_treated],
        method="block",
        seed=int(exp.id[-2:]) if seed is None else seed,
        strata=list(H.TIERS),
        covariates={"size": H.SIZES},
    )


# -- the calendar as planned ----------------------------------------------------------

#: NW-03 shared search with NW-01 on the same markets in the same weeks and was moved to
#: the autumn; AC-04 was moved past the onboarding rollout (notebook 2 §6).
RESCHEDULED: dict[str, tuple[int, int]] = {"AC-04": (42, 50), "NW-03": (40, 48)}
REFUSED: tuple[str, ...] = ()


def planned_calendar() -> tuple[H.Experiment, ...]:
    out = []
    for exp in H.CALENDAR:
        if exp.id in REFUSED or exp.kind == "rollout":
            continue
        if exp.id in RESCHEDULED:
            start, stop = RESCHEDULED[exp.id]
            why = "moved past the rollout" if exp.id == "AC-04" else "moved off NW-01's weeks"
            exp = H.Experiment(
                exp.id,
                exp.party,
                exp.channel,
                exp.kind,
                start,
                stop,
                exp.n_treated,
                exp.uplift,
                exp.randomized,
                why,
            )
        out.append(exp)
    return tuple(out)


@dataclass(frozen=True)
class Plan:
    experiment: H.Experiment
    estimand: Estimand
    candidate: DesignCandidate
    assigned: Assigned
    run: ExperimentRun

    @property
    def treated(self) -> tuple[str, ...]:
        return self.assigned.units_in("treated")

    @property
    def control(self) -> tuple[str, ...]:
        return self.assigned.units_in("control")


def plan_year(
    builders: Mapping[str, ExperimentBuilder], inputs: Mapping[str, DesignInputs]
) -> dict[str, Plan]:
    """Every experiment on the planned calendar, filed and committed in its party's scope."""
    plans: dict[str, Plan] = {}
    for exp in planned_calendar():
        est = lift_estimand(exp)
        cand = candidate(exp, inputs[exp.party])
        assigned = arms(exp)
        builder = builders[exp.party]
        run = builder.plan(
            exp.id,
            estimand=est,
            roles={
                "candidate": cand,
                "assignment": assigned.spec,
                "occupancy": H.occupancy(exp, H.MARKETS),
                "window": exp.window,
            },
            at=f"2026-W{max(exp.start - 6, 1):02d}",
        )
        run = builder.commit(run, at=f"2026-W{max(exp.start - 4, 1):02d}")
        plans[exp.id] = Plan(exp, est, cand, assigned, run)
    return plans


# -- the field -------------------------------------------------------------------------


def field_year(worlds: Mapping[str, H.World], plans: Mapping[str, Plan]) -> dict[str, H.Fielded]:
    """Apply the planned experiments to each party's year, in calendar order, stacking."""
    doses: dict[str, dict[str, Array]] = {p: dict(w.doses) for p, w in worlds.items()}
    fielded: dict[str, H.Fielded] = {}
    for plan in sorted(plans.values(), key=lambda p: p.experiment.start):
        exp = plan.experiment
        if exp.id == "GX-01":
            exp = H.Experiment(
                exp.id,
                exp.party,
                exp.channel,
                exp.kind,
                exp.start,
                exp.stop + 2,
                exp.n_treated,
                exp.uplift,
                exp.randomized,
                "extended two weeks",
            )
        f = H.field_experiment(
            worlds[exp.party], exp, plan.treated, plan.control, doses=doses[exp.party]
        )
        doses[exp.party] = f.doses
        fielded[exp.id] = f
    return fielded


def kpi_seen(w: H.World, f: H.Fielded) -> Array:
    """The KPI the analyst was handed for this experiment: the year's KPI under every
    experiment fielded so far, with this experiment's unreported cells as NaN."""
    return H.with_missing(w.kpi(f.doses), f.missing)


def did(
    kpi: Array,
    f: H.Fielded,
    *,
    post: slice | None = None,
    method: str = "difference_in_differences",
) -> MethodEstimate:
    """The experiment's estimate on the markets whose post period is complete."""
    exp = f.experiment
    window = exp.post if post is None else post
    rows = H.observed_markets(kpi, window)
    treated = [rows.index(i) for i in f.treated_indices() if i in rows]
    out = estimate(method, panel_arrays(kpi[rows], treated, exp.pre, window), mass=0.95)
    if not isinstance(out, MethodEstimate):
        raise RuntimeError(f"{exp.id}: {out}")
    return out


def lift_measurement(
    exp: H.Experiment, est: Estimand, e: MethodEstimate, *, source: str
) -> Measurement:
    """A holdout's effect is treated-minus-control with the channel *off* in the treated
    arm, so the lift the estimand names is its negative; a dose test's is its sign."""
    sign = 1.0 if exp.kind == "dose" else -1.0
    return Measurement(
        estimand=est,
        estimate=sign * e.effect,
        se=e.se,
        definition="wald",
        mass=0.95,
        method=e.method,
        n_units=e.n_treated + e.n_control,
        n_periods=e.n_post,
        source=source,
    )


def gx02_rule() -> StoppingRule:
    looks = LookSchedule(labels=("w32", "w34", "w36", "w38"), information=(0.25, 0.5, 0.75, 1.0))
    return StoppingRule(
        name="GX-02 O'Brien-Fleming", looks=looks, boundaries=(obrien_fleming(0.05, looks),)
    )


def weekly_path(
    kpi: Array, f: H.Fielded, weeks: Sequence[int]
) -> tuple[list[float], list[float], list[float]]:
    """``(z, lift, se)`` at each cumulative look, positive when the channel works."""
    zs, lifts, ses = [], [], []
    exp = f.experiment
    for k in weeks:
        e = did(kpi, f, post=slice(exp.start, exp.start + k))
        sign = 1.0 if exp.kind == "dose" else -1.0
        zs.append(sign * e.effect / e.se)
        lifts.append(sign * e.effect)
        ses.append(e.se)
    return zs, lifts, ses


@dataclass(frozen=True)
class Readout:
    """What notebook 3 filed for one experiment, and what it computed on the way."""

    plan: Plan
    fielded: H.Fielded
    run: ExperimentRun
    measurement: Measurement | None
    estimate: MethodEstimate | None
    detail: dict[str, Any]


def read_year(
    worlds: Mapping[str, H.World],
    builders: Mapping[str, ExperimentBuilder],
    plans: Mapping[str, Plan],
    fielded: Mapping[str, H.Fielded],
) -> dict[str, Readout]:
    """Read every experiment the way notebook 3 read it, and file it."""
    out: dict[str, Readout] = {}
    for exp_id, plan in plans.items():
        f = fielded[exp_id]
        exp = plan.experiment
        w = worlds[exp.party]
        builder = builders[exp.party]
        kpi = kpi_seen(w, f)
        run = builder.start(plan.run, at=f"2026-W{exp.start:02d}")
        detail: dict[str, Any] = {}
        if exp_id == "GX-02":
            zs, lifts, ses = weekly_path(kpi, f, (2, 4, 6, 8))
            rule = gx02_rule()
            path = monitor(rule, zs[:3], effects=lifts[:3], ses=ses[:3])
            run, correction = builder.read_stopped(run, path, plan.estimand, at="2026-W36")
            detail.update(path=path, correction=correction, z=zs, lifts=lifts, ses=ses)
            out[exp_id] = Readout(plan, f, run, builder.store.get(run.readout_hash), None, detail)
            continue
        if exp_id == "NW-03":
            # The analyst re-drew the assignment after commit and filed nothing (notebook 3 §8).
            redrawn = arms(exp, seed=99)
            run = run.model_copy(
                update={"roles": {**run.roles, "assignment": redrawn.spec.content_hash()}}
            )
            f = H.field_experiment(
                w, exp, redrawn.units_in("treated"), redrawn.units_in("control"), doses=f.doses
            )
            kpi = kpi_seen(w, f)
        e = did(kpi, f)
        measurement = lift_measurement(exp, plan.estimand, e, source=exp_id)
        if exp_id == "GX-01":
            longer = H.Experiment(
                exp.id,
                exp.party,
                exp.channel,
                exp.kind,
                exp.start,
                exp.stop + 2,
                exp.n_treated,
                exp.uplift,
                exp.randomized,
            )
            run = run.deviate(
                "occupancy",
                H.occupancy(longer, H.MARKETS),
                reason="the tv buy could not be restarted until week 18; the holdout ran ten weeks",
                state="asserted",
                at="2026-W16",
            ).deviate(
                "window",
                longer.window,
                reason="read over the ten weeks the holdout ran",
                state="asserted",
                at="2026-W18",
            )
            e = did(kpi, f, post=longer.post)
            measurement = lift_measurement(exp, plan.estimand, e, source=exp_id)
        if exp_id == "NW-02":
            run = run.deviate(
                "dose",
                f"{1.0 + exp.uplift * float(np.mean(list(f.delivered_fraction.values()))):.2f}x "
                "delivered on average",
                reason=(
                    "the platform cleared 65-85% of the doubled budget; the readout is the "
                    "effect of assigning the doubling, not of receiving it"
                ),
                state="asserted",
                at="2026-W24",
            )
        if exp_id == "AC-04":
            run = run.deviate(
                "population",
                Population(name="markets_whose_feed_stayed_up"),
                reason=(
                    "ten control markets went dark after week 44; the readout is over the "
                    "thirty that reported"
                ),
                state="unverified",
                at="2026-W50",
            )
        run = builder.read(run, measurement, at=f"2026-W{exp.stop:02d}")
        out[exp_id] = Readout(plan, f, run, measurement, e, detail)
    return out


def guardrail_frame(w: H.World, f: H.Fielded) -> dict[str, MethodEstimate]:
    """Each guardrail's difference-in-differences for this experiment."""
    exp = f.experiment
    out = {}
    for name, values in w.guardrail_values.items():
        rows = H.observed_markets(H.with_missing(values, f.missing), exp.post)
        treated = [rows.index(i) for i in f.treated_indices() if i in rows]
        e = estimate(
            "difference_in_differences", panel_arrays(values[rows], treated, exp.pre, exp.post)
        )
        assert isinstance(e, MethodEstimate)
        out[name] = e
    return out


def fielded_frame(worlds: Mapping[str, H.World], fielded: Mapping[str, H.Fielded]) -> pd.DataFrame:
    """One row per experiment: what ran, where, and the truth the world knows."""
    rows = []
    for exp_id, f in fielded.items():
        exp = f.experiment
        w = worlds[exp.party]
        truth = (w.mean(f.doses) - w.mean({**f.doses, exp.channel: w.doses[exp.channel]}))[
            f.treated_indices()
        ][:, exp.post].mean()
        rows.append(
            {
                "experiment": exp_id,
                "party": exp.party,
                "channel": exp.channel,
                "kind": exp.kind,
                "weeks": f"{exp.start}-{exp.stop}",
                "treated": len(f.treated),
                "true_lift": -truth if exp.kind != "dose" else truth,
            }
        )
    return pd.DataFrame(rows)


__all__ = [
    "ALLOCATIONS",
    "FIXED_COST",
    "MARKET_UNIT",
    "REFUSED",
    "RESCHEDULED",
    "VALUE_PER_KPI",
    "DesignInputs",
    "Plan",
    "Readout",
    "arms",
    "candidate",
    "cluster_design",
    "conversion",
    "design_inputs",
    "did",
    "experiment_se",
    "field_year",
    "fielded_frame",
    "guardrail_frame",
    "gx02_rule",
    "kpi_seen",
    "lift_estimand",
    "lift_measurement",
    "open_house",
    "plan_year",
    "planned_calendar",
    "programs",
    "read_year",
    "register_definitions",
    "weekly_path",
]
