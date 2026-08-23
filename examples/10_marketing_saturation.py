"""Marketing — contribution and marginal return, with the vocabulary bolted on top.

Why this file exists
    axiom's core has no idea what a channel is. Its nouns are Treatment, Dose, Unit,
    Outcome, Covariate, and a contract test fails the build if the words 'channel',
    'spend', 'geo' or 'ROAS' appear outside axiom.adapters. Marketing is ONE adapter
    over a domain-general core, not the thing the library is about.

    This example is therefore doing double duty: it is a working marketing analysis,
    and it is the proof that the vocabulary is a thin translation layer. Channel IS
    Treatment. Geo IS Unit. KPI IS Outcome. The same objects, renamed at the edge.

The question
    Given weekly spend by channel and geo, what did each channel contribute, and
    what would the next unit of spend return? Those are different questions and the
    second is the one budgets are set on.

Pillars: adapters (marketing), surface
"""

import numpy as np
import pandas as pd
from _walkthrough import Walkthrough

from axiom.adapters import (
    KPI,
    Channel,
    Geo,
    MarketingRoles,
    contribution,
    marginal_roas,
    marketing_spec,
    panel_from_marketing_frame,
    roas,
    role_map,
    spend,
)
from axiom.core import D, Intervention, Outcome, Treatment, Unit
from axiom.sim import DosePlan, surface_world
from axiom.surface import HillKernel, counterfactual_doses, fit, marginal_band, response_band

w = Walkthrough(
    field="Marketing",
    title="Which side of the bend are you standing on?",
    question="""What did each channel contribute, and what would the next unit of
        spend return? Those are different questions, the second is the one budgets
        are set on, and a single headline ROAS cannot tell them apart.""",
)

# ----------------------------------------------------------------------------------

w.step(
    "Check that the adapter is aliases and not a parallel type system",
    why="""This is the load-bearing claim of the whole design, so it is worth
        checking with `is` rather than asserting in prose. Channel IS Treatment. Geo
        IS Unit. KPI IS Outcome — the same objects, renamed at the edge. If these
        were separate classes, every function downstream would need a marketing
        variant, and the core would slowly acquire a domain.""",
    instead="""A MarketingTreatment class that wraps a Treatment. It reads more
        explicitly and it is the beginning of a fork: two type hierarchies that must
        be kept in step by hand, and a core that gradually learns what a channel
        is.""",
)
w.out(f"Channel is Treatment : {Channel is Treatment}")
w.out(f"Geo     is Unit      : {Geo is Unit}")
w.out(f"KPI     is Outcome   : {KPI is Outcome}")
w.out(f"spend('paid_search') -> {spend('paid_search')}")

# ----------------------------------------------------------------------------------

w.step(
    "Take the frame in the shape a planner actually hands you",
    why="""Weekly rows, one column per channel, a geo and a date. Nobody in marketing
        holds data in axiom's internal shape, so the adapter's real job is the
        translation, and role_map makes that translation inspectable rather than
        buried in a reshape.""",
    instead="""Asking the planner to reshape into long format first. That moves the
        error-prone step outside the library, where it is done differently by every
        analyst and reviewed by nobody.""",
)
rng = np.random.default_rng(4)
frame = pd.DataFrame(
    [
        {
            "geo": geo,
            "week": week,
            "revenue": 0.0,  # filled by the simulation below
            "paid_search": float(abs(rng.normal(38, 9))),
            "social": float(abs(rng.normal(17, 6))),
            "holiday": float(week in (10, 11)),
        }
        for geo in ("north", "south", "west")
        for week in range(16)
    ]
)
roles = MarketingRoles(
    kpi="revenue",
    channels=("paid_search", "social"),
    geo="geo",
    date="week",
    controls=("holiday",),
    kpi_dimension="currency",
    currency="USD",
)
mapped = role_map(roles)
panel = panel_from_marketing_frame(frame, roles)
w.out(f"role map    : unit {mapped.unit!r}, time {mapped.time!r}, outcome {mapped.outcome[1]!r}")
w.out(f"treatments  : {list(mapped.treatments)}")
w.out(f"panel       : {panel.units} geos x {len(panel.periods)} weeks")
w.out(f"balanced    : {panel.completeness().balanced}")

# ----------------------------------------------------------------------------------

w.step(
    "Let the spec default to saturation and carryover",
    why="""marketing_spec puts a Hill kernel and a carryover on every channel by
        default, because spend does neither of the two things a linear model assumes
        it does: it does not return proportionally forever, and it does not stop
        working at midnight on Sunday. Making these the defaults rather than options
        means the wrong model has to be chosen deliberately.""",
    instead="""A linear regression on spend with adstock bolted on afterwards. It
        fits, it reports a coefficient, and that coefficient is the average return
        over whatever spend range happened to occur — which is not the number any
        budget decision needs.""",
)
spec = marketing_spec(panel, seasonality=(4.0, 1), trend=True)
w.out(f"treatments : {spec.treatment_names}")
w.out(f"intercept  : {spec.intercept!r}")
w.out(f"kernels    : {({k: v.name for k, v in spec.kernels.items()})}")
w.out(f"carryover  : {({k: v.name for k, v in spec.carryover.items()})}")

# ----------------------------------------------------------------------------------

w.step(
    "Fit a world whose truth is known, so the numbers are checkable",
    why="""The point of this example is not to be persuasive about marketing, it is to
        show that the marketing quantities come out of the same fit() as the
        agriculture and clinical examples. A simulated channel with a known Hill
        curve — its half-saturation point deliberately inside the observed spend
        range, so the data actually sees the bend — makes every number below
        gradeable.""",
    instead="""Fitting the frame built above. It has no signal in it: the revenue
        column is zeros. A demonstration that fits noise and reports confident
        quantities is worse than no demonstration.""",
)
revenue_world = surface_world(
    n_units=3,
    n_periods=16,
    treatments=("paid_search",),
    outcome=Outcome(name="revenue", dimension=D.currency, unit="USD"),
    kernels=HillKernel(reference_dose=38.0, amplitude_scale=3.0),
    doses=DosePlan(scale=38.0, spread=0.9),
    intercept="shared",
    truth={"beta_paid_search": 3.0, "k_paid_search": 30.0, "s_paid_search": 1.8, "alpha": 1.5},
    noise_sd=0.15,
    seed=7,
)
result = fit(
    revenue_world.spec, revenue_world.panel, backend="laplace", draws=1200, chains=1, seed=7
)
w.out(f"converged      : {result.converged}")
w.out("true half-saturation spend : 30.0, inside the observed range")


def truth_at(dose: float) -> float:
    data = counterfactual_doses(
        revenue_world.surface, revenue_world.data, Intervention(doses={"paid_search": dose})
    )
    return float(revenue_world.surface.forward(data, revenue_world.theta).mean())


doses = np.linspace(0.0, 100.0, 30)
band = response_band(result, "paid_search", doses=doses.tolist(), mass=0.9, seed=0)
w.figure(
    "response",
    kind="band",
    data={
        "treatment": "paid_search",
        "doses": doses.tolist(),
        "mean": list(band.mean),
        "lower": list(band.lower),
        "upper": list(band.upper),
        "truth": [truth_at(float(d)) for d in doses],
        "mass": 0.9,
    },
    opt={"key": "@", "xLabel": "weekly spend", "yLabel": "revenue", "height": 300},
    title="The response curve the fit recovered",
    note="""S-shaped, not merely concave. There is a stretch at low spend where the
        curve is getting steeper, and everything surprising below follows from
        that.""",
    legend=(
        ("accent", "posterior mean"),
        ("accent-wash", "90% interval"),
        ("ink-2:dash", "the truth"),
    ),
)

# ----------------------------------------------------------------------------------

w.step(
    "Ask for the three named quantities",
    why="""Contribution, ROAS and marginal ROAS are three different questions and
        each is an estimand with its own definition, interval and provenance.
        Reporting them together makes the difference visible: the first is a total,
        the second is an average, the third is a derivative.""",
    instead="""Reporting ROAS alone, which is what a marketing dashboard does. It is
        an average over the whole observed spend range, so it answers 'was this
        channel worth running' and is routinely used to answer 'should we spend
        more' — a question it cannot address.""",
)
rows = []
for fn in (contribution, roas, marginal_roas):
    out = fn(result, "paid_search", mass=0.9, seed=0)
    if hasattr(out, "reason"):
        w.out(f"{fn.__name__:<18} refused: {out.reason}")
        continue
    w.out(
        f"{out.estimand_name:<26}{out.kind:<11}{out.summary.mean:>10.4f}   {out.summary.interval}"
    )
    rows.append([out.estimand_name, out.kind, f"{out.summary.mean:.4f}", str(out.summary.interval)])
w.table(["estimand", "kind", "mean", "90% interval"], rows)

avg = roas(result, "paid_search", mass=0.9, seed=0)
marg = marginal_roas(result, "paid_search", mass=0.9, seed=0)
w.out("")
w.out(f"average return per unit spent : {avg.summary.mean:.4f}")
w.out(f"return on the NEXT unit spent : {marg.summary.mean:.4f}")
w.out(f"ratio                         : {marg.summary.mean / avg.summary.mean:.2f}")

# ----------------------------------------------------------------------------------

w.step(
    "Then look at the whole marginal curve, not one summary of it",
    why="""The number that answers 'should we spend more' is the slope at the spend
        level you are actually at. marginal_band gives that at every level with its
        uncertainty, and on a Hill curve it is not monotone: below the inflection an
        extra unit returns MORE than the average, above it less.""",
    instead="""Comparing the single marginal ROAS to the single average ROAS and
        concluding 'we are past saturation'. One ratio at one implicit spend level
        cannot locate the bend, and the bend is the whole decision.""",
)
grid = [4, 8, 16, 24, 32, 40, 55, 75, 100]
marginal = marginal_band(result, "paid_search", doses=grid, mass=0.9, seed=0)
if hasattr(marginal, "reason"):
    w.out(f"marginal curve refused: {marginal.reason}")
else:
    w.table(
        ["weekly spend", "return on the next unit", "90% interval"],
        [
            [f"{d:.0f}", f"{m:.4f}", f"[{lo:.4f}, {hi:.4f}]"]
            for d, m, lo, hi in zip(
                marginal.doses, marginal.mean, marginal.lower, marginal.upper, strict=True
            )
        ],
    )
    peak = max(range(len(marginal.mean)), key=lambda i: marginal.mean[i])
    w.out(f"the marginal peaks around spend {marginal.doses[peak]:.0f} and falls after.")
    w.figure(
        "marginal",
        kind="band",
        data={
            "treatment": "paid_search",
            "doses": list(marginal.doses),
            "mean": list(marginal.mean),
            "lower": list(marginal.lower),
            "upper": list(marginal.upper),
            "mass": 0.9,
        },
        opt={
            "key": "@",
            "xLabel": "weekly spend",
            "yLabel": "revenue from the next unit of spend",
            "height": 300,
        },
        title="Return on the next unit, at every spend level",
        note=f"""It rises, peaks near {marginal.doses[peak]:.0f}, and falls. Left of
            the peak, more spend is worth more than the last unit was. Right of it,
            less. A headline ROAS reports one number for this entire curve.""",
        legend=(("accent", "marginal return"), ("accent-wash", "90% interval")),
    )

    base = float(band.mean[0])
    fine = np.linspace(4.0, 100.0, 25)
    fine_band = response_band(result, "paid_search", doses=fine.tolist(), mass=0.9, seed=0)
    average_curve = [(float(fine_band.mean[i]) - base) / float(fine[i]) for i in range(len(fine))]
    marginal_fine = marginal_band(result, "paid_search", doses=fine.tolist(), mass=0.9, seed=0)
    w.figure(
        "crossing",
        kind="lines",
        data={
            "spend": fine.tolist(),
            "average": average_curve,
            "marginal": list(marginal_fine.mean),
        },
        opt={
            "series": [
                {"label": "average", "x": "@spend", "y": "@average", "colour": "ink-2"},
                {"label": "marginal", "x": "@spend", "y": "@marginal", "colour": "accent"},
            ],
            "xLabel": "weekly spend",
            "yLabel": "revenue per unit of spend",
            "height": 300,
            "rightPad": 90,
        },
        title="Average and marginal return, on the same axis",
        note="""They cross. Where marginal is above average, the average is still
            rising and the channel is under-invested; where it falls below, every
            further unit drags the average down. The crossing point is the number a
            budget conversation is looking for, and neither summary statistic on its
            own can locate it.""",
        legend=(("ink-2", "average return"), ("accent", "return on the next unit")),
    )

# ----------------------------------------------------------------------------------

w.step(
    "Narrow the window, because the question usually has one",
    why="""'What is the marginal return' almost always means 'over the period I am
        planning for'. A window restricts the estimand to part of the horizon, and
        because it is part of the estimand rather than a filter applied afterwards,
        the resulting number carries which window it belongs to.""",
    instead="""Slicing the data and refitting. That changes the model as well as the
        question, and the two effects become impossible to separate — a smaller
        window means less data, and less data means a different fitted curve.""",
)
late = marginal_roas(result, "paid_search", window=(8, 16), mass=0.9, seed=0)
w.out(f"marginal over the whole horizon : {marg.summary.mean:.4f}")
w.out(f"marginal over weeks 8-16 only   : {late.summary.mean:.4f}")

# ----------------------------------------------------------------------------------

w.finding("""Everything above went through the same fit(), the same forward() and the same
    estimand machinery as the agriculture and clinical examples in this directory.
    The only marketing-specific code is the translation at the top and the three
    named quantities in the middle.""")
w.finding(f"""The headline numbers barely differ — marginal {marg.summary.mean:.4f} against
    average {avg.summary.mean:.4f}, a ratio of
    {marg.summary.mean / avg.summary.mean:.2f} — and that near-equality is itself
    misleading. The marginal curve behind those summaries runs from
    {marginal.mean[0]:.4f} at a spend of {marginal.doses[0]:.0f}, up to
    {max(marginal.mean):.4f} at {marginal.doses[peak]:.0f}, and down to
    {marginal.mean[-1]:.4f} at {marginal.doses[-1]:.0f}. It rises before it falls,
    because a Hill curve is S-shaped rather than merely concave — so the next unit of
    spend can be worth more than the last one was.""")
w.finding(f"""That is a factor of {max(marginal.mean) / marginal.mean[-1]:.0f} between the
    best and worst place to put the next unit, hidden inside a pair of summary
    statistics that agree with each other to two decimal places. 'Our ROAS is
    {avg.summary.mean:.3f}' does not say which side of the bend the budget is
    standing on, and it is the sentence most budgets are set with.""")
w.finding("""That is the design commitment made concrete. The mathematics is about doses and
    responses; the moment a domain's vocabulary leaks into it, the same code stops
    being usable for trials, fields, reactors and reserves — and the marketing
    analysis is not made any better by the leak.""")
