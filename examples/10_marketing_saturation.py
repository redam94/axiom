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
from axiom.core import D, Outcome, Treatment, Unit
from axiom.sim import DosePlan, surface_world
from axiom.surface import HillKernel, fit, marginal_band

# The adapter is aliases, not a parallel type system.
print("Channel is Treatment :", Channel is Treatment)
print("Geo     is Unit      :", Geo is Unit)
print("KPI     is Outcome   :", KPI is Outcome)
print("\nspend('paid_search') ->", spend("paid_search"))

# --- a marketing frame, in the shape a planner would hand you -----------------
rng = np.random.default_rng(4)
rows = [
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
frame = pd.DataFrame(rows)

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
print(
    f"\nrole map -> unit {mapped.unit!r}, time {mapped.time!r}, "
    f"outcome {mapped.outcome[1]!r}, treatments {list(mapped.treatments)}"
)

panel = panel_from_marketing_frame(frame, roles)
print(
    f"panel: {panel.units} geos x {len(panel.periods)} weeks, "
    f"balanced={panel.completeness().balanced}"
)

spec = marketing_spec(panel, seasonality=(4.0, 1), trend=True)
print(
    f"\nmarketing_spec built: treatments {spec.treatment_names}, " f"intercept {spec.intercept!r}"
)
print(f"  kernels  : {({k: v.name for k, v in spec.kernels.items()})}")
print(f"  carryover: {({k: v.name for k, v in spec.carryover.items()})}")
print("  saturating response and carryover are the defaults, because spend does")
print("  neither of the two things a linear model assumes it does.")

# --- fit a world where we know the truth, so the numbers are checkable --------
revenue_world = surface_world(
    n_units=3,
    n_periods=16,
    treatments=("paid_search",),
    outcome=Outcome(name="revenue", dimension=D.currency, unit="USD"),
    # a genuinely saturating channel: a Hill curve with its half-saturation point
    # inside the observed spend range, so the data actually sees the bend
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
print(f"\nfitted: converged={result.converged}")

print(f"\n{'quantity':<28}{'kind':<11}{'value':>10}   interval")
for fn in (contribution, roas, marginal_roas):
    out = fn(result, "paid_search", mass=0.9, seed=0)
    if hasattr(out, "reason"):
        print(f"{fn.__name__:<28}refused: {out.reason}")
        continue
    print(
        f"{out.estimand_name:<28}{out.kind:<11}{out.summary.mean:>10.3f}   "
        f"{out.summary.interval}"
    )

# Average and marginal return are different numbers, and a single headline ROAS
# hides which side of the curve you are standing on.
avg = roas(result, "paid_search", mass=0.9, seed=0)
marg = marginal_roas(result, "paid_search", mass=0.9, seed=0)
print(f"\naverage return per unit spent  : {avg.summary.mean:.4f}")
print(f"return on the NEXT unit spent  : {marg.summary.mean:.4f}")
print(f"ratio                          : {marg.summary.mean / avg.summary.mean:.2f}")

# The number that actually answers "should we spend more?" is the slope at the
# spend level you are at -- so look at the whole curve, not one summary.
band = marginal_band(
    result, "paid_search", doses=[8, 16, 24, 32, 40, 55, 75, 100], mass=0.9, seed=0
)
if hasattr(band, "reason"):
    print(f"\nmarginal curve refused: {band.reason}")
else:
    print("\nreturn on the next unit of spend, by spend level:")
    for dose, mean, lo, hi in zip(band.doses, band.mean, band.lower, band.upper, strict=True):
        print(f"  spend {dose:6.1f}   marginal {mean:7.4f}   [{lo:7.4f}, {hi:7.4f}]")
    peak = max(range(len(band.mean)), key=lambda i: band.mean[i])
    print(f"\n  the marginal peaks around spend {band.doses[peak]:.0f} and falls after.")
    print("  A Hill curve is S-shaped, not merely concave: below the inflection an")
    print("  extra unit returns MORE than the average, above it less. 'Our ROAS is")
    print("  0.04' does not say which side you are on; this curve does.")

# A window restricts the question to part of the horizon.
late = marginal_roas(result, "paid_search", window=(8, 16), mass=0.9, seed=0)
print(f"\nmarginal over weeks 8-16 only  : {late.summary.mean:.4f}")

print("""
Reading it
    Everything above went through the same fit(), the same forward(), and the same
    estimand machinery as the agriculture and clinical examples in this directory.
    The only marketing-specific code is the translation at the top and the three
    named quantities at the bottom.

    That is the design commitment made concrete: the mathematics is about doses and
    responses, and the moment a domain's vocabulary leaks into it, the same code
    stops being usable for trials, fields, reactors, or reserves.""")
