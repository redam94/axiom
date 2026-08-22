"""Agronomy — nitrogen and irrigation, and where the yield actually peaks.

The question
    A field trial varies nitrogen (kg/ha) and irrigation (mm) across plots. Yield
    responds to both, and saturates in both. Where is the optimum, and if water is
    rationed, how should the remaining budget be split?

Why a response surface and not a regression
    Fitting a straight line to a saturating response puts the predicted optimum at
    whichever corner of the design happens to be highest. The whole point of the
    surface layer is to keep the curvature, so the optimizer is asked about the
    shape the data actually support.

Pillars: surface (kernels, allocation, frontier)
"""

from axiom.sim import arms_world
from axiom.surface import (
    Bounds,
    HillKernel,
    allocate,
    central_composite,
    fit,
    frontier,
    response_band,
)

SEED = 0

# The dose box: nitrogen 0-200 kg/ha, irrigation 0-120 mm.
bounds = Bounds(treatments=("nitrogen", "irrigation"), low=(0.0, 0.0), high=(200.0, 120.0))

# A rotatable central composite design spends its plots where curvature is
# estimable, rather than spreading them evenly and learning the shape badly.
design = central_composite(bounds, alpha="rotatable", center_points=4, inscribed=True)
print(f"design: {design.kind}, {design.n} plots")
print(design.as_frame().round(1).to_string(index=False))

# A synthetic field whose truth we know, so recovery is checkable.
truth = {
    "alpha": 3.0,  # baseline yield, t/ha
    "beta_nitrogen": 4.2,
    "k_nitrogen": 90.0,
    "s_nitrogen": 2.0,
    "beta_irrigation": 2.6,
    "k_irrigation": 45.0,
    "s_irrigation": 1.6,
}
field = arms_world(
    n_units=design.n,
    treatments=("nitrogen", "irrigation"),
    kernels={
        "nitrogen": HillKernel(reference_dose=90.0, amplitude_scale=5.0),
        "irrigation": HillKernel(reference_dose=45.0, amplitude_scale=5.0),
    },
    doses=design.doses(),
    truth=truth,
    noise_sd=0.35,
    seed=SEED,
)

result = fit(field.spec, field.panel, backend="laplace", draws=2000, seed=SEED)
print(f"\nconverged: {result.converged}")

posterior = result.posterior
theta = {name: float(posterior.summary(name).mean) for name in posterior.names()}
print("\nparameter recovery (90% HDI):")
for name in ("beta_nitrogen", "k_nitrogen", "beta_irrigation", "k_irrigation"):
    s = posterior.summary(name, definition="hdi", mass=0.9)
    inside = s.interval.lower <= truth[name] <= s.interval.upper
    print(
        f"  {name:16s} truth {truth[name]:6.1f}   mean {s.mean:6.1f}   "
        f"{s.interval}  {'ok' if inside else 'MISSED'}"
    )

# Where does the response flatten out? The band is every posterior draw pushed
# through the same forward() the likelihood used.
band = response_band(result, "nitrogen", n_grid=9, mass=0.9, seed=SEED)
print("\nyield against nitrogen (irrigation at its observed levels):")
for dose, mean, lo, hi in zip(band.doses, band.mean, band.lower, band.upper, strict=True):
    print(f"  {dose:6.1f} kg/ha   {mean:5.2f} t/ha   [{lo:5.2f}, {hi:5.2f}]")

# Irrigation is rationed. Split a fixed total across the two inputs.
WATER_LIMITED = Bounds(treatments=("nitrogen", "irrigation"), low=(0.0, 0.0), high=(200.0, 60.0))
plan = allocate(
    result.surface, posterior, budget=180.0, bounds=WATER_LIMITED, objective="mean", seed=SEED
)
print("\nbest split of a 180-unit input budget under a 60 mm water cap:")
print("  " + ", ".join(f"{k} {v:.1f}" for k, v in plan.doses.items()))
print(f"  expected yield {plan.expected_outcome:.3f} t/ha  ({plan.status})")

# What is the next unit of budget worth? The shadow price falls as the surface
# saturates — which is the number a planner can actually act on.
budgets = [60.0, 120.0, 180.0, 240.0, 300.0]
front = frontier(result.surface, theta, budgets=budgets, bounds=WATER_LIMITED, seed=SEED)
print("\nbudget frontier:")
print(front.as_frame().round(3).to_string(index=False))
print("\nshadow price of the next input unit, by budget:")
for budget, price in zip(budgets, front.shadow_prices(), strict=True):
    print(f"  {budget:6.0f}  ->  {price:+.4f} t/ha per unit")
