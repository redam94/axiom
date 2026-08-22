"""Education — how much tutoring would a confounder have to explain to erase it?

The question
    Students who take extra tutoring score higher. How much of that is the
    tutoring? Prior attainment drives both who signs up and how they score, so the
    raw gap is the programme plus the sorting.

Two cases, deliberately
    First: prior attainment is recorded. axiom finds the back-door route, names the
    set to condition on, and the adjusted estimate lands on the truth.

    Second: it is not recorded -- which is the situation most school datasets are
    actually in. There is no unbiased estimate to be had. Rather than return one
    anyway, axiom downgrades the verdict, and the honest remaining move is to say
    how strong an unmeasured confounder would have to be to overturn the finding,
    and let a reader judge whether one that strong is plausible.

Pillars: identify (back-door), diagnose (sensitivity analysis)
"""

import numpy as np

from axiom.diagnose import bias_bounds, robustness_value, tipping_point
from axiom.identify import identify, minimal_adjustment_sets, ols
from axiom.sim import confounded_world, hidden_confounder_world

N, SEED = 5_000, 0

# --- case one: the confounder is measured ------------------------------------
world = confounded_world()
truth = world.total_effect("X", "Y")
verdict = identify(world.graph, "X", "Y")

print("CASE ONE - prior attainment is recorded")
print(f"  graph            : {world.graph.to_text()}")
print(f"  status           : {verdict.status} via {verdict.route}")
print(f"  condition on     : {sorted(verdict.adjustment_set)}")
print(f"  minimal sets     : {[sorted(s) for s in minimal_adjustment_sets(world.graph, 'X', 'Y')]}")

frame = world.observed(world.simulate(N, seed=SEED))
naive = ols(frame, "Y", "X")
adjusted = ols(frame, "Y", "X", covariates=verdict.adjustment_set)
print(f"\n  truth                     {truth:6.3f}")
for label, est in (("raw gap", naive), ("back-door adjusted", adjusted)):
    ci = est.ci(0.95)
    print(
        f"  {label:24s}  {est.estimate:6.3f}   [{ci.lower:6.3f}, {ci.upper:6.3f}]   "
        f"error {est.estimate - truth:+.3f}"
    )

# --- case two: it is not ------------------------------------------------------
hidden = hidden_confounder_world()
hidden_verdict = identify(hidden.graph, "X", "Y")
hidden_frame = hidden.observed(hidden.simulate(N, seed=SEED))
biased = ols(hidden_frame, "Y", "X")

print("\n\nCASE TWO - prior attainment was never recorded")
print(f"  columns you have : {hidden_frame.columns.tolist()}")
print(f"  status           : {hidden_verdict.status} ({hidden_verdict.route})")
print(f"  would need       : {sorted(hidden_verdict.unmeasured_required)}, which you do not have")
print(
    f"  fitting anyway   : {biased.estimate:.3f} against a truth of {truth:.3f} "
    f"({biased.estimate - truth:+.3f})"
)

# How strong would the thing you cannot see have to be?
df = biased.n - 2
rv = robustness_value(estimate=biased.estimate, se=biased.se, df=df, q=1.0, alpha=0.05)
print(f"\n  robustness value : {rv.rv:.1%}")
print(f"    an unmeasured confounder would have to explain {rv.rv:.1%} of the residual")
print("    variance in BOTH tutoring and scores to wipe the estimate out entirely.")

print("\n  the estimate under a confounder of each strength:")
print(f"  {'benchmark':<26}{'R2 w/ treat':>12}{'R2 w/ outcome':>15}{'adjusted':>11}")
for label, r2_treat, r2_out in (
    ("a tenth as strong", 0.039, 0.056),
    ("half as strong", 0.195, 0.281),
    ("as strong as attainment", 0.390, 0.562),
):
    b = bias_bounds(
        estimate=biased.estimate, se=biased.se, df=df, r2_yz_dx=r2_out, r2_dz_x=r2_treat, mass=0.95
    )
    print(f"  {label:<26}{r2_treat:>12.3f}{r2_out:>15.3f}{b.adjusted_estimate:>11.3f}")

# And does any of it change the decision that is actually on the table?
DECISION_THRESHOLD = 1.0
rng = np.random.default_rng(SEED)
draws = rng.normal(biased.estimate, biased.se, size=4000)
tp = tipping_point(draws, DECISION_THRESHOLD, np.linspace(0.0, 2.5, 26), certainty=0.9)
print(f"\n  decision: fund the programme if the effect exceeds {DECISION_THRESHOLD:.1f}")
print(f"    at zero bias      : {tp.decision_at_zero} (P = {tp.probability_at_zero:.3f})")
print(f"    flips once bias   : {tp.bias:.1f}")
print(f"    real bias here    : {biased.estimate - truth:.3f}")
print("    so on this decision, at this threshold, the confounding is not large")
print("    enough to change the call -- which is a far more useful sentence than")
print("    either 'it is significant' or 'we cannot know'.")
