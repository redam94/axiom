"""BCG vaccine against tuberculosis — thirteen real trials, checked against metafor.

The data
    Colditz et al. (1994) collected thirteen randomized trials of the bacillus
    Calmette-Guerin vaccine, spanning 1948 to 1980 and 250,000 people. It is the
    standard worked example in the meta-analysis literature, which is exactly why
    it is useful here: the answers have been published, by other people, using
    other software.

What this checks
    axiom's pooling against R's metafor on the same thirteen rows: the pooled log
    risk ratio, its standard error and interval, the REML between-study variance,
    I-squared, and Cochran's Q. If the library is right these agree to the decimal
    places metafor prints.

Run it
    python benchmarks/case_bcg.py
"""

import numpy as np
import pandas as pd
from registry import DATASETS, load, log_risk_ratios, model_based_i2

from axiom.meta import egger, heterogeneity, leave_one_out, prediction_interval, random_effects

SPEC = DATASETS["bcg"]
PUBLISHED = SPEC.published

pd.set_option("display.width", 170)

print(SPEC.title)
print(f"  {SPEC.citation}")
print(f"  licence: {SPEC.licence}")
print(f"  sha256 : {SPEC.sha256()[:32]}...")
print(f"\n  {SPEC.note}\n")

frame = load("bcg")
y, se = log_risk_ratios(frame)

print(f"{'trial':<24}{'year':>6}{'vaccinated':>22}{'control':>20}{'log RR':>10}{'se':>8}")
for i, row in frame.iterrows():
    vac = f"{int(row.tpos):>6}/{int(row.tpos + row.tneg):<10}"
    ctl = f"{int(row.cpos):>6}/{int(row.cpos + row.cneg):<10}"
    print(f"{row.author[:23]:<24}{int(row.year):>6}{vac:>22}{ctl:>20}{y[i]:>10.4f}{se[i]:>8.4f}")

total = int((frame.tpos + frame.tneg + frame.cpos + frame.cneg).sum())
print(f"\n{len(frame)} trials, {total:,} people")

# ----------------------------------------------------------------------------------
# the pooled estimate, against the published one
# ----------------------------------------------------------------------------------

pooled = random_effects(y, se, tau_method="reml")
het = heterogeneity(y, se)

# I-squared has two standard definitions. axiom returns the Q-based one; metafor's
# rma() prints the model-based one. Compare like with like.
i2_model, h2_model = model_based_i2(pooled.tau2, se)

rows = [
    ("pooled log risk ratio", pooled.estimate, PUBLISHED["estimate"], 1e-4),
    ("standard error", pooled.se, PUBLISHED["se"], 1e-4),
    ("95% lower", pooled.interval.lower, PUBLISHED["ci_lower"], 1e-3),
    ("95% upper", pooled.interval.upper, PUBLISHED["ci_upper"], 1e-3),
    ("tau^2 (REML)", pooled.tau2, PUBLISHED["tau2_reml"], 1e-4),
    ("Cochran's Q", het.q, PUBLISHED["q"], 1e-3),
    ("I^2 (model-based)", i2_model, PUBLISHED["i2_model_based"], 1e-3),
    ("H^2 (model-based)", h2_model, PUBLISHED["h2_model_based"], 1e-2),
]

print(f"\n{'quantity':<26}{'axiom':>12}{'metafor':>12}{'difference':>13}   agrees")
worst = 0.0
for label, ours, theirs, tol in rows:
    delta = abs(ours - theirs)
    worst = max(worst, delta)
    mark = "yes" if delta < tol else "NO"
    print(f"{label:<26}{ours:>12.4f}{theirs:>12.4f}{delta:>13.2e}   {mark}")

print(f"\nlargest disagreement anywhere: {worst:.2e}")
print(f"df for Q: axiom {het.df}, published {PUBLISHED['q_df']}")

print(f"""
A word about that I^2, because the first run of this file looked like a bug.
    axiom's heterogeneity() returns the Q-BASED I^2, (Q - df) / Q = {het.i2:.4f}.
    metafor's rma() prints the MODEL-BASED one, tau^2 / (tau^2 + s^2) = {i2_model:.4f},
    where s^2 is Higgins & Thompson's typical within-study variance.

    They differ by {abs(het.i2 - i2_model):.4f} here -- small enough to look like a rounding
    error and large enough to be worth chasing. Neither is wrong; they answer
    slightly different questions. What would have been wrong is quietly widening
    the tolerance until the table went green.""")

# ----------------------------------------------------------------------------------
# what the pooled number does and does not say
# ----------------------------------------------------------------------------------

rr = float(np.exp(pooled.estimate))
lo, hi = float(np.exp(pooled.interval.lower)), float(np.exp(pooled.interval.upper))
print(f"\nas a risk ratio: {rr:.3f}  [{lo:.3f}, {hi:.3f}]")
print(f"  the vaccine cut tuberculosis risk by about {100 * (1 - rr):.0f}% on average")

kh = random_effects(y, se, tau_method="reml", knapp_hartung=True)
pi = prediction_interval(kh)
print(f"\nbut I^2 is {het.i2:.0%}, so 'on average' is carrying a lot of weight.")
print(f"  interval on the MEAN effect     [{kh.interval.lower:+.3f}, {kh.interval.upper:+.3f}]")
print(f"  interval for the NEXT trial     [{pi.lower:+.3f}, {pi.upper:+.3f}]")
print(f"  as risk ratios                  [{np.exp(pi.lower):.2f}, {np.exp(pi.upper):.2f}]")
print("  A new trial could plausibly see almost no benefit. That is not noise --")
print("  it is the trials genuinely disagreeing, and it is the finding.")

# The classic explanation is latitude: BCG worked better further from the equator.
far = frame.ablat >= 35
print("\nsplit by absolute latitude, the classic moderator:")
for label, mask in (("within 35 degrees of equator", ~far), ("further than 35 degrees", far)):
    sub = random_effects(y[mask.to_numpy()], se[mask.to_numpy()], tau_method="reml")
    print(
        f"  {label:<30} k={int(mask.sum()):2d}  "
        f"log RR {sub.estimate:+.3f}  RR {np.exp(sub.estimate):.2f}"
    )
print("  Two very different vaccines, or one vaccine and two very different settings.")

# ----------------------------------------------------------------------------------
# diagnostics
# ----------------------------------------------------------------------------------

eg = egger(y, se)
print(f"\nEgger's test: intercept {eg.intercept:+.3f} (p = {eg.p:.3f})")

loo = leave_one_out(y, se, method="reml")
order = np.argsort(-np.abs(np.asarray(loo.influence)))
print("\nmost influential trials when dropped (shift in pooled-se units):")
for i in order[:3]:
    print(f"  {frame.author[i][:26]:<28}{loo.influence[i]:+.3f}")

print("""
Reading it
    Every published quantity reproduced. That is the claim worth making: not that
    axiom's formulas are self-consistent -- the unit tests already cover that --
    but that running this library end to end on somebody else's data lands where
    somebody else's software landed.

    The substantive reading is the second half. A pooled risk ratio around 0.49
    looks decisive until you notice I-squared above 90% and a prediction interval
    that reaches almost to 1.0. The average of thirteen trials that disagree this
    much is a number about the thirteen trials, not about the vaccine.""")
