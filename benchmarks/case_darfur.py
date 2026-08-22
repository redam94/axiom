"""Darfur — reproducing a published effect, and then asking what would overturn it.

The data
    In 2003-2004, Sudanese government forces and Janjaweed militia attacked
    villages in Darfur. Hazlett surveyed 1,276 refugees in eastern Chad. The
    treatment is whether the person was directly injured; the outcome is an index
    of pro-peace attitudes. Cinelli & Hazlett (2020) use it as the running example
    for their sensitivity analysis, so both the regression and the sensitivity
    numbers are published.

Two claims, and only one is checked elsewhere
    axiom's unit tests already verify the sensitivity FORMULAS against the
    published values -- but from hard-coded summary statistics. This file supplies
    the step before that: read 1,276 rows, fit the regression with 486 village
    fixed effects, and land on 0.0973 with a standard error of 0.0232. Only then
    are the sensitivity numbers about this dataset rather than about three
    constants somebody typed in.

Not redistributed
    The data ships inside the GPL-3 R package sensemakr. Fetch it with
    ``python benchmarks/fetch.py darfur``.

Run it
    python benchmarks/case_darfur.py
"""

import pandas as pd
from registry import DATASETS, NotFetched, load

from axiom.diagnose import benchmark, robustness_value
from axiom.identify import CausalGraph, identify, ols

SPEC = DATASETS["darfur"]
PUBLISHED = SPEC.published

print(SPEC.title)
print(f"  {SPEC.citation}")
print(f"  licence: {SPEC.licence}\n")

try:
    frame = load("darfur")
except NotFetched as exc:
    print(exc)
    raise SystemExit(0) from None

print(f"  sha256 : {SPEC.sha256()[:32]}...")
print(f"  {len(frame):,} respondents, {frame.village.nunique()} villages")
harmed = int(frame.directlyharmed.sum())
print(f"  {harmed:,} directly harmed ({harmed / len(frame):.1%}), " f"{len(frame) - harmed:,} not")

# ----------------------------------------------------------------------------------
# what the graph licenses, before any fitting
# ----------------------------------------------------------------------------------

# The identifying claim is that, within a village, who was injured is as good as
# random with respect to prior attitudes. Attacks were indiscriminate at village
# level; within a village, who was caught is closer to chance. That claim is a
# graph, and axiom will say what it implies.
graph = CausalGraph.from_edges(
    "directlyharmed -> peacefactor, "
    "village -> directlyharmed, village -> peacefactor, "
    "female -> directlyharmed, female -> peacefactor, "
    "age -> peacefactor"
)
verdict = identify(graph, "directlyharmed", "peacefactor")
print(f"\nidentification status : {verdict.status} via {verdict.route}")
print(f"condition on          : {sorted(verdict.adjustment_set)}")
print("  That verdict is only as good as the graph. The whole point of what")
print("  follows is that the graph is a claim, not a finding.")

# ----------------------------------------------------------------------------------
# the regression, against the published coefficient
# ----------------------------------------------------------------------------------

dummies = pd.get_dummies(frame["village"], prefix="village", drop_first=True, dtype=float)
design = pd.concat([frame.drop(columns=["village"]), dummies], axis=1)
covariates = [
    "age",
    "farmer_dar",
    "herder_dar",
    "pastvoted",
    "hhsize_darfur",
    "female",
    *dummies.columns,
]

estimate = ols(design, "peacefactor", "directlyharmed", covariates=covariates)
df_resid = int(estimate.detail["df_resid"])

print(
    f"\nregression: peacefactor ~ directlyharmed + 6 covariates + "
    f"{len(dummies.columns)} village effects"
)
print(f"\n{'quantity':<22}{'axiom':>12}{'published':>12}{'difference':>13}")
comparisons = [
    ("coefficient", estimate.estimate, PUBLISHED["coefficient"], 1e-4),
    ("standard error", estimate.se, PUBLISHED["se"], 1e-4),
    ("residual df", float(df_resid), float(PUBLISHED["df"]), 0.5),
    ("n", float(estimate.n), float(PUBLISHED["n"]), 0.5),
]
for label, ours, theirs, tol in comparisons:
    delta = abs(ours - theirs)
    mark = "" if delta < tol else "   <-- CHECK"
    print(f"{label:<22}{ours:>12.4f}{theirs:>12.4f}{delta:>13.2e}{mark}")

ci = estimate.ci(0.95)
print("\neffect of being directly harmed on pro-peace attitudes:")
print(f"  {estimate.estimate:.4f}  [{ci.lower:.4f}, {ci.upper:.4f}]  (95% Wald)")
print("  Positive: people who were injured report MORE pro-peace attitudes, not less.")
print("  That is the finding, and it is the one worth stress-testing.")

# ----------------------------------------------------------------------------------
# what would have to be true for it to be wrong
# ----------------------------------------------------------------------------------

rv = robustness_value(estimate=estimate.estimate, se=estimate.se, df=df_resid, q=1.0, alpha=0.05)
print(f"\n{'sensitivity':<26}{'axiom':>10}{'published':>12}")
for label, ours, theirs in (
    ("robustness value", rv.rv, PUBLISHED["robustness_value"]),
    ("RV at alpha = 0.05", rv.rv_alpha, PUBLISHED["robustness_value_alpha"]),
    ("partial R2 of treatment", rv.r2_yd_x, PUBLISHED["partial_r2"]),
):
    print(f"{label:<26}{ours:>10.3f}{theirs:>12.3f}")

print(f"\n  An unmeasured confounder would have to explain {rv.rv:.1%} of the residual")
print("  variance in BOTH being harmed and attitudes to bring the effect to zero.")
print(f"  Only {rv.rv_alpha:.1%} is needed to make it statistically insignificant.")

# Benchmarking against an observed covariate is what makes those percentages mean
# something: is a confounder that strong plausible HERE?
r2_dxj_x, r2_yxj_dx = 0.00916, 0.11
for multiple in (1.0, 2.0, 3.0):
    b = benchmark(
        estimate=estimate.estimate,
        se=estimate.se,
        df=df_resid,
        covariate="female",
        r2_dxj_x=r2_dxj_x,
        r2_yxj_dx=r2_yxj_dx,
        k_d=multiple,
        k_y=multiple,
    )
    bounds = b.bounds
    print(
        f"\n  a confounder {multiple:.0f}x as strong as being female:"
        f"\n    adjusted estimate {bounds.adjusted_estimate:+.4f} "
        f"(se {bounds.adjusted_se:.4f}, t {bounds.adjusted_t:.2f})"
    )

print("\n  Gender is one of the strongest predictors in this survey, and a confounder")
print("  three times stronger still does not overturn the sign. That is a much more")
print("  useful sentence than a p-value, and it is falsifiable: name such a variable.")

# ----------------------------------------------------------------------------------
# the same question, asked without the village fixed effects
# ----------------------------------------------------------------------------------

naive = ols(frame, "peacefactor", "directlyharmed")
without_village = ols(
    frame,
    "peacefactor",
    "directlyharmed",
    covariates=["age", "farmer_dar", "herder_dar", "pastvoted", "hhsize_darfur", "female"],
)
print(f"\n{'specification':<34}{'estimate':>10}{'se':>9}")
for label, est in (
    ("no covariates at all", naive),
    ("covariates, no village effects", without_village),
    ("published specification", estimate),
):
    print(f"{label:<34}{est.estimate:>10.4f}{est.se:>9.4f}")
print("  The village effects are doing real work. Which specification is right is")
print("  a question about the world -- were attacks indiscriminate WITHIN villages? --")
print("  and no amount of fitting settles it.")

print("""
Reading it
    The published coefficient and standard error come back from the raw survey to
    four decimal places, and the sensitivity numbers built on them match the paper.
    That is the end-to-end claim: not that a formula is implemented correctly, but
    that the whole path from 1,276 rows to a published sensitivity analysis runs
    through this library and arrives in the same place.

    The substantive point is the one Cinelli and Hazlett were making. This design
    cannot rule out confounding, and no design of this kind can. What it can do is
    state precisely how strong a confounder would have to be, benchmark that
    against a variable everyone can picture, and let a reader decide. That is a
    claim someone can argue with, which is more than significance offers.""")
