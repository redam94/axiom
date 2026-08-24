"""LaLonde's NSW — the experiment, and what happens when you throw it away.

The data
    The National Supported Work Demonstration randomly assigned disadvantaged
    workers to a subsidised training programme in the mid-1970s. Because
    assignment was random, the difference in 1978 earnings between arms is the
    causal effect, full stop. Dehejia & Wahba's 445-unit subsample puts it at
    about $1,794.

    LaLonde (1986) then did the thing that made the dataset famous: he threw the
    randomized controls away and replaced them with survey respondents from the
    PSID, then asked whether the econometric methods of the day could recover
    $1,794 from that. Mostly they could not, and often the sign was wrong.

Why it is here
    It is the cleanest available demonstration that an estimate can be careful,
    well-specified, plausible, and wrong -- and that nothing inside the data tells
    you which. That is the claim axiom's identification layer exists to make
    operational, so it should be checked against the canonical case.

Not redistributed
    CC BY-NC. Fetch with ``python benchmarks/fetch.py lalonde_nsw psid_controls``.

Run it
    python benchmarks/case_lalonde.py
"""

import pandas as pd
from registry import DATASETS, NotFetched, load

from axiom.diagnose import robustness_value
from axiom.identify import CausalGraph, identify, ols

SPEC = DATASETS["lalonde_nsw"]
PUBLISHED = SPEC.published
COVARIATES = ["age", "education", "black", "hispanic", "married", "nodegree", "re74", "re75"]

print(SPEC.title)
print(f"  {SPEC.citation}")
print(f"  licence: {SPEC.licence}\n")

try:
    experiment = load("lalonde_nsw")
    psid = load("psid_controls")
except NotFetched as exc:
    print(exc)
    raise SystemExit(0) from None

treated = experiment[experiment.treat == 1]
randomized_controls = experiment[experiment.treat == 0]
print(
    f"  randomized sample : {len(experiment)} people "
    f"({len(treated)} treated, {len(randomized_controls)} control)"
)
print(f"  PSID comparison   : {len(psid):,} survey respondents, none in the programme")

# ----------------------------------------------------------------------------------
# the experimental benchmark
# ----------------------------------------------------------------------------------

experimental = ols(experiment, "re78", "treat")
ci = experimental.ci(0.95)
print("\nEXPERIMENTAL BENCHMARK (randomization does the work)")
print(
    f"  effect on 1978 earnings  ${experimental.estimate:,.0f}  "
    f"[${ci.lower:,.0f}, ${ci.upper:,.0f}]"
)
print(f"  published                ${PUBLISHED['experimental_ate']:,.0f}")
print(
    f"  difference               ${abs(experimental.estimate - PUBLISHED['experimental_ate']):,.2f}"
)

adjusted = ols(experiment, "re78", "treat", covariates=COVARIATES)
print(f"\n  adjusting for covariates anyway: ${adjusted.estimate:,.0f} " f"(se {adjusted.se:,.0f})")
print("  Barely moves, which is what randomization is supposed to buy: the")
print("  covariates were already balanced, so conditioning on them changes little.")

balance = pd.DataFrame(
    {
        "treated": treated[COVARIATES].mean(),
        "randomized control": randomized_controls[COVARIATES].mean(),
        "PSID comparison": psid[COVARIATES].mean(),
    }
).round(1)
print(f"\ncovariate balance:\n{balance.to_string()}")
print("  The randomized arms match. The PSID group is a different population:")
print("  older, better educated, and earning an order of magnitude more before")
print("  the programme even started.")

# ----------------------------------------------------------------------------------
# now discard the experiment
# ----------------------------------------------------------------------------------

observational = pd.concat([treated, psid], ignore_index=True)
print(f"\nOBSERVATIONAL VERSION: {len(treated)} treated vs {len(psid):,} PSID controls")

naive = ols(observational, "re78", "treat")
controlled = ols(observational, "re78", "treat", covariates=COVARIATES)

print(f"\n{'specification':<40}{'estimate':>12}{'se':>10}{'vs truth':>12}")
truth = experimental.estimate
for label, est in (
    ("experimental benchmark", experimental),
    ("PSID controls, no adjustment", naive),
    ("PSID controls, 8 covariates", controlled),
):
    print(f"{label:<40}{est.estimate:>12,.0f}{est.se:>10,.0f}" f"{est.estimate - truth:>12,.0f}")

print("\n  The unadjusted observational estimate says the programme destroyed")
print("  earnings. Controlling for everything available recovers some of it and")
print("  still misses the benchmark badly. Both regressions are correctly")
print("  computed; both answer a question other than the one asked.")

# ----------------------------------------------------------------------------------
# what axiom says before any of that is fitted
# ----------------------------------------------------------------------------------

print("\nWhat the identification layer says, from the graph alone:")

randomized_graph = CausalGraph.from_edges(
    "treat -> re78, age -> re78, education -> re78, re75 -> re78"
)
v_rand = identify(randomized_graph, "treat", "re78")
print(
    f"  randomized   : {v_rand.status} via {v_rand.route}, "
    f"adjust for {sorted(v_rand.adjustment_set) or 'nothing'}"
)

# In the observational version, whatever made someone enrol also drives earnings,
# and it is not in the file.
observational_graph = CausalGraph.from_edges(
    "treat -> re78, age -> treat, age -> re78, education -> treat, education -> re78, "
    "re75 -> treat, re75 -> re78, motivation -> treat, motivation -> re78",
    # the node exists in the world and not in the file, which is the entire point
    unmeasured=["motivation"],
)
v_obs = identify(observational_graph, "treat", "re78")
print(
    f"  observational: {v_obs.status} ({v_obs.route}), "
    f"needs {sorted(v_obs.unmeasured_required)} which is not in the file"
)
print("  'motivation' stands for whatever led a person to enrol. It is not a")
print("  variable anyone collected, and it is the whole problem.")

# ----------------------------------------------------------------------------------
# how strong would the confounder have to be?
# ----------------------------------------------------------------------------------

df = int(controlled.detail["df_resid"])
gap = truth - controlled.estimate
q = gap / controlled.estimate if controlled.estimate else 1.0
rv = robustness_value(estimate=controlled.estimate, se=controlled.se, df=df, q=1.0, alpha=0.05)
rv_gap = robustness_value(estimate=controlled.estimate, se=controlled.se, df=df, q=abs(q))

print(f"\nsensitivity of the observational estimate (${controlled.estimate:,.0f}):")
print(f"  to bring it to zero, a confounder needs partial R2 of {rv.rv:.1%} in both")
print(f"  to move it the ${abs(gap):,.0f} to the benchmark, it needs {rv_gap.rv:.1%}")
print("  Here we happen to know the answer, because the experiment was run. In")
print("  every real application you do not, which is why the number to report is")
print("  the strength required rather than the estimate alone.")

print(f"""
Reading it
    axiom reproduces the experimental benchmark to the dollar: ${experimental.estimate:,.0f}
    against a published ${PUBLISHED['experimental_ate']:,.0f}. That is the part this file
    checks.

    The part it demonstrates is older and more important. Two analyses of the same
    treated people, differing only in who they were compared against, disagree by
    thousands of dollars and in sign. No diagnostic run on the observational data
    flags it, because nothing is wrong with the arithmetic. The randomized
    benchmark is the only reason anyone knows which answer was wrong -- and it
    exists here only because somebody ran the experiment.

    That is the argument for the whole first pillar: decide whether the data can
    answer the question BEFORE fitting, because afterwards it looks fine either
    way.""")
