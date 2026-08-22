"""Public health — sizing a cluster-randomized water and sanitation trial.

The question
    You cannot randomize a chlorination programme household by household: neighbours
    share water, share pathogens, and would talk. So you randomize whole villages,
    and measure children within them.

The trap
    Outcomes within a village are correlated. Two hundred children in ten villages
    are worth far fewer than two hundred independent children, and a trial sized as
    though they were independent will be badly underpowered while looking fine on
    paper. The correction is the design effect, and it is brutal: even a small
    intra-cluster correlation costs you most of your sample when clusters are large.

Pillars: design (cluster randomization)
"""

from axiom.core import Unit
from axiom.design import (
    ClusterDesign,
    cluster_mde,
    cluster_power,
    clusters_needed,
    design_effect,
    effective_sample_size,
    sample_size,
)

OUTCOME_SD = 1.0  # standardized: diarrhoea episodes per child-year
TARGET_EFFECT = 0.25  # the reduction worth detecting
ICC = 0.03  # small, and typical for village-level health outcomes
CLUSTER_SIZE = 40  # children measured per village

# The randomization unit is a village, and axiom wants that said out loud: a
# ClusterDesign carries the Unit it randomizes, not just the arithmetic.
VILLAGE = Unit(name="village", kind="cluster")

# What the naive calculation says, ignoring clustering entirely.
naive = sample_size(effect=TARGET_EFFECT, sd=OUTCOME_SD, power=0.8, alpha=0.05)
print("if children were independent (they are not):")
print(f"  n = {naive.n} children ({naive.n_treated} treated / {naive.n_control} control)")
print(f"  power {naive.power:.3f}")

# What clustering actually costs.
de = design_effect(CLUSTER_SIZE, ICC)
print(f"\ndesign effect at m = {CLUSTER_SIZE}, ICC = {ICC}:  {de:.2f}x variance inflation")
print(f"  an ICC of only {ICC} inflates the variance by {de:.2f} because the clusters")
print("  are large -- the penalty is (m - 1) * ICC, so cluster SIZE is what hurts.")

need = clusters_needed(
    effect=TARGET_EFFECT, sd=OUTCOME_SD, cluster_size=CLUSTER_SIZE, icc=ICC, power=0.8, alpha=0.05
)
print(
    f"\nclusters needed: {need.n} villages "
    f"({need.n_treated} treated / {need.n_control} control)"
)
print(f"  that is {need.n * CLUSTER_SIZE:,} children to do the work of {naive.n}")

ess = effective_sample_size(need.n, CLUSTER_SIZE, ICC)
print(f"  effective sample size: {ess:.0f} independent observations")

# The design as an object, so power and MDE are read off the same thing.
design = ClusterDesign(unit=VILLAGE, n_clusters=need.n, cluster_size=CLUSTER_SIZE, icc=ICC)
power = cluster_power(design, effect=TARGET_EFFECT, sd=OUTCOME_SD)
mde = cluster_mde(design, sd=OUTCOME_SD, power=0.8)
print(f"\nrealized design: {design.n_clusters} clusters of {design.cluster_size}")
print(f"  power for an effect of {TARGET_EFFECT}: {power.power:.3f}")
print(f"  smallest effect detectable at 80% power: {mde.effect:.3f}")

# The real design question is not "how many children" but "how to spend a fixed
# field budget": more villages, or more children per village?
print("\nfewer, larger clusters vs more, smaller ones (about 1,600 children either way):")
print(f"  {'villages':>9}{'per village':>13}{'design effect':>15}{'power':>9}")
for n_clusters, size in ((10, 160), (20, 80), (40, 40), (80, 20), (160, 10)):
    d = ClusterDesign(unit=VILLAGE, n_clusters=n_clusters, cluster_size=size, icc=ICC)
    p = cluster_power(d, effect=TARGET_EFFECT, sd=OUTCOME_SD)
    print(f"  {n_clusters:>9}{size:>13}{design_effect(size, ICC):>15.2f}{p.power:>9.3f}")

print("""
Reading it
    The same number of children bought very different amounts of information. Ten
    villages of 160 is nearly worthless; 160 villages of 10 is a real trial. Field
    cost usually runs the other way -- villages are expensive to reach, children
    within them are cheap -- so this is the trade-off the budget conversation is
    actually about, and it is worth having before the protocol is written rather
    than after the results come back inconclusive.""")
