"""Conservation ecology — does a result from one reserve apply to another?

The question
    A grazing-exclusion trial ran in one reserve and worked. A second reserve wants
    to know what to expect. The two differ in rainfall, soil, and the composition of
    the herbivore community. Is the first reserve's number the second reserve's
    number, and if not, what would make it so?

Why this is not a statistics question
    No amount of data from the source reserve tells you whether it transports. That
    depends on *what differs* between the two populations and where those
    differences sit relative to the causal path. axiom takes a selection diagram --
    the graph, annotated with where the populations differ -- and reads off whether
    the effect transports, and what you would have to measure in the target to
    license it.

Pillars: identify (transportability)
"""

from axiom.identify import (
    directly_transportable,
    identify,
    ols,
    s_admissible_sets,
    selection_diagram,
    transport_verdict,
    trivially_transportable,
)
from axiom.sim import transport_pair

N, SEED = 6_000, 0

source, target = transport_pair()
print("source reserve:", source.graph.to_text())
print("target reserve:", target.graph.to_text())
print("  the two populations differ in the distribution of Z, not in the mechanism")

diagram = selection_diagram(source.graph)
print(f"\nselection diagram: {diagram.to_text()}")

# Three increasingly demanding questions, all answerable from the diagram alone.
print(f"\ndirectly transportable  : {directly_transportable(source.graph, 'X', 'Y')}")
print("   (would mean the source estimate is the target estimate, untouched)")
print(f"trivially transportable : {trivially_transportable(source.graph, 'X', 'Y')}")
print("   (would mean you can get it from target data alone and ignore the source)")

verdict = transport_verdict(source.graph, "X", "Y")
print(f"\ntransport verdict : {verdict.status}")
for field in ("route", "reason"):
    value = getattr(verdict, field, None)
    if value:
        print(f"  {field:15s}: {value}")

sets = s_admissible_sets(source.graph, "X", "Y")
print(f"\nS-admissible sets : {[sorted(s) for s in sets]}")
print("  measure these in the TARGET, and the source effect can be re-weighted")
print("  onto the target population. This is the shopping list the second reserve")
print("  needs before it can borrow the first reserve's answer.")

# What it costs to ignore all of this: take the source number at face value.
source_frame = source.observed(source.simulate(N, seed=SEED))
target_frame = target.observed(target.simulate(N, seed=SEED + 17))  # a different draw

v = identify(source.graph, "X", "Y")
source_est = ols(source_frame, "Y", "X", covariates=v.adjustment_set)
target_est = ols(target_frame, "Y", "X", covariates=v.adjustment_set)
target_truth = target.total_effect("X", "Y")

print(f"\nsource reserve, adjusted   {source_est.estimate:6.3f} +/- {source_est.se:.3f}")
print(f"target reserve, adjusted   {target_est.estimate:6.3f} +/- {target_est.se:.3f}")
print(f"target truth               {target_truth:6.3f}")

print("""
Reading it
    In this pair the mechanism is shared and only Z's distribution differs, so the
    conditional effect carries across once you condition on the S-admissible set --
    which is exactly what the verdict says and what the two adjusted estimates
    show. Change the diagram so the populations differ in how X acts rather than
    in who is in them, and the same call refuses instead.

    The useful output is not the number. It is the shopping list: transport is a
    claim about which covariates you collected in the new site, decided before
    anyone flies out there.""")
