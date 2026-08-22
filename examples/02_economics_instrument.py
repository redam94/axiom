"""Labour economics — does a training programme raise earnings?

The question
    Workers choose whether to enrol in a training programme. Enrolment is not
    random: the same drive and circumstances that lead someone to enrol also raise
    their earnings directly. Comparing enrolees to non-enrolees measures the
    programme plus the selection.

The move
    Find something that shifts enrolment without touching earnings except through
    enrolment — distance to the nearest training centre, a lottery, an
    administrative cutoff. axiom looks for such an instrument in the graph rather
    than taking one on trust, and checks its relevance instead of assuming it.

What is not testable
    Exclusion — that the instrument affects earnings only through enrolment — is an
    assumption, not a finding. axiom records it as one, in the verdict, unverified,
    which is where a referee should look first.

Pillars: identify (instrumental variables)
"""

from axiom.identify import (
    identify,
    ols,
    two_stage_least_squares,
    weak_instrument_check,
)
from axiom.sim import iv_world

N, SEED = 5_000, 1

world = iv_world()
print("graph:", world.graph.to_text())
print("measured:", sorted(world.graph.measured))

# Ask before estimating: is the effect recoverable, and by what route?
verdict = identify(world.graph, "X", "Y")
print(f"\nstatus     : {verdict.status}")
print(f"route      : {verdict.route}")
print(f"instrument : {verdict.instrument}   (found in the graph, not nominated by hand)")
for a in verdict.verdict.assumptions:
    print(f"assumption : {a.name}  [{a.state}]")
    print(f"             {a.statement}")
    print(f"             challenged by: {a.challenged_by}")

frame = world.observed(world.simulate(N, seed=SEED))
truth = world.total_effect("X", "Y")

naive = ols(frame, "Y", "X")
iv = two_stage_least_squares(frame, "Y", "X", instruments=[verdict.instrument])

print(f"\ntruth                       {truth:6.3f}")
for label, est in (("naive OLS", naive), ("2SLS", iv)):
    ci = est.ci(0.95)
    print(
        f"{label:26s}  {est.estimate:6.3f} +/- {est.se:.3f}   "
        f"[{ci.lower:6.3f}, {ci.upper:6.3f}]   error {est.estimate - truth:+.3f}"
    )

# A weak instrument is its own source of bias, so relevance is checked rather
# than assumed. The first-stage F is the diagnostic.
relevance = weak_instrument_check(iv)
print(f"\nfirst-stage F = {iv.detail['first_stage_f']:.1f}")
print(f"relevance     : {relevance.name} -> {relevance.state}")

print("""
Reading it
    OLS overstates the programme by the selection it cannot see, and no sample
    size fixes that -- the bias does not shrink with n. 2SLS recovers the effect
    because the instrument moves enrolment for reasons unrelated to earnings.
    What you are buying with 2SLS is a much wider interval: the honest price of
    only using the part of the variation you can defend.""")
