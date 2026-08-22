"""Epidemiology — the effect of an exposure you cannot deconfound.

The question
    Does an exposure raise disease risk? Lifestyle and genetics drive both the
    exposure and the outcome, and nobody measured them. There is no set of recorded
    covariates that closes the back door, so back-door adjustment is not available
    at any sample size.

The move
    If the exposure acts entirely through a measured intermediate -- a biomarker,
    a deposited dose, a mediating physiological quantity -- the effect is still
    recoverable. The front-door route splits the problem in two: exposure to
    mediator (unconfounded, because the confounder does not touch the mediator
    directly), and mediator to outcome (deconfoundable by conditioning on the
    exposure). This is the structure behind the classic smoking-tar-cancer
    argument.

    axiom reads the route off the graph and names the mediator, rather than
    requiring you to know which of the three routes applies before you start.

Pillars: identify (front-door)
"""

from axiom.identify import CausalGraph, frontdoor_linear, identify, ols
from axiom.sim import frontdoor_world

N, SEED = 8_000, 0

world = frontdoor_world()
print("graph:", world.graph.to_text())
print("measured:", sorted(world.graph.measured))
print("  X <-> Y is an unmeasured common cause: lifestyle, genetics, everything")
print("  nobody wrote down. It is not in the data and never will be.")

verdict = identify(world.graph, "X", "Y")
print(f"\nstatus    : {verdict.status}")
print(f"route     : {verdict.route}")
print(f"mediators : {sorted(verdict.mediators)}")
print(f"graph hash: {verdict.graph_hash[:16]}...")
print("  The structural claim -- that X reaches Y only through M -- is not a separate")
print("  assumption axiom adds. It IS the graph you drew, and the graph is hashed into")
print("  the verdict, so a result cannot later be quoted against a different one.")

frame = world.observed(world.simulate(N, seed=SEED))
truth = world.total_effect("X", "Y")

naive = ols(frame, "Y", "X")
front = frontdoor_linear(frame, "Y", "X", mediators=sorted(verdict.mediators))

print(f"\ntruth                        {truth:6.3f}")
for label, est in (("naive (confounded)", naive), ("front-door", front)):
    ci = est.ci(0.95)
    print(
        f"{label:26s}   {est.estimate:6.3f} +/- {est.se:.3f}   "
        f"[{ci.lower:6.3f}, {ci.upper:6.3f}]   error {est.estimate - truth:+.3f}"
    )

# The structural claim is load-bearing, and you can watch it bear the load: add a
# direct exposure-to-outcome edge and there is no route left in the graph at all.
also_direct = CausalGraph.from_edges("X -> M, M -> Y, X -> Y, X <-> Y")
blocked = identify(also_direct, "X", "Y")
print("\nthe same graph, plus a direct exposure-to-outcome edge:")
print(f"  status : {blocked.status}")
print(f"  reason : {blocked.verdict.reason}")

print("""
Reading it
    No amount of covariate adjustment could have produced the front-door number,
    because the confounder is not in the data. What made it recoverable was a
    structural claim -- that the exposure reaches the outcome only through the
    measured mediator. That claim is the graph, it is a claim about the world
    rather than about the data, and no fit can check it for you -- which is why
    the script adds a direct edge above and watches the route disappear.

    That is the pattern worth taking away: when the answer is not identified from
    what you measured, the fix is a defensible structural assumption or a new
    measurement, never a more elaborate estimator.""")
