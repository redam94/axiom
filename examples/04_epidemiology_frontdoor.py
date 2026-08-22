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

from _walkthrough import Walkthrough

from axiom.identify import CausalGraph, frontdoor_linear, identify, ols
from axiom.sim import LinearSCM, frontdoor_world

N, SEED = 8_000, 0

w = Walkthrough(
    field="Epidemiology",
    title="Recovering an effect nothing in the data can deconfound",
    question="""Lifestyle and genetics drive both the exposure and the disease, and
        nobody measured them. No covariate adjustment can work. Is the effect gone,
        or is there another way in?""",
)

# ----------------------------------------------------------------------------------

w.step(
    "First establish that the obvious route is genuinely closed",
    why="""Before reaching for anything clever it is worth confirming there is
        nothing to be had the ordinary way. On a graph with only the exposure, the
        outcome and an unmeasured common cause, identify() refuses — and the refusal
        is typed, with a reason, rather than being an empty result the caller has to
        interpret.""",
    instead="""Assuming the problem is a missing covariate and going looking for
        proxies. Proxies for an unmeasured confounder do not close the back door
        unless they close it completely, and 'we adjusted for a rich set of
        covariates' is the sentence that most often stands in for a route nobody
        checked.""",
)
bare = CausalGraph.from_edges("X -> Y, X <-> Y")
bare_verdict = identify(bare, "X", "Y")
w.out(f"graph  : {bare.to_text()}")
w.out(f"status : {bare_verdict.status}")
w.out(f"reason : {bare_verdict.verdict.reason}")

# ----------------------------------------------------------------------------------

w.step(
    "Add the one structural claim that changes everything",
    why="""The claim is that the exposure reaches the outcome only through a measured
        intermediate — a biomarker, a deposited dose. That is not an extra assumption
        axiom bolts on; it IS the graph, and the graph is hashed into the verdict, so
        a result can never later be quoted against a different one. With the mediator
        present the front-door route opens and the effect becomes recoverable
        despite the confounder still being invisible.""",
    instead="""An instrument. It would work too, but it requires finding something
        that moves the exposure and touches the outcome no other way — and in
        observational epidemiology that thing usually does not exist, while the
        mediator often has already been measured for other reasons.""",
)
world = frontdoor_world()
verdict = identify(world.graph, "X", "Y")
w.out(f"graph      : {world.graph.to_text()}")
w.out(f"measured   : {sorted(world.graph.measured)}")
w.out(f"status     : {verdict.status}")
w.out(f"route      : {verdict.route}")
w.out(f"mediators  : {sorted(verdict.mediators)}")
w.out(f"graph hash : {verdict.graph_hash[:16]}...")
w.say("""The route works by splitting the problem in two. Exposure to mediator is
    unconfounded, because the hidden common cause does not touch the mediator
    directly. Mediator to outcome is deconfoundable by conditioning on the exposure.
    Neither half needs the confounder, so their product does not either — this is the
    structure behind the smoking-tar-cancer argument.""")

# ----------------------------------------------------------------------------------

w.step(
    "Estimate it both ways on the same 8,000 people",
    why="""The naive regression is what a careful analyst would report from these
        columns if nobody had drawn the graph, and it is confidently wrong. The
        front-door estimate uses no additional data at all — same rows, same columns
        — only the structural claim.""",
    instead="""Reporting the front-door number alone. The distance between the two is
        the size of the confounding, which is a quantity the study can now put a
        number on despite never having measured the confounder.""",
)
frame = world.observed(world.simulate(N, seed=SEED))
truth = world.total_effect("X", "Y")
naive = ols(frame, "Y", "X")
front = frontdoor_linear(frame, "Y", "X", mediators=sorted(verdict.mediators))

rows = []
for label, est in (("naive (confounded)", naive), ("front-door", front)):
    ci = est.ci(0.95)
    w.out(
        f"{label:20s} {est.estimate:6.3f} +/- {est.se:.3f}  "
        f"[{ci.lower:6.3f}, {ci.upper:6.3f}]  error {est.estimate - truth:+.3f}"
    )
    rows.append(
        {
            "label": label,
            "estimate": est.estimate,
            "lower": ci.lower,
            "upper": ci.upper,
            "note": f"standard error {est.se:.3f}",
            "bad": abs(est.estimate - truth) > 2 * est.se,
        }
    )
w.out(f"{'truth':20s} {truth:6.3f}")
w.figure(
    "estimates",
    kind="intervals",
    data={"rows": rows},
    opt={
        "rows": "@rows",
        "truth": truth,
        "truthLabel": "the truth",
        "xLabel": "effect of the exposure on disease risk",
        "labelWidth": 155,
        "rowHeight": 38,
    },
    title="Same rows, same columns, one structural claim apart",
    note="""No covariate adjustment could have produced the lower interval, because
        the confounder is not in the data. What produced it was a claim about how the
        world is wired.""",
)

# ----------------------------------------------------------------------------------

w.step(
    "Break the claim on purpose and watch the route disappear",
    why="""A load-bearing assumption should be shown bearing the load. Adding a
        direct exposure-to-outcome edge — a second pathway that skips the mediator —
        leaves a graph with no identifying route at all, and identify() says so
        rather than falling back on the front-door formula regardless.""",
    instead="""Testing the assumption against the data. It cannot be tested against
        the data: both graphs imply the same joint distribution over the measured
        variables here. That is exactly why it has to be argued substantively and
        recorded explicitly.""",
)
also_direct = CausalGraph.from_edges("X -> M, M -> Y, X -> Y, X <-> Y")
blocked = identify(also_direct, "X", "Y")
w.out(f"graph  : {also_direct.to_text()}")
w.out(f"status : {blocked.status}")
w.out(f"reason : {blocked.verdict.reason}")

# ----------------------------------------------------------------------------------

w.step(
    "And measure what believing it wrongly would cost",
    why="""Refusing on a graph is one thing; knowing the size of the mistake is
        another. Here the same analysis runs on a family of worlds that each have a
        small direct path the analyst does not know about, so the exclusion claim is
        false by a controlled amount. The front-door estimator keeps returning the
        mediated part and misses the direct part entirely.""",
    instead="""Treating the assumption as binary — 'it holds' or 'the analysis is
        invalid'. Assumptions fail by degrees, and an estimator that degrades
        smoothly is usable under a claim that is nearly true, which is the only kind
        of claim epidemiology ever gets.""",
)
leaks = [0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]
truths, estimates = [], []
for leak in leaks:
    leaky = LinearSCM.from_text(
        f"X -> M: 1.2, M -> Y: 1.5, X -> Y: {leak}, X <-> Y: 1.0", name="leaky"
    )
    leaky_frame = leaky.observed(leaky.simulate(N, seed=SEED))
    truths.append(float(leaky.total_effect("X", "Y")))
    estimates.append(float(frontdoor_linear(leaky_frame, "Y", "X", mediators=["M"]).estimate))
w.table(
    ["hidden direct path", "true total effect", "front-door says", "error"],
    [
        [f"{d:.2f}", f"{t:.3f}", f"{e:.3f}", f"{e - t:+.3f}"]
        for d, t, e in zip(leaks, truths, estimates, strict=True)
    ],
)
w.figure(
    "leak",
    kind="lines",
    data={"leak": leaks, "truth": truths, "estimate": estimates},
    opt={
        "series": [
            {"label": "the truth", "x": "@leak", "y": "@truth", "colour": "ink-2"},
            {"label": "front-door", "x": "@leak", "y": "@estimate", "colour": "accent"},
        ],
        "xLabel": "strength of the direct path the analyst assumed away",
        "yLabel": "total effect",
        "height": 300,
        "rightPad": 90,
    },
    title="How wrong the answer gets as the exclusion claim fails",
    note="""The front-door line is flat: it recovers the mediated part correctly and
        is blind to everything else, so the error is exactly the size of the path
        that was assumed away. That is a good failure mode — it is bounded by
        something a subject expert can reason about.""",
    legend=(("ink-2", "true total effect"), ("accent", "what front-door returns")),
)

# ----------------------------------------------------------------------------------

w.finding(f"""The naive estimate was {naive.estimate:.2f} against a truth of {truth:.2f};
    the front-door estimate was {front.estimate:.2f}. Nothing was measured in between.
    The gap of {naive.estimate - truth:+.2f} is the confounding, now quantified by a
    study that never observed the confounder.""")
w.finding("""What made it recoverable was a structural claim, not a better estimator. That
    is the pattern worth taking away: when the answer is not identified from what you
    measured, the fix is a defensible claim about the world or a new measurement.
    A more elaborate fit is never the fix, because the problem is not in the
    fitting.""")
w.finding(f"""And the claim really is load-bearing. Add a direct edge and the graph offers
    no route at all; leave the edge in the world but out of the analysis and the
    error is the size of the edge — {estimates[-1] - truths[-1]:+.2f} when the direct
    path is as strong as {leaks[-1]:.1f}. Both behaviours are what you want: refuse
    when the structure is wrong, degrade proportionally when it is only slightly
    wrong.""")
