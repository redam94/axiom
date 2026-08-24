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

from _walkthrough import Walkthrough

from axiom.identify import (
    identify,
    ols,
    two_stage_least_squares,
    weak_instrument_check,
)
from axiom.sim import LinearSCM, iv_world

N, SEED = 5_000, 1

w = Walkthrough(
    field="Labour economics",
    title="An effect you can only see sideways",
    question="""Does a training programme raise earnings, when the people who enrol
        are exactly the people who would have earned more anyway?""",
)

# ----------------------------------------------------------------------------------

w.step(
    "Write down the world, including the part nobody measured",
    why="""The obstacle here is not a missing column, it is a structure: drive and
        circumstance push both enrolment and earnings, and they were never recorded.
        Writing that down as an edge — X <-> Y, a latent common cause — is what makes
        the problem statable. An analysis that cannot say what is confounding it
        cannot say what would fix it either.""",
    instead="""Starting from the estimator. 'We ran 2SLS with distance as an
        instrument' skips the only question that determines whether the number means
        anything: does the graph you believe in actually license that move?""",
)
world = iv_world()
w.out(f"graph    : {world.graph.to_text()}")
w.out(f"measured : {sorted(world.graph.measured)}")
w.out("")
w.out("X <-> Y is the bidirected edge: a common cause of enrolment and earnings")
w.out("that is not in the data and never will be. Z is the shifter -- distance to")
w.out("the nearest centre, say -- which moves enrolment for reasons of geography.")

# ----------------------------------------------------------------------------------

w.step(
    "Ask whether the effect is recoverable at all, before estimating anything",
    why="""identify() is a question about the graph, not about the data, and it can
        be asked before a single row is collected. It returns the route, the
        instrument it found, and — the part that matters most in a referee report —
        the assumptions the route rests on, each tagged with whether the data can
        check it.""",
    instead="""Nominating the instrument yourself and checking the first-stage F. That
        tests relevance, which is the assumption you can test, and quietly skips
        exclusion, which is the one that actually fails in practice.""",
)
verdict = identify(world.graph, "X", "Y")
w.out(f"status     : {verdict.status}")
w.out(f"route      : {verdict.route}")
w.out(f"instrument : {verdict.instrument}   (found in the graph, not nominated by hand)")
w.table(
    ["assumption", "state", "what it claims", "what would challenge it"],
    [
        [a.name, a.state, a.statement, ", ".join(a.challenged_by) or "—"]
        for a in verdict.verdict.assumptions
    ],
    caption="""The verdict carries its own assumptions. 'unverified' is not a
        weasel word — it is the correct state for a claim no dataset can settle.""",
)

# ----------------------------------------------------------------------------------

w.step(
    "Estimate it twice: once naively, once through the instrument",
    why="""Both estimators are run on the same 5,000 workers so the difference
        between them is the method and nothing else. The naive comparison is not a
        straw man — it is what a well-run analysis of this dataset would report if
        nobody had drawn the graph.""",
    instead="""Reporting only the instrumented estimate. The gap between the two is
        the size of the selection, and it is the most informative number on the
        page: it says how much of the raw programme effect was never the
        programme.""",
)
frame = world.observed(world.simulate(N, seed=SEED))
truth = world.total_effect("X", "Y")
naive = ols(frame, "Y", "X")
iv = two_stage_least_squares(frame, "Y", "X", instruments=[verdict.instrument])

rows = []
for label, est in (("naive OLS", naive), ("2SLS", iv)):
    ci = est.ci(0.95)
    w.out(
        f"{label:12s} {est.estimate:6.3f} +/- {est.se:.3f}   "
        f"[{ci.lower:6.3f}, {ci.upper:6.3f}]   error {est.estimate - truth:+.3f}"
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
w.out(f"{'truth':12s} {truth:6.3f}")

w.figure(
    "estimates",
    kind="intervals",
    data={"rows": rows},
    opt={
        "rows": "@rows",
        "truth": truth,
        "truthLabel": "the truth",
        "xLabel": "effect of the programme on earnings",
        "labelWidth": 110,
        "rowHeight": 38,
    },
    title="What each method says the programme is worth",
    note="""The naive interval does not merely miss the truth, it excludes it
        confidently — the interval is narrow and in the wrong place. 2SLS lands on
        the truth and pays for it in width.""",
    legend=(("accent", "95% interval"), ("boundary", "excludes the truth")),
)

# ----------------------------------------------------------------------------------

w.step(
    "Watch what more data does to each one",
    why="""This is the distinction the whole example turns on. Noise shrinks as the
        square root of n; bias does not shrink at all. Re-running both estimators
        across sample sizes from 250 to 32,000 draws the difference rather than
        asserting it — the naive line converges, confidently, on the wrong
        number.""",
    instead="""Saying 'OLS is biased' and moving on. Every analyst agrees with that
        sentence in the abstract and then, under deadline, argues that a large
        enough sample makes it academic. The chart is the answer to that argument.""",
)
sizes = [250, 500, 1_000, 2_000, 4_000, 8_000, 16_000, 32_000]
naive_path, iv_path = [], []
for n in sizes:
    f = world.observed(world.simulate(n, seed=SEED))
    naive_path.append(float(ols(f, "Y", "X").estimate))
    iv_path.append(float(two_stage_least_squares(f, "Y", "X", instruments=["Z"]).estimate))
w.table(
    ["workers", "naive OLS", "2SLS"],
    [
        [f"{n:,}", f"{a:.3f}", f"{b:.3f}"]
        for n, a, b in zip(sizes, naive_path, iv_path, strict=True)
    ],
)
w.figure(
    "convergence",
    kind="lines",
    data={"n": sizes, "naive": naive_path, "iv": iv_path},
    opt={
        "series": [
            {"label": "naive OLS", "x": "@n", "y": "@naive", "colour": "boundary"},
            {"label": "2SLS", "x": "@n", "y": "@iv", "colour": "accent"},
        ],
        "xLabel": "workers in the sample",
        "yLabel": "estimated effect",
        "hline": truth,
        "hlineLabel": "the truth",
        "height": 300,
        "rightPad": 80,
    },
    title="More data does not fix a bias",
    note="""Both lines settle down. Only one settles on the right answer. Everything
        the naive estimator gains from a larger sample it spends on being more
        certain of the selection it cannot see.""",
    legend=(
        ("boundary", "naive OLS"),
        ("accent", "2SLS"),
        ("ink-3:dash", "the truth"),
    ),
)

# ----------------------------------------------------------------------------------

w.step(
    "Check the instrument is actually doing work",
    why="""A weak instrument is not a neutral loss of precision. When the first stage
        barely moves, 2SLS becomes biased towards OLS and its intervals stop
        covering, so a weak-instrument analysis can be worse than the naive one it
        was meant to replace. The first-stage F is the standard diagnostic, and
        axiom returns it as an assumption with a state rather than a number to
        eyeball.""",
    instead="""Assuming relevance because the instrument is clever. Cleverness is an
        argument for exclusion, which cannot be tested; relevance is the part that
        can be, so it should be.""",
)
relevance = weak_instrument_check(iv)
w.out(f"first-stage F : {iv.detail['first_stage_f']:.1f}")
w.out(f"relevance     : {relevance.name} -> {relevance.state}")
w.out("")

# The same structure with the instrument's first stage turned almost off, so the
# cost of weakness is visible rather than argued for.
weak_world = LinearSCM.from_text("Z -> X: 0.05, X -> Y: 2.0, X <-> Y: 1.0", name="weak-iv")
weak_frame = weak_world.observed(weak_world.simulate(N, seed=SEED))
weak_iv = two_stage_least_squares(weak_frame, "Y", "X", instruments=["Z"])
weak_check = weak_instrument_check(weak_iv)
w.out("a nearly irrelevant instrument (Z -> X of 0.05 instead of 1.0):")
w.out(f"  first-stage F : {weak_iv.detail['first_stage_f']:.1f}")
w.out(f"  relevance     : {weak_check.state}")
w.out(f"  estimate      : {weak_iv.estimate:.3f} +/- {weak_iv.se:.3f}")

strong_ci, weak_ci = iv.ci(0.95), weak_iv.ci(0.95)
w.figure(
    "relevance",
    kind="intervals",
    data={
        "rows": [
            {
                "label": "strong instrument",
                "estimate": iv.estimate,
                "lower": strong_ci.lower,
                "upper": strong_ci.upper,
                "note": f"first-stage F = {iv.detail['first_stage_f']:.0f}",
            },
            {
                "label": "weak instrument",
                "estimate": weak_iv.estimate,
                "lower": weak_ci.lower,
                "upper": weak_ci.upper,
                "note": f"first-stage F = {weak_iv.detail['first_stage_f']:.1f}",
                "bad": True,
            },
        ]
    },
    opt={
        "rows": "@rows",
        "truth": truth,
        "truthLabel": "the truth",
        "xLabel": "effect of the programme on earnings",
        "labelWidth": 150,
        "rowHeight": 38,
    },
    title="The same estimator, on an instrument that barely moves enrolment",
    note="""Same graph, same route, same code. Only the strength of the first stage
        changed, and the answer became useless — which is why relevance is checked
        rather than assumed.""",
)

# ----------------------------------------------------------------------------------

w.finding(f"""OLS overstates the programme by {naive.estimate - truth:+.2f}, and the chart of
    sample sizes shows that no amount of data would have helped: the naive line
    converges on {naive_path[-1]:.2f} against a truth of {truth:.2f}. Bias is not a
    small-sample problem.""")
w.finding(f"""What 2SLS bought was accuracy, and what it cost was width: its interval is
    {(iv.ci(0.95).upper - iv.ci(0.95).lower) / (naive.ci(0.95).upper - naive.ci(0.95).lower):.1f}
    times wider than the naive one. That is the honest price of using only the part
    of the variation in enrolment that geography explains, and it is the right trade
    — a wide interval around the truth is usable, a narrow one around the wrong
    number is not.""")
w.finding("""The assumption table is the part to take to a referee. Relevance was checked
    and passed. Exclusion — that distance to a training centre affects earnings only
    through enrolment — is recorded as unverified, because it is unverifiable from
    this data. If it fails, everything above fails with it, and the verdict says so
    rather than burying it in a methods appendix.""")
