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
from _walkthrough import Walkthrough

from axiom.diagnose import bias_bounds, robustness_value, tipping_point
from axiom.identify import identify, minimal_adjustment_sets, ols
from axiom.sim import confounded_world, hidden_confounder_world

N, SEED = 5_000, 0
DECISION_THRESHOLD = 1.0

w = Walkthrough(
    field="Education",
    title="How strong would the thing you cannot see have to be?",
    question="""Students who take extra tutoring score higher, and the students who
        sign up were already ahead. When prior attainment is in the data this is a
        solved problem. When it is not — which is most school datasets — what is
        left to say?""",
)

# ----------------------------------------------------------------------------------

w.step(
    "Case one: the confounder is recorded, so ask the graph what to condition on",
    why="""An adjustment set is a property of the graph, not a matter of taste. The
        back-door criterion says exactly which variables block the non-causal paths,
        and minimal_adjustment_sets enumerates every sufficient set — so the choice
        of controls becomes a derivation rather than a habit.""",
    instead="""Putting every available column on the right-hand side. That is the
        default in practice and it is unsafe in both directions: conditioning on a
        collider opens a path that was closed, and conditioning on a mediator
        removes part of the effect you were trying to measure.""",
)
world = confounded_world()
truth = world.total_effect("X", "Y")
verdict = identify(world.graph, "X", "Y")
w.out(f"graph        : {world.graph.to_text()}")
w.out(f"status       : {verdict.status} via {verdict.route}")
w.out(f"condition on : {sorted(verdict.adjustment_set)}")
w.out(f"minimal sets : {[sorted(s) for s in minimal_adjustment_sets(world.graph, 'X', 'Y')]}")

frame = world.observed(world.simulate(N, seed=SEED))
naive = ols(frame, "Y", "X")
adjusted = ols(frame, "Y", "X", covariates=verdict.adjustment_set)
rows = []
for label, est in (("raw gap", naive), ("back-door adjusted", adjusted)):
    ci = est.ci(0.95)
    w.out(
        f"{label:20s} {est.estimate:6.3f}  [{ci.lower:6.3f}, {ci.upper:6.3f}]  "
        f"error {est.estimate - truth:+.3f}"
    )
    rows.append(
        {
            "label": label,
            "estimate": est.estimate,
            "lower": ci.lower,
            "upper": ci.upper,
            "note": f"error {est.estimate - truth:+.3f}",
            "bad": abs(est.estimate - truth) > 2 * est.se,
        }
    )
w.figure(
    "adjusted",
    kind="intervals",
    data={"rows": rows},
    opt={
        "rows": "@rows",
        "truth": truth,
        "truthLabel": "the truth",
        "xLabel": "effect of tutoring on score",
        "labelWidth": 150,
        "rowHeight": 38,
    },
    title="With prior attainment in the data, adjustment does the whole job",
    note="""One column moves the answer from confidently wrong to correct. That is
        also the measure of what is lost in case two, where the column does not
        exist.""",
)

# ----------------------------------------------------------------------------------

w.step(
    "Case two: it was never recorded, so the verdict is a refusal",
    why="""The same graph with prior attainment unmeasured has no back-door
        adjustment set at all. axiom downgrades the verdict and names what is
        missing rather than returning a number with a caveat attached — a number
        with a caveat attached gets quoted; a refusal does not.""",
    instead="""Fitting on the columns you have and putting 'unobserved confounding
        cannot be ruled out' in the limitations section. That sentence appears in
        almost every observational paper and changes no reader's behaviour, because
        it does not say how much confounding would matter.""",
)
hidden = hidden_confounder_world()
hidden_verdict = identify(hidden.graph, "X", "Y")
hidden_frame = hidden.observed(hidden.simulate(N, seed=SEED))
biased = ols(hidden_frame, "Y", "X")
w.out(f"columns you have : {hidden_frame.columns.tolist()}")
w.out(f"status           : {hidden_verdict.status} ({hidden_verdict.route})")
w.out(f"would need       : {sorted(hidden_verdict.unmeasured_required)}, which you do not have")
w.out(
    f"fitting anyway   : {biased.estimate:.3f} against a truth of {truth:.3f} "
    f"({biased.estimate - truth:+.3f})"
)

# ----------------------------------------------------------------------------------

w.step(
    "Say how strong the invisible thing would have to be",
    why="""The remaining honest question is not 'what is the effect' but 'how much
        confounding would it take to change the conclusion'. The robustness value is
        the share of residual variance in BOTH tutoring and scores that an
        unmeasured variable would have to explain to wipe the estimate out. It turns
        an unanswerable question into one a subject expert can actually judge.""",
    instead="""Reporting the raw estimate with a wider interval to 'account for'
        confounding. Widening an interval models noise, and confounding is not
        noise: it moves the centre, and no amount of extra width puts it back.""",
)
df = biased.n - 2
rv = robustness_value(estimate=biased.estimate, se=biased.se, df=df, q=1.0, alpha=0.05)
w.out(f"robustness value : {rv.rv:.1%}")
w.say(f"""An unmeasured confounder would have to explain {rv.rv:.1%} of the residual
    variance in both tutoring and scores to reduce the estimate to zero. Prior
    attainment — the very variable we hid — explains roughly 39% of tutoring and 56%
    of scores, which is nowhere near enough. So the honest reading is that this
    estimate survives a confounder as strong as the one we know about, and would
    only collapse under one substantially stronger. That is a claim a head of
    department can argue with; 'unobserved confounding cannot be ruled out' is not.""")

BENCH = (
    ("a tenth as strong as attainment", 0.039, 0.056),
    ("half as strong", 0.195, 0.281),
    ("as strong as attainment", 0.390, 0.562),
)


def surviving(r2_treat: float, r2_out: float) -> float:
    """What the estimate becomes under a confounder of exactly this strength."""
    return bias_bounds(
        estimate=biased.estimate,
        se=biased.se,
        df=df,
        r2_yz_dx=r2_out,
        r2_dz_x=r2_treat,
        mass=0.95,
    ).adjusted_estimate


w.table(
    ["benchmark confounder", "R2 with tutoring", "R2 with score", "estimate becomes"],
    [
        [label, f"{r2_treat:.3f}", f"{r2_out:.3f}", f"{surviving(r2_treat, r2_out):.3f}"]
        for label, r2_treat, r2_out in BENCH
    ],
)

grid_t = np.linspace(0.01, 0.55, 28)
grid_o = np.linspace(0.01, 0.80, 28)
contour = [[surviving(float(t), float(o)) for t in grid_t] for o in grid_o]
worst = min(min(row) for row in contour)
strongest = surviving(BENCH[-1][1], BENCH[-1][2])
w.figure(
    "contour",
    kind="heatmap",
    data={
        "treat": grid_t.tolist(),
        "outcome": grid_o.tolist(),
        "z": contour,
    },
    opt={
        "key": "@",
        "x": "treat",
        "y": "outcome",
        "z": "z",
        "xLabel": "share of tutoring the confounder explains",
        "yLabel": "share of score the confounder explains",
        "zLabel": "estimate after adjusting for it",
        "pct": True,
        "height": 340,
        "marks": [
            {"x": r2_treat, "y": r2_out, "label": label}
            for label, r2_treat, r2_out in (
                ("a tenth", 0.039, 0.056),
                ("half", 0.195, 0.281),
                ("as strong", 0.390, 0.562),
            )
        ],
    },
    title="Every confounder that could exist, and what it would do to the answer",
    note=f"""Darker is a larger surviving estimate. The three marks are confounders
        benchmarked against prior attainment itself, and even the strongest of them
        leaves the estimate at {strongest:.2f}. The worst corner of this
        whole grid — a confounder explaining 55% of tutoring and 80% of scores —
        still leaves {worst:.2f}, comfortably above the {DECISION_THRESHOLD:.0f} the
        decision turns on.""",
    legend=(("accent", "surviving estimate"), ("boundary", "benchmark confounders")),
)

# ----------------------------------------------------------------------------------

w.step(
    "Then ask the only question that pays: does any of it change the decision?",
    why="""A finding is not a decision. The programme gets funded if the effect
        exceeds 1.0 with 90% certainty, so the useful sensitivity analysis walks the
        bias up until that call flips, rather than until the estimate reaches zero.
        Sensitivity should be measured against the threshold in play, not against
        the null.""",
    instead="""Testing against zero. Nothing on this table was ever going to be
        decided at zero — a tutoring programme with an effect of 0.2 is as
        unfundable as one with an effect of 0, so a robustness value computed
        against zero answers a question nobody asked.""",
)
rng = np.random.default_rng(SEED)
draws = rng.normal(biased.estimate, biased.se, size=4000)
tp = tipping_point(draws, DECISION_THRESHOLD, np.linspace(0.0, 2.5, 26), certainty=0.9)
w.out(f"decision        : fund if the effect exceeds {DECISION_THRESHOLD:.1f}")
w.out(f"at zero bias    : {tp.decision_at_zero} (P = {tp.probability_at_zero:.3f})")
w.out(f"flips once bias : {tp.bias:.1f}")
w.out(f"real bias here  : {biased.estimate - truth:.3f}")
w.figure(
    "tipping",
    kind="lines",
    data={
        "bias": list(tp.bias_grid),
        "p": list(tp.probabilities),
    },
    opt={
        "series": [{"label": "P(effect > 1.0)", "x": "@bias", "y": "@p", "colour": "accent"}],
        "xLabel": "bias assumed away, in score points",
        "yLabel": "posterior probability the bar is cleared",
        "yPct": True,
        "hline": 0.9,
        "hlineLabel": "90% certainty required",
        "height": 290,
        "rightPad": 110,
    },
    title="How much bias the funding decision can absorb",
    note=f"""The call survives until an assumed bias of {tp.bias:.1f} score points.
        The bias actually present is {biased.estimate - truth:.2f} — a factor of
        {tp.bias / (biased.estimate - truth):.1f} of headroom, which is a far more
        useful sentence than either 'it is significant' or 'we cannot know'.""",
    legend=(("accent", "P(effect > 1.0)"), ("ink-3:dash", "the certainty required")),
)

# ----------------------------------------------------------------------------------

w.finding(f"""With the confounder measured, adjustment moved the estimate from
    {naive.estimate:.2f} to {adjusted.estimate:.2f} against a truth of {truth:.2f}.
    That is the easy case, and it is worth running first precisely because it
    calibrates the hard one: it shows what the missing column was worth.""")
w.finding(f"""With it hidden, no estimator recovers the truth — the estimate sits
    {biased.estimate - truth:+.2f} out and stays there. What replaced the point
    estimate was not a wider interval but a different kind of statement: a
    confounder would need to explain {rv.rv:.1%} of both sides to erase the effect,
    and {tp.bias:.1f} points of bias to change the funding call.""")
w.finding("""This one survives. That is not the interesting part — the interesting part is
    that it survives by an amount anyone can check. Two different sensitivity
    questions, one against zero and one against the funding bar, both came back with
    room to spare, and both came back as quantities rather than as a hedge in a
    limitations paragraph.""")
w.finding("""It would not always. Run the same three steps on a weaker programme and the
    contour's marks move into the pale region, the tipping bias falls below the bias
    a benchmark confounder would produce, and the correct output is 'do not fund on
    this evidence'. The value of the machinery is that it returns that answer as
    readily as this one.""")
