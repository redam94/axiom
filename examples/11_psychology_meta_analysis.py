"""Behavioural science — pooling a literature that disagrees with itself.

The question
    Twenty published studies test the same behavioural nudge. The effects range from
    trivial to large. What does the literature say, how much do the studies really
    disagree, and what should you expect if you run study twenty-one?

Three things that get skipped
    The pooled mean is the least interesting output. What a replication team
    actually needs is the PREDICTION interval -- where a new study would land --
    which is much wider than the interval on the mean and is the number that
    predicts replication success.

    Heterogeneity is not a nuisance to be averaged away; a high I-squared means the
    single pooled number is answering a question nobody asked.

    And small-study effects deserve a look before any of it is believed.

Pillars: meta (pooling, heterogeneity, small-study effects, influence)
"""

from _walkthrough import Walkthrough

from axiom.build import MetaBuilder
from axiom.meta import (
    baujat,
    egger,
    funnel_data,
    heterogeneity,
    leave_one_out,
    prediction_interval,
    random_effects,
)

# Twenty nudge studies: effect size and standard error, as a review would collect
# them. Larger studies are more precise, and the smaller ones scatter wider --
# partly by sampling and partly, as the funnel below suggests, by what got
# published at all.
STUDIES = [
    ("s01", 0.09, 0.041, 1180),
    ("s02", 0.31, 0.152, 88),
    ("s03", 0.15, 0.033, 1840),
    ("s04", 0.22, 0.131, 118),
    ("s05", 0.19, 0.058, 602),
    ("s06", 0.48, 0.186, 61),
    ("s07", 0.13, 0.028, 2510),
    ("s08", 0.05, 0.104, 187),
    ("s09", 0.21, 0.049, 842),
    ("s10", 0.42, 0.168, 72),
    ("s11", 0.11, 0.031, 2080),
    ("s12", 0.34, 0.088, 258),
    ("s13", 0.17, 0.126, 128),
    ("s14", 0.24, 0.044, 1030),
    ("s15", 0.39, 0.147, 94),
    ("s16", 0.06, 0.030, 2240),
    ("s17", 0.28, 0.076, 344),
    ("s18", 0.51, 0.198, 54),
    ("s19", 0.16, 0.052, 742),
    ("s20", 0.12, 0.097, 214),
]

w = Walkthrough(
    field="Behavioural science",
    title="What study twenty-one will actually find",
    question="""Twenty published studies test the same nudge and disagree with each
        other. Pooling them gives a number. The question is whether that number is
        the one a replication team should power against.""",
)

# ----------------------------------------------------------------------------------

w.step(
    "Build the corpus through a vocabulary that refuses the wrong pool",
    why="""'standardized_contrast' is not free text: poolable quantities are a
        controlled vocabulary, so two reviews cannot quietly pool an odds ratio with
        a standardized mean difference. Pass something outside the vocabulary and the
        record refuses to be built, at collection time rather than at interpretation
        time.""",
    instead="""A dataframe of estimates and standard errors. It pools anything you put
        in it, including things that are not the same kind of number, and the mistake
        surfaces — if it ever does — in review.""",
)
builder = MetaBuilder().name("nudge-replication").family("behavioural_nudge")
for study, estimate, se_i, n in STUDIES:
    builder = builder.study(
        study,
        estimate=estimate,
        se=se_i,
        read="experiment",
        contributor=study,
        quantity="standardized_contrast",
        n=n,
        source=f"paper-{study}",
    )
corpus = builder.build()
y, se = corpus.arrays()
w.out(f"corpus   : {corpus.name}, k = {len(corpus)} studies")
w.out(f"quantity : {corpus.records[0].quantity}")
w.out(f"effects  : {min(y):.2f} to {max(y):.2f}")
w.out(f"n        : {min(n for _, _, _, n in STUDIES):,} to {max(n for _, _, _, n in STUDIES):,}")

# ----------------------------------------------------------------------------------

w.step(
    "Pool it three ways, because the estimators disagree",
    why="""The between-study variance has to be estimated, and DerSimonian-Laird,
        Paule-Mandel and REML do it differently. When they agree, quoting one is
        fine. When they do not, the disagreement is information about how much of the
        answer is coming from the tau estimator rather than from the studies — and it
        is worth knowing before quoting any of them.""",
    instead="""Using DerSimonian-Laird because it is the default in most software. It
        is known to understate tau-squared when heterogeneity is high, which is
        exactly the regime this literature is in.""",
)
pooled_rows = []
for method, label in (("dl", "DerSimonian-Laird"), ("pm", "Paule-Mandel"), ("reml", "REML")):
    re = random_effects(y, se, tau_method=method)
    w.out(
        f"{label:<20}{re.estimate:>9.4f}  se {re.se:.4f}  tau2 {re.tau2:.5f}   "
        f"[{re.interval.lower:.3f}, {re.interval.upper:.3f}]"
    )
    pooled_rows.append(
        {
            "label": label,
            "estimate": float(re.estimate),
            "lower": float(re.interval.lower),
            "upper": float(re.interval.upper),
            "note": f"tau^2 = {re.tau2:.5f}",
        }
    )
w.figure(
    "estimators",
    kind="intervals",
    data={"rows": pooled_rows},
    opt={
        "rows": "@rows",
        "zero": True,
        "xLabel": "pooled effect",
        "labelWidth": 150,
        "rowHeight": 36,
        "mass": 95,
    },
    title="Three ways of estimating the between-study variance",
    note="""Close enough here that the choice does not change the conclusion — which
        is itself worth establishing rather than assuming, and takes one loop.""",
)

# ----------------------------------------------------------------------------------

w.step(
    "Draw all twenty, because the pooled number hides them",
    why="""A forest plot is not decoration. The pooled mean is a summary of a
        distribution, and the only way to see whether that distribution has a centre
        worth summarising is to look at the studies. Here the effects run from 0.05
        to 0.51 and the small studies sit systematically to the right.""",
    instead="""Reporting the pooled estimate and I-squared as two numbers. I-squared
        of 70% and I-squared of 70% look identical in a table and can mean a tidy
        spread or two distinct clusters, which imply completely different next
        steps.""",
)
het = heterogeneity(y, se)
kh = random_effects(y, se, tau_method="reml", knapp_hartung=True)
w.out(f"Q  = {het.q:.1f} on {het.df} df (p = {het.p_value:.4f})")
w.out(f"I2 = {het.i2:.0%}")
w.out(
    f"pooled (REML, Knapp-Hartung) = {kh.estimate:.4f} "
    f"[{kh.interval.lower:.3f}, {kh.interval.upper:.3f}]"
)
w.say(f"""An I-squared of {het.i2:.0%} means most of the variation between these studies
    is real disagreement rather than sampling noise. The pooled mean is a summary of
    a distribution, not an estimate of one shared number — and that distinction is
    what the next two steps are about.""")
order = sorted(range(len(STUDIES)), key=lambda i: STUDIES[i][1])
weights = [1.0 / (s * s) for s in se]
total_weight = sum(weights)
w.figure(
    "forest",
    kind="intervals",
    data={
        "rows": [
            {
                "label": STUDIES[i][0],
                "estimate": float(y[i]),
                "lower": float(y[i] - 1.96 * se[i]),
                "upper": float(y[i] + 1.96 * se[i]),
                "weight": weights[i] / total_weight,
                "note": f"n = {STUDIES[i][3]:,}, se {se[i]:.3f}",
            }
            for i in order
        ],
        "pooled": {
            "estimate": float(kh.estimate),
            "lower": float(kh.interval.lower),
            "upper": float(kh.interval.upper),
        },
    },
    opt={
        "rows": "@rows",
        "pooled": "@pooled",
        "pooledLabel": "pooled (REML)",
        "weightBy": "weight",
        "zero": True,
        "xLabel": "standardized effect",
        "labelWidth": 90,
        "rowHeight": 26,
        "mass": 95,
    },
    title="Every study, sorted by effect, sized by weight",
    note="""The dot size is the study's weight in the pool. Notice that the largest
        dots — the most precise studies — sit at the left end, and the studies at the
        right are the small ones. That pattern is what the funnel below is for.""",
    legend=(("accent", "95% interval, sized by weight"),),
)

# ----------------------------------------------------------------------------------

w.step(
    "Then compute the number a replication team actually needs",
    why="""The interval on the mean says where the average of this literature sits.
        The prediction interval says where a NEW study would land, and it includes
        the between-study variance rather than averaging it away. When heterogeneity
        is real these are very different numbers, and only the second one predicts
        replication.""",
    instead="""Powering the replication off the pooled mean. It is the single most
        common way a replication ends up underpowered for the effect it actually
        meets: the mean is the centre of a distribution the new study is drawn
        from, not the value it will take.""",
)
pi = prediction_interval(kh)
ratio = (pi.upper - pi.lower) / (kh.interval.upper - kh.interval.lower)
w.out(f"pooled mean        {kh.estimate:.4f}  [{kh.interval.lower:.3f}, {kh.interval.upper:.3f}]")
w.out(f"NEXT study lands in           [{pi.lower:.3f}, {pi.upper:.3f}]")
w.out(f"the prediction interval is {ratio:.1f}x wider")
w.figure(
    "prediction",
    kind="intervals",
    data={
        "rows": [
            {
                "label": "where the mean is",
                "estimate": float(kh.estimate),
                "lower": float(kh.interval.lower),
                "upper": float(kh.interval.upper),
                "note": "95% interval on the pooled mean",
            },
            {
                "label": "where study 21 lands",
                "estimate": float(kh.estimate),
                "lower": float(pi.lower),
                "upper": float(pi.upper),
                "note": "95% prediction interval",
                "colour": "s2",
            },
        ]
    },
    opt={
        "rows": "@rows",
        "zero": True,
        "xLabel": "standardized effect",
        "labelWidth": 160,
        "rowHeight": 42,
        "mass": 95,
    },
    title="Two intervals that are routinely confused",
    note=f"""Same centre, {ratio:.1f} times the width. A replication powered off the
        mean is powered for the wrong quantity, and the bottom of the prediction
        interval — {pi.lower:.2f} — is where an honest power calculation has to
        start.""",
    legend=(("accent", "interval on the mean"), ("s2", "prediction interval")),
)

# ----------------------------------------------------------------------------------

w.step(
    "Check whether the small studies are telling a different story",
    why="""Egger's test asks whether effect size is associated with precision. A
        non-zero intercept means the imprecise studies report systematically larger
        effects — the signature of publication bias. The funnel plot is the same
        question drawn, and it shows the asymmetry rather than compressing it into a
        p-value.""",
    instead="""Treating a significant Egger test as proof of publication bias. It is
        not: real heterogeneity correlated with study size looks identical. Small
        studies often run more intensive versions of an intervention, and that would
        produce the same picture with nobody hiding anything.""",
)
eg = egger(y, se)
w.out(f"Egger intercept {eg.intercept:+.3f} +/- {eg.se:.3f}")
w.out(f"t = {eg.t:+.2f} on {eg.df} df, p = {eg.p:.4f}")
fu = funnel_data(y, se, kh)
w.figure(
    "funnel",
    kind="funnel",
    data={
        "y": [float(v) for v in y],
        "se": [float(v) for v in se],
        "labels": [r.study for r in corpus.records],
        "pooled": float(fu.pooled),
        "contours": [
            {
                "mass": c.mass,
                "se": [float(v) for v in c.se],
                "lower": [float(v) for v in c.lower],
                "upper": [float(v) for v in c.upper],
            }
            for c in fu.contours
        ],
        "max_se": float(max(se)),
    },
    opt={"key": "@", "xLabel": "standardized effect", "height": 360},
    title="The funnel, with its contours",
    note="""Precise studies sit at the top and should scatter symmetrically about the
        pooled line. Here the bottom-left of the funnel is empty: there are no small
        studies reporting small effects, which is either what publication looks like
        or what a genuine relationship between study size and effect looks like.""",
)

# ----------------------------------------------------------------------------------

w.step(
    "And ask which studies are carrying the answer",
    why="""Leave-one-out says how far the pooled estimate moves when each study is
        dropped; the Baujat plot separates two different reasons a study matters —
        contributing heterogeneity, and influencing the result. A study high on both
        axes is one whose inclusion decision is doing real work and should be argued
        for rather than defaulted into.""",
    instead="""Running a sensitivity analysis that drops studies by quality rating.
        Quality ratings are largely uncorrelated with influence, so that procedure
        removes studies that do not matter and keeps the ones that do.""",
)
loo = leave_one_out(y, se, method="reml")
influence = sorted(
    zip((r.study for r in corpus.records), loo.influence, strict=True),
    key=lambda t: abs(t[1]),
    reverse=True,
)
by_estimate = {s: e for s, e, _, _ in STUDIES}
by_n = {s: n for s, _, _, n in STUDIES}
w.table(
    ["study", "shift when dropped", "effect", "n"],
    [
        [study, f"{shift:+.3f}", f"{by_estimate[study]:.2f}", f"{by_n[study]:,}"]
        for study, shift in influence[:5]
    ],
    caption="The five most influential studies, in pooled-se units.",
)
bj = baujat(y, se)
worst = max(range(len(bj.q_contribution)), key=lambda i: bj.q_contribution[i])
w.out(
    f"largest contributor to heterogeneity: {corpus.records[worst].study} "
    f"(Q share {bj.q_contribution[worst]:.2f})"
)
w.figure(
    "baujat",
    kind="scatter",
    data={
        "x": [float(v) for v in bj.q_contribution],
        "y": [float(v) for v in bj.influence],
        "labels": [r.study for r in corpus.records],
    },
    opt={
        "key": "@",
        "x": "x",
        "y": "y",
        "labels": "labels",
        "xLabel": "contribution to heterogeneity",
        "yLabel": "influence on the pooled estimate",
        "height": 330,
    },
    title="Baujat: which studies matter, and for which reason",
    note="""Far right means a study disagrees with the rest. High up means dropping it
        moves the answer. The top-right corner is where an inclusion decision changes
        the conclusion, and those are the studies to argue about by name.""",
)

# ----------------------------------------------------------------------------------

w.finding(f"""Reported as one number this literature says the nudge works: {kh.estimate:.2f},
    interval comfortably clear of zero. Reported honestly it says something more
    useful and less quotable.""")
w.finding(f"""The studies genuinely disagree — {het.i2:.0%} of the variation is real rather
    than sampling — effect size tracks precision in a way consistent with publication
    bias (p = {eg.p:.4f}), and study twenty-one could land anywhere in
    [{pi.lower:.2f}, {pi.upper:.2f}]. The bottom of that range is about a quarter of
    the pooled mean and small enough that a replication powered off {kh.estimate:.2f}
    would miss it.""")
w.finding("""None of that needed a different estimator. It needed the three quantities that
    usually end up in an appendix: heterogeneity, the prediction interval, and a
    small-study check — plus two plots, because I-squared and an Egger p-value are
    summaries of pictures and the pictures are where the shape is.""")
