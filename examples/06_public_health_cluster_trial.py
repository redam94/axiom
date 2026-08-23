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

from _walkthrough import Walkthrough

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
VILLAGE = Unit(name="village", kind="cluster")

w = Walkthrough(
    field="Public health",
    title="The same children, worth five times as much",
    question="""A chlorination programme has to be randomized by village, because
        neighbours share water and would talk. How many villages, how many children
        in each, and why is that second question the one that decides the trial?""",
)

# ----------------------------------------------------------------------------------

w.step(
    "Name the thing being randomized, in the design itself",
    why="""The unit of randomization is a village and the unit of measurement is a
        child, and almost every mistake in this literature comes from letting those
        two blur. A ClusterDesign carries the Unit it randomizes rather than just the
        arithmetic, so a power calculation cannot quietly be performed on the wrong
        one.""",
    instead="""Randomizing households and hoping contamination is small. Neighbours
        share a water source and a conversation; a household-randomized chlorination
        trial measures the programme minus whatever leaked across the fence, and
        nothing in the analysis can recover the difference.""",
)
w.out(f"randomization unit : {VILLAGE.name} ({VILLAGE.kind})")
w.out("measurement unit   : child")
w.out(f"outcome            : diarrhoea episodes per child-year, sd {OUTCOME_SD}")
w.out(f"effect worth having: {TARGET_EFFECT} sd")
w.out(f"intra-cluster corr : {ICC}")

# ----------------------------------------------------------------------------------

w.step(
    "Get the naive answer first, so the damage is measurable",
    why="""The independent-children calculation is not a straw man — it is what a
        sample-size calculator returns if nobody tells it about villages, and it is
        what ends up in a great many protocols. Computing it explicitly gives the
        rest of the walkthrough something to be a multiple of.""",
    instead="""Skipping straight to the clustered formula. Then the design effect is
        a number in a spreadsheet rather than a factor of five, and factors of five
        are what get a budget conversation started.""",
)
naive = sample_size(effect=TARGET_EFFECT, sd=OUTCOME_SD, power=0.8, alpha=0.05)
w.out("if children were independent (they are not):")
w.out(f"  n = {naive.n} children ({naive.n_treated} treated / {naive.n_control} control)")
w.out(f"  power {naive.power:.3f}")

# ----------------------------------------------------------------------------------

w.step(
    "Then look at what clustering costs, and where the cost comes from",
    why="""The design effect is 1 + (m - 1) x ICC. Read that formula: the
        correlation enters once, the cluster SIZE enters as a multiplier. An ICC of
        0.03 sounds negligible and is negligible in clusters of five; in clusters of
        two hundred it multiplies your variance by seven. Size is the lever, and it
        is the one most protocols treat as an afterthought.""",
    instead="""Reporting the ICC alone and calling it small. 'Our ICC is only 0.03'
        is the single most common way a cluster trial talks itself into being
        underpowered — the number is meaningless without the cluster size beside
        it.""",
)
de = design_effect(CLUSTER_SIZE, ICC)
w.out(f"design effect at m = {CLUSTER_SIZE}, ICC = {ICC}: {de:.2f}x variance inflation")

sizes = [5, 10, 20, 40, 80, 120, 160, 200]
curves = {icc: [design_effect(m, icc) for m in sizes] for icc in (0.01, 0.03, 0.05, 0.10)}
w.table(
    ["children per village"] + [f"ICC {icc}" for icc in curves],
    [[str(m)] + [f"{curves[icc][i]:.2f}" for icc in curves] for i, m in enumerate(sizes)],
    caption="Variance inflation. Every entry is a multiplier on the sample you need.",
)
w.figure(
    "deff",
    kind="lines",
    data={"m": sizes, **{f"icc{int(icc * 100):02d}": curves[icc] for icc in curves}},
    opt={
        "series": [
            {"label": f"ICC {icc}", "x": "@m", "y": f"@icc{int(icc * 100):02d}"} for icc in curves
        ],
        "xLabel": "children measured per village",
        "yLabel": "design effect",
        "height": 300,
        "rightPad": 80,
    },
    title="The penalty is on cluster size, not on the correlation",
    note="""Every line is straight, because the design effect is linear in cluster
        size. Doubling the ICC doubles the slope; doubling the cluster size doubles
        the penalty outright. The choice a protocol actually controls is the
        horizontal axis.""",
)

# ----------------------------------------------------------------------------------

w.step(
    "Size the trial for the design you are really running",
    why="""clusters_needed inverts the whole thing: given the effect, the ICC and the
        cluster size, how many villages. The effective sample size is the number
        worth quoting beside it — it says how many independent children all those
        real children are actually worth, which is the honest denominator for
        everything downstream.""",
    instead="""Inflating the naive n by the design effect and dividing by the cluster
        size. It gets you close, and it hides that both power and the minimum
        detectable effect should be read off the same design object rather than
        recomputed by hand from a rounded intermediate.""",
)
need = clusters_needed(
    effect=TARGET_EFFECT, sd=OUTCOME_SD, cluster_size=CLUSTER_SIZE, icc=ICC, power=0.8, alpha=0.05
)
ess = effective_sample_size(need.n, CLUSTER_SIZE, ICC)
design = ClusterDesign(unit=VILLAGE, n_clusters=need.n, cluster_size=CLUSTER_SIZE, icc=ICC)
power = cluster_power(design, effect=TARGET_EFFECT, sd=OUTCOME_SD)
mde = cluster_mde(design, sd=OUTCOME_SD, power=0.8)
w.out(f"villages needed : {need.n} ({need.n_treated} treated / {need.n_control} control)")
w.out(f"children        : {need.n * CLUSTER_SIZE:,}")
w.out(f"worth about     : {ess:.0f} independent children")
w.out(f"power at {TARGET_EFFECT}   : {power.power:.3f}")
w.out(f"MDE at 80% power: {mde.effect:.3f}")
w.figure(
    "cost",
    kind="bars",
    data={
        "rows": [
            {
                "label": "if children were independent",
                "value": float(naive.n),
                "display": f"{naive.n:,}",
                "colour": "ink-3",
                "note": "what a calculator with no notion of villages returns",
            },
            {
                "label": "children actually enrolled",
                "value": float(need.n * CLUSTER_SIZE),
                "display": f"{need.n * CLUSTER_SIZE:,}",
                "colour": "accent",
                "note": f"{need.n} villages of {CLUSTER_SIZE}",
            },
            {
                "label": "independent children they are worth",
                "value": float(ess),
                "display": f"{ess:,.0f}",
                "colour": "boundary",
                "note": "the effective sample size",
            },
        ]
    },
    opt={"rows": "@rows", "xLabel": "children", "labelWidth": 220, "rowHeight": 44},
    title="What the trial enrols, and what it is worth",
    note=f"""Enrol {need.n * CLUSTER_SIZE:,} children to buy the information in
        {ess:.0f}. The gap is the design effect, and it is paid in field cost,
        consent, follow-up and time.""",
)

# ----------------------------------------------------------------------------------

w.step(
    "Now the question the budget is really about: shape, not size",
    why="""Fix the number of children at about 1,600 and vary only how they are
        grouped. This is the comparison that decides a cluster trial, and it is
        almost never drawn: the same 1,600 children give wildly different power
        depending on whether they sit in ten villages or a hundred and sixty.""",
    instead="""Asking 'how many children can we afford'. Total enrolment is close to
        irrelevant here compared with the grouping. A protocol that negotiates hard
        for 20% more children and accepts larger villages to get them can easily end
        up with less power than it started with.""",
)
shapes = [(10, 160), (20, 80), (40, 40), (80, 20), (160, 10)]
powers = []
for n_clusters, size in shapes:
    d = ClusterDesign(unit=VILLAGE, n_clusters=n_clusters, cluster_size=size, icc=ICC)
    powers.append(float(cluster_power(d, effect=TARGET_EFFECT, sd=OUTCOME_SD).power))
w.table(
    ["villages", "children each", "total children", "design effect", "power"],
    [
        [
            str(n),
            str(m),
            f"{n * m:,}",
            f"{design_effect(m, ICC):.2f}",
            f"{p:.3f}",
        ]
        for (n, m), p in zip(shapes, powers, strict=True)
    ],
)
w.figure(
    "shape",
    kind="bars",
    data={
        "rows": [
            {
                "label": f"{n} villages of {m}",
                "value": p,
                "display": f"{p:.2f}",
                "colour": "boundary" if p < 0.8 else "accent",
                "note": f"{n * m:,} children, design effect {design_effect(m, ICC):.2f}",
            }
            for (n, m), p in zip(shapes, powers, strict=True)
        ]
    },
    opt={"rows": "@rows", "xLabel": "power", "labelWidth": 170, "rowHeight": 42},
    title="Same 1,600 children, five different trials",
    note=f"""From {min(powers):.2f} to {max(powers):.2f} with no change in enrolment.
        Red is below the conventional 0.8 bar. Nothing about the intervention, the
        effect or the budget changed between these rows — only the grouping.""",
    legend=(("accent", "reaches 80% power"), ("boundary", "does not")),
)

# ----------------------------------------------------------------------------------

w.step(
    "Put field cost on it, because the trade-off runs the other way",
    why="""Villages are expensive to reach and children within them are cheap, so
        the design that maximises power is not the one that minimises cost. With an
        illustrative 4,000 per village of setup and 25 per child enrolled, a fixed
        budget buys a curve with an interior maximum — and that maximum is the
        design worth proposing.""",
    instead="""Choosing the highest-power shape from the previous step and taking it
        to the funder. 160 villages of ten wins on power and spends almost all of its
        money on reaching villages; the recommendation has to survive the budget, not
        just the power calculation.""",
)
SETUP_PER_VILLAGE, PER_CHILD, BUDGET = 4_000.0, 25.0, 200_000.0
options, opt_power, opt_labels = [], [], []
for n_clusters in range(10, 100, 5):
    affordable = (BUDGET - SETUP_PER_VILLAGE * n_clusters) / (PER_CHILD * n_clusters)
    if affordable < 2:
        continue
    m = int(affordable)
    d = ClusterDesign(unit=VILLAGE, n_clusters=n_clusters, cluster_size=m, icc=ICC)
    options.append(n_clusters)
    opt_power.append(float(cluster_power(d, effect=TARGET_EFFECT, sd=OUTCOME_SD).power))
    opt_labels.append(m)
w.table(
    ["villages", "children each, at this budget", "total children", "power"],
    [
        [str(n), str(m), f"{n * m:,}", f"{p:.3f}"]
        for n, m, p in zip(options, opt_labels, opt_power, strict=True)
        if n % 10 == 0
    ],
    caption=f"A fixed budget of {BUDGET:,.0f}, at {SETUP_PER_VILLAGE:,.0f} per village "
    f"and {PER_CHILD:.0f} per child.",
)
best = max(range(len(options)), key=lambda i: opt_power[i])
w.figure(
    "budget",
    kind="lines",
    data={"villages": options, "power": opt_power},
    opt={
        "series": [{"label": "power", "x": "@villages", "y": "@power", "colour": "accent"}],
        "xLabel": "villages, with children per village set by what the budget allows",
        "yLabel": "power",
        "hline": 0.8,
        "hlineLabel": "conventional 80%",
        "height": 290,
        "rightPad": 70,
    },
    title="What a fixed field budget can actually buy",
    note=f"""The curve turns over. Too few villages and the design effect eats the
        sample; too many and the per-village setup cost leaves no children to
        measure. The best buy here is {options[best]} villages of {opt_labels[best]},
        at power {opt_power[best]:.2f}.""",
    legend=(("accent", "power at this budget"), ("ink-3:dash", "the conventional bar")),
)

# ----------------------------------------------------------------------------------

w.finding(f"""An ICC of {ICC} — a number most protocols would describe as negligible —
    turned {naive.n} children into {need.n * CLUSTER_SIZE:,}. The correlation was
    never the problem. The cluster size was: at forty children per village the design
    effect is {de:.2f}, and it would be {design_effect(160, ICC):.2f} at a hundred and
    sixty.""")
w.finding(f"""The same 1,600 children bought power of {min(powers):.2f} or {max(powers):.2f}
    depending only on how they were grouped. Ten villages of 160 is close to
    worthless; 160 villages of 10 is a real trial. That is a factor a protocol
    controls completely and usually decides by convenience.""")
w.finding(f"""Once field cost is on the table the answer moves again, and it stops being
    'more villages'. At {BUDGET:,.0f} the curve peaks at {options[best]} villages of
    {opt_labels[best]} and falls away on both sides. This is the conversation worth
    having before the protocol is written rather than after the results come back
    inconclusive — and it needs three numbers the statistics alone never supplies:
    the ICC, the cost of reaching a village, and the cost of a child.""")
