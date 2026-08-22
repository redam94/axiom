"""Product analytics — is this A/B test worth running at all?

The question
    A team wants to test a checkout change. The usual conversation is about sample
    size and how long it will take. The prior question, almost never asked, is
    whether the test can change what anyone does.

    If you would ship the change whatever the result, the test is theatre. If the
    decision genuinely hangs in the balance, the test has value, and that value can
    be put in the same units as its cost.

The move
    Value of information. EVPI is the ceiling -- what you would pay to know the
    truth outright. EVSI is what a test of a given precision is actually worth. Both
    fall to nearly zero when the prior already settles the decision, which is the
    quantitative version of "we already know this".

Pillars: design (value of information, candidate scoring)
"""

import numpy as np
from _walkthrough import Walkthrough

from axiom.design import (
    DecisionSpec,
    DesignCandidate,
    EconomicInputs,
    ValuePerOutcome,
    difference_se,
    eig_gaussian,
    evaluate_candidate,
    evoi_gaussian,
    pareto_front,
    sample_size,
)

MONTHLY_USERS = 250_000
VALUE_PER_CONVERSION = 12.0  # USD of margin
THRESHOLD = 0.004  # +0.4pp conversion, the bar for rolling out
HORIZON_MONTHS = 12  # a checkout change ships and stays shipped
BASE_RATE = 0.052
OUTCOME_SD = (BASE_RATE * (1 - BASE_RATE)) ** 0.5
PRIOR_MEAN, PRIOR_SD = 0.004, 0.0035

w = Walkthrough(
    field="Product analytics",
    title="The worst-powered design is the one worth running",
    question="""Before asking how many users an A/B test needs, ask whether it can
        change what anyone does — and then whether the precision it buys is worth
        more than the traffic it withholds.""",
)

# ----------------------------------------------------------------------------------

w.step(
    "State the decision and, carefully, the stake",
    why="""The stake is the whole horizon the decision governs, not one month. A
        checkout change ships and stays shipped, so the value of getting it right is
        twelve months of margin on the whole base. Getting this wrong is the single
        most common way a value-of-information calculation comes out saying 'never
        test': the cost of the test is immediate and real, so the benefit has to be
        counted over the life of the decision it informs.""",
    instead="""Valuing the decision at one month of impact. It makes every test look
        unaffordable, which produces the same answer as never running the
        calculation — and it is wrong for a different reason than the intuition it
        agrees with.""",
)
decision = DecisionSpec(
    name="roll_out_checkout_change",
    threshold=THRESHOLD,
    value_per_outcome_unit=VALUE_PER_CONVERSION * MONTHLY_USERS * HORIZON_MONTHS,
    numeraire="USD",
)
w.out(f"decision : roll out if the lift beats {THRESHOLD:.1%}")
w.out(f"stakes   : {decision.value_per_outcome_unit:,.0f} USD per unit of conversion rate")
w.out(f"           over {HORIZON_MONTHS} months and {MONTHLY_USERS:,} users a month")
w.out(
    f"so a {THRESHOLD:.1%} lift is worth " f"{THRESHOLD * decision.value_per_outcome_unit:,.0f} USD"
)

# ----------------------------------------------------------------------------------

w.step(
    "Ask what a test is worth under three different beliefs",
    why="""EVPI is the ceiling: what you would pay to know the truth outright. EVSI
        is what a test of a given precision is actually worth. Both collapse when the
        prior already settles the decision — which is the quantitative version of 'we
        already know this', and it is a sentence a team can act on in a way it cannot
        act on a hunch.""",
    instead="""Starting from the sample size. A sample-size calculator will happily
        size a test for a decision that is already made; it has no way to represent
        the fact that the result cannot change anything.""",
)
n_pilot = 40_000
se_pilot = difference_se(n_pilot, sd=OUTCOME_SD)
BELIEFS = (
    ("already sure it works", 0.011, 0.0020),
    ("genuinely on the fence", 0.004, 0.0035),
    ("already sure it does not", -0.004, 0.0020),
)
for label, prior_mean, prior_sd in BELIEFS:
    ev = evoi_gaussian(decision, prior_mean, prior_sd, se_pilot)
    w.out(
        f"{label:<28}EVPI {ev.evpi:>11,.0f}   EVSI {ev.evsi:>11,.0f}   "
        f"EIG {eig_gaussian(prior_sd, se_pilot):>6.3f}"
    )

sweep = np.linspace(-0.006, 0.014, 61)
sweep_evsi, sweep_evpi = [], []
for m in sweep:
    ev = evoi_gaussian(decision, float(m), PRIOR_SD, se_pilot)
    sweep_evsi.append(float(ev.evsi))
    sweep_evpi.append(float(ev.evpi))
w.figure(
    "belief",
    kind="lines",
    data={"mean": sweep.tolist(), "evsi": sweep_evsi, "evpi": sweep_evpi},
    opt={
        "series": [
            {"label": "EVPI", "x": "@mean", "y": "@evpi", "colour": "ink-2"},
            {"label": "EVSI", "x": "@mean", "y": "@evsi", "colour": "accent"},
        ],
        "xLabel": "what the team already believes the lift is",
        "yLabel": "value of the information, USD",
        "height": 300,
        "rightPad": 70,
    },
    title="A test is only worth something near the decision boundary",
    note=f"""Both curves peak where the prior mean sits on the rollout bar of
        {THRESHOLD:.1%} and fall away in both directions. Far from the bar the
        decision is already made, and buying information about it returns almost
        nothing — however many users are thrown at it and however tight the resulting
        confidence interval looks in the readout.""",
    legend=(("ink-2", "EVPI: knowing the truth outright"), ("accent", "EVSI: this test")),
)

# ----------------------------------------------------------------------------------

w.step(
    "Take the live case and size it the ordinary way",
    why="""The on-the-fence case is the only one where a test earns its keep, so it
        is the one to size. This step is the conventional power calculation, run
        exactly as it normally would be, so that the next two steps have something
        familiar to disagree with.""",
    instead="""Nothing — this step is not in dispute. A minimum detectable effect and
        80% power is the right way to turn a decision-relevant effect size into a
        number of users. The argument is about what happens next.""",
)
mde = 0.003
sized = sample_size(effect=mde, sd=OUTCOME_SD, power=0.8, alpha=0.05)
se_at_n = difference_se(sized.n, sd=OUTCOME_SD)
ev = evoi_gaussian(decision, PRIOR_MEAN, PRIOR_SD, se_at_n)
w.out(f"powering for a {mde:.1%} lift:")
w.out(f"  n = {sized.n:,} users ({sized.n_treated:,} / {sized.n_control:,})")
w.out(f"  power {sized.power:.3f}, experiment se {se_at_n:.5f}")
w.out(
    f"  EVPI {ev.evpi:,.0f} USD, EVSI {ev.evsi:,.0f} USD "
    f"({ev.evsi / ev.evpi:.0%} of the ceiling)"
)
w.out(f"  posterior sd after the test {ev.preposterior_sd:.5f}, down from {PRIOR_SD:.5f}")

# ----------------------------------------------------------------------------------

w.step(
    "Then watch precision stop paying",
    why="""More precision costs more exposure, and EVSI is bounded above by EVPI no
        matter how large the test. Plotting the two together shows where the curve
        flattens: past that point extra users buy a tighter interval that does not
        change the decision, and a tighter interval that changes nothing is worth
        nothing.""",
    instead="""Choosing the sample size that reaches 95% power. Power is a property of
        the test, not of the decision; it keeps improving long after the decision has
        stopped being in doubt, and nothing in a power calculation ever says
        'enough'.""",
)
ns = [10_000, 20_000, 40_000, 80_000, 160_000, 320_000, 640_000, 1_280_000]
evsis = []
for n in ns:
    e = evoi_gaussian(decision, PRIOR_MEAN, PRIOR_SD, difference_se(n, sd=OUTCOME_SD))
    evsis.append(float(e.evsi))
w.table(
    ["test size", "experiment se", "EVSI", "% of EVPI"],
    [
        [
            f"{n:,}",
            f"{difference_se(n, sd=OUTCOME_SD):.5f}",
            f"{e:,.0f}",
            f"{e / ev.evpi:.0%}",
        ]
        for n, e in zip(ns, evsis, strict=True)
    ],
)
w.figure(
    "returns",
    kind="lines",
    data={"n": ns, "evsi": evsis},
    opt={
        "series": [{"label": "EVSI", "x": "@n", "y": "@evsi", "colour": "accent"}],
        "xLabel": "users in the test",
        "yLabel": "value of what the test tells you, USD",
        "hline": float(ev.evpi),
        "hlineLabel": "EVPI — the ceiling",
        "height": 300,
        "rightPad": 70,
    },
    title="Doubling the test does not double what it is worth",
    note=f"""Each doubling of exposure buys less than the last, and the curve is
        asymptotic to the value of knowing the truth outright. Between
        {ns[-2]:,} and {ns[-1]:,} users the test gains
        {evsis[-1] - evsis[-2]:,.0f} USD of value for double the traffic.""",
    legend=(("accent", "EVSI at this size"), ("ink-3:dash", "EVPI")),
)

# ----------------------------------------------------------------------------------

w.step(
    "Now count what the test costs, including the part nobody bills for",
    why="""Showing someone a different checkout page costs nothing to serve, so the
        cost per unit of exposure is zero — but withholding the change from the
        holdout arm forgoes the lift on those users for the duration, and that is a
        real cost the power calculation never mentions. Putting the price of a
        conversion on the record, with its source, is what lets the two be compared
        at all.""",
    instead="""Comparing designs on run cost alone. Engineering time and analyst time
        are the visible costs; the invisible one is usually larger, and it scales
        with exactly the thing a power calculation wants to maximise — how much
        traffic you hold back.""",
)
value = ValuePerOutcome(
    value=VALUE_PER_CONVERSION,
    outcome_unit="conversion",
    numeraire="USD",
    source="finance, margin per converted user, 2026 H1",
)
economics = EconomicInputs(
    value_per_outcome=value,
    dose_per_period=float(sized.n),
    discount_rate=0.0,
    dose_unit="exposure",
    dose_cost_per_unit=0.0,
    marginal_value_ratio=PRIOR_MEAN,
)
w.out(f"the price is on the record: {value.ledger_line().statement}")

candidates = [
    DesignCandidate(
        name="two_week_50_50",
        method="cluster_based_regression",
        n_units=sized.n,
        n_periods=2,
        holdout_fraction=0.5,
        experiment_se=se_at_n,
        cost=18_000.0,
        cooldown_periods=0,
    ),
    DesignCandidate(
        name="one_week_90_10",
        method="cluster_based_regression",
        n_units=sized.n,
        n_periods=1,
        holdout_fraction=0.1,
        experiment_se=se_at_n * 1.6,
        cost=9_000.0,
        cooldown_periods=0,
    ),
    DesignCandidate(
        name="four_week_50_50",
        method="cluster_based_regression",
        n_units=sized.n * 2,
        n_periods=4,
        holdout_fraction=0.5,
        experiment_se=se_at_n * 0.7,
        cost=34_000.0,
        cooldown_periods=1,
    ),
]
scores = [
    evaluate_candidate(c, decision, prior_mean=PRIOR_MEAN, prior_sd=PRIOR_SD, economics=economics)
    for c in candidates
]
front = {s.name for s in pareto_front(scores, objectives=("net_value", "-cost", "eig"))}
w.table(
    ["design", "power", "EVSI", "holdout cost", "run cost", "net", "on the front"],
    [
        [
            s.name,
            f"{s.power:.2f}",
            f"{s.evsi:,.0f}",
            f"{s.opportunity_cost:,.0f}",
            f"{s.cost:,.0f}",
            f"{s.net_value:,.0f}",
            "yes" if s.name in front else "",
        ]
        for s in scores
    ],
)
w.figure(
    "net",
    kind="bars",
    data={
        "rows": [
            {
                "label": s.name.replace("_", " "),
                "value": float(s.net_value),
                "display": f"{s.net_value:,.0f}",
                "colour": "accent" if s.net_value > 0 else "boundary",
                "note": f"power {s.power:.2f}, EVSI {s.evsi:,.0f}, "
                f"holdout cost {s.opportunity_cost:,.0f}",
            }
            for s in scores
        ]
    },
    opt={"rows": "@rows", "xLabel": "net value, USD", "labelWidth": 160, "rowHeight": 44},
    title="What each design is worth once everything is counted",
    note="""Red is net-negative: the design costs more than the decision it improves
        is worth. Hover any bar for the power it achieves — the ordering by net value
        is close to the reverse of the ordering by power.""",
    legend=(("accent", "net positive"), ("boundary", "net negative")),
)
by_name = {s.name: s for s in scores}
w.figure(
    "power_vs_net",
    kind="dumbbell",
    data={
        "rows": [
            {
                "label": s.name.replace("_", " "),
                "a": float(s.evsi),
                "b": float(s.net_value),
            }
            for s in scores
        ]
    },
    opt={
        "rows": "@rows",
        "xLabel": "USD",
        "aLabel": "EVSI, before costs",
        "bLabel": "net value",
        "labelWidth": 160,
        "rowHeight": 44,
    },
    title="How much of each design's value survives its costs",
    note="""The length of each bar is what the design spends — run cost plus the
        upside withheld from the holdout arm. The design with the largest gross value
        gives most of it back.""",
    legend=(("ink-2", "EVSI, gross"), ("accent", "net of run and holdout cost")),
)

# ----------------------------------------------------------------------------------

low, high = by_name["one_week_90_10"], by_name["four_week_50_50"]
winner = max(scores, key=lambda s: s.net_value)
w.finding(f"""The winner is the WORST-powered design on the list. '{low.name}' has power
    {low.power:.2f} — a number that would be rejected out of hand in a design
    review — and nets {low.net_value:,.0f} USD. '{high.name}' has power
    {high.power:.2f}, near certainty, and nets {high.net_value:,.0f}.""")
w.finding(f"""Nothing is wrong with the power calculation. It is answering a different
    question. Holding half the traffic away from a change the team mostly believes in
    forgoes {high.opportunity_cost:,.0f} USD of upside, and running four weeks
    instead of one costs more besides — neither of which appears anywhere in a
    sample-size formula. Once both are counted, the extra precision is being bought
    at a price higher than the decision it improves is worth.""")
w.finding("""Precision is an input, not the objective. Past the point where more of it stops
    changing what you would do, buying more is just spending — and the first chart in
    this walkthrough says where that point is before a single user is exposed.""")
