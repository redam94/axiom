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

# The decision: roll the change out to the whole base if the lift clears the bar
# that makes it worth the engineering and support cost.
MONTHLY_USERS = 250_000
VALUE_PER_CONVERSION = 12.0  # USD of margin
THRESHOLD = 0.004  # +0.4pp conversion, the bar for rolling out
HORIZON_MONTHS = 12  # a checkout change ships and stays shipped

# The stake is the whole horizon, not one month. Getting this wrong is the most
# common way a value-of-information calculation comes out saying "never test":
# the cost of the test is real and immediate, so the benefit has to be counted
# over the life of the decision it informs.
decision = DecisionSpec(
    name="roll_out_checkout_change",
    threshold=THRESHOLD,
    value_per_outcome_unit=VALUE_PER_CONVERSION * MONTHLY_USERS * HORIZON_MONTHS,
    numeraire="USD",
)

BASE_RATE = 0.052
OUTCOME_SD = (BASE_RATE * (1 - BASE_RATE)) ** 0.5

print(f"decision : roll out if the lift beats {THRESHOLD:.1%}")
print(
    f"stakes   : {decision.value_per_outcome_unit:,.0f} USD per unit of conversion "
    f"rate over {HORIZON_MONTHS} months"
)
print(
    f"           i.e. a {THRESHOLD:.1%} lift is worth "
    f"{THRESHOLD * decision.value_per_outcome_unit:,.0f} USD"
)

# Three states of belief about the same change, and what a test is worth in each.
print(f"\n{'prior belief':<34}{'EVPI':>12}{'EVSI':>12}{'EIG':>9}")
BELIEFS = (
    ("already sure it works", 0.011, 0.0020),
    ("genuinely on the fence", 0.004, 0.0035),
    ("already sure it does not", -0.004, 0.0020),
)
n_pilot = 40_000
se_pilot = difference_se(n_pilot, sd=OUTCOME_SD)
for label, prior_mean, prior_sd in BELIEFS:
    ev = evoi_gaussian(decision, prior_mean, prior_sd, se_pilot)
    eig = eig_gaussian(prior_sd, se_pilot)
    print(f"{label:<34}{ev.evpi:>12,.0f}{ev.evsi:>12,.0f}{eig:>9.3f}")

print("""
  The middle row is the only one where a test earns its keep. In the other two the
  prior has already made the decision, so buying information about it returns
  almost nothing -- however many users you throw at it, and however tight the
  resulting confidence interval looks in the readout.""")

# Take the live case and size it properly.
PRIOR_MEAN, PRIOR_SD = 0.004, 0.0035
mde = 0.003
sized = sample_size(effect=mde, sd=OUTCOME_SD, power=0.8, alpha=0.05)
print(f"\nfor the on-the-fence case, powering for a {mde:.1%} lift:")
print(
    f"  n = {sized.n:,} users ({sized.n_treated:,} / {sized.n_control:,}), "
    f"power {sized.power:.3f}"
)

se_at_n = difference_se(sized.n, sd=OUTCOME_SD)
ev = evoi_gaussian(decision, PRIOR_MEAN, PRIOR_SD, se_at_n)
print(f"  experiment se {se_at_n:.5f}")
print(
    f"  EVPI {ev.evpi:,.0f} USD, EVSI {ev.evsi:,.0f} USD "
    f"({ev.evsi / ev.evpi:.0%} of the ceiling)"
)
print(
    f"  posterior sd after the test would be {ev.preposterior_sd:.5f}, " f"down from {PRIOR_SD:.5f}"
)

# More precision costs more exposure. Where does it stop paying?
print(f"\n{'test size':>12}{'se':>10}{'EVSI':>12}{'% of EVPI':>12}")
for n in (20_000, 40_000, 80_000, 160_000, 320_000, 640_000):
    se_n = difference_se(n, sd=OUTCOME_SD)
    e = evoi_gaussian(decision, PRIOR_MEAN, PRIOR_SD, se_n)
    print(f"{n:>12,}{se_n:>10.5f}{e.evsi:>12,.0f}{e.evsi / ev.evpi:>11.0%}")

# Now the part a sample-size calculator cannot do: net of what the test costs.
value = ValuePerOutcome(
    value=VALUE_PER_CONVERSION,
    outcome_unit="conversion",
    numeraire="USD",
    source="finance, margin per converted user, 2026 H1",
)
print(f"\nthe price is on the record: {value.ledger_line().statement}")

# The "dose" here is exposure to the new checkout, measured in users per period.
# Showing someone a different page costs nothing to serve, so the cost per unit of
# dose is zero -- but withholding it from the holdout arm forgoes the lift on those
# users, and that is a real cost the power calculation never mentions.
economics = EconomicInputs(
    value_per_outcome=value,
    dose_per_period=float(sized.n),
    discount_rate=0.0,
    dose_unit="exposure",
    dose_cost_per_unit=0.0,
    marginal_value_ratio=PRIOR_MEAN,
)
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

print(
    f"\n{'design':<20}{'EVSI':>12}{'holdout cost':>15}{'run cost':>11}"
    f"{'net':>12}{'power':>8}  front"
)
for s in scores:
    print(
        f"{s.name:<20}{s.evsi:>12,.0f}{s.opportunity_cost:>15,.0f}{s.cost:>11,.0f}"
        f"{s.net_value:>12,.0f}{s.power:>8.3f}  {'yes' if s.name in front else ''}"
    )

winner = max(scores, key=lambda s: s.net_value)
print(f"\nhighest net value: {winner.name}")

by_name = {s.name: s for s in scores}
low, high = by_name["one_week_90_10"], by_name["four_week_50_50"]
print(f"""
Reading it
    The winner is the WORST-powered design on the list. '{low.name}' has power
    {low.power:.2f} -- a number that would be rejected out of hand in a design review --
    and nets {low.net_value:,.0f} USD. '{high.name}' has power {high.power:.2f}, near
    certainty, and nets {high.net_value:,.0f}.

    Nothing is wrong with the power calculation. It is answering a different
    question. Holding half the traffic away from a change you mostly believe in
    forgoes {high.opportunity_cost:,.0f} USD of upside, and running four weeks instead of
    one costs more besides -- neither of which appears anywhere in a sample-size
    formula. Once both are counted, the extra precision is being bought at a price
    higher than the decision it improves is worth.

    Precision is an input, not the objective. Past the point where more of it stops
    changing what you would do, buying more is just spending.""")
