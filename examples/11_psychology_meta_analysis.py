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

from axiom.build import MetaBuilder
from axiom.meta import (
    baujat,
    egger,
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

builder = MetaBuilder().name("nudge-replication").family("behavioural_nudge")
for study, estimate, se, n in STUDIES:
    builder = builder.study(
        study,
        estimate=estimate,
        se=se,
        read="experiment",
        contributor=study,
        quantity="standardized_contrast",
        n=n,
        source=f"paper-{study}",
    )
# 'standardized_contrast' is not free text: poolable quantities are a controlled
# vocabulary, so two reviews cannot quietly pool things that are not the same kind
# of number. Pass something outside it and the record refuses to be built.
corpus = builder.build()
y, se = corpus.arrays()
print(f"corpus: {corpus.name}, k = {len(corpus)} studies")
print(f"  quantity: {corpus.records[0].quantity}")
print(f"  raw effects run from {min(y):.2f} to {max(y):.2f}")

# Three estimators of the between-study variance. They can disagree, and when they
# do it is worth knowing before quoting one of them.
print(f"\n{'tau estimator':<22}{'pooled':>9}{'se':>8}{'tau^2':>9}   95% interval")
for method, label in (("dl", "DerSimonian-Laird"), ("pm", "Paule-Mandel"), ("reml", "REML")):
    re = random_effects(y, se, tau_method=method)
    print(
        f"{label:<22}{re.estimate:>9.4f}{re.se:>8.4f}{re.tau2:>9.5f}   "
        f"[{re.interval.lower:.3f}, {re.interval.upper:.3f}]"
    )

het = heterogeneity(y, se)
print(
    f"\nheterogeneity: Q = {het.q:.1f} on {het.df} df (p = {het.p_value:.4f}), "
    f"I2 = {het.i2:.0%}"
)
print(f"  I2 of {het.i2:.0%} means most of the variation between these studies is")
print("  real disagreement rather than sampling noise. The pooled mean is a summary")
print("  of a distribution, not an estimate of one shared number.")

# The number a replication team actually wants.
kh = random_effects(y, se, tau_method="reml", knapp_hartung=True)
pi = prediction_interval(kh)
print(
    f"\npooled mean            {kh.estimate:.4f}  "
    f"[{kh.interval.lower:.3f}, {kh.interval.upper:.3f}]  (95%, Knapp-Hartung)"
)
print(f"NEXT study will land in [{pi.lower:.3f}, {pi.upper:.3f}]  (95% prediction)")
ratio = (pi.upper - pi.lower) / (kh.interval.upper - kh.interval.lower)
print(f"  the prediction interval is {ratio:.1f}x")
print("  wider than the interval on the mean. Powering a replication off the mean")
print("  is how a study ends up underpowered for the effect it actually meets.")

# Small-study effects: do the imprecise studies report systematically larger
# effects than the precise ones?
eg = egger(y, se)
print(
    f"\nEgger's test: intercept {eg.intercept:+.3f} +/- {eg.se:.3f}, "
    f"t = {eg.t:+.2f} on {eg.df} df, p = {eg.p:.4f}"
)
print("  a non-zero intercept means effect size is associated with precision --")
print("  the signature of publication bias, though not proof of it: real")
print("  heterogeneity correlated with study size looks identical.")

# Which studies are carrying the answer?
loo = leave_one_out(y, se, method="reml")
influence = sorted(
    zip((r.study for r in corpus.records), loo.influence, strict=True),
    key=lambda t: abs(t[1]),
    reverse=True,
)
print("\nmost influential studies (shift in pooled-se units when dropped):")
for study, shift in influence[:5]:
    est = dict((s, e) for s, e, _, _ in STUDIES)[study]
    n = dict((s, n) for s, _, _, n in STUDIES)[study]
    print(f"  {study}  {shift:+.3f}   (effect {est:.2f}, n = {n})")

bj = baujat(y, se)
worst = max(range(len(bj.q_contribution)), key=lambda i: bj.q_contribution[i])
print(
    f"\nlargest contributor to heterogeneity: "
    f"{corpus.records[worst].study} (Q share {bj.q_contribution[worst]:.2f})"
)

print(f"""
Reading it
    Reported as one number this literature says the nudge works: {kh.estimate:.2f}, interval
    comfortably clear of zero. Reported honestly it says something more useful and
    less quotable.

    The studies genuinely disagree ({het.i2:.0%} of the variation is real, not sampling),
    effect size tracks precision in a way consistent with publication bias
    (p = {eg.p:.4f}), and study twenty-one could land anywhere in
    [{pi.lower:.2f}, {pi.upper:.2f}] -- the bottom of which is about a quarter of the pooled
    mean and small enough that a replication powered off {kh.estimate:.2f} would miss it.

    None of that needed a different estimator. It needed the three quantities that
    usually end up in an appendix: heterogeneity, the prediction interval, and a
    small-study check.""")
