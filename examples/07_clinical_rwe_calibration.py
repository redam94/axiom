"""Clinical research — using a trial to correct a registry.

The question
    A disease registry has tens of thousands of patient-years and is confounded:
    sicker patients get the aggressive therapy, so the raw association understates
    the benefit. A randomized trial has a few hundred patients, is clean, and is far
    too small and too short to answer the question the formulary actually asks.

    You want both. The registry has the coverage; the trial has the identification.

The operation
    Calibration folds the trial's answer into the registry model. axiom offers two
    routes -- through the prior, or through the likelihood -- and does not pretend
    they are the same operation. It reports how each fit then agrees with the trial,
    so the disagreement is visible rather than averaged away.

Pillars: calibrate (measurement, both routes, agreement)
"""

import numpy as np
from _walkthrough import Walkthrough

from axiom.calibrate import Measurement, agreement, derive_prior, fit_calibrated
from axiom.core import Intervention, Population, TimeWindow
from axiom.data import Panel
from axiom.estimands import Estimand, Level, Quantity, realize
from axiom.sim import DosePlan, surface_world
from axiom.surface import GeometricCarryover, HillKernel, fit, response_band

TRUTH = {"beta_dose": 8.0, "alpha": 4.0, "k_dose": 50.0, "s_dose": 2.0, "lam_dose": 0.4}
HIGH, NONE, MAX_LAG = 100.0, 0.0, 4

w = Walkthrough(
    field="Clinical research",
    title="Nearly right, and still too sure of itself",
    question="""A registry has the coverage and the confounding; a trial has the
        identification and no reach. Folding one into the other is the obvious move.
        The question is which of the two available ways to do it, and what each one
        leaves you believing.""",
)

# ----------------------------------------------------------------------------------

w.step(
    "Build the registry, including the severity nobody recorded",
    why="""Confounding by indication is the defining problem of registry evidence:
        sicker patients are dosed harder, so the raw association understates the
        benefit. Simulating it explicitly — with a severity term that enters both the
        dose and the outcome and is never handed to the model — is what makes the
        rest of this checkable, because the true amplitude is known.""",
    instead="""Simulating clean data and calling the result a demonstration of
        calibration. Calibration on unconfounded data does nothing interesting; the
        whole operation exists to correct a bias, so the bias has to be there.""",
)
registry = surface_world(
    n_units=6,
    n_periods=24,
    treatments=("dose",),
    kernels=HillKernel(reference_dose=50.0, amplitude_scale=10.0),
    carryover={"dose": GeometricCarryover(max_lag=MAX_LAG)},
    doses=DosePlan(scale=50.0, spread=0.8, zero_fraction=0.05),
    intercept="shared",
    truth=TRUTH,
    noise_sd=2.0,
    seed=0,
)
spec = registry.spec
rng = np.random.default_rng(11)
frame = registry.panel.frame
dose = frame["dose"].to_numpy(dtype=np.float64)
z_dose = (dose - dose.mean()) / dose.std()
severity = 0.45 * z_dose + 0.9 * rng.standard_normal(dose.size)
observed = Panel(frame.assign(y=frame["y"] + 1.0 * severity), registry.panel.roles)
w.out(f"registry : {registry.n_units} clinics x {registry.n_periods} periods")
w.out(f"true amplitude : {TRUTH['beta_dose']:.1f}")
w.out(f"correlation(dose, unmeasured severity) = {float(np.corrcoef(dose, severity)[0, 1]):.3f}")
w.out("")
w.out("The severity column exists in this script and is never passed to fit().")
w.out("That is the whole of the confounding: real, moderate, and invisible.")

# ----------------------------------------------------------------------------------

w.step(
    "Say precisely what the trial measured, as an object",
    why="""A trial result is not a number, it is a number attached to a question:
        which contrast, over which window, in which population, at which level. The
        Estimand records all of that and hashes it, so a measurement can only be
        folded into a model that answers the same question. The hash is what stops a
        first-period per-patient contrast being quietly used to constrain a
        cumulative population-level one.""",
    instead="""Carrying the trial as 'the treatment effect was 12.4'. That is the
        form in which evidence is normally transferred between studies, and it is
        the form in which the window, the population and the estimand silently
        change on the way.""",
)
trial_estimand = Estimand(
    name="first_period_response",
    quantity=Quantity(kind="contrast"),
    treatment=spec.treatment("dose"),
    intervention=Intervention(doses={"dose": HIGH}),
    reference=Intervention(doses={"dose": NONE}),
    outcome=spec.outcome,
    population=Population(name="trial_patients"),
    window=TimeWindow(start=0, stop=1, basis="cumulative"),
    level=Level(unit="individual"),
    dimension=spec.outcome_dimension,
)
diff = registry.forward({"dose": HIGH}) - registry.forward({"dose": NONE})
trial_truth = float(np.mean(diff[:, 0]))
se = 0.04 * trial_truth
measurement = Measurement(
    estimand=trial_estimand,
    estimate=trial_truth + se * float(np.random.default_rng(3).standard_normal()),
    se=se,
    method="randomized_contrast",
    n_units=280,
    n_periods=1,
    source="TRIAL-2026-A",
)
w.out(f"trial estimate : {measurement.estimate:.3f} +/- {measurement.se:.3f}")
w.out(f"interval       : {measurement.interval}")
w.out(f"n              : {measurement.n_units} patients, {measurement.n_periods} period")
w.out(f"target hash    : {measurement.target[:16]}...")
w.out(
    f"window         : periods {trial_estimand.window.start} to "
    f"{trial_estimand.window.stop}, {trial_estimand.window.basis}"
)

# ----------------------------------------------------------------------------------

w.step(
    "Fit the registry alone, and notice it is not merely uncertain",
    why="""The failure mode worth recognising is tight and wrong. If the
        uncalibrated fit were simply noisy, its interval would cover the truth and
        the trial would be a precision upgrade. It does not: the truth sits several
        posterior standard deviations away, which is the signature of confounding
        rather than sampling error, and no amount of registry data would fix it.""",
    instead="""Judging the fit by its convergence diagnostics and its posterior
        predictive check. Both pass here. A confounded model fits its own data
        beautifully — that is what makes confounding dangerous rather than
        obvious.""",
)


def amplitude(result):
    draws = result.posterior.flat("beta_dose")
    return float(draws.mean()), float(draws.std(ddof=1))


uncalibrated = fit(spec, observed, backend="laplace", draws=1000, seed=1)
mean0, sd0 = amplitude(uncalibrated)
ag0 = agreement(uncalibrated, measurement, seed=0)
w.out(f"converged      : {uncalibrated.converged}")
w.out(f"amplitude      : {mean0:.3f} +/- {sd0:.3f}")
w.out(f"truth          : {TRUTH['beta_dose']:.1f}")
w.out(f"drift vs truth : {(mean0 - TRUTH['beta_dose']) / sd0:+.1f} posterior sd")
w.out(f"agreement with the trial : z = {ag0.z:+.2f}, verdict {ag0.verdict}")

# ----------------------------------------------------------------------------------

w.step(
    "Route one: turn the trial into a prior, then refit",
    why="""derive_prior converts the trial's statement about a PREDICTION into a
        statement about a PARAMETER, using a design factor that carries the
        conversion. Every step of that conversion writes a ledger line, so the
        assumption that made it possible is recoverable later — this is the transfer
        that most often disappears into a footnote.""",
    instead="""Setting an informative prior by hand to match the trial. It produces a
        similar posterior and records nothing: six months later there is no way to
        tell which prior came from evidence and which came from an analyst's
        judgement about what the answer ought to be.""",
)
realized = realize(trial_estimand, uncalibrated, assume_identified=True, keep_draws=True)
calibrated = derive_prior(
    [measurement],
    spec,
    "dose",
    beta_draws=uncalibrated.posterior.flat("beta_dose"),
    contribution_draws=np.asarray(realized.draws, dtype=np.float64).reshape(-1),
)
prior_route = fit(calibrated.spec, observed, backend="laplace", draws=1000, seed=1)
w.out(f"design factor   : {calibrated.design_factor:.4f}")
w.out(f"parameter       : {calibrated.parameter}")
w.out(f"new prior       : {calibrated.prior}")
w.out("")
w.out("ledger:")
for line in calibrated.ledger_lines:
    w.out(f"  {line.kind}  [{line.assumption.name}]")

# ----------------------------------------------------------------------------------

w.step(
    "Route two: add the trial to the log density instead",
    why="""fit_calibrated puts the measurement into the objective as a constraint, so
        the trial competes with the registry data rather than preceding it. When the
        two disagree the posterior lands between them, weighted by their precisions,
        instead of the registry being fitted inside a prior the trial already
        fixed.""",
    instead="""Assuming this is the same operation as route one with the arithmetic
        rearranged. It is not, and the two do not agree — constraining a parameter
        and constraining a prediction are different requests. Which you want depends
        on what the model is for.""",
)
likelihood_route = fit_calibrated(
    spec, observed, [measurement], backend="laplace", draws=1000, seed=1
)
mean2, sd2 = amplitude(likelihood_route)
w.out(f"amplitude : {mean2:.3f} +/- {sd2:.3f}")

# ----------------------------------------------------------------------------------

w.step(
    "Compare all three on the two things that matter",
    why="""Distance from the truth and agreement with the trial are different
        questions and a fit can pass one while failing the other. Reporting the drift
        in posterior standard deviations rather than in raw units is what exposes
        that: a fit can land closest in absolute terms and still be the worst on this
        column, because it left itself too little uncertainty to be wrong in.""",
    instead="""Picking a winner. axiom reports the gap between the two routes rather
        than choosing, because the choice depends on what the model will be used
        for — a formulary decision about this dose wants the constrained prediction,
        a model that will be extrapolated to other doses wants the constrained
        parameter.""",
)
fits = (
    ("uncalibrated", uncalibrated),
    ("prior route", prior_route),
    ("likelihood route", likelihood_route),
)
rows, drift_rows, agree_rows = [], [], []
for label, result in fits:
    mean, sd = amplitude(result)
    ag = agreement(result, measurement, seed=0)
    drift = (mean - TRUTH["beta_dose"]) / sd
    w.out(
        f"{label:<18}{mean:>9.3f}{sd:>8.3f}  drift {drift:+6.1f} sd   "
        f"z vs trial {ag.z:+.2f}  {ag.verdict}"
    )
    rows.append(
        {
            "label": label,
            "estimate": mean,
            "lower": mean - 1.96 * sd,
            "upper": mean + 1.96 * sd,
            "note": f"posterior sd {sd:.3f}, drift {drift:+.1f} sd",
            "bad": abs(drift) > 2,
        }
    )
    drift_rows.append(
        {
            "label": label,
            "value": abs(drift),
            "display": f"{abs(drift):.1f} sd",
            "colour": "boundary" if abs(drift) > 2 else "accent",
            "note": f"amplitude {mean:.2f} against a truth of {TRUTH['beta_dose']:.1f}",
        }
    )
    agree_rows.append(
        {
            "label": label,
            "value": abs(float(ag.z)),
            "display": f"{ag.z:+.2f}",
            "colour": "boundary" if ag.verdict != "agrees" else "accent",
            "note": f"verdict: {ag.verdict}",
        }
    )
w.figure(
    "amplitude",
    kind="intervals",
    data={"rows": rows},
    opt={
        "rows": "@rows",
        "truth": TRUTH["beta_dose"],
        "truthLabel": "true amplitude",
        "xLabel": "dose amplitude (beta_dose)",
        "labelWidth": 145,
        "rowHeight": 38,
    },
    title="What each fit believes about the amplitude",
    note="""Worth reading carefully: none of the three intervals contains the true
        amplitude. Calibration bought distance, not coverage — the uncalibrated centre
        is furthest out and both routes move a long way towards the truth without
        arriving. And the two routes do not contain each other's centre either, which
        is the difference between constraining a parameter and constraining a
        prediction showing up as a number.""",
)
w.figure(
    "drift",
    kind="bars",
    data={"rows": drift_rows},
    opt={
        "rows": "@rows",
        "xLabel": "distance from the truth, in posterior standard deviations",
        "labelWidth": 145,
        "rowHeight": 40,
    },
    title="Being close is not the same as being calibrated",
    note="""This is the column that catches overconfidence. A fit that lands nearest
        in raw units can still sit furthest away when measured in its own
        uncertainty, and that is a failure a recovery test exists to find.""",
    legend=(("accent", "within 2 sd of the truth"), ("boundary", "further than that")),
)
w.figure(
    "agreement",
    kind="bars",
    data={"rows": agree_rows},
    opt={"rows": "@rows", "xLabel": "|z| against the trial", "labelWidth": 145, "rowHeight": 40},
    title="And whether each fit still argues with the trial",
    note="""The uncalibrated registry openly disagrees with the trial. Both routes
        reconcile with it — which is the operation working, and is separate from
        whether either of them is right about the truth.""",
)

doses = np.linspace(0.0, 120.0, 25)
truth_curve = [float(registry.forward({"dose": float(d)}).mean()) for d in doses]
curves = {
    key: list(response_band(result, "dose", doses=doses.tolist(), mass=0.9, seed=0).mean)
    for key, (_, result) in zip(("uncal", "prior", "likelihood"), fits, strict=True)
}
w.figure(
    "curves",
    kind="lines",
    data={"doses": doses.tolist(), "truth": truth_curve, **curves},
    opt={
        "series": [
            {"label": "the truth", "x": "@doses", "y": "@truth", "colour": "ink-2"},
            {"label": "uncalibrated", "x": "@doses", "y": "@uncal", "colour": "boundary"},
            {"label": "prior route", "x": "@doses", "y": "@prior", "colour": "s3"},
            {"label": "likelihood", "x": "@doses", "y": "@likelihood", "colour": "accent"},
        ],
        "xLabel": "dose",
        "yLabel": "expected outcome",
        "height": 320,
        "rightPad": 110,
    },
    title="The whole dose-response, before and after the trial is folded in",
    note="""Calibration is not a correction applied to one number. It moves the entire
        curve, including at doses the trial never tested — which is exactly why the
        estimand the trial measured had to be pinned down before any of this
        began.""",
)

# ----------------------------------------------------------------------------------

mean1, sd1 = amplitude(prior_route)
w.finding(f"""The uncalibrated registry was not merely uncertain, it was confidently wrong:
    amplitude {mean0:.2f} against a truth of {TRUTH['beta_dose']:.1f}, which is
    {abs(mean0 - TRUTH['beta_dose']) / sd0:.1f} posterior standard deviations out,
    and it openly disagreed with the trial (z = {ag0.z:+.2f}). Tight and wrong is the
    signature of confounding rather than noise.""")
w.finding(f"""Both routes cut most of that bias and both then reconcile with the trial:
    {mean1:.2f} and {mean2:.2f} against an uncalibrated {mean0:.2f}. Neither reaches
    the truth, and neither interval covers it. One trial of 280 patients does not
    undo confounding in a registry — it bounds it, and that is a smaller claim than
    the word 'calibrated' invites.""")
w.finding(f"""They also disagree with each other, by {abs(mean1 - mean2):.2f}. That
    disagreement is the useful part rather than a defect to be averaged away: it is
    the size of the difference between the two questions being asked, and it is
    reported rather than resolved on your behalf.""")
w.finding(f"""The prior route lands CLOSEST in absolute terms and still scores worst on
    drift, because the derived prior inherited the trial's precision and left the
    posterior very tight: {sd1:.3f} against {sd2:.3f}. A calibrated model can be
    nearly right and still overconfident. That is exactly the failure a recovery test
    against a known truth exists to catch, and the reason the design factor and the
    moment-matching wrote themselves into the ledger rather than disappearing into
    the fit.""")
