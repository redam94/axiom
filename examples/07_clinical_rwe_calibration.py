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

from axiom.calibrate import Measurement, agreement, derive_prior, fit_calibrated
from axiom.core import Intervention, Population, TimeWindow
from axiom.data import Panel
from axiom.estimands import Estimand, Level, Quantity, realize
from axiom.sim import DosePlan, surface_world
from axiom.surface import GeometricCarryover, HillKernel, fit

TRUTH = {"beta_dose": 8.0, "alpha": 4.0, "k_dose": 50.0, "s_dose": 2.0, "lam_dose": 0.4}
HIGH, NONE, MAX_LAG = 100.0, 0.0, 4

# The registry: a panel of clinics over time, with a saturating dose-response and
# carryover, and an unmeasured severity that drives both dosing and outcome.
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
# Confounding-by-indication of a realistic strength: sicker patients are dosed
# harder, but severity is far from a deterministic function of dose.
severity = 0.45 * z_dose + 0.9 * rng.standard_normal(dose.size)
observed = Panel(frame.assign(y=frame["y"] + 1.0 * severity), registry.panel.roles)
print(f"registry: {registry.n_units} clinics x {registry.n_periods} periods")
print(
    f"  correlation(dose, unmeasured severity) = " f"{float(np.corrcoef(dose, severity)[0, 1]):.3f}"
)

# The trial measured one specific thing: a first-period contrast, per patient.
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
print(f"\ntrial: {measurement.estimate:.3f} +/- {measurement.se:.3f}  " f"{measurement.interval}")
print(f"  it is an estimate OF a specific estimand, hashed: {measurement.target[:16]}...")


def amplitude(result):
    draws = result.posterior.flat("beta_dose")
    return float(draws.mean()), float(draws.std(ddof=1))


# Route zero: don't calibrate at all.
uncalibrated = fit(spec, observed, backend="laplace", draws=1000, seed=1)

# Route one: convert the trial into a prior on the amplitude, then refit.
realized = realize(trial_estimand, uncalibrated, assume_identified=True, keep_draws=True)
calibrated = derive_prior(
    [measurement],
    spec,
    "dose",
    beta_draws=uncalibrated.posterior.flat("beta_dose"),
    contribution_draws=np.asarray(realized.draws, dtype=np.float64).reshape(-1),
)
prior_route = fit(calibrated.spec, observed, backend="laplace", draws=1000, seed=1)

# Route two: add the trial to the log density as a constraint, so it competes
# with the registry data rather than preceding it.
likelihood_route = fit_calibrated(
    spec, observed, [measurement], backend="laplace", draws=1000, seed=1
)

print(f"\n{'fit':<20}{'amplitude':>12}{'sd':>8}{'vs truth':>11}{'z vs trial':>13}  verdict")
for label, result in (
    ("uncalibrated", uncalibrated),
    ("prior route", prior_route),
    ("likelihood route", likelihood_route),
):
    mean, sd = amplitude(result)
    ag = agreement(result, measurement, seed=0)
    drift = (mean - TRUTH["beta_dose"]) / sd
    print(f"{label:<20}{mean:>12.3f}{sd:>8.3f}{drift:>+11.1f}{ag.z:>+13.2f}  {ag.verdict}")

print(f"\ntrue amplitude: {TRUTH['beta_dose']:.1f}")
print(f"design factor used by the prior route: {calibrated.design_factor:.4f}")
print(f"new prior on {calibrated.parameter}: {calibrated.prior}")
for line in calibrated.ledger_lines:
    print(f"  ledger: {line.kind}  [{line.assumption.name}]")

print("""
Reading it
    The uncalibrated registry is not merely uncertain, it is confidently wrong: the
    truth sits nearly three posterior standard deviations from its mean, and it
    openly disagrees with the trial. That combination -- tight and wrong -- is the
    signature of confounding rather than noise, and it is what the agreement test
    is for.

    Both routes cut most of that bias and both then reconcile with the trial. They
    do not agree with each other, and that is the useful part: constraining a
    PARAMETER and constraining a PREDICTION are different requests. Which you want
    depends on what the model is for, and axiom reports the gap rather than
    choosing on your behalf.

    Note the 'vs truth' column is in posterior standard deviations. The prior route
    lands CLOSEST in absolute terms and still scores worst on that column, because
    the derived prior inherited the trial's precision and left the posterior very
    tight. A calibrated model can be nearly right and still overconfident, which is
    exactly the failure a recovery test on a synthetic world is there to catch --
    and the reason the design factor and the moment-matching both wrote themselves
    into the ledger above rather than disappearing into the fit.""")
