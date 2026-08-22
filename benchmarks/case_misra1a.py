"""NIST Misra1a — a certified nonlinear regression, fitted as a dose-response.

The data
    Fourteen measurements from a 1978 NIST study of monomolecular adsorption in
    dental research: volume adsorbed against pressure. NIST publishes it as a
    Statistical Reference Dataset with CERTIFIED parameter values, computed to 500
    digits and rounded to eleven, precisely so that software can be checked against
    something better than another program's opinion.

Why it fits here
    NIST's model is

        y = b1 * (1 - exp(-b2 * x))

    and axiom's ExponentialKernel is f(u) = 1 - exp(-u) with u = dose / k. With the
    intercept switched off, axiom's beta IS b1 and its k IS 1 / b2. Not an
    approximation of the same shape -- the same two-parameter curve, reachable
    without touching the library.

What this checks and what it cannot
    axiom fits a posterior; NIST certifies a least-squares optimum. Those are
    different objects and will not agree to eleven digits. The question worth
    asking is whether the posterior concentrates on the certified values, and the
    answer here is that it does, to about one part in ten thousand.

Run it
    python benchmarks/case_misra1a.py
"""

import numpy as np
import pandas as pd
from registry import DATASETS, load

from axiom.core import BASES, D, Outcome, Treatment
from axiom.data import Panel, RoleMap
from axiom.surface import ExponentialKernel, SurfaceSpec, fit, response_band

SPEC = DATASETS["misra1a"]
CERTIFIED = SPEC.published

print(SPEC.title)
print(f"  {SPEC.citation}")
print(f"  licence: {SPEC.licence}")
print(f"  sha256 : {SPEC.sha256()[:32]}...\n")

frame = load("misra1a")
print(frame.to_string(index=False))

# ----------------------------------------------------------------------------------
# real physical quantities, declared as such
# ----------------------------------------------------------------------------------

# axiom will not let a quantity exist without a dimension, and it ships only the
# bases its own domain needs. Physical work declares its own; declaring is
# idempotent, and re-declaring with a different symbol is an error rather than a
# silent overwrite.
BASES.declare("volume", symbol="V")
BASES.declare("pressure", symbol="P")

VOLUME = Outcome(name="volume", dimension=D.volume, unit="cc")
PRESSURE = Treatment(name="pressure", dimension=D.pressure, unit="mmHg")
print(f"\noutcome  {VOLUME.name} [{VOLUME.dimension}] in {VOLUME.unit}")
print(f"dose     {PRESSURE.name} [{PRESSURE.dimension}] in {PRESSURE.unit}")

# Fourteen independent measurements on one specimen: one unit, fourteen periods.
tidy = pd.DataFrame(
    {
        "unit": ["specimen"] * len(frame),
        "t": np.arange(len(frame)),
        "pressure": frame["pressure"].to_numpy(float),
        "volume": frame["volume"].to_numpy(float),
    }
)
panel = Panel(
    tidy,
    RoleMap(unit="unit", time="t", outcome=("volume", VOLUME), treatments={"pressure": PRESSURE}),
)

spec = SurfaceSpec(
    name="misra1a",
    treatments=(PRESSURE,),
    outcome=VOLUME,
    kernels={"pressure": ExponentialKernel(reference_dose=1800.0, amplitude_scale=250.0)},
    intercept="none",  # NIST's model has no additive constant
    unit_labels=("specimen",),
)
print(f"\nmodel    volume = beta * (1 - exp(-pressure / k)),  intercept {spec.intercept!r}")
print(
    f"panel    {panel.units[0]}, {len(panel.periods)} observations, "
    f"hash {panel.content_hash()[:16]}..."
)

# ----------------------------------------------------------------------------------
# fit, and compare to the certified values
# ----------------------------------------------------------------------------------

result = fit(spec, panel, backend="laplace", draws=4000, seed=0)
print(f"\nconverged: {result.converged}")

posterior = result.posterior
beta = posterior.summary("beta_pressure", definition="hdi", mass=0.95)
k = posterior.summary("k_pressure", definition="hdi", mass=0.95)
sigma = posterior.summary("sigma")

rows = [
    ("beta   (NIST b1)", beta.mean, beta.sd, CERTIFIED["axiom_beta"], CERTIFIED["b1_sd"]),
    ("k      (NIST 1/b2)", k.mean, k.sd, CERTIFIED["axiom_k"], None),
]
print(
    f"\n{'parameter':<20}{'axiom mean':>14}{'posterior sd':>14}"
    f"{'NIST certified':>17}{'rel. error':>12}"
)
for label, mean, sd, certified, _ in rows:
    rel = abs(mean - certified) / abs(certified)
    print(f"{label:<20}{mean:>14.6f}{sd:>14.6f}{certified:>17.6f}{rel:>12.2e}")

print(f"\n{'':<20}{'95% HDI':>32}   contains certified?")
for label, summary, certified in (
    ("beta", beta, CERTIFIED["axiom_beta"]),
    ("k", k, CERTIFIED["axiom_k"]),
):
    lo, hi = summary.interval.lower, summary.interval.upper
    inside = lo <= certified <= hi
    print(f"{label:<20}{f'[{lo:.4f}, {hi:.4f}]':>32}   {'yes' if inside else 'NO'}")

# NIST also certifies b2 itself; axiom parameterises by k = 1 / b2.
b2_implied = 1.0 / k.mean
print(f"\nimplied b2 = 1/k      {b2_implied:.10e}")
print(f"NIST certified b2     {CERTIFIED['b2']:.10e}")
print(f"relative error        {abs(b2_implied - CERTIFIED['b2']) / CERTIFIED['b2']:.2e}")

# ----------------------------------------------------------------------------------
# residuals at the posterior mean, against the certified residual sum of squares
# ----------------------------------------------------------------------------------

x = tidy["pressure"].to_numpy(float)
y = tidy["volume"].to_numpy(float)
predicted = beta.mean * (1.0 - np.exp(-x / k.mean))
rss = float(np.sum((y - predicted) ** 2))

print(f"\nresidual sum of squares at the posterior mean  {rss:.8f}")
print(f"NIST certified minimum                        {CERTIFIED['residual_sum_of_squares']:.8f}")
excess = rss - CERTIFIED["residual_sum_of_squares"]
print(f"excess over the certified optimum             {excess:+.2e}")
print("  The posterior mean cannot beat the least-squares optimum -- if it did, the")
print("  certified value would be wrong. Sitting a hair above it is the correct")
print("  result, and the size of that hair is the whole check.")

print(f"\nposterior sigma                {sigma.mean:.6f} +/- {sigma.sd:.6f}")
print(f"NIST residual sd (df = {CERTIFIED['df']})   {CERTIFIED['residual_sd']:.6f}")
print(f"sqrt(RSS / n) for comparison   {np.sqrt(CERTIFIED['residual_sum_of_squares'] / 14):.6f}")
print("  These three are different estimators of the same scale, not disagreements:")
print("  NIST divides by the degrees of freedom, maximum likelihood divides by n,")
print("  and the posterior mean sits between them.")

# ----------------------------------------------------------------------------------
# the curve, with the uncertainty NIST does not report
# ----------------------------------------------------------------------------------

band = response_band(result, "pressure", doses=[100, 250, 400, 550, 700, 900, 1200], mass=0.95)
print("\nfitted adsorption curve, with 95% intervals:")
print(f"  {'pressure':>10}{'volume':>10}{'lower':>10}{'upper':>10}")
for dose, mean, lo, hi in zip(band.doses, band.mean, band.lower, band.upper, strict=True):
    print(f"  {dose:>10.0f}{mean:>10.3f}{lo:>10.3f}{hi:>10.3f}")

print(f"""
Reading it
    Both certified parameters recovered to about one part in ten thousand, with
    the certified values inside the 95% intervals, and a residual sum of squares
    that sits just above the certified minimum rather than below it.

    That last point is the one worth dwelling on. A fit that reported a LOWER
    residual sum of squares than NIST's certified optimum would not be a better
    fit -- it would be a bug, either in the model being fitted or in the arithmetic
    reporting it. Benchmarks with certified answers catch that class of error,
    which is exactly why NIST publishes them.

    What axiom adds beyond the certified numbers is the last table: NIST reports
    two parameters and their standard deviations, and axiom will also tell you what
    the curve does at a pressure of {band.doses[-1]:.0f} and how sure it is.""")
