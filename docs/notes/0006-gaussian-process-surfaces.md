# 0006 — Gaussian-process surfaces, and the recovery world that was never identified

**Kind**: decision + deviation. **Opened**: 2026-08-21. **Status**: landed on `develop`;
part of the 1.1.0 surface.

Two things, and they are the same thing twice: a design that does not pin down what it is
trying to measure produces numbers that look fine and are not.

## Part 1 — `test_nuts_interval_coverage`, chased down

`tests/recovery/test_surface_recovery.py::test_nuts_interval_coverage` asserts that a 90 %
HDI for `beta_a` covers the truth at the nominal rate over 60 seeded worlds. It was
failing at **29/60** against an exact acceptance region of `[45, 60]`, reproduced with an
identical hit count on a clean `develop` worktree — deterministic, not flaky.

**It was not the sampler.** The evidence, in the order it was collected:

| check | result |
|---|---|
| posterior mean over 10 seeds | 12.4 against a truth of 10.0 — *bias*, not underdispersion |
| `converged` | `False` everywhere (ESS 139–343 against a 400 threshold), but `rhat ≤ 1.03`, 0 divergences |
| 4× the sampling (2000 tune, 2000 draws, 4 chains) | `rhat` 1.002, ESS ≈ 2000, 0 divergences — **same bias**, 1/5 covered |
| the same 10 worlds under **Laplace** | 4/10 covered, same upward bias — and Laplace has no jax in it |
| `world.mean == forward(truth)` | exactly equal; the simulator and the model evaluate the same tree |
| profile log-likelihood in `beta_a`, others at truth | peaks **exactly** at 10.0, and sharply (Δ25 at 10.5) |

So both backends agree, they agree with each other on the wrong answer, the generative and
fitted models are the same tree, and conditionally on the other parameters the truth is
recovered to the decimal. The problem is the **joint geometry**, and `axiom.design` already
ships the instrument for it. `identifiability_ridge` on the world as written:

```
condition number 847,  corr(alpha, beta_a) = -0.93,  corr(beta_a, s_a) = -0.97
```

The intercept, the amplitude and the Hill shape are nearly collinear. The fitted curve is
within 1.4 mmHg of the truth across the observed dose range and the biggest gap is *at
dose zero* — with `zero_fraction=0.05` almost nothing pins the intercept, so `alpha` slides
down while `beta_a` slides up and `s_a` flattens to compensate.

**Fixed-truth coverage is a frequentist property and a nearly singular design does not
have it.** Nothing is wrong with the posterior; it is an honest posterior over a ridge.

### The fix is a design change, which is the point

Widening the dose distribution and putting a fifth of the rows at zero dose
(`spread=0.8 → 1.3`, `zero_fraction=0.05 → 0.20`) is the whole change. Measured, at
`n = 24` worlds under Laplace:

| dose plan | rows | condition | `corr(beta, s)` | coverage of alpha / beta / k / s / lam |
|---|---|---|---|---|
| as written | 160 | 847 | −0.967 | 10, 11, 17, 11, 20 |
| **widened** | 160 | **119** | −0.891 | **22, 21, 22, 22, 21** |
| bigger panel, narrow doses | 720 | 440 | −0.949 | 22, 20, 22, 21, 20 |
| bigger *and* widened | 720 | 79 | −0.861 | 22, 24, 19, 23, 20 |

Four and a half times the data barely moves the condition number; changing where the doses
go moves it by a factor of seven. The NUTS tier now passes at `n = 60`.

`test_the_recovery_world_is_identified` guards the decision: it asserts the condition
number stays under 200 and no pair exceeds 0.95, so narrowing the dose plan fails *that*
test with a diagnosis rather than failing the coverage test with a mystery.

### What this says about the repository

The Phase 3 record called this tier green and the code is unchanged since, so either the
dependency set moved under it or the original run was not what it was recorded as. Either
way the useful conclusion is not about that: a recovery test on an unidentified world is a
test that can only be passed by luck, and the repository had one for nine phases without
noticing. Recovery worlds should carry an identifiability assertion; this one now does.

## Part 2 — `GaussianProcessKernel`

Notes [0004](0004-case-study-hypertension.md) and [0005](0005-basis-response-families.md)
walked the same path once already: the case study needed a shape the library could not
express, and the answer was a more flexible family. A spline is more flexible than a Hill
curve and still asks the analyst to place knots. The GP asks for less: stationarity, and a
smoothness scale it will estimate.

### 0006.1 — It is the Hilbert-space approximation, and the docstring says so

An exact GP needs a multivariate normal over the latent function. `core.Likelihood`
evaluates elementwise families — normal, lognormal, Student-t, Poisson — so an exact GP is
not expressible, and pretending otherwise would be the kind of quiet substitution this
repository exists to avoid. What ships is the reduced-rank construction (Solin & Särkkä
2020; Riutort-Mayol et al. 2023): on a bounded interval the Laplacian eigenfunctions are a
fixed sine basis, and a stationary GP is recovered by giving basis `j` the prior standard
deviation `sqrt(S(w_j))`, with `S` the covariance's spectral density.

```
f(u) = beta · Σ_j sqrt(S(w_j; ell)) · z_j · [ phi_j(u) − phi_j(0) ]
phi_j(u) = sin(w_j (u + L)) / sqrt(L),   w_j = pi j / (2 L),   z_j ~ N(0, 1)
```

Written non-centred, because that is the version that samples. Subtracting `phi_j(0)`
keeps `response(0) = 0` like every other family, so the surface's intercept still means
the response at zero dose instead of competing with the GP for it — the exact confound
Part 1 is about.

Three covariances ship: squared exponential, Matérn 3/2, Matérn 5/2.

### 0006.2 — One amplitude, so the classic invariant survives

Unlike the basis families of note 0005, this declares exactly **one** amplitude. The sign
of the response lives in the `z` coefficients, so `beta` is a positive GP marginal scale
and `response == beta · saturation` holds node for node. `calibrate`'s prior route works on
it unchanged, and no part of `surface.forward` needed a second exception.

The lengthscale and the coefficients take the `shape` role — dimensionless, and
`parameter_roles` classifies them `nonlinear`, which is correct: `beta · z_j` is bilinear
and the response is genuinely nonlinear in `ell`. A GP is not a linear model and the roles
should not claim it is.

### 0006.3 — `sin` and `cos` in `core.ApplyFn`

Two more entries, four touch points each (the `Literal`, the numpy interpreter, the jax
interpreter, the latex renderer), exactly as `relu` and `step` in note 0005. The basis is
a sine and its derivative is a cosine; there was no way around needing both.

### 0006.4 — The approximation is measured, not asserted

`n_basis` and `boundary_factor` decide whether the thing is a GP at all, and getting them
wrong does not raise — it fits a different prior than the one you asked for. So the class
ships the measurement:

- `implied_covariance(ell, u)` — the covariance the basis actually implies,
  `Σ_j S(w_j) phi_j(u) phi_j(u')`.
- `exact_covariance(ell, u)` — the stationary covariance it is approximating.
- `covariance_error(ell)` — the worst absolute gap between them, on a unit scale.
- `sufficient_for(ell, tolerance=0.02)` — the same thing as a yes or no.

The defaults were **chosen by running that sweep**, not from a remembered rule of thumb:

| `n_basis` | `boundary_factor` | usable lengthscale band (error ≤ 0.02) |
|---|---|---|
| 16 | 3.0 | 0.13 – 0.70 |
| **24** | **3.0** | **0.10 – 0.70** |
| 32 | 4.0 | 0.13 – 1.00 |
| 96 | 8.0 | 0.03 – 2.00 |

`n_basis` sets the *short* lengthscale limit and `boundary_factor` the *long* one, and they
trade — raising the boundary costs basis functions to hold the short end. The shipped
defaults (24, 3.0) cover 0.10–0.70 on the `u` scale, which is the 90 % interval of the
default lengthscale prior. Outside that band the approximation, not the data, limits the
fit, and `covariance_error` says so.

### 0006.5 — What it recovers

`tests/recovery/test_surface_recovery.py` gains a world whose truth is outside every
parametric family the library ships — a rise, a turn, and a slow drift. At 300 units with
noise 0.8 the GP recovers the noise scale it was given (0.79), tracks the curve to 6.7 %
of its span, and settles on a lengthscale the basis can represent. Fitted with a Hill
kernel instead, the same data give `sigma = 1.43`: the misfit has nowhere else to go.

In the case study's oldest age band
(`nbs/case-studies/hypertension/04-the-dose-response.ipynb` §3b, §6):

| family | `sigma` | worst arm-mean residual | slope at 40 mg | sign change |
|---|---|---|---|---|
| truth | 0.5 | — | −0.835 | 16 mg |
| `hill` | 7.37 | 7.95 | 0.000 | **never (monotone)** |
| `polynomial(3)` | 6.04 | 3.16 | −1.109 | 18 mg |
| `spline`, 3 knots | 6.21 | 4.54 | −0.442 | 18 mg |
| `piecewise_linear` | 6.02 | 3.13 | −0.619 | 16 mg |
| `gaussian_process` | 5.90 | **1.70** | −0.027 | 13 mg |

The marginal effect is the decisive column, not the residual. A monotone family's slope
cannot change sign, so at 40 mg the Hill fit reports zero in the band where the drug is
raising pressure — not a bad fit, an inexpressible one.

The GP wins on residual and finds the sign change earliest, and its slope at the *top* of
the range is attenuated (−0.03 against a true −0.84) because a stationary GP reverts
toward its prior at the edge of the data. That is the price of not committing to a shape,
and the notebook says it rather than showing the winning row and stopping.

## Left open

- **Edge behaviour.** The GP's attenuation at the boundary of the dose range is real and
  currently only documented. A linear mean function, or a boundary-aware prior, would fix
  it; neither is written.
- **Multi-treatment GPs.** One GP per treatment composes through the existing interaction
  machinery, but a genuine 2-D GP over a dose *pair* would need a tensor-product basis and
  is not there.
- **Choosing `n_basis` automatically** from the lengthscale prior. `sufficient_for` makes
  it checkable but nothing solves for it.
- **SBC for the GP.** The recovery test is fixed-truth; a rank test over prior draws would
  be the stronger check, and per Part 1 it should carry an identifiability assertion.
