# 0044 — One diagonal matrix

*Kind: decision. Opened 2026-08-30. Status: implemented on
`feature/exponential-families`. Extends the design math past the Gaussian it
was written against, and closes the gap where a model could be fitted but not
designed for.*

Before this, `axiom.core` fitted four likelihood families and `axiom.design`
computed information for one. You could fit a Poisson model, simulate a Poisson
outcome, and then discover that `design.identifiability` returned `Unsupported`
and `design.structural` had a scalar `noise_sd` in its signature. The design
layer was the only one still assuming a Gaussian.

## D44.1 — the link function is not in the formula

For any exponential family, with mean `mu_i`, variance function `V`, and
dispersion `phi`, the score is

```
dl/dtheta = sum_i (y_i - mu_i) / (phi V(mu_i)) . dmu_i/dtheta
```

and since `Var[y_i] = phi V(mu_i)`,

```
FI = sum_i (dmu_i/dtheta)(dmu_i/dtheta)' / (phi V(mu_i))  =  J' W J
```

with `w_i = 1 / (phi V(mu_i))`. One diagonal weight per row is the entire
difference between a Gaussian design and a Poisson one.

The reason this is cheap *here* rather than merely standard is the shape of
`J`. axiom differentiates the **mean** with respect to `theta`, not a linear
predictor with respect to a coefficient vector, so any link the mean expression
contains is already inside `J` — which `surface.linearize` and `jax.jacfwd`
compute exactly, under rule 3. **No inverse links, no deviance, no IRLS.** The
usual GLM apparatus exists to get from a linear predictor to a mean; axiom
starts at the mean.

Nothing downstream of the matrix changed. D-optimality, the ridge, expected
posterior sd and point exchange all read `J' W J` without knowing which family
produced it, because `FI = J'J / noise_sd²` is `V = 1, phi = noise_sd²`.

`core.variance_weight` is the one place a family's variance is written down.
`core.information_weight` (a fitted model) and `design.Weighting.at` (a
candidate design, where there is no fitted model to read a mean off) both go
through it.

| family | `phi · V(mu)` | `w` |
|---|---|---|
| `normal` | `sigma²` | `1 / sigma²` |
| `student_t` | — | `(df+1) / ((df+3) sigma²)` |
| `lognormal` | `(sigma · mu)²` | `1 / (sigma · mu)²` |
| `gamma` | `(cv · mu)²` | `1 / (cv · mu)²` |
| `poisson` | `mu` | `1 / mu` |
| `binomial` | `mu(1-mu) / n` | `n / (mu(1-mu))` |

Two of those rows are worth a sentence each. **`student_t` is not an
exponential family**, but its location information is a constant multiple of
the Gaussian one — the score is bounded, which is what buys the robustness — so
it fits the same shape, and tends to `1/sigma²` as `df` grows. That limit is
the check to make when touching this code. **`lognormal` and `gamma` share a
weight**, because both put a constant coefficient of variation on a positive
mean; they differ in their density, not in this geometry.

The whole claim is tested against the definition rather than against itself:
`test_the_weight_is_the_variance_of_the_score` simulates 20 000 datasets per
family and checks `sum_i w_i (dmu_i/da)²` against `Var[score]` to Monte-Carlo
tolerance.

## D44.2 — a weighting is a rule, not a vector

`Weighting` is a `Spec` holding a family, a dispersion, a `df` and a trials
column name — no arrays. That is deliberate. The weights are a function of the
mean, so they are different at every candidate design a point-exchange search
visits; what is constant, and what a result should carry as provenance under
rule 4, is the *rule* for computing them. `FisherInformation` and
`IdentifyingDesign` each record one, and `FisherInformation.__add__` refuses to
add across two different ones: information adds over independent rows, but only
rows in the same units of information.

`noise_sd` keeps its exact meaning — `FI = J' W J / noise_sd²`, with `W = I`
by default — so every existing call is unchanged and every existing number is
bit-identical. Passing both a `weighting` and a `noise_sd` other than 1 would
apply a dispersion twice, and raises.

## D44.3 — where the weight must not go

In `design.identifiability` the sensitivity matrix used to be built as
`observation.evaluate(theta) / noise_sd`, dividing *before* forming the central
difference. With a constant `noise_sd` that is the same as dividing after. With
a weight that depends on the mean it is not:

```
d(sqrt(w) mu)/dtheta  =  sqrt(w) dmu/dtheta  +  mu d sqrt(w)/dtheta
```

and the second term is not part of the Fisher information. The differences are
now formed unweighted and the rows scaled by `sqrt(w)` afterwards
(`Observation.root_weights`). The round-off test that decides whether a column
is exactly zero is invariant to the scaling — both the difference and the floor
carry the same factor — so every existing verdict is unchanged, which the
existing tests confirm.

`design.structural` never had this problem: it built `J` unweighted and applied
`/ noise_sd²` to the finished matrix.

## D44.4 — the gap a weight cannot close

`power.difference_se` is `sd / sqrt(n · a · (1 - a))`: one outcome standard
deviation, shared by both arms. That is not a rescalable Gaussian assumption —
it is a formula that has already collapsed both arms into a single `sd`, and no
per-row weight reaches inside it. A binary outcome's variance is a function of
its own mean, so the arms differ unless `p_control == p_treated`, and if they
were equal there would be no effect to power for.

The error therefore grows with the effect being powered for, which is the wrong
direction for an error to run. `proportion_difference_se` is the correct
`sqrt(p_t(1-p_t)/n_t + p_c(1-p_c)/n_c)`, and feeds `power_from_se` like any
other design standard error. At 0.50 vs 0.52 the shortcut is harmless; at 0.20
vs 0.60 it overstates the SE by 10%.

That was only half the fix, and the other half is the half that matters when
planning. `mde` and `sample_size` do not take a standard error — they take an
`sd` and *build* one, and `mde` in particular inverts a **fixed** standard
error. That is exactly right when the variance does not depend on the mean.
For a proportion the treated arm's variance moves as the effect grows, so the
standard error has to be solved for jointly with the effect, and freezing it at
the null is optimistic wherever `p_control < 0.5`:

| base rate | n | frozen-SE MDE | power it really has | correct MDE |
|---|---|---|---|---|
| 0.05 | 2000 | 0.0273 | 0.707 | 0.0309 |
| 0.10 | 400 | 0.0840 | 0.679 | 0.0989 |
| 0.20 | 400 | 0.1121 | 0.735 | 0.1218 |
| 0.50 | 400 | 0.1401 | 0.815 | 0.1374 |

At a 10% base rate you are told you can detect an 8.4-point lift at 80% power
and you actually have 68%. The bottom row is why the mistake survives: at
`p = 0.5` the error reverses and the Gaussian answer is mildly *conservative*,
so it only bites where the base rate is low — which is where most conversion
studies live.

`proportion_mde` root-finds on the real thing, and takes a `direction`, because
a fall and a rise of the same size are not equally detectable: they land on
different variances (from 0.20 at n = 400, a fall of 0.099 is as detectable as
a rise of 0.122). `proportion_sample_size` takes both proportions, so its
variance is fixed and the standard error falls exactly as `n^-1/2` — that one
is an ordinary integer search. `proportion_power` completes the trio.

`difference_se`, `mde` and `sample_size` keep their behaviour and now name the
proportion function in their docstrings. Nothing can detect a binary outcome
from a float, so a signpost is the whole of what is available.

## D44.5 — a Poisson likelihood could not be sampled at all

Fixed on the way past, and worth recording because of how it hid.
`interpret.pytensor`'s scale closure was `theta[lik.scale or ""]`, which raises
`KeyError('')` for any family without a scale. It is evaluated eagerly at every
call of the compiled density — but `compile_log_density` returns the closure
without calling it, so **building** a Poisson model succeeded and only
**sampling** one failed. No test called it, so the bug sat behind a green
suite. The closure is now family-aware (it returns the trial counts for
`binomial`, which is what that family's second argument is).

## What this does not do

* **`surface_world` cannot simulate a binomial outcome.** Its panel is
  `mean + noise` on one outcome column, and a binomial needs a second column of
  trial counts that a success probability is not measured in. It raises, saying
  so and naming the two ways round it. Binomial is fully supported for fitting
  and for design; only the convenience generator refuses it.
* **`simulated_identifiability` draws `normal`, `poisson`, `gamma` and
  `lognormal`.** `student_t` and `binomial` are refused rather than
  approximated by a normal.
* **The designs are locally optimal.** The weights depend on `mu`, hence on
  `theta`. This is not new: `structural`'s docstring has always said the
  information on a nonlinear surface is local (review B11) and prescribed
  averaging over prior draws, because a Hill kernel makes `J` depend on `theta`
  whatever the likelihood is. Non-Gaussian families widen that dependence
  rather than introducing it — which is the main reason this was a cheap change
  rather than a new problem class.

## What it is worth

`nbs/design/10-exponential-families.ipynb` runs the same eight-row
dose-ranging design under both assumptions on one Hill surface. The Gaussian
weighting spends a row at dose 10 and two at the top dose; the Poisson
weighting returns both and buys replicates at the half-saturation point
instead, having worked out that the top of the curve — where a Gaussian sees a
perfectly good observation — is where a count is noisiest. The expected
posterior sd of `k_a` is 1.26 under the Gaussian arithmetic and 5.23 under the
right one.

A study powered on the first number comes back inconclusive and nobody can say
why.
