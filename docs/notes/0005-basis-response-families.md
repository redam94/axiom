# 0005 — Basis response families: polynomial, spline, piecewise linear

**Kind**: decision. **Opened**: 2026-08-21. **Status**: landed on `develop`; part of the
1.1.0 surface.

## Why

Note [0004](0004-case-study-hypertension.md) §0004.3 recorded a gap the HYPER-3 case
study walked into: every response kernel `axiom.surface` shipped is **monotone and
saturating**, and the case study's oldest age stratum has a dose-response that *turns
over*. The Hill fit could not describe it, and failed informatively — an arm-mean
residual of 8 mmHg at the top dose against ≤1.5 in the strata where the family fits.
That note deferred the fix pending a second use case. There is no second use case: a
dose that helps and then harms is the ordinary reason a phase-II trial exists, and a
library for dose-response that cannot express it is missing a family, not an extra.

Three land together, because they are the same idea at three levels of structure:

| family | basis on `u = dose / reference_dose` | what it is for |
|---|---|---|
| `PolynomialKernel(degree=d)` | `u, u², …, u^d` | one smooth bend, few dose levels |
| `SplineKernel(knots=…)` | natural cubic, fixed knots | several bends, linear extrapolation |
| `PiecewiseLinearKernel(knots=…)` | `u, (u−t₁)₊, …` | a slope that changes at a stated dose |

## 0005.1 — A basis family declares several amplitudes, and that is the whole difference

The shipped kernels were `amplitude · saturation(dose)`: one signed-positive amplitude
carrying the outcome dimension, times a dimensionless shape. Three parts of the library
read that structure — `surface.model.parameter_roles` (amplitude ⇒ *linear*),
`surface.forward` (interactions), and `calibrate.prior` (the prior route rewrites the
amplitude).

A basis family cannot be written that way, and forcing it would have been the wrong
trade. Writing `beta · (u + c₂u² + c₃u³)` with a single amplitude and signed *shape*
coefficients keeps the old invariant, but the response is then bilinear in `(beta, c)`,
so every shape coefficient is `nonlinear` and `linearize` becomes an approximation.
Declaring one **signed amplitude per basis function** instead makes the response linear
in every parameter it has — which is the entire practical advantage of a basis
expansion. `surface.linearize` is then exact, `design_matrix` is the real design matrix,
and the alphabet-optimality criteria in `surface.designs` mean what they say.

So `roles` is what tells the two kinds apart: **more than one `"amplitude"`** is a basis
family. Consequences, all of them fixed rather than worked around:

- **`ResponseKernel` gained `saturation_derivative`.** `surface.forward` previously built
  `∂f/∂c` for an interaction by dividing `derivative` by "the" amplitude, and raised if a
  family did not have exactly one. Every family now ships `d saturation / d dose` in
  closed form, so an interaction differentiates one side without assuming anything about
  the parameter count, and the `LinearKernel` special case in `forward` is gone. For the
  five single-amplitude families `derivative` is now *defined* as
  `amplitude · saturation_derivative`, so the two cannot drift; a test asserts they are
  the same tree.
- **`saturation` is the identity shape `u`.** As for `LinearKernel`, whose exception this
  extends: no single parameter factors out of a basis response, so the dimensionless
  shape an interaction multiplies is the reduced dose itself. Documented in the module
  docstring and in `ResponseKernel`.
- **`calibrate.derive_prior` refuses.** The prior route rewrites the one amplitude a
  randomized contrast identifies; a basis family has no such target. `_amplitude_name`
  now raises naming the count and pointing at the likelihood route
  (`calibrate.attach`), which constrains a function of the whole coefficient vector.
  Rule 5: a typed refusal, not a silently wrong choice of `beta1`.

## 0005.2 — Basis coefficients are signed; that needed a second prior set

`AMPLITUDE_PRIOR_FAMILIES` is positive-support only, because a saturating kernel's
amplitude is an asymptote with a sign convention. A basis coefficient is the opposite:
the family earns its keep by letting coefficients go negative. `BASIS_PRIOR_FAMILIES`
adds `normal`, which is also the default (`normal(0, amplitude_scale)`). Passing a
positive family instead is how a caller asks a basis family for a monotone fit, so the
set is a superset rather than a replacement.

## 0005.3 — `relu` and `step` in `core.ApplyFn`

A truncated-power spline basis needs `(u − t)₊`. The node set had `softplus` but no
hinge, and a softplus approximation would have made the knot width a parameter and the
family not a spline. Two entries were added to `ApplyFn` — `relu` and `step` — with one
line each in the numpy interpreter, the jax interpreter, and the latex renderer. Nothing
else in `core` is aware of them; `Apply`'s dimensionless-argument rule already covers
them.

Both take the value `0` at `x = 0`, identically under numpy and jax, so a
piecewise-linear derivative evaluated exactly *at* a knot reports the slope arriving into
it. `relu` raised to a power `p ≥ 2` is `C^(p−1)`, so a cubic spline's kink is invisible
to a gradient; `step` has zero gradient a.e., which is correct.

A truncated-power basis is not the best-conditioned way to write a spline — a B-spline
basis is — but it is *exact* in the node set, it differentiates in closed form, and at
the three to six knots a dose-finding study supports the conditioning is not the binding
constraint. `surface.check_linearization` and `design.identifiability_ridge` are the
things to consult if it becomes one. A B-spline basis would need a recursive
construction the node set cannot express without an `Opaque`, and `Opaque` costs the jax
path; that is the trade, and it is recorded here rather than discovered later.

## 0005.4 — The natural cubic spline, not the unrestricted one

`SplineKernel` is the standard reduced natural-cubic basis (Hastie, Tibshirani &
Friedman, *ESL* 5.2.1) with the constant term dropped, since the surface supplies the
intercept:

```
B_1(u)     = u
B_{k+1}(u) = d_k(u) − d_{K−1}(u),                      k = 1 … K−2
d_k(u)     = [ (u−t_k)₊³ − (u−t_K)₊³ ] / (t_K − t_k)
```

`K` knots give `K − 1` coefficients. "Natural" is the point: the fit is **linear outside
the boundary knots**, so extrapolating past the highest tested dose gives a straight line
rather than whatever a cubic's leading term wants. A unit test pins both tails (second
difference zero above the last knot and below the first) and contrasts them with a cubic
polynomial over the same range, which does not have the property.

Every basis function vanishes at `u = 0` because knots are validated strictly positive,
so `response(0) = 0` holds for these families exactly as for the others, and the trial
intercept keeps its meaning.

## 0005.5 — `reference_dose` is the coordinate, not a parameter

For the saturating families `reference_dose` only centres the prior on `k`. For a basis
family there is no `k`: `reference_dose` *defines* `u`, and the knots are given in dose
units and divided by it. The docstrings say what to set it to — the top of the dose range
you intend to fit — because leaving it at `1.0` with doses in the tens makes `u³` four
orders of magnitude larger than `u`, and no single prior scale is sensible for both. That
is a real footgun and the only one these families have; it is called out on every class.

Unit invariance still holds, differently: rescaling the dose unit means moving
`reference_dose` and the knots with it (spec fields), not rescaling a parameter.
`test_unit_invariance_of_shapes` covers both routes.

## What it bought, measured

Refitting HYPER-3's oldest age stratum, where the true response reverses above 20 mg
(`nbs/case-studies/hypertension/04-the-dose-response.ipynb` §3b):

| family | residual scale `sigma` | worst arm-mean residual | parameter roles |
|---|---|---|---|
| `hill` | 7.37 | 7.95 mmHg | linear + nonlinear |
| `polynomial(3)` | 6.04 | 3.16 | all linear |
| `spline`, 3 knots | 6.21 | 4.54 | all linear |
| `piecewise_linear`, 2 knots | 6.02 | 3.13 | all linear |

In the two strata whose response really is monotone the Hill curve fits as well on fewer
parameters, and its `k` and `beta` mean something a pharmacologist can argue with where a
spline coefficient does not. The notebook draws the ordinary conclusion rather than a
triumphal one: use the family whose shape assumption you are willing to defend, and let a
fit that cannot hold the shape say so.

## Recovery, and one sharp edge in `sim`

`tests/recovery/test_surface_recovery.py` gains a world per family whose response
genuinely turns over, and recovers every coefficient inside a 90 % HDI at 400 units. Its
negative control is the same world fitted with a Hill kernel: at low noise the misfit
has nowhere to go but `sigma`, which comes out four times the basis family's.

The edge: those worlds state `truth` explicitly rather than taking `truth_mode="centre"`.
The centre of a signed coefficient prior is **zero**, so a basis family at the default
truth mode builds a world in which the treatment does nothing, and any recovery or SBC
test written on it is vacuously green. The recovery test says so in its docstring; it is
a property of centring a signed prior, not a bug in `sim`, and the alternative — an
off-centre "centre" — would be worse.

## Left open

- **A B-spline basis**, per 0005.3, if conditioning ever binds before sample size does.
- **Monotone-constrained splines** (I-splines, or a positive-coefficient parameterization
  of the increments). Passing a positive `amplitude_prior` to `PiecewiseLinearKernel`
  constrains the *slope changes* to be positive, which is convexity, not monotonicity —
  the useful constraint needs a different basis, not a different prior.
- **Knot placement as a design question.** `design.design_to_identify` will place doses
  for a fixed knot set, but nothing chooses the knots; quantiles of the observed dose
  distribution are the usual default and the notebooks set them by hand.
- **`calibrate`'s prior route for basis families** — currently a refusal pointing at the
  likelihood route. A contrast at one dose does identify a linear functional of the
  coefficient vector, so a constrained version is possible and is not written.
