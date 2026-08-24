# 0019 — Writing an adapter: the seam, the five parts, and where things actually go

**Kind**: decision + progress. **Opened**: 2026-08-23. **Status**: landed on `develop`.

Rule 2 forbids domain vocabulary outside `adapters/` and gate 3 enforces it on every
identifier in `src/axiom`. That rule has a corollary nobody had written down: **`adapters/`
is the supported extension point**, and until now there was exactly one adapter, which meant
the pattern was a sample of size one and several of its choices looked like conventions
rather than decisions.

`axiom.adapters.agronomy` is a second worked domain — a fertilizer trial — written to be read
as a template rather than because a fertilizer trial was needed. `nbs/adapters/02-agronomy.ipynb`
is the tutorial; this note is what the exercise settled.

## 0019.1 — An adapter is five parts, and none of them is mathematics

1. **Vocabulary** — aliases (`Plot = Unit`), base dimensions (`BASES.declare`), and entity
   constructors (`nutrient_rate`, `soil_test`) that fill in the dimension, the unit and the
   numeraire so a user never has to be right about them.
2. **A roles spec** — a `Spec` naming which columns play which part, plus one function
   translating it to `RoleMap`. The only place the domain's column names exist.
3. **A loader** — the domain's file shape to a `Panel`. Selects, validates, hands off; imputes
   nothing.
4. **A preset** — the `SurfaceSpec` a practitioner would have written, through
   `build.SurfaceBuilder`, with every default argued in the docstring.
5. **Domain quantities** — the numbers the domain asks for, assembled from
   `estimands.Estimand` + `realize`.

If a sixth thing appears — a solver, a likelihood, a transform — it belongs *below* the
adapter and under a general name. The test in 0019.3 is how to tell.

## 0019.2 — Adapters are submodules now, not a flat union

`axiom.adapters.__init__` used to re-export marketing's symbols flat: `role_map`, `spend`,
`roas`. A second adapter makes that untenable immediately — every adapter wants to call its
translator `role_map` and its preset `spec`, and the collisions are not incidental, they are
structural, because the whole point of an adapter is that it uses the *obvious* name.

So the interface is now the submodule: `adapters.marketing`, `adapters.agronomy`, and
`__all__` names them. Marketing's flat names stay because they were public before this was
understood; nothing new should follow them. Gate 12 reads `axiom.adapters.__all__`, so the
notebook series covers the module names, and each adapter's own `__all__` is covered by its
own notebook as a matter of discipline rather than of gate.

## 0019.3 — Where does this go? One test, applied repeatedly

> **Would this make sense to somebody in another field?**

If yes it goes below the adapter, under a general name. The exercise produced a clean
instance of getting this right and one of getting it wrong in the other direction.

**Right.** The classical nitrogen-response curve is Mitscherlich's `Y = A(1 − exp(−kN))`. It
has a name, a century of literature, and an obvious home in an agronomy adapter — and it was
already in the library as `surface.ExponentialKernel`, because "saturating, concave
everywhere, one scale parameter" is a shape, not an agronomic idea. The adapter's job was to
*find* it and make it the default. **Look before you add** is the single highest-value habit
here; a domain expert's first instinct is always that their curve is theirs.

**Wrong in the other direction.** `nbs/case-studies/rutherford/scattering.py` has every
ingredient of an adapter — a hand-built `SupportsForward`, a base dimension, a dozen domain
functions — and is correctly not one. It describes *one experiment*, not a class of them. An
adapter earns its place when the second user shows up; before that it is a case study, and
the cost of getting this wrong is a public API maintained forever for one reader.

The full table:

| you are writing | it goes | the question |
|---|---|---|
| "a plot is a unit" | adapter | is it a translation? |
| a domain's file format | adapter | would another field ever read this file? |
| which facets a domain's number implies | adapter | does the domain disagree with the default? |
| a response shape | `surface` | would another field recognize the curve? |
| a new functional of the response | `estimands` | is it a functional at a stated intervention? |
| a design criterion | `design` | is it about what data to collect? |
| one study's synthetic world | `nbs/case-studies/` | is it *this* dataset, or any dataset of this shape? |

## 0019.4 — The adapter's real contribution is choosing the facets

This is the finding, and it was not obvious before there were two adapters to compare.

The four agronomy quantities are the four `QuantityKind`s with no arithmetic added. What the
adapter contributes is the **facets**, and the two adapters choose oppositely on both of the
ones that matter:

| facet | marketing | agronomy | because |
|---|---|---|---|
| `level.unit` | `aggregate` | `individual` | revenue is **extensive** and sums over geos; a yield in t/ha is **intensive** and summing twenty-eight of them gives twenty-eight times the answer |
| `window.basis` | `cumulative` | `per_period` | contribution over a campaign adds up; three harvests of four tonnes a hectare is four tonnes a hectare a *year*, not twelve |

Get either wrong and every number is off by a factor of `n_units` or `n_periods` — and
**nothing catches it**, because the dimensions and the unit strings still agree. `TimeWindow`
defaults to `basis="cumulative"`, which is the right default for the domain the parent came
from and the wrong one here, so `agronomy._window` overrides it rather than leaving a caller
to discover it. That is the shape of the general lesson: an adapter's defaults are a claim
about what its domain means, and each one deserves a sentence in a docstring.

A third facet decision, smaller but real: `roas` is dimensionless **only** for a
currency-valued outcome, so it checks and returns `Unsupported` otherwise. Agronomic
efficiency needs no such guard, because a rate and a yield are both a mass per unit area and
the ratio is dimensionless by construction — which is exactly why agronomy already reports kg
grain per kg N as a bare number. Same estimand, opposite precondition.

## 0019.5 — A quantity `estimands` cannot express, and what to do about it

Every `Estimand` is a functional of the response **at a stated intervention**. The number an
agronomist actually wants is the inverse: the *rate at which* a functional takes a stated
value — where the marginal product falls to the price ratio. No facet expresses an inversion.

It was not forced into an `Estimand`. `EconomicOptimum` is its own `Spec`, and being one is
what lets it carry what it needs: the prices, the dose range searched, `bracketed`, and a
`detail` line saying the interval came from **inversion of the marginal-product band** and is
therefore not a symmetric error bar. It is built from `surface.marginal_band` and
`surface.response_band` — public API, every draw through the same `forward` — so rule 3 holds
and no optimizer was written.

The case that made it worth doing properly: make nitrogen cheap enough and the optimum moves
past the highest rate anybody applied. The arithmetic will happily extrapolate. It returns
`Unsupported` naming `dose_range`, reporting what was searched and what the marginal product
was at the top of it. That is rule 5, and it is the difference between a recommendation and a
fabrication.

## 0019.6 — The checklist a new adapter has to satisfy

Discovered the hard way, in this order:

1. **Gate 3** exempts `adapters/`, so the domain words are legal — but only there. A helper
   that leaks a domain word into `surface` or `design` fails.
2. **Gate 4** wants an example of every `Spec` in `tests/contracts/_factories.py`. Three new
   specs, three new entries; the gate fails naming the class.
3. **Gate 10** rejects an undimensioned entity shipped in `src/axiom`. This is why
   `soil_test` declares dimensionless rather than leaving it `None` — a soil index has no
   dimension, and saying so is different from not saying.
4. **Gate 12** wants every name in the subpackage's `__all__` used in a code cell under
   `nbs/<subpackage>/`. A new adapter is a new notebook, in the same commit.
5. **`make gates`, `make format lint types`, `make notebooks`** — all four before the commit.

## 0019.7 — One default changed on evidence

`trial_spec` defaults to `intercept="shared"`, not `"hierarchical"`, and the reason is worth
recording because it is the third time this has come up.

Plots differ; that is why fields are blocked, and hierarchical is the better description. But
a field trial is many plots and few seasons, and a hierarchical intercept over twenty-eight
plots with three harvests each makes the Laplace mode search return `Unverified` — the
Hessian at the mode is not positive definite. Note [0009](0009-case-study-tutoring.md) records
the same failure for a school trial, from a different cause (a unit-level dose collinear with
the unit intercept). Here the rates *do* vary within a plot, so it is thinness rather than
collinearity, and the remedy is a sampler rather than a reparameterization.

`"shared"` puts the plot-to-plot variance in the residual, which is honest and is what the
design was already paying for. The notebook shows the hierarchical attempt failing as a typed
value and every downstream quantity declining to produce a number, which is a better
demonstration of the failure discipline than any passing cell.

## 0019.8 — Follow-ups

1. Marketing's flat re-exports are now a deprecated shape. Worth a decision on whether to
   keep them past 2.0.
2. `EconomicOptimum` generalizes: "the dose at which a functional reaches a threshold" is
   domain-free, and a marketing adapter wants the same thing under the name *the spend where
   marginal ROAS hits the hurdle rate*. If a third adapter needs it, it belongs in
   `axiom.surface` next to `frontier` rather than being written twice.
3. The inversion interval is a confidence set by construction and its coverage has not been
   checked. `diagnose.coverage` could check it, and should before anybody quotes it in
   anger.
