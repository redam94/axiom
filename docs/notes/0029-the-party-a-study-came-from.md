# 0029 — Twelve studies from three clients are not twelve draws

*Kind: decision. Opened 2026-08-27. Status: implemented on
`feature/party-level-pooling`. Closes item 3 of the 1.3 backlog in
[0027](0027-scope-and-the-experiment-lifecycle.md) §D27.6.*

`meta.pool` offered two ways to group records into effects, and for a house
running repeated experiments for a handful of parties both were wrong:

| `effect_key` | says | is wrong when |
|---|---|---|
| `"contributor"` | a party's studies all estimate one effect | a party's studies differ, which is why it ran four of them |
| `"study"` | every record its own effect, all exchangeable | two studies in one party's markets are more alike than two across parties |

The damage lands on `mu`. `"study"` estimates one `tau` and credits the family
mean with as many independent units as there are studies. Pool twelve studies
from three parties and the interval on `mu` is the interval twelve independent
markets would have earned. In the notebook's simulation that interval is
`[0.28, 0.68]`; the nested one is `[0.00, 0.90]`, 2.25 times wider, and the
second is the honest one. An overconfident family mean is exactly the number a
house quotes to the next client.

`"contributor"` is the honest half-measure: it counts three units, and buys that
by asserting a party's studies are one effect, so its `tau` is again a mixture
of within and between and there is no within-party spread to report.

## D29.1 — a third key, two variance components

```
alpha_p ~ N(mu, tau_party)                              p = 1..P
theta_g ~ N(alpha_p(g), tau)                            g = 1..G
y_i     ~ N(theta_g(i) + x_i·gamma + delta·m_i,  se_i)
```

`tau` keeps its name and changes its meaning: under the nested tree it is the
between-study sd *within* a party. `tau_party` is new. `PoolResult` gains
`tau_party`, `alphas` (one summary per party) and `n_parties`, all `None`/empty
under the two-level trees where there is one variance component and `tau` is it.

The nesting needs no schema change. `StudyRecord` already carries `study` and
`contributor`, and `Corpus` already refuses a repeated study id, so a study
belongs to exactly one party by construction.

## D29.2 — the party level stays in the tree, because a diagonal likelihood cannot hold it

The `"marginal"` parametrization integrates `theta` out exactly by rotating each
effect's records onto the direction of `u = 1/se`: one row carrying `mu/se_g`
with scale `sqrt(1 + tau²/se_g²)`, and `n_g − 1` contrasts with unit scale. The
nested tree keeps that rotation, applied within each *study*, and replaces the
`mu` the first row loads with that study's `alpha_p`.

`alpha` could not also be integrated out. After the within-study rotation the
study rows of one party have covariance `diag(1 + tau²·n_g²) + tau_party²·n·nᵀ`
— diagonal plus rank one — and diagonalizing it needs a rotation along `n`
whose effect depends on `tau`. A rotation is data; `tau` is a parameter. So
`alpha` stays a parameter of shape `(P,)` and the likelihood stays diagonal,
which is the only shape `Likelihood(scale_expr=...)` can express.

## D29.3 — non-centered, and the price of it

The party level is non-centered: the sampled parameter is `z_p ~ N(0, 1)` and
`alpha_p = mu + tau_party · z_p`, reconstructed from the draws after the fit
(the same move `_with_effects` already makes for `theta`). With the centered
form a sampler produced 27 divergent transitions in 4000 draws on the notebook's
corpus; non-centered it is 17, and the geometry is the one HMC is known to want.

The price is that `laplace` can no longer fit a **free** `tau_party`. The
likelihood sees only the product `tau_party · z_p`, so `(tau_party, z)` lies on
a multiplicative ridge that only the priors break; HMC traverses it, and a
Gaussian expanded at one point of a curved ridge does not. Measured on simulated
corpora with a true `tau_party` of 0.4:

| parties | `laplace` `tau_party` | `numpyro` |
|---|---|---|
| 3 | 1.26 | — |
| 4 | 1.39 | 0.42 |
| 8 | 1.42 | — |
| 16 | 1.48 | — |
| 30 | 1.80 | — |

Three to four times the truth, and *worse* with more parties, so it is not a
small-sample calibration story that more data fixes. `pool` returns
`Unsupported` for that combination naming both ways out — fix `tau_party`, or
use a sampler — rather than the number. `tests/unit/test_meta_pool_nested.py`
pins both the refusal and the fact that the refused fit really is wrong, so
nobody relaxes the guard on the theory that it was precious.

With `tau_party` fixed the ridge is gone (it is a constant), the model is
linear-Gaussian in `(mu, z, gamma, delta)` conditional on `tau`, and `laplace`
is exact — which is what the fast tests and the notebook use. Fixing it is also
a defensible modelling choice on its own: with three parties the data say very
little about the spread between them.

## D29.4 — the nested tree is the only one that gets `delta` and heterogeneity together

`delta`, the model-read provenance offset, is identified by a contributor's two
reads of one *shared effect* (`meta.bias`). That is why `PoolSpec` has always
refused `bias_term` with `effect_key="study"`: there, every record is its own
effect and no two reads share one.

Under the nested tree a party's reads still share its `alpha`, so `delta` stays
identified while the studies below it are free to differ. The three-way choice
is now:

| | `delta` | within-party heterogeneity |
|---|---|---|
| `"contributor"` | yes | no |
| `"study"` | no | yes (conflated with between-party) |
| `"nested"` | yes | yes, separated |

`PoolSpec`'s validator was widened accordingly: `bias_term` is refused only for
`effect_key="study"`.

## D29.5 — shrinkage is toward the party, and says so

An effect's conditional mean is `(1 − B)·ȳ_g + B·target` with
`B = se_g²/(se_g² + tau²)`. `target` was always `mu`; under the nested tree it
is `alpha_p(g)`, which is where the information about that study actually is.
`EffectShrinkage` gains `party` and `toward` so the table can be read without
knowing which tree produced it — empty and `None` under the two-level trees,
where the target is `mu` and always was.

## D29.6 — what this does not do

**Two levels, not many.** Party over study is the nesting a book of business
has. A third grouping — region within party, or study within programme within
party — is not expressible, and generalizing to arbitrary nesting would mean
either a rotation per level (which fails at the second, see D29.2) or leaving
every level in the tree and requiring a sampler for all of them. Neither is
worth doing before something needs it.

**No classical three-level estimator.** `meta.classical` still stops at the
two-level DerSimonian-Laird / Paule-Mandel / REML family. A three-level REML
over `(tau², tau_party²)` is a genuinely sampler-free route to the same two
variance components and would remove the `Unsupported` above for anyone who
only wants point estimates and a Wald interval on `mu`. It is the obvious next
item in this area and is not in this change.

**Nothing consumes `tau_party` yet.** `meta.priors.prior_from_pool` hands the
family mean forward as the next study's prior; for a *new party* the right
handoff is `N(mu, sqrt(tau_party² + tau²))` and for another study of an
*existing* party it is `N(alpha_p, tau)` — two different priors that the pool
can now tell apart and the handoff cannot yet ask for.
