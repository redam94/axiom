# 0003 — Where axiom deliberately differs from mmm-framework

One entry per place the port changes a number, a name, or a behaviour on
purpose (CLAUDE.md porting protocol §6; roadmap Phase 9). Entries that
change a *number* are linked from `tests/golden/parent_values.json` through
`superseded_by` where a parent fixture exists; the parent repository was not
available on the machine that built Phases 5–8, so those modules were ported
by specification and the fixture's `_pending` list records what was never
captured (no value there is superseded — there is nothing to supersede).

| # | Where | Parent behaviour | axiom behaviour | Why | Decision |
|---|---|---|---|---|---|
| 1 | `surface.kernels.HillKernel` | `x^s / (k^s + x^s)` on raw doses | `x̃^s / (k̃^s + x̃^s)` with `x̃ = x / reference_dose` | a dimensioned base under a real exponent is not dimensionally sound; the reparametrization keeps the curve and makes `k` carry the dose unit | 0002.18 |
| 2 | every interval | mixed ETI/HDI with no label | `Interval(definition, mass)` on every number | provenance commitment #4 | 0002.3 |
| 3 | `surface.forward` on carryover | any 1-D array convolved as time | `(n_units, n_periods)` required; `steady_state()` for row-wise callers | silent wrong numbers in the allocator and design code | 0002.19 |
| 4 | `estimands` `ratio` | two definitions under one name | `Δagg(Y)/Δagg(X)`, basis-invariant | one name, one number | 0002.20 |
| 5 | `design.methods` intervals | normal critical value | Student-t at the recorded df, still labelled `wald` | the A/A gate found 7–10 % false positives at nominal 5 % on small panels | 0002.21 |
| 6 | `design.methods.synthetic_control` SE | `sd(placebo)/√n_T` | per-donor placebo noise with the `‖w̄‖²` donor-overlap term | the naive SE under-covers with several treated units sharing donors | 0002.21 |
| 7 | `design.methods.switchback` SE | HAC Bartlett | HAC plus the unit-fixed-effect dof factor `NT/(NT−N−1)`; no block bootstrap | 7.4 % → 5.8 % at nominal 5 % | 0002.22 |
| 8 | `design.economics.mid_horizon_factor` | midpoint `(1+r)^{-(n-1)/2}` | exact mean of the discount weights | the midpoint is the first-order approximation of the mean | 0002.24 |
| 9 | `planning/design.py` geo design | `geo` | `design.cluster` | the vocabulary gate; a geo is one kind of cluster | review A5 |
| 10 | `planning/cpa.py` | cost per acquisition | `design.precision` cost per outcome unit | generalization | ledger |
| 11 | `calibrate.prior.design_factor` | (formula not in the ledger) | `mean(contribution / beta)` over draws — the only candidate that reproduces the fixture | golden at 1e-12 | 0002.25 |
| 12 | `calibrate` likelihood route | graph attachment to `BayesianMMM` | `core.Constraint` on `ModelSpec` (numpy and jax) | backend-agnostic | 0002.25 |
| 13 | `calibrate` transfer corrections | a 75-cell notebook | typed `Correction` operators with ledger lines | promoted to library code with tests | 0002.26 |
| 14 | `meta.pool` | centered PyMC hierarchy | marginal model (theta integrated out), exact conditional draws; centered form refused with free tau under Laplace | the funnel has no mode | 0002.29 |
| 15 | `meta.bias` delta | identified by exchangeability | identified only within dual-read contributors; otherwise kept out of the mean and flagged | no silent identification | 0002.30 |
| 16 | `meta.privacy` `spend` | `spend(eps)` | `charge(...)` | banned token | 0002.31 |
| 17 | `meta.store` | sqlite | content-addressed directory via `io.ArtifactRegistry` | no database | ledger |
| 18 | `infer` default backend | NumPyro | Laplace (sampler-free), NumPyro behind `[numpyro]`, PyMC deferred to 1.1 | dependency budget; honest `Unverified` | D2 |
| 19 | `diagnose.refute.placebo_treatment` rule | interval contains zero | tail share of the placebo posterior beyond the original | a one-signed amplitude prior cannot straddle zero | 0002.34 |
| 20 | `diagnose.sbc` ECDF band | — | DKW (Massart) band, conservative | simpler than Säilynoja et al. 2022; documented | 0002.32 |
| 21 | `diagnose.coverage` criterion | "88–92 %" | the exact Clopper–Pearson region at alpha 0.01 with N stated | a percentage band is not a test at finite N | 0002.32 |
| 22 | `planning/{pacing,calendar,forecast,variance,payback}`, `finance/*`, `validation/{results,builders,config,charts}` | shipped | dropped | out of scope per the charter | ledger |

## Charter criteria measured at 1.0

- **Import weight.** `import axiom.core` adds ~60 ms on top of its four
  dependencies (numpy + scipy.stats + pandas + pydantic, ~400–750 ms cold on
  the build machine); gate 1 bounds it relative to `import pandas`. No
  sampler, plotting, or convex library appears in `sys.modules`.
- **Size.** `src/axiom` is 38.5 k lines including docstrings, 32.8 k
  non-blank; 34.7 k excluding `adapters`, `viz`, `sim`, `build`. The charter
  said 30 k for "core"; the overrun is docstrings and typed result Specs,
  and is recorded rather than trimmed.
- **Golden.** Every fixture value reproduces at its recorded tolerance. The
  `_pending` keys were never captured from the parent; the corresponding
  axiom code is tested against `axiom.sim` truth and published references
  (Cinelli–Hazlett 2020, Viechtbauer 2005, Talts 2018) instead.
