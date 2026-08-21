# 00 — Charter

Status: **decided**. Written 2026-08-20.

## The thing being built

`axiom` is a Bayesian toolkit for causal decision science. It supports one loop,
end to end:

```
        declare an estimand
                 |
                 v
    is it identified?  ----no---->  what would identify it?
                 | yes                        |
                 v                            v
      fit a response surface  <---  design an experiment
                 |                            |
                 v                            v
        make a decision  <-----  calibrate on the result
                 |                            |
                 +------> pool into meta-evidence <-+
```

Each arrow is a package. Nothing else is in scope.

## Why a new repo instead of a subset of mmm-framework

The parent repo is ~177,000 lines across 32 subpackages. The five that matter
here (`estimands`, `calibration`, `planning`, `diagnostics`, `benchmarks`)
total ~24,000 lines, and every one of them reaches into `model/base.py` — a
5,375-line `BayesianMMM` class — for the posterior it operates on. Stripping the
platform out in place would mean rewriting that seam anyway, in a repo that also
has to keep a FastAPI server, a React frontend, a LangGraph agent, and a 37k-line
reporting engine compiling. The seam is the work; the fork is not.

The seam, stated once: **the causal/design/calibration layer should depend on a
posterior, not on a model class.** `axiom.core.protocols` defines that
dependency in about 120 lines. Everything downstream is written against it.

## Estimands are the transfer key

The loop above only closes if a quantity measured by an experiment and a
quantity read off a model can be compared. That comparison *is* the product;
everything else is machinery around it. So `axiom` takes a hard position, and it
is the second seam in this repo:

**An `Estimand` must be a complete transferability key.** Two estimands denote
the same quantity if and only if all eight facets match.

| Facet | What it fixes |
|---|---|
| `quantity` | the functional — contrast, marginal derivative, ratio, elasticity, area |
| `intervention` | what is set, to what value, over what support |
| `outcome` | which outcome, at what aggregation, in what dimension |
| `population` | the target population, and the covariate distribution defining it |
| `window` | the time window, and the time basis — per-period rate or cumulative |
| `level` | the unit of analysis: individual, cluster, or aggregate |
| `conditioning` | the strata it is conditional on; empty for a marginal quantity |
| `dimension` | the derived dimension, asserted against the declaration |

Nothing about an estimand may be implicit in the model that produced it. The
producers differ by construction — one is an experiment, one is a fitted
surface, one is a pooled literature — so any facet left to convention is a facet
that silently differs. In the parent repo most of these live in a variable name.

Where facets differ, the difference must be expressible as a finite set of
named, falsifiable assumptions, or the transfer is blocked:

| Facet differs | Assumption required | Challenged by |
|---|---|---|
| `population` | S-admissibility given Z (transportability) | overlap; moderator interaction in `meta` |
| `window` | dynamics stationary; carryover contained in the window | half-life against window length |
| `intervention` | the response surface is correct *between* the two dose levels | curvature; the chord-versus-marginal correction |
| `level` | linear aggregation, or an explicit aggregation model | the Jensen gap |
| `outcome` | commensurability or surrogate validity | usually not falsifiable; must be asserted |
| `conditioning` | effect homogeneity across the collapsed strata | interaction test |

Note that the chord-versus-marginal correction already in scope below falls out
of this table as the `intervention` row. The abstraction subsumes machinery that
was going to be built anyway rather than sitting beside it.

`Estimand.transfer_to(target)` returns a `TransferPlan` carrying
`status ∈ {identified, downgraded, blocked}` — deliberately the same vocabulary
as an identification `Verdict`, because transport *is* an identification
problem: d-separation on a selection diagram. The answer is derived from a
graph, not asserted in prose.

### Dimensions are necessary, not sufficient

Every quantity carries a dimension, and dimensional consistency is enforced. It
buys two things, and the smaller one is catching unit errors.

The larger one is structural. A nonlinear kernel requires a dimensionless
argument, so a saturation kernel is not `f(x)` but `f(x / k)` with `k` in dose
units. That forces every kernel to declare which of its parameters are
**scales** — carrying units, elicited per population — and which are **shapes**,
dimensionless. That split is the meta-analysis story: shape parameters are the
Buckingham Pi groups that pool across studies with different currencies, time
grids, and populations, while scale parameters are local and cross a population
boundary only with a stated assumption. It is also the differential-equation
story: `d(state)/dt` has dimension `[state]/T` and every term on the right-hand
side must match it, which is checkable before anything is sampled.

**The limit, stated so it is not overclaimed:** a passing dimension check is a
*necessary* condition for transfer and never a sufficient one. Outcome-per-dose
measured in three cities and outcome-per-dose measured nationally have identical
dimensions and are different estimands. Dimensions settle the `dimension` row of
the facet table. The other seven rows are settled by the graph and the ledger.

## Scope

### In

- **Causal identification.** DAGs, d-separation, back-door adjustment sets,
  front-door and IV verdicts, role assignment (confounder / mediator / collider /
  proxy), the honest downgrade when an adjustment variable is unmeasured, and
  linear estimators for the routes the graph says are open.
- **Declarative estimands.** Named, versioned, content-hashed counterfactual
  quantities realized from a posterior. An estimand names all eight facets that
  determine whether two quantities are the same quantity, plus its interval
  definition, and can produce a typed transfer plan against another estimand.
- **Dimensional typing and model expression.** A declarable base-dimension
  registry, unit conversion within a dimension, and a model expression tree
  whose dimensions are verified by abstract interpretation at spec-construction
  time. Regression, systems of equations, and ODEs are instances of one tree,
  and `forward()` is one interpreter over it.
- **Response-surface methodology.** Parametric dose–response families (Hill,
  logistic, exponential, power) with carryover, a Bayesian surface model with
  interaction terms, classical and optimal designs (central-composite,
  Box–Behnken, D/A/E-optimal, space-filling), steepest ascent, canonical
  analysis, and constrained allocation on the fitted surface.
- **Experimental design and planning.** MDE and power, expected information
  gain, expected value of information, method selection across geo lift /
  synthetic control / TBR / GBR / ghost ads / switchback / DiD, A/A and A/B
  simulation, structural-parameter identification design, opportunity cost and
  net value, Pareto-front design selection, and program sequencing.
- **Experimental calibration and transport.** Evidence records, the prior route
  and the in-graph likelihood route, selection diagrams and transport verdicts,
  the typed `TransferPlan` between a source and a target estimand,
  chord-versus-marginal corrections, and an assumption ledger — which is now a
  typed diff between two estimand specs rather than free text.
- **Meta-analysis.** Bayesian random-effects pooling with moderators, a
  provenance bias term identified by contributors reporting both a model read
  and an experimental read, evidence-to-prior handoff, and privacy-gated
  publication (k-anonymity, dominance, an epsilon ledger, DP releases).
- **Trust machinery.** SBC, interval coverage, weak-identification diagnostics,
  Cinelli–Hazlett sensitivity benchmarking, decision-scale tipping points,
  prior-to-posterior contraction, specification curves, refutation, backtesting.
- **Simulation.** Data-generating processes with known causal ground truth, used
  by every recovery test in the suite.

### Out

Permanently, not "later":

- Any web application, HTTP API, or frontend.
- Any LLM or agent framework.
- A report generator. `axiom` returns typed results; rendering is someone
  else's job. (`axiom.viz` is a thin optional figure helper, not a report engine.)
- Authentication, tenancy, sessions, job queues, object stores.
- A marketing-mix model as such. The MMM is a *configuration* of
  `axiom.surface` + `axiom.adapters.marketing`, not a class in this repo.
- Extension model families (nested / multivariate / combined / structural-nested),
  LTV/CLV, Excel configuration, ad-platform connectors, data-studio pipelines.

### Deliberately deferred

- Non-parametric surfaces (GP / BART). The API should not preclude them; the
  first release does not ship them. See `05-open-decisions.md`.
- Discrete latent structure (LCA, mixtures). Needs the PyMC backend.
- Sequential / adaptive designs beyond the response-surface bandit ported in
  Phase 5.

## Success criteria

The repo is version 1.0 when all of these hold:

1. `pip install axiom` installs four dependencies and `import axiom` takes
   under 400 ms with no sampler in `sys.modules`.
2. Core is under 30,000 lines and no core module mentions a marketing noun.
3. Every ported computation reproduces its mmm-framework value to the tolerance
   recorded in `tests/golden/` — or the difference is a documented, deliberate
   correction with a note in `docs/notes/`.
4. Every `Estimand` populates all eight facets, and for every pair differing in
   exactly one facet, `transfer_to` returns a plan naming that facet's licensing
   assumption or `status="blocked"` with a reason. No facet returns a silent pass.
5. Every declared estimand's expression tree derives to its declared dimension,
   no nonlinear node takes a dimensioned argument, and every shipped `Equation`
   balances.
6. Every estimator in `identify`, `surface`, `design`, `calibrate`, and `meta`
   has a recovery test against `axiom.sim` ground truth.
7. SBC rank-uniformity and nominal interval coverage pass for the surface model
   and the meta-analysis model.
8. A complete analysis — graph, estimand, surface fit, design, calibration,
   meta contribution — round-trips through `axiom.io.serialize` and reproduces
   its numbers, with no pickle in the path.
9. The twelve gates in `make gates` are green in CI on every commit.
10. Every public symbol in every subpackage appears, executed, in that
    subpackage's notebook series under `nbs/`, and `make notebooks` is green
    in CI. A subpackage with an undemonstrated API is not complete.

## What "lighter weight" means here, in numbers

Measured against the parent repo's own environment on 2026-08-20:

| | Parent (`mmm-framework`) | `axiom` core | `axiom[numpyro]` |
|---|---|---|---|
| Declared runtime deps | 18 | 4 | 8 |
| On-disk (excl. numpy/scipy/pandas) | ~430 MB | ~4 MB | ~256 MB |
| Compiler needed at runtime | yes (PyTensor/C++) | no | no |
| Source LOC | ~177,000 | ~25,000 target | — |

The parent's heaviest single items are `jaxlib` (228 MB, via NumPyro),
`llvmlite` (113 MB, via Numba/PyMC), `scipy` (85 MB), `plotly` (51 MB),
`statsmodels` (44 MB), `nutpie` (24 MB). Core keeps exactly one of those.

Note the honest part: NumPyro is **not** lighter than PyMC on disk. It is
chosen as the default backend because it needs no C++ toolchain at runtime
(the parent's `~/.pytensorrc` clang trap is a recurring support cost), and
because the response-surface layer wants `jax.grad` through the same function
the likelihood uses. The weight win comes from making *either* backend optional,
not from picking one.
