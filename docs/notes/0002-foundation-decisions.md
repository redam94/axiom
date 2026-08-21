# 0002 — Decisions taken while building the foundation

Each entry: what was decided, why, and which plan file should be updated to
match. Numbered per entry so later notes can cite `0002.3`.

## 0002.1 — Foundation layer order is `core < data < io`

`01-architecture.md` draws `core data io` on one row and says "core imports
nothing from axiom". `io` must import both `core` (specs) and `data`
(`Panel`), so the row has an order. `data` may import `core`; `io` may import
`core` and `data`; `infer` may import all three. `test_layering.py` encodes
this. *Update:* `01-architecture.md` diagram caption.

## 0002.2 — `Analysis` lives in `io`, not `core`

Review D1 resolves D7 as "yes, `core/analysis.py`". But an `Analysis` holds a
`Panel`, and `core` may not import `data`. The container is
`axiom.io.analysis.Analysis`: frozen, holds `(specs, panel, posterior,
evidence, ledger, provenance)`, delegates, and contains no math. The rule from
D7 stands; only the home moved. *Update:* `05-open-decisions.md` D7.

## 0002.3 — Measurement units are strings; `Unit` is the observation unit

The vocabulary table reserves `Unit` for the randomization / observation unit
(geo, patient, plot). A unit of measure (USD, day) is therefore a plain `str`
field named `unit` on entities and on `Dimension`-aware columns, and the
`UnitSystem` maps unit strings to base dimensions and registers conversions.
There is no `Unit`-of-measure class. This keeps the one confusing name out of
the API rather than in it.

## 0002.4 — `Interval` fields are `definition` and `mass`

`04-contracts-and-testing.md` gate 6 speaks of `interval_definition` and
`hdi_prob`. The shipped type is `Interval(lower, upper, definition, mass)`
with both required and no defaults (review C8: the gate is constructive — a
provenance-less interval cannot be built). `hdi_prob` was the parent's name
and is wrong for an ETI; `mass` is definition-neutral. The runtime half of
gate 6 (every result carries an `Interval`) lands with the estimand registry
in 1b/Phase 4. *Update:* gate 6 wording.

## 0002.5 — Serialized form and hash canon

`Spec.to_json()` writes `{"spec": "<module>:<Class>", "schema_version": "..",
"data": {...}}`. `content_hash()` is blake2b-256 over
`json.dumps(envelope, sort_keys=True, separators=(",", ":"), ensure_ascii=False)`
with the `data` produced by pydantic's JSON mode (so floats canonicalize via
`repr`, `Fraction` as `"p/q"` strings, enums as values). Specs hold no arrays;
anything array-valued is referenced by hash (the `Panel`'s, the posterior's).
Loading a spec whose `schema_version` differs from the class's
`SCHEMA_VERSION` applies a registered migration or raises
`SchemaVersionError` naming both versions — never coerces (review E).

## 0002.6 — Declared base dimensions travel with the analysis

`BaseRegistry` is process-global and mutable by design (a domain declares
its bases). A spec naming an undeclared base fails construction, including
on load. So `io` records the registry's non-default declarations in
`manifest.json` and re-declares them before replaying specs. A stored
analysis is therefore self-describing about its dimension bases.

## 0002.7 — Unit conversion is a ledger line, not an `Assumption`

D6 says a registered unit conversion is automatic "with a ledger line". The
line is a `LedgerLine(kind="unit_conversion", ...)` in `core.verdict`, which
is the same record type the Phase 6 ledger appends. An `Assumption` is
reserved for things that can be false; a registered conversion factor is a
fact of the unit system. A ledger line may carry an `Assumption` or not.

## 0002.8 — Seed contract

Public functions take `seed: int | None`; internals take
`numpy.random.Generator`. `Posterior` and every sampled result record the
seed in their provenance. (Review E; adopted as written.)

## 0002.9 — Panel storage format

`analysis.axiom/panel.csv` with `float_format="%.17g"` plus dtypes in the
manifest, because `pyarrow` is not in the core budget and CSV at 17
significant digits round-trips IEEE doubles exactly. Revisit if panels grow
past what CSV handles comfortably; the manifest records the format so a
second format can be added without a version break.

## 0002.10 — `Panel` reports completeness, never imputes

Review D5 adopted. `Panel.completeness()` returns a typed `Completeness`
spec (balanced or not, missing cells, gaps per unit). Nothing in `data/`
fills a gap.

## 0002.11 — `Posterior` is defined in `core`, re-exported from `infer`

The roadmap places the plain `Posterior` in `infer/posterior.py`, and
`io.serialize` writes `posterior.npz`. But `infer` sits above `io` in the
layer diagram, so `io` cannot import it. The sampler-free `Posterior` is a
dict of arrays implementing `SupportsPosterior` — foundation material — so it
lives in `core/posterior.py`; `axiom.infer.Posterior` is the same class.
Backends (Phase 3) stay in `infer` and *produce* a `core.Posterior`.
*Update:* `01-architecture.md` package map, `03-roadmap.md` Phase 1.

## 0002.12 — axiom's typed errors subclass `Exception`, not `ValueError`

Pydantic catches `ValueError` (and `AssertionError`) raised inside a
validator and re-raises it as `ValidationError`, which would erase
`DimensionError` / `UndeclaredBaseError` the moment they were raised during
spec construction — exactly where they are raised. So `SpecError` and its
subclasses derive from `Exception` and propagate untouched. Plain
`ValueError` is still used for ordinary field validation (a negative mass, a
backwards window) where pydantic's `ValidationError` is the right type.

## 0002.13 — Composition over inheritance, protocols over ABCs (standing rule)

Set by Matthew 2026-08-21 and recorded in CLAUDE.md "Style". Applied
retroactively: `Entity` is now a `Protocol` and the five entity specs are
flat classes sharing `EntityName` and one validator function;
`Unsupported`/`Blocked`/`Unverified` are flat classes sharing `NonEmptyStr`
(which `Assumption` and `LedgerLine` also use). The only inheritance left is
`X(Spec)`, which pydantic requires, and exception hierarchies. A unit test
asserts the entity classes' MRO is ``[cls, Spec, BaseModel, ...]``.

## 0002.14 — Expression-tree node set as shipped

Applied from the review: `Pow` with a rational constant exponent is legal on
any base and gives `dim ** q`; a variable exponent requires a dimensionless
base and exponent (A3). `ODESystem` is dimension-check-only — `value` raises
`NotImplementedError` (C1). There is no `Deriv` node (C2); `ODESystem`
carries the derivative structurally as `(states, rhs, time)` and the checker
enforces `dim(rhs_i) == dim(state_i) / dim(t)`. `Expr` is a pydantic
discriminated union on a `node` literal, so there is no node base class
(0002.13) and the tree serializes. *Update:* `01-architecture.md` node table.

## 0002.15 — Estimand facet representation

The `intervention` facet is three fields — `treatment`, `intervention`,
`reference` — compared together; `Intervention.version` (B2) and
`Level.interference` (B2) are sub-fields and gate 11 parametrizes over them.
`dimension` is a required field that the validator derives and checks, so it
can never differ alone; gate 11 records that explicitly. The rule table
`_RULES` is a dict keyed by facet and asserted equal to `FACETS` at import,
so a ninth facet without a rule fails before any test runs. `FacetDiff`
refuses to exist with neither an assumption nor a block — the "no silent
pass" property is constructive, not tested-for.

## 0002.16 — `sim` sits above the domain layer; graph-established sets are `identified`

Two calls made while integrating Phase 2:

1. `sim` moves from layer 4 to layer 5 in `test_layering.py` so `LinearSCM`
   can *compose* a `CausalGraph` (`graph: CausalGraph`) rather than duplicate
   its seven fields. Nothing in the pillars imports `sim`; `diagnose`
   (layer 6) and tests do. *Update:* `01-architecture.md` layer diagram.
2. A back-door set or an S-admissible set that the graph establishes yields
   status `identified`, in both `identify.verdict` and `identify.transport`.
   Positivity / overlap of the adjustment set is a *data* property and is
   reported by `diagnose.overlap` (review B12), not smuggled into the graph
   verdict as an unverified assumption — the `Verdict` invariant (identified
   carries no unverified assumption) forces the choice to be explicit.
3. When no sufficient transportability condition holds, the transport
   verdict is `unsupported` (the complete sID algorithm is not implemented),
   not `blocked`; `blocked` is reserved for a proposed set (`given=...`) that
   the graph shows is *not* S-admissible. Roadmap Phase 2 criterion 4
   ("blocked unadjusted") is read as `given=()`.

## 0002.17 — Tree extensions for Phase 3, and the backend contract

Added to `core.expr` so that carryover, normalization, and panel hierarchy
are *expressions* rather than opaque functions (which the jax interpreter
could not see): a vector `Const` (≤ 4096 structural values — a lag index,
knots; data still lives in the `Panel`), `Reduce(op, arg)` along the last
axis, and `Gather(source, index)` for a vector `Param` indexed by an integer
column. `Param` gains `shape`; `Prior.hyper` values may name another
parameter, which is how a hierarchy is declared and validated for closure
and cycles in `ModelSpec`.

`core.model.ModelSpec` = mean expression + outcome column + likelihood +
every parameter's prior. `Backend.sample` takes a `ModelSpec` (review A2);
the NumPyro backend runs NUTS on `potential_fn = -compile_log_density`, so
the sampler sees exactly the tree the numpy evaluator sees, and gate 9 is
the numerical agreement `value == jax` / `log_density == compile_log_density`
rather than an AST check (C8). Inference happens in unconstrained
coordinates with the Jacobian applied (B13); `constrain` returns it.

## 0002.18 — Phase 3 review outcomes folded into core

`Reduce` gained `keepdims` and `Convolve` accepts `(..., L)` kernels with
broadcasting leading axes, after a reviewer showed the original
`Div(raw, Reduce(raw))` normalization silently mixed draws when carryover
parameters carried a draw axis. Hill/Power kernels are parameterized as
`x̃^s / (k̃^s + x̃^s)` (both sides divided by a unit constant of the dose
dimension) so jax gradients in `k` are finite at zero dose for `s < 1`.
Laplace returns `Unverified` rather than a posterior when the mode search
does not converge or the Hessian is not positive definite; an explicit
`allow_unverified=True` is the only way to get jittered draws. Every
tolerance in `infer.laplace` and `surface.ascent` is relative to the problem
scale — a reviewer produced wrong modes, wrong SDs, and sign-flipped
Hessians from absolute floors on non-unit or non-default-unit problems,
which is precisely the unit-invariance the dimension system promises.

## 0002.19 — Carryover surfaces and row grids; units at the panel boundary

A reviewer showed that a `Surface` with carryover would convolve *any* dose
array's last axis as time — so the allocator and the optimal-design code,
which evaluate candidate allocations as independent rows, were silently
wrong. The contract is now explicit: with carryover declared, `forward` /
`linearize` require `(n_units, n_periods)` arrays and refuse 1-D grids;
`Surface.steady_state()` (the same spec with `NoCarryover`, valid because
weights sum to one) is what row-wise callers use, and `allocate` /
`frontier` return `Unsupported` naming it when handed a carryover surface.

`prepare()` also requires the panel's entities to carry the *same unit
string* as the spec's, not merely the same dimension — the only place the
unit tier of D6 is checked before a fit — and equispaced periods whenever
carryover is declared. Nuisance conventions resolved at fit time
(`LinearTrend` origin/scale) are recorded in `FitResult.provenance` and
reused for prediction panels. Default treatment names in `sim` are `a`/`b`
— no marketing nouns even where the vocabulary gate would not object.

## 0002.20 — Estimand functionals are defined over the aggregated outcome and dose

A reviewer showed `ratio` meant two different things in `estimands.graph`
(per-row `Δmean/Δdose`) and `estimands.evaluate` (`Δagg(Y)/Δagg(X)`), and
that `transfer_to` emitted a basis correction for a ratio that this
realization makes basis-invariant. Ratified: every functional is taken over
`agg(Y)` and `agg(X)` as the `window` (basis) and `level` facets define them.
`contrast` and `area` scale with the basis and get the
`per_period_to_cumulative` correction; `ratio`, `marginal`, `elasticity`
are basis-invariant and do not. The `Quantity` docstring is the reference.
`estimands.graph` returns `Unsupported` where a per-row expression cannot
express the aggregated functional (scale-mode ratios), rather than a
different number under the same name.

## 0002.21 — Method estimates are comparable numbers with stated assumptions

Every experiment method in `design.methods` returns the same quantity — the
average effect per treated unit per post period — as a `MethodEstimate`
carrying a `wald` interval, the SE method by name, the degrees of freedom
and every intermediate in `detail`. Methods whose natural quantity is
cumulative (time-based regression) put the total in `detail`. Each
`MethodSpec` names its `core.Assumption`s (facet `"method"`, default
`unverified`) and its data requirements; `estimate()` checks the
requirements and returns `Unsupported` before an estimator sees data it
cannot use. Where an SE is an estimated variance with finite degrees of
freedom, the interval's critical value is Student-t at the recorded df
(recorded as `detail["critical"]`), still labelled `wald` because that is
what it is: estimate ± critical × se.

## 0002.22 — The A/A gate decides a method's status, not its author

`design.simulate.calibrate_registry` runs every registered method on A/A
panels (no effect) from one documented DGP — unit intercepts, stationary
AR(1) common period shocks, iid noise, Bernoulli switchback assignment and
ghost exposure — and returns a *new* registry in which a method whose
false-positive count lies outside `clopper_pearson(n, α, 1e-3)` or whose
rate leaves the roadmap's [3 %, 7 %] is `"experimental"`. `METHODS` itself
records the status measured by `tests/recovery/test_aa_calibration.py` on
the gate panel (40 units, 24 periods, 12 pre, 500 simulations), with the
measured rate in a comment; the gate asserts the rule in both directions —
a stable method must be inside, an outside method must be experimental —
so a method can neither be shipped quietly nor demoted silently.

## 0002.23 — Structural design differentiates `forward`, nothing else

`design.structural.fisher_information` builds the Jacobian of the
response with respect to the structural parameters by differentiating a
`SupportsForward` object's `forward` (jax when available and x64 is on,
central differences otherwise — the two agree to 1e-6 and the unit tests
assert it). There is no numpy mirror of the transform chain in `design`;
the information matrix is `Jᵀ J / σ²` plus the prior precision, and a
constant dose schedule is reported as `Unsupported` (singular on the
correlation form) rather than as a large number. Schedules (`pulse`,
`alternating`, `random_switchback`) are the design variable: temporal
contrast is what identifies carryover, which is why `schedule` feeds
`structural`. `contrast_score` is a heuristic ordering, not a substitute
for the information matrix — a reviewer found a pattern where the two
disagree.

## 0002.24 — Value of information carries its numeraire

`design.economics` takes one scalar, `ValuePerOutcome(value, outcome_unit,
numeraire, source)`, and every monetary output (`OpportunityCost`,
`ExperimentValue`, `CandidateScore`, `LearningPriority`) carries the
numeraire string; opportunity cost is *signed* (a prior that says the
treatment is net-negative makes withholding it a gain), discounting uses
explicit per-period weights, and `mid_horizon_factor` is the exact mean of
those weights rather than the midpoint approximation. The parent's finance
reporting (`finance/*`, `planning/{pacing,calendar,forecast,variance,
payback}`) is dropped per the ledger.

## 0002.25 — Two calibration routes, one evidence record

A `calibrate.Measurement` is an `Estimand` plus an estimate, its SE, the
interval definition, the method, and a source; it is the only way evidence
enters. The **prior route** (`derive_prior`) combines measurements of the
same target by inverse variance, converts the target quantity to the
amplitude scale through the design factor — `mean(contribution / beta)` over
posterior draws, the formula that reproduces the parent's fixture exactly
(ratio-of-means, median-of-ratios and regression-through-origin do not) —
moment-matches a lognormal or gamma prior, and returns a new `SurfaceSpec`
whose kernel carries an explicit `amplitude_prior`. It moves the amplitude
and nothing else. The **likelihood route** (`attach`, `fit_calibrated`)
adds a `core.Constraint` — a scalar expression over the model's parameters
built by `estimands.estimand_expr` at the experiment's doses and reduced to
the estimand's aggregate per 0002.20 — to `ModelSpec.constraints`; it moves
the whole curve. `log_likelihood` and the jax `compile_log_density` add the
same term, so gate 9 covers it. Both routes write ledger lines with the
counterfactual number.

## 0002.26 — Corrections are typed operators that carry their own ledger line

Every scope correction in `calibrate.transfer` returns a `Correction` —
kind (multiplicative, additive, SE scale), the operator value, the number
before and after, and a `LedgerLine` whose `assumption` is a typed
`Assumption` on the estimand facet it serves. Chord-to-marginal and
dose-path accumulation go through `surface.forward` / `surface.marginal` and
`core.causal_convolve`; there is no second transform chain. `resolve`
folds the plan's lines and the corrections into one `Ledger`, replacing
the plan's uncorrected line for a facet with the correction's; a facet
nobody corrected is recorded as `uncorrected: source estimate read as-is`
rather than omitted. `Ledger.check_complete(plan)` is the gate rule: every
differing facet in exactly one line, with an assumption and a
counterfactual. A transport verdict that is blocked blocks the transfer.

## 0002.27 — `check.agreement` compares, it does not decide

`calibrate.check.agreement` realizes the measurement's estimand on the
calibrated producer and reports the z-score against the combined
uncertainty, whether the point lies in the posterior interval, and a
three-way verdict at explicit, argument-controlled thresholds (|z| < 1
agrees, < 2 tension, otherwise disagrees). It never alters a fit; a
disagreement is a reason to revisit the transfer plan, which is why it
returns the realized result alongside.

## 0002.28 — Pooling is over dimensionless quantities, and the record says so

A `meta.StudyRecord` carries a `Dimension` and a catalogued poolable
quantity; `normalize` refuses any record that is not dimensionless (or
whose catalogue quantity is not) unless a *licensed* `estimands.TransferPlan`
for that study is supplied, in which case the plan's hash, status and
assumptions are written into the record's detail. Two studies reporting the
same shape in different units therefore cannot meet in a pool by accident.
The classical path (`meta.classical`: fixed effect, DerSimonian–Laird,
Paule–Mandel, REML, Knapp–Hartung, prediction intervals) imports only the
four core dependencies and runs in the core-only CI job.

## 0002.29 — The Bayesian pool is fit on the marginal, not the funnel

The literal hierarchical tree (`theta_i ~ N(mu, tau)`) is built and
works with `tau` fixed, but with `tau` free its joint density has no mode —
Laplace returns `Unverified` rather than a number, and the test pins that
refusal. The default parametrization therefore integrates `theta` out
exactly (a per-effect whitening so contributors with several records
share one effect), fits `(mu, tau, gamma, delta)`, and draws `theta` from
its exact conditional; shrinkage matches `se²/(se² + tau²)` to under 1 %.
To express the marginal scale `sqrt(se_i² + tau²)`, `core.Likelihood`
gained `scale_expr` (exactly one of `scale`/`scale_expr`), resolved
identically by the numpy and jax densities (gate 9) and laid out by Laplace
through `ModelSpec.data_columns`; `surface`'s predictive sampler refuses a
`scale_expr` likelihood by name instead of failing on a key.

## 0002.30 — The provenance bias term is identified within contributor or not at all

`delta` (model read minus experiment read) enters the mean only when at
least one contributor in the family reports both reads — effects are per
contributor by default, so a dual read shares `theta` and identifies
`delta` within contributor. With no dual-read contributor the exchangeability
of model-only and experiment-only contributors is refused as an
identification route: `delta_identification` is `blocked` with the reason,
the `PoolResult` flags `delta.identified = False`, and the posterior sd
equals the prior sd (asserted at 1.005×). The prior handoff
(`prior_from_pool`) returns a `core.Prior` and a `LedgerLine` — `meta`
does not import `calibrate`; the predictive target folds `tau` in.

## 0002.31 — Privacy refuses, it does not clip

`check_cell` returns `Blocked` for a cell below `k` contributors or failing
the dominance rules; no number is produced. The epsilon ledger is immutable:
`charge` returns a new ledger or `Blocked`, never a partial charge, and
`used + remaining == budget` exactly. Releases are clipped-mean statistics
with the clip bounds, sensitivity, mechanism, noise scale (analytic
Gaussian per Balle & Wang 2018, or Laplace) and seed on the record;
`carry_forward` re-uses a release for free only while contributor churn
(Jaccard distance) stays under the stated threshold. The parent's `spend`
is `charge` here because `spend` is a banned token.
