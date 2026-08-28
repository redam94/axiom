# 0027 — The invariants hold inside one analysis and stop at its edge

*Kind: decision + progress. Opened 2026-08-26. Status: D27.1–D27.5 implemented on
`feature/program-scope-and-lifecycle`; D27.6 is the 1.3 backlog.*

Read as one analysis, `axiom` is strict to the point of pedantry. A dimension
mismatch raises. A facet difference is a `TransferPlan` with a status and a named
assumption. Every interval carries its definition and its mass, every transfer
carries a `LedgerLine`, every spec carries a content hash.

Read as a *program* — one party running many experiments, or one house running
experiments for many parties and comparing the answers — almost none of that
holds. `ArtifactRegistry` is a flat `root/<hash>.json` and an `index.tsv` of
`hash\ttype`. It has no notion of whose artifact it is. Two parties' corpora in
one root are indistinguishable, and the only thing standing between one party's
panel and another party's pool is that somebody remembered to use two
directories. `StudyRecord` has a `contributor` — the party that produced the
number — and free text for `period` and `source`, and nothing that says which
line of work the number belongs to.

The same thing is true along the other axis. `StudyBuilder` builds a plan into a
`SimulationSpec`, a `DesignCandidate`, a `Schedule`; after the study runs,
`build_measurement()` builds a `calibrate.Measurement`. The two ends are never
compared. Nothing records that the study that was read is the study that was
planned, and nothing records what changed if it is not. `io.Analysis` is the
container a notebook holds at the end, not a thing with a life.

So the question "is this readout the analysis we said we would run?" has no
answer in the library, and the question "may these two readouts be compared?"
has a very good answer that rests on an unrecorded premise. Across one party
that premise is usually true. Across thirty it is the thing that is wrong.

The failure mode is exactly the one the facet table was written to kill —
a quantity that differs from another in a way nobody wrote down — one level up,
where the facets do not reach.

## D27.1 — a `Program` is a scope, and every artifact belongs to exactly one

`io.Program` is a `Spec` with a `party` and a `program`: the party the work is
for, and the line of work within it. Its `scope` is `party/program`, and both
halves are tokens (`[a-z0-9][a-z0-9._-]*`), so a scope is a path component that
cannot climb out of its root.

The vocabulary is general on purpose (rule 2). A commercial house reads `party`
as the client, a trial reads it as the site, a consortium reads it as the
member. `adapters/` may alias it; nothing under `io/` says client.

`io.Catalog` is the store over scopes. `Catalog.store(program)` returns a
`ProgramStore` — the same `put`/`get`/`__iter__` surface as `ArtifactRegistry`,
restricted to one scope, and physically backed by `root/<party>/<program>/`.
`ArtifactRegistry` is unchanged and still the primitive underneath; a `Catalog`
is a mapping from scope to registry plus one index across all of them.

A `ProgramStore.get` for a digest that exists in the catalog under a *different*
scope raises `CrossScopeError` naming both scopes. It does not return the
artifact and it does not return `None`. Rule 5: a typed failure that says what
happened, because "no such artifact" and "that is somebody else's artifact" are
different facts and the second one is the one worth an alert.

## D27.2 — crossing a scope boundary is a transfer, and a transfer takes a ledger line

`Catalog.transfer(digest, source=..., target=..., line=...)` is the only way an
artifact reaches a second scope, and the `LedgerLine` argument is not optional.
The line is stored in the target scope as an artifact in its own right, and the
target's index entry records the digest of the line and the scope it came from.

This is the same rule the charter states for a facet difference, applied to the
boundary the facets do not describe. A pooled prior derived from party A's
studies that ends up informing party B's model is a real, defensible move; it is
also a move somebody must be able to find six months later, and now they can:
`Catalog.crossings()` is the list.

## D27.3 — the index carries columns, because a scan is not a query

The catalog's `index.tsv` has a header and seven columns: `digest`, `type`,
`scope`, `label`, `created`, `derived_from`, `tags`. `CatalogEntry` is the row.
`Catalog.find(...)` filters on scope, type, label, and tags without opening a
single artifact.

The flat registry's `index.tsv` answered one question — what is in here — and
answered it by reading every file to learn anything else. At thirty parties and
a few hundred runs each that is the difference between a catalog and a
directory.

## D27.4 — `ExperimentRun` binds a plan to its readout, and the plan freezes

`io.ExperimentRun` is a `Spec` holding an experiment id, its scope, a `stage`,
a mapping of role to content hash, and two append-only sequences: `deviations`
and `ledger`. Stages are `designed → committed → running → read → calibrated →
pooled`, with `abandoned` reachable from any of them. Transitions are methods
that return a new run; an illegal one raises `LifecycleError` naming both
stages.

The `commit()` transition is the one that matters. It computes `plan_hash` over
the role→hash mapping as it stands, and from that point on any change to a
planned role must go through `deviate()`, which requires a `Deviation` — what
changed, from what, to what, and why. A run that reaches `read` with an empty
`deviations` tuple is making a strong claim, and it is a checkable one.

`read()` takes the measurement's hash and the estimand hash it was computed
for. If that estimand hash is not the one committed, the run does not silently
accept it: the readout attaches and the conformance verdict below turns
`blocked`. The library's job is to record the mismatch, not to prevent the user
from having made it.

## D27.5 — conformance is a `Verdict`, in the vocabulary already in use

`ExperimentRun.conformance()` returns a `core.Verdict`:

| | |
|---|---|
| `identified` | the readout's roles hash to what was committed; no deviations |
| `downgraded` | deviations recorded, each carried as an `Assumption` with the state the deviation was filed under |
| `blocked` | the readout answers a different estimand than the one committed, or the plan was never committed |
| `unverified` | the run has not reached `read` |

This is deliberate reuse and not a pun. The charter's argument for
`transfer_to` returning the identification vocabulary is that transport *is* an
identification problem. The argument here is the same shape: whether two
readouts may be compared is a question about what licenses the comparison, and
the answer belongs in the same four words with the same rule that a
`downgraded` answer must name at least one assumption. A conformance verdict
travels into `calibrate` and `meta` as a first-class fact about the number,
which is what makes cross-party comparison something other than an act of
faith.

## D27.6 — what this does not do

The scope and lifecycle layer is plumbing. It makes the rest of the backlog
possible to state; it does not do any of it. Recorded here as the 1.3 list, in
the order they bite a house running experiments across many parties. **All eight
were closed on 2026-08-27** — item 7 only in part, as its entry says — and are
struck through; what remains in the area is listed after them, and each closing
note ends with its own "what this does not do".

1. ~~**Assignment is computed, never executed or audited.**~~ **Closed
   2026-08-27** by `design.assign` (hash / block / re-randomize, a rule that
   re-derives without its roster) and `diagnose.delivery` (sample ratio,
   exposure rate, balance, at the alpha a weekly check can afford).
   [0030](0030-the-experiment-that-was-actually-run.md).
2. ~~**Experiment collision is unmodeled.**~~ **Closed 2026-08-27** by
   `design.collision`: disjoint / concurrent / confounded rather than a
   boolean, the concurrent case priced in the units power speaks, and the
   confounded pairs fed to `recommend` as a constraint net value cannot buy
   off. [0033](0033-two-experiments-in-the-same-markets.md).
3. ~~**`meta.pool` has no party level.**~~ **Closed 2026-08-27** by
   `effect_key="nested"`: a second variance component, `mu`'s uncertainty
   scaled to the number of parties rather than the number of studies, and
   `delta` identified alongside within-party heterogeneity for the first time.
   [0029](0029-the-party-a-study-came-from.md).
4. ~~**A stopped estimate is flagged, not corrected.**~~ **Closed 2026-08-27**
   by `design.stopped_estimate`, which inverts the stage-wise ordered tail for
   the median-unbiased estimate and its interval — the correction
   `STOPPED_ESTIMATE_BIAS.challenged_by` had been asking for since Phase 5.
   [0028](0028-the-estimate-a-stopped-study-may-report.md).
5. ~~**No program-level error control.**~~ **Closed 2026-08-27** by
   `design.program`: four routes with `e_bh` for the arbitrary dependence a
   book of experiments actually has, guardrails counted as the decisions they
   are, and `n · alpha` printed on every report so `none` is a choice somebody
   made. [0035](0035-eight-wrong-go-decisions-a-quarter.md).
6. ~~**Monitoring is group-sequential only.**~~ **Closed 2026-08-27** by
   `design.anytime`: a normal-mixture confidence sequence valid at every look at
   once, ~55 % wider than a fixed-sample interval, against the 19.4 % error rate
   a fixed-sample interval actually has when it is read ten times.
   [0034](0034-the-look-nobody-scheduled.md).
7. ~~**Delivery and compliance are asserted.**~~ **Partly closed 2026-08-27**
   by `identify.compliance`: the intention-to-treat and complier effects are
   reported together, with the population under each and the assumptions that
   license the second. Still open within this item: **attrition** (a selection
   problem, not a compliance one) and **continuous exposure** (a partially
   delivered dose is a local average derivative, not a complier type).
   [0031](0031-assigned-is-not-received.md).
8. ~~**No outcome-definition registry.**~~ **Closed 2026-08-27** by
   `io.DefinitionRegistry`: versions that are content rather than intent, drift
   reported as a field-level diff, and a party that never registered a term
   recorded as `unverified` rather than agreeing by silence.
   [0036](0036-what-this-party-means-by-conversion.md).

What closing them opened, each recorded in its own note:

* a **classical three-level estimator** in `meta.classical` — a REML over
  `(tau², tau_party²)` would give the same two variance components without a
  sampler, and would remove the `Unsupported` a free `tau_party` currently
  returns under `laplace` ([0029](0029-the-party-a-study-came-from.md) §D29.6);
* a **prior handoff that knows which party it is for** — `N(alpha_p, tau)` for
  another study of an existing party against
  `N(mu, sqrt(tau_party² + tau²))` for a new one. The pool can now tell those
  apart and `meta.priors.prior_from_pool` cannot yet ask;
* ~~a **`population` facet that can hold a latent subpopulation**~~ — closed by
  [0032](0032-a-population-you-can-size-and-cannot-list.md), which also decided
  that a *pool* must be stricter than a *transfer*: an assumption attached to
  one number is not a licence to average it with another;
* **attrition and continuous exposure** — a partly-delivered dose is a local
  average derivative rather than a complier type, and missing outcomes are a
  selection problem the delivery check measures and nothing models
  ([0031](0031-assigned-is-not-received.md) §D31.6);
* a **factorial interaction check** — `CONCURRENT_EXPERIMENTS.challenged_by`
  names it and nothing implements it
  ([0033](0033-two-experiments-in-the-same-markets.md) §D33.5);
* **online error control** — `design.program` is a batch procedure and a book
  whose experiments finish continuously wants alpha-investing
  ([0035](0035-eight-wrong-go-decisions-a-quarter.md) §D35.7);
* a **corpus that resolves its own definitions** — `meta.commensurable` still
  takes estimands as an argument rather than resolving each record's outcome
  through the registry for its scope
  ([0036](0036-what-this-party-means-by-conversion.md) §D36.6).

The ergonomic front — a `build` entry point that takes the typed `design`
objects and commits a run in one call — is also deferred. `ExperimentRun` takes
`Spec`s by role, which is what keeps it in `io` and below everything it
records; a builder over it belongs in `build` with the rest of them. With
`design.stopped_estimate` landed, that entry point now has a specific first
job: hand `build_measurement` the corrected pair and file a `"stopped_early"`
`Deviation` in the same call ([0028](0028-the-estimate-a-stopped-study-may-report.md)).
