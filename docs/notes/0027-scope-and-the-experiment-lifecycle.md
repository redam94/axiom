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
the order they bite a house running experiments across many parties:

1. **Assignment is computed, never executed or audited.** `match_clusters`
   returns a seeded `Assignment` with its `pre_smd`; there is no unit-level
   hash bucketing, no stratified or blocked randomization, no re-randomization
   against a balance criterion, and — the expensive one — no sample-ratio
   or delivery check anywhere in the repo. `diagnose/` checks the model. Nothing
   checks that the experiment was delivered.
2. **Experiment collision is unmodeled.** `no_interference` is a named
   `method_assumption` and is never tested. `portfolio.recommend` has no
   exclusion constraint, so nothing stops it proposing two experiments that
   share units and weeks.
3. **`meta.pool` has no party level.** `effect_key` is `"contributor"` or
   `"study"`: one shared effect for all of a party's records, or every record
   independent. Neither is right for repeated studies in one market. The fix is
   a third key and a nested tree — study within party within family.
4. **A stopped estimate is flagged, not corrected.** `STOPPED_ESTIMATE_BIAS`
   names its own remedy in `challenged_by` — a median-unbiased or stage-wise
   ordered estimate — and that estimator does not exist, so an early-stopped
   readout is pooled as though it were unbiased.
5. **No program-level error control.** Holm and Benjamini-Hochberg exist in
   `diagnose/structure.py` for one analysis. Nothing controls the error rate
   across parties × treatments × guardrails, and nothing reports the expected
   number of wrong go-decisions per period.
6. **Monitoring is group-sequential only.** No confidence sequences and no
   e-values, so there is no honest always-on readout for somebody who will look
   whenever they like.
7. **Delivery and compliance are asserted.** No ITT/treated-on-treated pair, no
   dose-assigned versus dose-delivered distinction in the readout estimators,
   no attrition model. Two parties with different delivery quality produce
   readouts that are the same estimand on paper and not in fact.
8. **No outcome-definition registry.** `StudyRecord.estimand_hash` is the right
   hook and defaults to `""`. Optional provenance is absent provenance once
   there are thirty parties and one of them has quietly redefined its outcome.

The ergonomic front — a `build` entry point that takes the typed `design`
objects and commits a run in one call — is also deferred. `ExperimentRun` takes
`Spec`s by role, which is what keeps it in `io` and below everything it
records; a builder over it belongs in `build` with the rest of them.
