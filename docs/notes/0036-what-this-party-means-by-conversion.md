# 0036 — Silence is not agreement

*Kind: decision. Opened 2026-08-27. Status: implemented on
`feature/definition-registry`. Closes item 8 of the 1.3 backlog in
[0027](0027-scope-and-the-experiment-lifecycle.md) §D27.6 — the last of the
eight.*

[0032](0032-a-population-you-can-size-and-cannot-list.md) gave `meta.pool` a
check that refuses to average two quantities, and it works — given the
estimands. An estimand names a `core.Outcome`, which is a name, a dimension, a
unit and an aggregation. This note closes the gap one level further back:
**who decided what that outcome is, and when did they change it?**

`StudyRecord.estimand_hash` is the hook and it defaults to the empty string.
0027 §D27.6.8 puts it plainly — *optional provenance is absent provenance once
there are thirty parties and one of them has quietly redefined its outcome.*

## D36.1 — two failures, not one

They look alike and want different machinery:

* **Disagreement across parties.** Party A's "conversion" counts a trial
  signup; party B's counts a paid one. Both file under the same quantity name.
  The comparison that would catch it lives in a `Spec` neither record carries.
* **Drift within a party.** A definition changes in week nine. Records on either
  side of the change are the same name, the same party, and different
  quantities, and nothing distinguishes them.

`DefinitionRegistry` answers the first with `consensus` and the second with
`changes`.

## D36.2 — versions are content, not intent

Registering an identical spec returns the version that already exists.
Registering a different one under the same name is version `n + 1` with
`supersedes` pointing at the digest it replaced.

That is the whole versioning policy and it is deliberate. A pipeline that
registers its definitions on every run must not manufacture a version per run,
and a person who changes an outcome must not be able to avoid a version by
forgetting to bump one. Content-addressing already gives both properties for
free; the registry only has to not throw them away.

## D36.3 — a change is a diff, not a flag

`changes` returns the field-level `Spec.diff` between consecutive versions.
"Somebody redefined conversion in week nine" comes back as

```
2026-03-02: v1 -> v2
   aggregation: 'sum' -> 'mean'
```

which is a sentence an analyst can act on, where a boolean is not. `Spec.diff`
already existed and had no callers outside its own tests; this is the first
place in the repository where it does real work.

## D36.4 — silence is not agreement

`Consensus.verdict()` has three answers and the third is the one that matters:

| | |
|---|---|
| `identified` | every scope's current definition hashes the same |
| `blocked` | two or more definitions, with the field-level diff of each dissent |
| `unverified` | some scope has **never registered** the term |

A party that has not said what it means by "conversion" has not agreed with
anybody. Recording that as agreement is precisely the failure the registry
exists to prevent, and it is the failure a naive implementation makes by
default, because absent data compares equal to nothing. `missing` carries the
scopes and the reason says *"silence is not agreement"*.

## D36.5 — generic over `Spec`, and composed over the `Catalog`

A definition is any `Spec` a party has agreed a term means: an `Outcome`, a
`Treatment`, a whole `RoleMap`. `Definition.type_name` records which, and
`consensus` reports a type disagreement as one rather than diffing across types.

The registry is composed over `io.Catalog` rather than owning storage.
Definitions live in their party's scope, appear in the catalog index like
everything else, and cross a party boundary only through a ledgered
`Catalog.transfer` — which is exactly the behaviour a shared definition should
have. The registry needs no state of its own: `DefinitionRegistry(Catalog(root))`
on a fresh process reads the whole history back, and a test pins it.

## D36.6 — what this does not do

**Nothing consumes it yet.** `meta.commensurable` takes estimands as an
argument; it does not resolve an `Outcome` name against a registry, and
`StudyRecord.estimand_hash` still defaults to empty. The wiring — a corpus that
carries its scope, a check that resolves each record's outcome through the
registry for that scope and refuses a record whose definition drifted mid-family
— is the obvious next step and is a `meta` change rather than an `io` one.

**No effective dates.** `Definition.registered` is when the definition was
*recorded*, not when it took effect. A party that changes its outcome on the
first of March and registers it on the ninth has records in between that the
registry will attribute to the old version. Recording an effective range
(rather than a point) would fix it and would need every consumer to ask "which
version was in force when this record was measured" rather than "what is
current".

**No cross-programme consensus.** `consensus` compares `Program`s, which are
`party/programme` pairs. One party running two programmes with two different
definitions of "conversion" is a real and common situation, and the registry
sees it only if somebody passes both programmes in the list.

---

With this, all eight items of the 1.3 backlog in
[0027](0027-scope-and-the-experiment-lifecycle.md) §D27.6 are closed, along with
the follow-on that closing them named. What remains in that area is recorded at
the end of each note rather than as a list: the `build` entry point that would
file an `ArmAssignment`, an `Occupancy` and a `StoppedEstimate` onto an
`ExperimentRun` in one call; attrition and continuous exposure
([0031](0031-assigned-is-not-received.md)); the classical three-level estimator
([0029](0029-the-party-a-study-came-from.md)); the factorial interaction check
([0033](0033-two-experiments-in-the-same-markets.md)); and the online error
control a book whose experiments finish continuously would want
([0035](0035-eight-wrong-go-decisions-a-quarter.md)).
