# 0032 — A population you can size and cannot list

*Kind: decision + deviation. Opened 2026-08-27. Status: implemented on
`feature/latent-populations`. Closes the follow-on
[0031](0031-assigned-is-not-received.md) §D31.6 named as the most valuable
remaining item in that area.*

Note 0031 built the ITT/complier pair and ended by saying what it could not do:

> `Estimand` has no vocabulary for "the compliers" as a population, because that
> population cannot be described by a covariate distribution — only sized.

So `transfer_to` could not see the difference between an effect over everybody
and an effect over the compliers, `meta.pool` could not refuse to average them,
and the whole eight-facet apparatus — the thing the charter calls the transfer
key — was blind to the most common way two field experiments stop measuring the
same thing.

## D32.1 — a fourth kind of population, beside the strata

`core.Population` was a name and `strata`: covariate levels and weights. That is
the population you can point at. `core.LatentSelection` is the one you cannot:

```python
class LatentSelection(Spec):
    kind: EntityName        # "complier", "always_taker", "never_taker", "survivor"
    instrument: EntityName  # what defines the response type
    exposure: EntityName    # the variable whose response defines it
    share: float | None     # identified size, when known
```

and `Population.latent: LatentSelection | None`. A latent population is sized
and never listed: the complier share falls out of two exposure rates, and no
covariate distinguishes a member from a never-taker.

**Identity is `(kind, instrument, exposure)` and deliberately not `share`.** A
complier is a unit whose exposure responds to *this* instrument, so a letter and
a phone call recruit different people. Two studies of the letter in different
markets have different complier shares and the same stratum; a study of the
letter and a study of the call have neither. `LatentSelection.same_stratum`
is that comparison, and it is what the facet rule branches on.

`kind` is free text with a documented convention rather than a `Literal`,
because truncation-by-death survivors are a principal stratum too (Frangakis and
Rubin) and should not need a code change.

## D32.2 — the facet rule gained a case; it did not change the one that was there

| source | target | verdict |
|---|---|---|
| no latent | no latent | unchanged: `s_admissibility` |
| latent | same stratum | `s_admissibility` if the populations otherwise differ |
| latent | different stratum | **blocked** |
| latent | no latent (either way) | `latent_type_homogeneity`, plus `s_admissibility` if the populations differ too |

Two *different* latent strata are blocked rather than downgraded, and the reason
is worth stating precisely: a bridging assumption has to be about something, and
there is nothing here to be about. No observable distinguishes a member of
either set, so no condition over observables can license reading one as the
other. That is the charter's rule — *"a finite set of named, falsifiable
assumptions, or the transfer is blocked"* — applied to a case where the set is
empty.

`latent_type_homogeneity` is not falsifiable either, and it is still an
assumption rather than a block, because it is at least *namable*: it says the
effect on the compliers is the effect on everyone. Its `challenged_by` is exact
about why the data cannot settle it — a never-taker is by construction never
exposed under this instrument, so no design using it observes the effect being
assumed equal; only a different instrument can, and that recruits a different
subpopulation.

## D32.3 — pooling is not transferring, and the pool's rule is stricter

This is the decision that makes the rest work, and it took two attempts to get
right. `transfer_to` *downgrades* a complier effect read as a whole-population
effect. If `meta.pool` had simply honoured transfer statuses, a corpus mixing
one party's ITT with another's complier effect would have been licensed, and the
promise in 0031 would not have been kept.

The resolution is not to make the transfer stricter. It is that the two
operations are different:

* a **transfer** attaches a named assumption to *one* number and carries it in
  that number's ledger, where a reader sees it;
* a **pool** takes a precision-weighted average. The weights come from standard
  errors; nothing in them refers to a population; and the assumption, if anyone
  had named it, would apply to a quantity that never appears in the output.

So `meta.commensurable` blocks on a latent population where `transfer_to`
downgrades, and every other downgrade is allowed and *named* on the result.
Between-population heterogeneity is what a random-effects pool is for; a
subpopulation no covariate describes is not.

## D32.4 — the check is optional, the silence about it is not

`pool(spec, corpus, estimands=...)` takes a mapping from `StudyRecord.study` to
the `Estimand` behind it. Requiring it would have broken every existing caller
and every notebook, and making it optional risks exactly what note 0027 §D27.6.8
complains about — *optional provenance is absent provenance*.

The compromise is that the result always says which it was.
`PoolResult.detail["commensurability"]` is either the check's ledger line or
`"unchecked: no estimands supplied, so nothing compared what the records
measure"`. A record with no estimand supplied lands in
`Commensurability.unchecked` and is never counted as passing.

## D32.5 — a deliberate hash change

Adding a field to `core.Population` changes the serialized form of every
`Population`, therefore of every `Estimand` that holds one, therefore of every
`Spec` that holds an estimand. `Population(name="enrolled")` hashes differently
today than it did yesterday.

Nothing in the repository pins a hash *value* — every test compares one hash to
another — and no artifacts are published, so this costs nothing now. It would
not be free later: `io.ArtifactRegistry.get` verifies that a stored artifact
still hashes to its filename, so a registry written before this change would
report its populations as corrupt. `Population.SCHEMA_VERSION` is left at `"1"`
because the change is backward-compatible for *loading* — a missing `latent`
key takes its default — and a version bump without a migration would turn a
loadable file into a `SchemaVersionError`.

Recorded here rather than in the deviations note (0003) because it is not a
ported number: it is the cost of a structural change, and the next person to
find a "corrupt artifact" in a pre-1.3 registry should find this paragraph.

## D32.6 — what this does not do

**No estimand registry behind `StudyRecord`.** `estimand_hash` is still a
string that defaults to empty, and `commensurable` takes the estimands as an
argument rather than resolving hashes against an `EstimandRegistry`. Wiring
those together — so a corpus carries its own estimands and the check needs no
second argument — is the obvious next step and is a `meta`/`io` change, not this
one.

**No share-weighted reconciliation.** Knowing the complier share means the ITT
and the complier effect are related by `itt = complier × share`, so in principle
one *could* be converted to the other and pooled. That is a `calibrate`
`Correction`, exactly like `chord_to_marginal`, and it would move the case from
blocked to downgraded-with-a-correction. It is a real thing to want and it is
not here, because the conversion is only valid under the same
`latent_type_homogeneity` the block exists to surface — it would make the
assumption invisible again, one layer further in.

**Only the population facet knows about latency.** An `intervention` facet
distinguishing assigned-dose from delivered-dose would close the other half of
0031's pair. `Intervention` has a `version` field that could carry it, and
nothing does.
