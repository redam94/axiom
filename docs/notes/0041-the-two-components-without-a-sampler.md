# 0041 — The two components without a sampler, and the prior that knows whose it is

*Kind: decision. Opened 2026-08-28. Status: implemented on
`feature/meta-completions`. Closes the two follow-ons
[0029](0029-the-party-a-study-came-from.md) §D29.6 named and the wiring
[0036](0036-what-this-party-means-by-conversion.md) §D36.6 left open.*

Three gaps in `meta`, all opened by earlier work, all closed here.

## D41.1 — a sampler-free three-level fit

[0029](0029-the-party-a-study-came-from.md) §D29.3 refuses `laplace` on a
nested pool with a free `tau_party`, and the refusal is measured and correct.
It leaves a house that wants the two variance components and a Wald interval,
with nothing installed beyond the four core dependencies, with nowhere to go —
`meta.classical` stopped at two levels.

`meta.three_level` is that route. The marginal covariance
`Sigma = diag(se²) + tau_study²·(same study) + tau_party²·(same party)` is
block-diagonal by party, small, and perfectly ordinary to invert, so the
restricted likelihood is profiled over both variance components by direct
optimization and the mean falls out of GLS at the optimum. It is exact rather
than approximate.

On the notebook's twelve records over three parties, against `tau = 0.15`
within and `0.40` between:

| fit | mu | interval | width | tau within | tau_party |
|---|---|---|---|---|---|
| two-level REML | 0.484 | [0.28, 0.69] | 0.41 | 0.326 | — |
| three-level REML | 0.476 | [0.08, 0.88] | **0.80** | 0.174 | 0.334 |

The same doubling of the interval on `mu` that 0029 got from the sampler, and a
test pins that the two agree.

**What it does not give** is uncertainty *about* the variance components — the
classical/Bayesian trade the rest of `meta.classical` already makes, and one
that matters more at the party level than the study level because the number of
parties is usually small. `meta.pool` with a sampler is the answer that
quantifies it, and the module docstring says so rather than letting a point
estimate of `tau_party` from four parties read as a settled number.

## D41.2 — two next studies, two priors

`prior_from_pool` handed a pooled posterior forward with two targets, `mu` and
`predictive`. A pool with a party level can tell two more questions apart:

* `target="new_party"` — the next study is for a party nobody has seen. It
  crosses both levels: `N(E[mu], sqrt(E[tau_party²] + E[tau²] + sd[mu]²))`.
* `target="same_party"` — another study for a party already in the corpus,
  named by `party`. It crosses one: `N(E[alpha_p], sqrt(E[tau²] + sd[alpha_p]²))`.

In the notebook the second is **2.3 times tighter** than the first, which is the
whole reason the party level was worth fitting: a house that has run four
experiments for a client should not start the fifth from the same prior it would
use for a client it has never met.

Asking for either against a two-level pool raises. A prior that says "for this
party" out of a pool that never knew about parties is exactly the silent-facet
failure this line of work exists to prevent, and returning the two-level answer
under a party-shaped name would have been the easiest way to reintroduce it.

## D41.3 — the corpus resolves its own definitions

[0036](0036-what-this-party-means-by-conversion.md) built the registry and left
the wiring: `commensurable` compared whatever estimands the caller supplied and
never asked what each party had on record.

It now takes `definitions`, `programs` and `term` together, resolves each
record's party's *current* definition of that term, and reports a party that
defines it differently as an `Incompatibility` on the `outcome` facet. The test
that matters shows the two checks disagreeing: the estimands are **identical**,
`commensurable` without definitions says the corpus is poolable, and with them
it is blocked because one party's "conversion" is a mean where the others' is a
sum.

The three arguments are supplied together or not at all, and a party with no
`Program` or no registered definition raises rather than being skipped. A check
that silently ignores the records it cannot resolve is the check 0027 §D27.6.8
complained about, one layer up.

## D41.4 — what remains in `meta`

**No uncertainty on the classical variance components** (D41.1), and no
Knapp-Hartung analogue for the three-level interval.

**`term` is one term.** A corpus whose records differ on *two* definitions needs
two calls. Resolving a whole `RoleMap` per party is the natural generalization
and needs a decision about what "the same roles" means.

**Nothing populates the registry from an analysis.** `io.DefinitionRegistry`
takes a `Spec` somebody hands it; nothing extracts the `Outcome` from a fitted
`Analysis` and registers it. That is the `build` entry point again, which now
has a fifth job.
