# 0030 — A well-diagnosed wrong number

*Kind: decision. Opened 2026-08-27. Status: implemented on
`feature/assignment-and-delivery`. Closes item 1 of the 1.3 backlog in
[0027](0027-scope-and-the-experiment-lifecycle.md) §D27.6.*

`axiom.diagnose` had eleven modules and every one of them asked whether a
*model* deserved belief: does the procedure cover, does the posterior predict,
does the graph survive refutation, is the parameter weakly identified. None of
them asked the question that comes first — **did the units arrive in the arms
the design put them in?** — and a model diagnosed to three decimal places on a
split that silently ran 51/49 is a well-diagnosed wrong number.

The other half of the same gap was on the design side. `match_clusters` returns
a matched-pair `Assignment` with a seed and a balance number, which is a design
*calculation*. Nothing in the repo decided which arm a unit was in, and nothing
could be asked afterwards to prove what it had decided.

Two modules: `design.assign` runs the assignment, `diagnose.delivery` checks it
was delivered.

## D30.1 — the arm is a function of the unit, not a row somebody kept

`bucket(unit, salt=...)` is blake2b — the hash `Spec.content_hash` already uses
— over the salt and the unit id, read as a uniform draw in `[0, 1)`.
`arm_for(unit, allocation, salt=...)` reads the arm off it against the
allocation's cumulative shares. That function has the property the whole
operational problem needs and that a stored roster does not:

* it needs no state, so two systems that never talk agree;
* it needs no roster, so a unit that first appears in week three lands where it
  always would have;
* it needs no ordering, so a re-export cannot change it;
* it is re-derivable by anyone with the salt, which is the audit.

The price, stated on the result in `detail["realized_vs_target"]`: a hash split
is binomial, so the realized shares are only approximately the target ones. On
4000 units at 50/25/25 the notebook gets 0.509/0.242/0.249.

`method="block"` is the answer when the split has to be exact: permuted blocks
within each stratum, exact at the end of every block and exact within every
stratum. It buys that with a roster and an ordering, both recorded.

## D30.2 — the rule is a `Spec`; the roster is not

`assign` returns an `Assigned` — a frozen dataclass carrying the per-unit arms
as an array — whose `spec` is an `ArmAssignment`. The `ArmAssignment` holds
everything needed to re-derive the arms (the allocation, the method, the salt or
seed, the stratum labels' hash) plus what came out (counts, balance table,
draws used), and a `roster_hash` over the unit ids. It does **not** hold the
roster: a spec holds no large arrays (note 0002 §5), and an assignment of a
hundred thousand units is a large array.

This is the same split `meta.pool` uses — `Pooled` carries the `Posterior`,
`PoolResult` is the spec — and it means the *rule* is the thing that travels,
hashes, and files in a `Catalog`.

`Assigned.verify()` puts the two back together: it re-derives every arm from the
rule and returns a `core.Verdict`. `identified` when they reproduce;
`blocked` naming the first unit that moved and how many did. A boolean would
not have been enough — an audit needs to see which unit.

## D30.3 — re-randomization is a design change, and says so

`method="rerandomize"` draws block assignments until the worst standardized
difference is under a threshold (Morgan and Rubin 2012). It works: on the
notebook's corpus the worst standardized difference falls from 0.0117 to 0.0037
in 135 draws.

It also changes the design, and this is the part that gets forgotten.
Re-randomization narrows the estimator's sampling distribution relative to
complete randomization, so a Wald interval computed as though the design were
completely randomized is **conservative** — valid, but with power left on the
table. `RERANDOMIZED` rides on every such result with the two analyses that
recover it in `challenged_by`: adjust for the balancing covariates, or test over
the set of assignments the threshold would have accepted.

A threshold that is never met returns the best draw with `balance_met=False`
and a ledger line saying which, rather than raising or passing silently. The
best of a thousand draws is a usable design; a silent pass is not.

## D30.4 — three checks, each the challenge to an assumption already named

`design/methods/registry.py` has named `random_assignment` and `exposure_known`
since Phase 5, with `challenged_by` clauses nobody had implemented. Same shape
as note [0028](0028-the-estimate-a-stopped-study-may-report.md): the assumption
names the check, and the module is the check.

| check | tests | challenges |
|---|---|---|
| `sample_ratio` | chi-square goodness of fit of arm counts against the allocation | `random_assignment` |
| `delivery` | chi-square on the arm-by-exposed table: is there one exposure rate? | `exposure_known` |
| `balance` | one-way F per covariate, Holm-corrected | `random_assignment` |

`check_delivery` runs every check its arguments support and reports which ones
ran in `checks_run` and in the verdict's `route`. What is not supplied is never
silently passed.

## D30.5 — alpha is 0.001, because the check runs every week

The convention is not 0.05 and the reason is not statistical delicacy. These
checks run on *every* experiment, so at 0.05 one experiment in twenty trips one
of them with nothing wrong. A check that cries wolf weekly is a check nobody
reads, and an unread check is a worse failure than the one it was guarding
against. `SRM_ALPHA = 0.001` is the default and is a named constant so the
choice is visible rather than a literal in a signature.

The three-arm case in the notebook is the demonstration: a high arm 6 % light is
`p = 0.0022` — a mismatch at 0.05 and 0.01, not at 0.001. Both readings are
defensible. Only one of them can be the default.

## D30.6 — a passing check leaves the assumption `unverified`

Not `satisfied`. `Assumption.satisfied()` exists and means code checked the
condition; a sample-ratio check that does not fire has checked one consequence
of random assignment and found nothing, which is absence of evidence. So:

* pass → `random_assignment` at `unverified`, and the ledger line says so;
* fail → `.violated()`.

`DeliveryReport.verdict()` is `identified` or `blocked` and **never
`downgraded`**. A downgrade is what an assumption buys you across a facet
difference. Nothing licenses a readout past a split that did not happen, so
that status is not on the menu here, and the ledger carries the violated
assumption rather than a licence.

## D30.7 — `ArmAllocation`, not `Allocation`

`axiom.surface` already exports an `Allocation`: a dose per treatment and the
objective it achieves, returned by the optimizer. Two public classes called
`Allocation` in one package is the kind of thing that reads fine in each
module's own tests and confuses everybody afterwards. The arm-shares spec is
`ArmAllocation`, which also pairs with `ArmAssignment` and `ArmCount`.

## D30.8 — what this does not do

**No compliance model.** `delivery` reports *whether* the arms were reached at
different rates. It does not estimate what to do about it: there is still no
ITT / treated-on-treated pair, no dose-assigned versus dose-delivered
distinction in the readout estimators, and no attrition model. That is item 7 of
the 1.3 backlog and this module is its input, not its answer.

**No panel-level entry point.** The checks take counts and arrays. Reading arm
labels and exposure flags off a `data.Panel` is an adapter-shaped job and is
not here; `arm_counts` is the one convenience, and it refuses an arm label the
allocation does not name rather than counting it as zero.

**Balance only sees what was recorded.** The F test is the weakest of the three
and the docstring says so twice. A clean balance table says nothing about the
covariates nobody wrote down, which is most of them.

**Nothing files the assignment yet.** `ArmAssignment` is a `Spec`, so it stores
in a `Catalog` and can be a role on an `io.ExperimentRun` — `"assignment"` is
already named as an example role in that module's docstring — but no code puts
it there. That is the `build` entry point deferred in
[0027](0027-scope-and-the-experiment-lifecycle.md), which now has a third
specific job.
