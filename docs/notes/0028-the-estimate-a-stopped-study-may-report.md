# 0028 — Naming a bias is not correcting it

*Kind: decision. Opened 2026-08-27. Status: implemented on
`feature/stopped-estimate-correction`. Closes item 4 of the 1.3 backlog in
[0027](0027-scope-and-the-experiment-lifecycle.md) §D27.6.*

`design.sequential` has been emitting this since Phase 5:

```python
STOPPED_ESTIMATE_BIAS = Assumption(
    name="stopped_estimate_bias",
    statement="an estimate reported at the look that stopped the study is biased away "
              "from the null; its naive interval does not have the nominal coverage",
    challenged_by="a median-unbiased or stage-wise-ordered estimate computed for the same path",
)
```

The statement is true and the challenge was never implemented, so every stopped
readout left the module carrying a ledger line that said the number was wrong
and the number that was. Rule 4 was satisfied — the provenance travelled — and
nothing downstream acts on an assumption's prose. `calibrate.attach` folds the
estimate into a prior; `meta.pool` inverse-variance weights it. Across a
programme where winners stop early and losers run to the end, that is a
systematic upward drift in the pooled evidence, assembled out of individually
well-documented numbers.

`design.stopped` is the `challenged_by` clause, written.

## D28.1 — stage-wise ordering, because the ordering must not depend on the drift

Correcting a stopped estimate means choosing a total order on the sample space —
which of two possible outcomes counts as more extreme — and reading the estimate
off the tail probability that order induces. Three orderings are standard
(Jennison and Turnbull §8.5): stage-wise (sample-space), MLE, and
likelihood-ratio.

Stage-wise wins here on one criterion: it is the only one whose ordering does
not depend on the unknown drift. An outcome `(j, z')` is at least as extreme as
the observed `(k, z)` when it crossed the **upper** edge of the continuation
region at some `j < k`, or when it survived to `k` and landed at `z' >= z`. That
set is fixed before any parameter is named, so the tail probability

```
p(θ) = Σ_{j<k} P_θ(cross upper at j)  +  P_θ(survive to k, B_k ≥ z·√t_k)
```

is a clean function of `θ` alone — strictly increasing, which makes all three
numbers inversions of one monotone function:

| solve | gives |
|---|---|
| `p(θ) = 0.5` | the median-unbiased estimate |
| `p(θ) = (1 − mass)/2` | the interval's lower limit |
| `p(θ) = (1 + mass)/2` | its upper limit |

The set is well defined whichever boundary fired and whether or not one did, so
one construction serves an efficacy stop, a harm stop, a futility stop, and a
study that ran to the end — which is biased too, by the looks it could have
stopped at but did not. An outcome continuing past look `k` necessarily sat
inside the continuation region there, hence below `z` for an upper stop, so the
formula needs no separate term for later looks; that is what makes it one
expression rather than a case analysis.

## D28.2 — the same integrator, not a second one

`p(θ)` is the recursion `design.sequential` already runs for
`crossing_probabilities`, stopped at the observed look and asked for the upper
region rather than every labelled region. `stopped.py` imports `_advance`,
`_between`, `_grid_for` and `_limits` from `sequential` rather than copying
them. Rule 3 is written about `forward`, but its reason — two implementations of
one piece of mathematics drift apart, and the drift is a documented bug class in
the parent — applies to the Armitage-McPherson-Rowe recursion exactly as well.

The module is separate from `sequential.py` because it is a separate topic and
`sequential.py` is already a thousand lines, not because the mathematics is
separate.

## D28.3 — every boundary conditions, binding or not

`crossing_probabilities` takes `binding_only=True` by default: the convention
that lets a non-binding futility boundary be conservative about type I error.
The correction uses **every** boundary the rule carries, and the difference is
not academic.

In `nbs/design/07`, a five-look rule stops for harm at look 3 with `Z = −3.05`.
The naive drift is −3.94. The corrected one is −1.20, and the corrected interval
spans zero where the naive one does not. Almost all of that comes from the
*futility* boundary — the non-binding one — because at the first two looks it
sits at −1.5 and −0.8 while the harm boundary sits at −2.12 and −2.31, so
futility, not harm, is the lower edge of the continuation region. A study whose
drift was really −3.94 would have been abandoned before look 3 with high
probability. Surviving to look 3 is evidence against that drift, and a
correction that ignored the non-binding boundary would not see it.

The rule as run is what happened. The error arithmetic is entitled to pretend
otherwise; an estimate is not.

## D28.4 — `stagewise` is a fourth interval definition

`core.IntervalDefinition` gains `"stagewise"` beside `eti`, `hdi` and `wald`.
Gate 6's constructive half is that no interval in axiom has its meaning in a
variable name, and a stage-wise interval is not a Wald interval — it is not
symmetric about the estimate, and on the harm stop above it is nearly twice as
wide. Labelling it `wald` would be the exact failure the gate exists to prevent.
`core.interval()` names `design.stopped_estimate` in its error message so the
route from "I have a monitored study" to "I have an interval" is discoverable
from the failure.

## D28.5 — what it does not do, stated so it is not overclaimed

**At the first look the correction is exactly zero.** Stage-wise ordering
conditions on not having stopped earlier, and at look 1 nothing precedes. The
estimate is unchanged and the interval is the fixed-sample one. This is a
property of the ordering, not a gap in the implementation: a first-look stop is
corrected by a bias-adjusted MLE, which is a different estimator with a
different justification and is not implemented here. A caller who stops at the
first look and wants a correction does not get one from this module, and the
docstring says so rather than returning a number that looks adjusted.

**Median-unbiased is not mean-unbiased.** The simulation in
`tests/unit/test_design_stopped.py` records both: at drift 2 under a four-look
Pocock rule the corrected estimate has median 2.02 against the naive 2.12, and
its *mean* is still above 2. The interval covers 95.0 % against a nominal 95 %.

**The correction is small when the rule rarely stops early.** Stopping at
`z = 2.5` under four-look Pocock, the drift falls by 0.14, 0.19 and 0.22 at
looks 2, 3 and 4. Under O'Brien-Fleming that `z` does not stop the study until
look 3, where the correction is 0.03. This is the right answer — a rule that
almost never stops early has almost nothing to condition on — and it means the
module is not a large effect on most well-designed trials. It is a large effect
on the ones with tight early boundaries, which are the ones that stop early,
which are the ones whose numbers get pooled.

## What is still open

Nothing consumes `StoppedEstimate` yet. `build.StudyBuilder.build_measurement`
still takes `(estimate, se)` from wherever the caller found them, and the
obvious next move is for a monitored study to hand it the corrected pair and the
`stopped_estimate` ledger line together — at which point
`io.ExperimentRun.deviate` has a `"stopped_early"` role to record and
[0027](0027-scope-and-the-experiment-lifecycle.md) §D27.6.3's party-level pool
has a number worth pooling. Those are the next two backlog items, in that order.
