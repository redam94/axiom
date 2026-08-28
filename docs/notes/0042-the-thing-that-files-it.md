# 0042 — The thing that files it

*Kind: decision. Opened 2026-08-28. Status: implemented on
`feature/build-experiment-front`. Closes the `build` entry point deferred since
[0027](0027-scope-and-the-experiment-lifecycle.md) §D27.6, which had accumulated
five specific jobs.*

Every note since 0027 ends with a version of the same sentence.
[0030](0030-the-experiment-that-was-actually-run.md): *"nothing files the
assignment yet."* [0033](0033-two-experiments-in-the-same-markets.md):
*"nothing files an `Occupancy`."* [0028](0028-the-estimate-a-stopped-study-may-report.md):
the corrected estimate should reach `build_measurement` with its ledger line.
[0040](0040-the-experiments-that-finish-on-tuesdays.md): nothing collects
readouts from runs as they reach `read`.
[0041](0041-the-two-components-without-a-sampler.md): nothing registers an
analysis's own outcome.

Five jobs, one object.

## D42.1 — a facade, not a layer

`ExperimentBuilder` is a frozen dataclass holding a `Catalog` and a `Program`
and nothing else. Every method takes the run it acts on and returns the next
one, exactly as `io.ExperimentRun` does; the builder holds only *where things
go*.

**It adds no statistics.** Every number it moves was computed by the module that
owns it. What it adds is that the plan, the assignment, the occupancy, the
readout and the correction end up in one scope under one experiment id with the
run's conformance verdict standing over them — which is the whole argument of
0027 and was, until now, something a caller assembled by hand and could assemble
wrong.

`plan` files each typed role spec under its role name and records the hashes;
`commit`, `start` and `read` walk the lifecycle and file as they go; `runs` and
`readouts` read back.

## D42.2 — `read_stopped` is the one place it does more than plumb

A study that crossed a boundary has a biased naive estimate and
`design.stopped_estimate` corrects it. Two moves then belong together:

* the `Measurement` carries the **median-unbiased** number, not the naive one;
* a `"stopped_early"` `Deviation` records what the naive number was and why it
  was replaced.

Doing them separately is exactly how they come apart — a corrected estimate
filed against a plan that still says the study ran its length, which is the
failure `ExperimentRun` exists to catch and would then not catch. `read_stopped`
does both, files the `StoppedEstimate` itself, appends its ledger line, and the
conformance verdict drops to `downgraded` on its own.

In the notebook the naive drift is 3.00 and the filed one is 2.77.

## D42.3 — `readouts` skips what the run itself blocks

`readouts_across` assembles the input `design.program` and `design.online` take.
It returns one readout per experiment, from the latest filed version, **and only
for runs whose conformance is licensed**.

A readout the run itself calls `blocked` — a plan changed with no deviation
filed, a measurement answering a different estimand — is not evidence about
anything. Including it in a book's error control would quietly restore the
failure `io.ExperimentRun` was built to prevent, one layer further out, and a
test pins that it is skipped.

The `Readout` carries no p-value or e-value. Those come from the analysis, not
from the lifecycle, and inventing one here would have been the module's first
statistical claim.

## D42.4 — what this does not do

**No p-values.** As above: `readouts` returns the identity and the party, and
the caller attaches the evidence. That is a real seam and it is deliberate;
closing it means deciding where a run's *test statistic* lives, which nothing
has decided.

**No `Analysis` bridge.** `io.Analysis` is the container a notebook holds at the
end of a fit, and `ExperimentBuilder` never sees one. Filing an `Analysis`
against a run — so the posterior, the panel hash and the ledger travel with the
readout — is the obvious next connection and is an `io` decision about what an
`Analysis` is *for* rather than a `build` one.

**One party per builder.** A house running eleven parties holds eleven builders,
and `readouts_across` is the only thing that spans them. That is the right
default — a builder that could write into any scope would undo
[0027](0027-scope-and-the-experiment-lifecycle.md) §D27.1 — but it means
programme-level operations are a loop the caller writes.
