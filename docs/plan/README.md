# Implementation plan

Read in order.

| | |
|---|---|
| [00 — Charter](00-charter.md) | What is being built, why an estimand is a transfer key, what is out of scope, what "lighter weight" means in numbers, and the nine conditions for 1.0. |
| [01 — Architecture](01-architecture.md) | The package map, the layering rule, the core vocabulary, the dimension system, the model expression tree and its interpreters, the protocol seam that replaces `BayesianMMM`, and the serialization format. |
| [02 — Porting ledger](02-porting-ledger.md) | Every file in `mmm-framework/src/` with a verdict: port, port-and-extend, rewrite, extract, or drop. ~177k LOC in, ~28k LOC out. |
| [03 — Roadmap](03-roadmap.md) | Nine phases with deliverables, exit criteria, and the CI gate that keeps each from regressing. ~68 working days to 1.0. |
| [04 — Contracts and testing](04-contracts-and-testing.md) | The twelve contract gates, the golden-fixture protocol, the recovery-test rule, the notebook-series rule, and the CI matrix. |
| [05 — Open decisions](05-open-decisions.md) | Nine decisions, eight still open, each with what it blocks and a recommended position. D6 is resolved and kept for the record. |
| [06 — Plan review](06-review.md) | A three-lens critique (statistician, engineer, user) with proposed edits, none yet applied. |

## Decisions already locked

- Name: `axiom`.
- Relationship to `mmm-framework`: **clean-room rewrite, port selectively.** Not
  a fork, not a dependency.
- Vocabulary: **domain-general.** Treatment / dose / unit / outcome / covariate.
  Marketing is one adapter, enforced by a gate.
- Core dependencies: numpy, scipy, pandas, pydantic. Samplers are extras.
- Default backend: NumPyro, behind a swappable protocol.
- **An `Estimand` is a complete transferability key.** Eight facets, and a
  differing facet needs a named licensing assumption or the transfer is blocked.
- **Base dimensions are declarable, not fixed**, and dimension checking is
  abstract interpretation over one model expression tree — the same tree the
  likelihood, the design math, and the serializer read.
- **Every public API is demonstrated in a notebook.** Each subpackage ships a
  notebook series under `nbs/<subpackage>/` as a phase deliverable; gate 12
  fails on an uncovered symbol, and `make notebooks` executes them all in CI.
