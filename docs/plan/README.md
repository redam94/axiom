# Implementation plan

Read in order.

| | |
|---|---|
| [00 — Charter](00-charter.md) | What is being built, what is out of scope, what "lighter weight" means in numbers, and the seven conditions for 1.0. |
| [01 — Architecture](01-architecture.md) | The package map, the layering rule, the core vocabulary, the protocol seam that replaces `BayesianMMM`, and the serialization format. |
| [02 — Porting ledger](02-porting-ledger.md) | Every file in `mmm-framework/src/` with a verdict: port, port-and-extend, rewrite, extract, or drop. ~177k LOC in, ~28k LOC out. |
| [03 — Roadmap](03-roadmap.md) | Nine phases with deliverables, exit criteria, and the CI gate that keeps each from regressing. ~64 working days to 1.0. |
| [04 — Contracts and testing](04-contracts-and-testing.md) | The nine contract gates, the golden-fixture protocol, the recovery-test rule, and the CI matrix. |
| [05 — Open decisions](05-open-decisions.md) | Eight decisions deliberately not yet made, each with what it blocks and a recommended position. |

## Decisions already locked

- Name: `axiom`.
- Relationship to `mmm-framework`: **clean-room rewrite, port selectively.** Not
  a fork, not a dependency.
- Vocabulary: **domain-general.** Treatment / dose / unit / outcome / covariate.
  Marketing is one adapter, enforced by a gate.
- Core dependencies: numpy, scipy, pandas, pydantic. Samplers are extras.
- Default backend: NumPyro, behind a swappable protocol.
