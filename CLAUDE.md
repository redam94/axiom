# CLAUDE.md — axiom developer guide

**axiom** is a Bayesian toolkit for causal decision science: identification,
experimental design and calibration, response-surface methodology, and evidence
meta-analysis. It is a clean-room rewrite that ports selected mathematics from
`../mmm-framework`; it is *not* a fork and does not depend on it.

- **Author**: Matthew Reda · **Python**: 3.12+
- **Status**: pre-implementation. Read `docs/plan/` before writing any code.

## Read these first

| Question | File |
|---|---|
| What is in scope, what is not, when is it 1.0? | `docs/plan/00-charter.md` |
| Where does this code go, what does it import? | `docs/plan/01-architecture.md` |
| Where does this port come from, verbatim or rewritten? | `docs/plan/02-porting-ledger.md` |
| What phase are we in, what is the exit criterion? | `docs/plan/03-roadmap.md` |
| What test do I have to write? | `docs/plan/04-contracts-and-testing.md` |
| Is this question already known-open? | `docs/plan/05-open-decisions.md` |

## The five rules

1. **The dependency budget is a hard constraint.** Core is numpy, scipy,
   pandas, pydantic. If you need a sampler, you are in `axiom.infer` or behind
   a lazy import. `make gates` fails otherwise.
2. **No marketing vocabulary outside `adapters/`.** Treatment, dose, unit,
   outcome, covariate. Not channel, spend, geo, KPI, ROAS.
3. **One `forward()`.** The likelihood, the DGP, the design math, and the
   optimizer all call `axiom.surface.forward`. Never reimplement the transform
   chain "for speed" — that drift is a documented bug class in the parent.
4. **Every number carries its provenance.** Interval definition and mass on
   every interval. A ledger line on every evidence transfer. A content hash on
   every spec.
5. **Never swallow an exception.** `except Exception` must re-raise, log at
   warning or above, or return a typed failure. Four separate wrong-number bugs
   in the parent repo trace to this.

## Quick commands

```bash
uv sync --group dev      # core + all extras + tooling
make fast_tests          # everything except -m slow
make gates               # the nine contract tests; run before every commit
make format lint types   # black, ruff, mypy --strict — all three are CI gates
```

## Porting protocol

When moving something out of `mmm-framework`:

1. Find its row in `docs/plan/02-porting-ledger.md`. If it has no row, it is out
   of scope until the ledger says otherwise.
2. Check `tests/golden/parent_values.json` for its recorded values. Port against
   them. A golden failure is a stop.
3. Rename the vocabulary as you go, not afterward.
4. Delete the parent's marketing-specific branches rather than generalizing
   them; a general function with a marketing special case is not general.
5. Write the recovery test against `axiom.sim` in the same commit. A ported
   estimator with no recovery test does not land.
6. If the port deliberately changes a number, write a note in `docs/notes/` and
   link it from the fixture's `superseded_by`.

## Layering

```
viz  adapters  →  build  diagnose  →  meta  calibrate  design
                                            →  surface  estimands  identify
                                            →  infer
                                            →  core  data  io
```

Imports point down only. `tests/contracts/test_layering.py` enforces it.
