# 0001 — Phase 1a: foundation progress

Branch: `feature/phase-1a-foundation` (cut from `develop` 2026-08-21).
Scope is the Phase 1a split proposed in `docs/plan/06-review.md` §F:
dimensions, spec, verdict and result types, protocols, intervals, stats,
entities, `data/`, `io/`, `infer/posterior.py`, the gates that cover them,
and the notebook series for `core`, `data`, `io`, `infer` (the parts that
exist). The expression tree, its interpreters, and `estimands/spec.py` are
Phase 1b and follow on a separate branch.

Decisions made along the way are in [0002](0002-foundation-decisions.md).

## Log

### 2026-08-21 — opened

- Read the plan and the review. Applying from the review before writing code:
  A1 (`core/verdict.py`, one `Assumption` type), A3 (`Pow` rule — matters for
  1b, recorded now), C8 (gate 6 constructive: `Interval` cannot be built
  without `definition` and `mass`), C9 (gate 1 relative threshold), E (schema
  migration rule, typed failure types in `core/result.py`, seed contract,
  hashing rules), D1 (`Analysis` container now, not Phase 4).
- No golden fixture touches Phase 1a, and `../mmm-framework` is not checked
  out in this environment; everything here is written fresh against the plan.
  `core/intervals.py`, `core/stats.py`, `data/*` are marked PORT in the ledger
  — they are ported *by specification* (the ledger row's description), not by
  copying. When the parent is available, diff behaviour and add golden cases
  for `intervals` (ETI/HDI on a fixed draw set).

### 2026-08-21 — Phase 1a code complete on the branch

Landed (all of it `mypy --strict`, black, ruff clean; 112 tests pass; all 9
notebooks execute under nbmake):

- `core/spec.py` — envelope, blake2b content hash, `diff`, subclass
  discovery, migration registry, `SchemaVersionError` / `UnknownSpecError`.
- `core/dimensions.py` — `Dimension` over `Fraction`, declarable `BASES`
  (`D` alias), `UnitSystem` with transitive conversions returning a
  `LedgerLine`.
- `core/verdict.py` — `Status`, `Assumption`, `Verdict` (invariants enforced
  at construction), `LedgerLine`. `core/result.py` — `Unsupported`,
  `Blocked`, `Unverified`, `is_failure`.
- `core/intervals.py` — `Interval` (definition + mass required), `eti`,
  `hdi`, `summarize`. `core/stats.py` — Clopper–Pearson acceptance region,
  ESS (Geyer IPS), MCSE, z-score.
- `core/entities.py` — the vocabulary, `Population` with strata weights,
  `TimeWindow` with basis, `Intervention` with `version` (review B2).
- `core/protocols.py` — `SupportsPosterior`, `SupportsIntervention`,
  `Capability`, `PredictiveDraws`, `missing_capabilities`.
- `core/posterior.py` — `Posterior` with npz round-trip (re-exported from
  `infer`; see 0002.11).
- `data/` — `RoleMap`, `Panel` (validates, sorts, reports `Completeness`,
  canonical CSV hash), `ScalingParameters` / `fit_scaling`.
- `io/` — `Analysis` container, `save_analysis` / `load_analysis`
  (`analysis.axiom` directory, manifest records bases and units),
  `Provenance`, `ArtifactRegistry`.
- Gates 1 (relative threshold), 2 (layer table), 3 (token matching), 4
  (factory table + cross-process hash), 5, 8, 12. Gate 6 is constructive via
  `Interval`; 7, 9, 10, 11 wait for the expression tree and estimands.
- Notebooks: `nbs/core/01,02,04,05,06`, `nbs/data/01,02`, `nbs/io/01`,
  `nbs/infer/01`. `nbs/core/03-expression-tree` is reserved for 1b.

Things found while building, each recorded in 0002: pydantic converts a
`ValueError` raised inside a validator into `ValidationError`, so axiom's
typed errors subclass `Exception`, not `ValueError` (0002.12); pandas
`read_csv` needs `float_precision="round_trip"` to reproduce 17-digit floats
(0002.9); `Posterior` had to move below `io` (0002.11).

## Status

| Module | State | Notes |
|---|---|---|
| `core/spec.py` | done | |
| `core/dimensions.py` | done | |
| `core/verdict.py`, `core/result.py` | done | review A1 + E |
| `core/intervals.py` | done | golden case vs parent still to add when parent is available |
| `core/stats.py` | done | |
| `core/entities.py` | done | |
| `core/protocols.py` | partial | `SupportsForward` / `SupportsEstimands` wait for 1b (`Expr`, `Estimand`) |
| `core/posterior.py` (+ `infer` re-export) | done | |
| `data/{roles,frame,scale}.py` | done | |
| `io/{serialize,provenance,registry,analysis}.py` | done | zip single-file artifact (review E) not yet; directory only |
| gates 1, 2, 3, 4, 5, 8, 12 | done | 6 constructive; 7, 9, 10, 11 need 1b+ |
| `nbs/core`, `nbs/data`, `nbs/io`, `nbs/infer` | done for 1a | |

## Next (Phase 1b, new branch `feature/phase-1b-expression-tree`)

1. `core/expr.py` — the closed node set (with C1: `ODESystem` dimension-check
   only; C2: no general `Deriv`; A3: rational `Pow` on dimensioned bases).
2. `core/interpret/{value,dimension,latex}.py`; `SupportsForward`.
3. `estimands/spec.py` — facets, derived dimension, `transfer_to` →
   `TransferPlan` built on `core.verdict`.
4. Gates 10 and 11; `nbs/core/03-expression-tree`, `nbs/estimands/01,02`.
5. Plan-doc edits listed under "*Update:*" in 0002.
