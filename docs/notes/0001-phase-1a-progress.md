# 0001 — Implementation progress log

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

### 2026-08-21 — Phase 1b complete on `feature/phase-1b-expression-tree`

- `core/expr.py` — `Const`, `Data`, `Param` (+`Prior`), `Add`, `Mul`, `Div`,
  `Pow`, `Apply`, `Convolve`, `Link`, `Opaque`, `Equation`, `System`,
  `ODESystem`; `Expr` is a discriminated union (no node base class);
  `children` / `walk` / `params` / `data_names` / `node_path`.
- `core/interpret/{dimension,value,latex}.py` — dispatch-table interpreters.
  `dimension` raises naming the node path; `value` is `forward()`; `latex`
  degrades to `Unsupported` on `Opaque`.
- `core/protocols.py` — `SupportsForward` (`expr` + `forward`; `linearize`
  joins in Phase 3).
- `estimands/spec.py` — `Quantity`, `Level`, `Estimand` (eight facets, all
  required, derived dimension asserted), `FacetDiff`, `TransferPlan`,
  `transfer_to` with a rule per facet (`_RULES`, asserted complete against
  `FACETS`).
- Gates 10 and 11 (11 parametrized over facets *and* the B2 sub-fields);
  unit tests; `nbs/core/03-expression-tree`, `nbs/estimands/01,02`.
- 196 tests pass; 12 notebooks execute; gate 12 green for `core`, `data`,
  `io`, `infer`, `estimands`.

All seven Phase 1 exit criteria hold (criterion 3's "pickle absent" is gate
5; criterion 5 is `test_checker_and_evaluator_agree_on_200_random_trees`).

## Deferred from Phase 1 (tracked)

- Golden case for `intervals` against the parent — parent not available.
- Single-file `.axiom` zip artifact (review E) — directory format only.
- `SupportsEstimands`, `EstimandResult`, realization — Phase 4 by design.

### 2026-08-21 — Phase 2 (`identify`) complete on `feature/phase-2-identify`

Built with a four-agent parallel workflow (backdoor · front-door/IV ·
transport · sim+estimators), each followed by an independent skeptical
review against the literature and random-graph oracles, then a fix round
verified by fresh agents. Landed:

- `identify/graph.py` — `CausalGraph` (Spec; bidirected edges as hidden
  common causes; `unmeasured`, `selection`, `feedback`), Bayes-ball
  d-separation, `G_{x̲}` / `G_{x̄}` surgery, `from_edges` parser.
- `identify/backdoor.py` — back-door criterion, enumeration, canonical set
  and polynomial existence test (van der Zander et al. 2014), roles.
- `identify/frontdoor.py` — front-door criterion, (conditional) instruments.
- `identify/transport.py` — selection diagrams, direct / S-admissible /
  trivial transportability (B&P 2014 Defs 6–8, Thm 2), `TransportVerdict`.
- `identify/verdict.py` — `identify()` with route preference, alternatives,
  typed assumptions per route, B4 feedback downgrade.
- `identify/estimators.py` — OLS, 2SLS, linear front-door (multivariate
  first stage for the delta-method SE); `identify/endogeneity.py` — DWH and
  Hausman, degenerate branches return `Unverified`.
- `sim/scm.py`, `sim/worlds.py` — `LinearSCM` composing a `CausalGraph`,
  seven named worlds with exact truths and interventional means.
- Recovery tests (`tests/recovery/test_identify_recovery.py`,
  `test_transport_recovery.py`) with negative controls; 507 tests total.
- Notebooks `nbs/identify/01–03`, `nbs/sim/01`.

Reviewer-found defects fixed before merge: transport `blocked` overclaimed
(B&P's own Examples 4/8 are transportable) → trivial route + `unsupported`
fall-through; status convention unified (0002.16); multi-mediator front-door
SE understated by ~12% → multivariate first stage; outcome allowed on the
right-hand side → refused; minimal-set search capped on the wrong candidate
pool → canonical-set enumeration.

**Deferred from Phase 2:** the ~25-DAG golden verdict corpus (parent not
available — `_pending` in the fixture); `identify/narrative.py` dropped per
review C5; sID recursion (Thm 3) not implemented — the verdict says so.
Exit criterion 1 (golden verdicts) is therefore open until the parent is
captured; criteria 2–5 hold.

## Next (Phase 3 — `surface` deterministic half + `infer`)

1. `core/expr.py` — the closed node set (with C1: `ODESystem` dimension-check
   only; C2: no general `Deriv`; A3: rational `Pow` on dimensioned bases).
2. `core/interpret/{value,dimension,latex}.py`; `SupportsForward`.
3. `estimands/spec.py` — facets, derived dimension, `transfer_to` →
   `TransferPlan` built on `core.verdict`.
4. Gates 10 and 11; `nbs/core/03-expression-tree`, `nbs/estimands/01,02`.
5. Plan-doc edits listed under "*Update:*" in 0002.
