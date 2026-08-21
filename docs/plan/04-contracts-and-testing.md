# 04 — Contracts and testing

## The five kinds of test

| Directory | Marker | Question it answers | Runs in CI |
|---|---|---|---|
| `tests/unit/` | — | Does this function do what its docstring says? | always |
| `tests/contracts/` | — | Do the repo's design decisions still hold? | always, `make gates` |
| `tests/golden/` | `golden` | Does the port still reproduce `mmm-framework`? | always |
| `tests/recovery/` | `recovery` | Does the estimator recover known truth? | always; heavy cases `slow` |
| — | `slow` | Anything that samples for more than ~20 s | nightly + pre-release |

## The gates

Nine contract tests. Each one exists because a specific decision in this plan
would otherwise erode silently.

### 1. `test_import_weight.py`
`import axiom` must not put `jax`, `jaxlib`, `numpyro`, `pymc`, `pytensor`,
`arviz`, `plotly`, `cvxpy`, or `statsmodels` into `sys.modules`, and must
complete under 400 ms. Runs in a subprocess so it measures a cold import.

*Protects:* the dependency budget, which is the whole "lighter weight" premise.

### 2. `test_layering.py`
Walks the AST of every module and asserts the import graph respects the layer
diagram in `01-architecture.md`. `core` imports nothing from `axiom`; `infer`
imports only `core`; the four pillars do not import each other except
`design → surface` and `calibrate → estimands`.

*Protects:* the ability to keep the core light and the pillars separable. The
parent's `planning → reporting` and `garden → agents` edges are exactly the
cycles that made its subsetting hard.

### 3. `test_no_domain_vocabulary.py`
Greps identifiers (not comments, not docstrings) in `src/axiom/` outside
`adapters/` for `channel|spend|roas|roi|kpi|geo|dma|impression|creative|
media|campaign|brand`. Any hit fails.

*Protects:* the domain-general decision, which is otherwise re-litigated on
every port of a marketing-named function.

### 4. `test_spec_roundtrip.py`
Discovers every `Spec` subclass by walking `axiom`, builds one with a
hypothesis-style factory, and asserts JSON round-trip equality and hash
stability across two subprocesses with different `PYTHONHASHSEED`.

*Protects:* serializability, which the user named as a thing worth keeping.

### 5. `test_no_pickle.py`
AST-inspects `axiom.io` and asserts no import of `pickle`, `cloudpickle`, or
`dill`, and no `__reduce__` definitions anywhere in `src/axiom/`.

*Protects:* the "no executable state in a saved analysis" decision, and heads
off the parent's documented cross-environment cloudpickle failures.

### 6. `test_interval_provenance.py`
Every function returning an interval returns a type carrying
`interval_definition` and `hdi_prob`. Checked by walking return annotations and
by a runtime pass over the estimand registry.

*Protects:* design commitment #4. The parent shipped `contribution_roi` in two
places at two masses and two interval definitions before it noticed.

### 7. `test_capability_degradation.py`
For every `Capability`, construct a surface lacking it, request an estimand that
needs it, and assert `status="unsupported"` with a non-empty `reason` — never a
number, never an exception.

*Protects:* the typed replacement for the parent's `garden/contract.py` runtime
`inspect` checks.

### 8. `test_no_silent_degradation.py`
Fails on any `except Exception:` or bare `except:` whose handler does not either
re-raise, log at warning-or-above, or return a typed failure. Ruff's `BLE001` is
on for the same reason; this test catches the ones with a `# noqa`.

*Protects:* against the failure class the parent hit four separate times —
SLSQP non-convergence, NumPyro pickling, fingerprint memoization, and API
surface rot — each a broad `except` turning a hard failure into a wrong number.

### 9. `test_structural_uses_forward.py`
Asserts `design/structural.py` and `surface/model.py` both reach
`surface.forward()` and neither reimplements the transform chain.

*Protects:* the specific drift the parent documents in
`planning/identification.py`, where a numpy re-implementation "byte-mirrors" the
model and must be kept in sync by hand.

## Golden fixtures

`tests/golden/parent_values.json`:

```json
{
  "_meta": {
    "source_repo": "mmm-framework",
    "source_commit": "f1813bc",
    "captured": "2026-08-20",
    "python": "3.12",
    "numpy": "2.3.5"
  },
  "calibration.experiment::design_factor::case_01": {
    "args": {"...": "..."},
    "value": 0.7314,
    "rtol": 1e-10
  }
}
```

Rules:

- A golden test that fails is a **stop**, not a warning. Either the port is
  wrong, or the difference is deliberate — in which case it gets a note in
  `docs/notes/`, the fixture is updated with a `superseded_by` field, and the
  note is linked from the fixture.
- Tolerances are per-value and explicit. Anything ported as "bit-stable" gets
  `rtol=1e-12`; anything reimplemented gets a documented looser bound.
- The manifest is never regenerated wholesale. Regeneration requires the parent
  at the recorded commit.

## Recovery tests

The rule: **every estimator that returns a number gets a recovery test.** Build
the world in `axiom.sim` with known ground truth, run the estimator, assert it
lands. This is what makes a rewrite safe when the golden fixture cannot cover a
case.

Three tiers:

1. **Point recovery** — the estimate is within tolerance of truth. Fast.
2. **Interval coverage** — over N replications, the nominal 90% interval covers
   truth 88–92% of the time. Slow; nightly.
3. **Calibration** — SBC rank uniformity. Slow; pre-release.

Each recovery test must also have a **negative control**: a world where the
estimator *should* fail, asserted to fail. A coverage test that cannot fail is
not testing coverage. The parent's coverage diagnostics exist precisely because
a 90% interval covering 50% of the truth shipped unnoticed.

## CI matrix

| Job | Install | Runs | Trigger |
|---|---|---|---|
| `core` | `pip install -e .` | `tests/contracts`, `tests/unit`, `-m "not slow"` and skipping anything needing a backend | every push |
| `full` | `uv sync --group dev` | everything `not slow` | every push |
| `slow` | `uv sync --group dev` | `-m slow` | nightly + tags |
| `lint` | dev | `black --check`, `ruff`, `mypy --strict` | every push |

The `core` job is the load-bearing one. It is what proves the four-dependency
claim, and it is the job that fails when someone adds a top-level `import jax`.
