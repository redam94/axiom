# 0002 — Decisions taken while building the foundation

Each entry: what was decided, why, and which plan file should be updated to
match. Numbered per entry so later notes can cite `0002.3`.

## 0002.1 — Foundation layer order is `core < data < io`

`01-architecture.md` draws `core data io` on one row and says "core imports
nothing from axiom". `io` must import both `core` (specs) and `data`
(`Panel`), so the row has an order. `data` may import `core`; `io` may import
`core` and `data`; `infer` may import all three. `test_layering.py` encodes
this. *Update:* `01-architecture.md` diagram caption.

## 0002.2 — `Analysis` lives in `io`, not `core`

Review D1 resolves D7 as "yes, `core/analysis.py`". But an `Analysis` holds a
`Panel`, and `core` may not import `data`. The container is
`axiom.io.analysis.Analysis`: frozen, holds `(specs, panel, posterior,
evidence, ledger, provenance)`, delegates, and contains no math. The rule from
D7 stands; only the home moved. *Update:* `05-open-decisions.md` D7.

## 0002.3 — Measurement units are strings; `Unit` is the observation unit

The vocabulary table reserves `Unit` for the randomization / observation unit
(geo, patient, plot). A unit of measure (USD, day) is therefore a plain `str`
field named `unit` on entities and on `Dimension`-aware columns, and the
`UnitSystem` maps unit strings to base dimensions and registers conversions.
There is no `Unit`-of-measure class. This keeps the one confusing name out of
the API rather than in it.

## 0002.4 — `Interval` fields are `definition` and `mass`

`04-contracts-and-testing.md` gate 6 speaks of `interval_definition` and
`hdi_prob`. The shipped type is `Interval(lower, upper, definition, mass)`
with both required and no defaults (review C8: the gate is constructive — a
provenance-less interval cannot be built). `hdi_prob` was the parent's name
and is wrong for an ETI; `mass` is definition-neutral. The runtime half of
gate 6 (every result carries an `Interval`) lands with the estimand registry
in 1b/Phase 4. *Update:* gate 6 wording.

## 0002.5 — Serialized form and hash canon

`Spec.to_json()` writes `{"spec": "<module>:<Class>", "schema_version": "..",
"data": {...}}`. `content_hash()` is blake2b-256 over
`json.dumps(envelope, sort_keys=True, separators=(",", ":"), ensure_ascii=False)`
with the `data` produced by pydantic's JSON mode (so floats canonicalize via
`repr`, `Fraction` as `"p/q"` strings, enums as values). Specs hold no arrays;
anything array-valued is referenced by hash (the `Panel`'s, the posterior's).
Loading a spec whose `schema_version` differs from the class's
`SCHEMA_VERSION` applies a registered migration or raises
`SchemaVersionError` naming both versions — never coerces (review E).

## 0002.6 — Declared base dimensions travel with the analysis

`BaseRegistry` is process-global and mutable by design (a domain declares
its bases). A spec naming an undeclared base fails construction, including
on load. So `io` records the registry's non-default declarations in
`manifest.json` and re-declares them before replaying specs. A stored
analysis is therefore self-describing about its dimension bases.

## 0002.7 — Unit conversion is a ledger line, not an `Assumption`

D6 says a registered unit conversion is automatic "with a ledger line". The
line is a `LedgerLine(kind="unit_conversion", ...)` in `core.verdict`, which
is the same record type the Phase 6 ledger appends. An `Assumption` is
reserved for things that can be false; a registered conversion factor is a
fact of the unit system. A ledger line may carry an `Assumption` or not.

## 0002.8 — Seed contract

Public functions take `seed: int | None`; internals take
`numpy.random.Generator`. `Posterior` and every sampled result record the
seed in their provenance. (Review E; adopted as written.)

## 0002.9 — Panel storage format

`analysis.axiom/panel.csv` with `float_format="%.17g"` plus dtypes in the
manifest, because `pyarrow` is not in the core budget and CSV at 17
significant digits round-trips IEEE doubles exactly. Revisit if panels grow
past what CSV handles comfortably; the manifest records the format so a
second format can be added without a version break.

## 0002.10 — `Panel` reports completeness, never imputes

Review D5 adopted. `Panel.completeness()` returns a typed `Completeness`
spec (balanced or not, missing cells, gaps per unit). Nothing in `data/`
fills a gap.

## 0002.11 — `Posterior` is defined in `core`, re-exported from `infer`

The roadmap places the plain `Posterior` in `infer/posterior.py`, and
`io.serialize` writes `posterior.npz`. But `infer` sits above `io` in the
layer diagram, so `io` cannot import it. The sampler-free `Posterior` is a
dict of arrays implementing `SupportsPosterior` — foundation material — so it
lives in `core/posterior.py`; `axiom.infer.Posterior` is the same class.
Backends (Phase 3) stay in `infer` and *produce* a `core.Posterior`.
*Update:* `01-architecture.md` package map, `03-roadmap.md` Phase 1.

## 0002.12 — axiom's typed errors subclass `Exception`, not `ValueError`

Pydantic catches `ValueError` (and `AssertionError`) raised inside a
validator and re-raises it as `ValidationError`, which would erase
`DimensionError` / `UndeclaredBaseError` the moment they were raised during
spec construction — exactly where they are raised. So `SpecError` and its
subclasses derive from `Exception` and propagate untouched. Plain
`ValueError` is still used for ordinary field validation (a negative mass, a
backwards window) where pydantic's `ValidationError` is the right type.

## 0002.13 — Composition over inheritance, protocols over ABCs (standing rule)

Set by Matthew 2026-08-21 and recorded in CLAUDE.md "Style". Applied
retroactively: `Entity` is now a `Protocol` and the five entity specs are
flat classes sharing `EntityName` and one validator function;
`Unsupported`/`Blocked`/`Unverified` are flat classes sharing `NonEmptyStr`
(which `Assumption` and `LedgerLine` also use). The only inheritance left is
`X(Spec)`, which pydantic requires, and exception hierarchies. A unit test
asserts the entity classes' MRO is ``[cls, Spec, BaseModel, ...]``.
