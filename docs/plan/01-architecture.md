# 01 — Architecture

## Package map

```
src/axiom/
├── core/            vocabulary, protocols, specs, intervals      [no heavy deps]
├── data/            role-tagged Panel, scaling                   [no heavy deps]
├── identify/        DAGs, adjustment sets, IV/front-door, verdicts, estimators
├── estimands/       declarative counterfactual quantities + realization
├── surface/         dose-response kernels, carryover, RSM designs, optimization
├── design/          power, EIG, EVOI, methods registry, simulation, portfolio
├── calibrate/       evidence records, prior route, likelihood route, transfer, ledger
├── meta/            random-effects pooling, moderators, bias, priors, privacy
├── infer/           Backend protocol; NumPyro impl, PyMC impl, Laplace, convergence
├── diagnose/        SBC, coverage, weak-id, sensitivity, learning, spec curve, refute
├── build/           fluent builders over every spec
├── io/              serialize, provenance, artifact registry
├── sim/             DGPs with causal ground truth
├── adapters/        marketing.py (channel/spend/geo/KPI over the general core)
└── viz/             optional plotly figures
```

### The dependency rule

Arrows point *down only*. A module may import from any layer below it and never
from one above.

```
  viz    adapters                          (leaves; nothing imports them)
   |        |
  build   diagnose                         (compose everything below)
   |        |
  meta  calibrate  design                  (the four pillars; peers, no cross-imports
   |        |        |                      except design <- surface, calibrate <- estimands)
   +--------+--------+
            |
        surface   estimands   identify     (domain layer)
            |         |          |
            +---------+----------+
                      |
                    infer                  (sampler seam; optional deps live here)
                      |
                  core    data    io       (foundation; numpy/scipy/pandas/pydantic)
```

`tests/contracts/test_layering.py` enforces this by walking the import graph.

## Core vocabulary

`axiom.core.entities` replaces the parent's marketing nouns:

| mmm-framework | axiom | Meaning |
|---|---|---|
| channel | `Treatment` | The thing you can intervene on |
| spend / impressions | `Dose` | The intervention's magnitude, with units |
| geo / DMA | `Unit` | The randomization / observation unit |
| KPI / sales | `Outcome` | What you are trying to move |
| control variable | `Covariate` | Measured, not intervened on |
| panel | `Panel` | Units × time × (treatments, covariates, outcome) |
| ROAS | `Outcome` per `Dose` unit | An estimand, not a builtin |

`Dose` carries units and a `numeraire` so cost-per-outcome and value-of-
information arithmetic is unit-checked rather than by convention. The parent
learned this the hard way: it has a whole `finance/` package existing mostly to
answer "what currency is this number in".

## The seam: protocols instead of a model class

`axiom.core.protocols` is the entire contract the upper layers depend on. It
replaces `mmm_framework.model.base.BayesianMMM` (5,375 lines) as the thing
`estimands`, `calibrate`, `design`, and `diagnose` are written against.

```python
@runtime_checkable
class SupportsPosterior(Protocol):
    """Anything with draws you can ask questions of."""
    def draws(self, name: str) -> np.ndarray: ...          # (chain, draw, *shape)
    def names(self) -> frozenset[str]: ...
    def coords(self) -> Mapping[str, Sequence[Any]]: ...
    def n_draws(self) -> int: ...

@runtime_checkable
class SupportsIntervention(Protocol):
    """Anything you can ask a counterfactual of."""
    treatments: Sequence[Treatment]
    def predict_under(self, iv: Intervention, window: TimeWindow | None = None,
                      seed: int | None = None) -> PredictiveDraws: ...
    def capabilities(self) -> frozenset[Capability]: ...

@runtime_checkable
class SupportsEstimands(SupportsPosterior, SupportsIntervention, Protocol):
    declared_estimands: Sequence[Estimand]
    def evaluate(self, es: Sequence[Estimand] | None = None
                 ) -> dict[str, EstimandResult]: ...

@runtime_checkable
class SupportsForward(Protocol):
    """A deterministic response surface: the design layer's dependency."""
    def forward(self, dose: np.ndarray, theta: Mapping[str, np.ndarray]
                ) -> np.ndarray: ...
    def linearize(self, dose: np.ndarray, theta_at: Mapping[str, np.ndarray]
                  ) -> DesignMatrix: ...
```

Three things this buys that the parent does not have:

1. **The design layer never needs a fitted model.** `design.eig`,
   `design.power`, `design.structural` need a `SupportsForward` and a prior —
   both plain data. In the parent, `planning/identification.py` byte-mirrors
   `BayesianMMM.sample_channel_contributions` in numpy to avoid importing PyMC,
   and a comment admits the two must be kept in sync by hand. Here there is one
   `forward()` and both the likelihood and the design math call it.
2. **Capability degradation is typed.** The parent's `garden/contract.py` already
   invented this (`REQUIRED_ATTRS`, `RECOMMENDED_METHODS`, "absent ones must
   degrade gracefully") but enforced it with `inspect` at runtime. Here it is a
   `frozenset[Capability]` on the protocol and estimand realization returns
   `status="unsupported"` with a reason, checked statically.
3. **A posterior from anywhere works.** A NumPyro fit, a PyMC fit, a Laplace
   approximation, a stored netCDF, or a hand-built dict of draws all satisfy
   `SupportsPosterior`. The parent could only realize an estimand against a live
   `BayesianMMM`.

## Specs: frozen, versioned, hashed

Every configuration object in `axiom` inherits `axiom.core.spec.Spec`:

```python
class Spec(BaseModel, frozen=True):
    SCHEMA_VERSION: ClassVar[str]
    def content_hash(self) -> str: ...        # blake2b over canonical JSON
    def diff(self, other: Self) -> SpecDiff: ...
    def to_json(self) -> str: ...
    @classmethod
    def from_json(cls, s: str) -> Self: ...
```

This generalizes three things the parent has in separate places and re-derives
each time: `config/spec_diff.py`, the estimand schema version, and the graph
fingerprint engine. One base class, one hash, one diff, and the provenance
record in `axiom.io.provenance` is just the set of content hashes an analysis
touched.

## Serialization: no pickle in the path

The parent serializes with `cloudpickle` and has a documented failure mode
("ensure cloudpickle versions match across environments") plus a NumPyro pickle
bug in its history. `axiom.io.serialize` writes:

```
analysis.axiom/
├── manifest.json        format version, content hashes, axiom version, timestamps
├── specs/               one JSON per Spec (graph, surface, estimands, design, ...)
├── posterior.nc         netCDF via [netcdf] extra, or posterior.npz without it
└── evidence/            calibration records + assumption ledger, JSONL
```

Loading replays the specs and rehydrates the posterior. No executable state is
stored, so a load cannot execute arbitrary code and a version bump cannot
silently change what a saved analysis means. The cost is that a bespoke
user-written surface must be re-importable by qualified name — the same trade
the parent's Model Garden already makes with `model_class_qualname`.

## Inference backends

`axiom.infer.Backend` is a small protocol:

```python
class Backend(Protocol):
    name: str
    def sample(self, model: ModelFn, data: Mapping, *, draws: int, tune: int,
               chains: int, seed: int) -> Posterior: ...
    def optimize(self, model: ModelFn, data: Mapping, *, seed: int) -> PointEstimate: ...
    def laplace(self, model: ModelFn, data: Mapping, *, draws: int,
                seed: int) -> Posterior: ...
```

Two implementations. `numpyro_backend` is the default. `pymc_backend` exists for
models NumPyro serves badly.

**Port the Laplace fix.** The parent's branch `laplace-covariance-and-hierarchy-config`
(commits `5f0aef6`, `f1813bc`) fixes a real bug: taking the covariance from
BFGS's `hess_inv` instead of real curvature produced `inf` draws, which showed up
downstream as ROI exactly 0.0 and NaN intervals. The fix optimizes with
`trust-ncg` and takes the Hessian at the mode. `axiom.infer.laplace` must ship
that behavior from its first commit, along with the `nonfinite_draw_frac`
diagnostic that made the bug visible.

## What the marketing adapter is

`axiom.adapters.marketing` is aliases and presets, target ~400 lines:

```python
Channel = Treatment
Spend   = Dose(units="currency", numeraire="USD")

ROAS  = Estimand(name="roas",  quantity=Contrast(...), ...)
MROAS = Estimand(name="mroas", quantity=MarginalDose(...), ...)
CPA   = Estimand(name="cpa",   ...)

def panel_from_mff(path, config) -> Panel: ...   # the one MFF concession
```

`tests/contracts/test_no_domain_vocabulary.py` greps `src/axiom/` outside
`adapters/` for `channel|spend|roas|roi|kpi|geo|dma|impression|creative` in
identifiers and fails on a hit. Docstring prose may use them as *examples*; the
gate checks code, not comments.
