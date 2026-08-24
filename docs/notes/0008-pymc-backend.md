# 0008 — PyMC as a third backend, and the PyTensor interpreter under it

**Kind**: decision. **Opened**: 2026-08-21. **Status**: landed on `develop`; part of the
1.1.0 surface. Closes "PyMC backend (D2)" on the 1.1 backlog in
[0001](0001-phase-1a-progress.md).

The ask was "use PyMC for Bayesian models — I like that it has multiple backends". Two
things had to be true for that to mean anything: PyMC had to run **the model axiom
already has**, not a second copy of it written in PyMC, and *multiple backends* had to be
a checkable claim rather than a list of names in a docstring.

## Part 1 — `core.interpret.pytensor`: the third interpreter

### 0008.1 — A backend is an interpreter, not a model definition

`ModelSpec` is an expression tree. `core.interpret` already walks it twice: `value` (numpy,
what Laplace and every closed-form path uses) and `jax` (what the NumPyro backend uses).
The obvious way to get PyMC is to write the model again with `pm.Normal(...)` calls — and
that is exactly the drift rule 3 exists to forbid. So PyMC gets a third interpreter,
`core.interpret.pytensor`, and the PyMC "model" is one `pm.Flat` per free parameter in
unconstrained space plus a single `pm.Potential` holding the compiled log density.

The consequence worth stating: **there is no PyMC-specific prior code**. `_log_prior_pt`,
`_log_constraint_pt` and `_likelihood_pt` are the same arithmetic as the numpy and jax
paths, transcribed into `pt.*`. Deliberately *not* `pymc.logp` — reaching for PyMC's own
distribution logps would mean the priors in a PyMC fit come from PyMC and the priors in a
Laplace fit come from axiom, which is two definitions of the model with one name.

`tests/unit/test_pymc_backend.py` pins the interpreter against `core.value` across 13
model configurations — nine kernels (including the polynomial, spline, piecewise-linear
and GP families from [0005](0005-basis-response-families.md) and
[0006](0006-gaussian-process-surfaces.md), which are what exercise `relu`, `step`, `sin`
and `cos`), three carryovers, all three intercept layouts, and a two-treatment model with
interactions and nuisance. Agreement is ≤ 2e-11, and the gradient matches jax's.

### 0008.2 — Two bugs the equivalence test caught that nothing else would have

**`pt.as_tensor_variable(0.0)` is float32.** On a *Python* float it ignores
`pytensor.config.floatX` and gives float32, and a float32 hyperparameter costs the log
density about eight digits — the first equivalence run was off by 2.4e-8, which is small
enough to look like ordinary floating-point noise and is not. Every scalar constant in
the module now goes through a `_f64` helper that builds the constant as an explicit
float64 numpy array. Agreement after that is exact where the arithmetic is exact.

**`Gather` must not ravel its index.** A hierarchical intercept's index is `(n_units, 1)`
so the gathered vector broadcasts down the time axis — that shape *is* the layout contract
in `surface.model`. Flattening the index (which reads as harmless normalization) silently
transposes the result; here it surfaced as `Incompatible Elemwise input shapes
[(1, 3), (3, 12)]`, but a model where the two axes happened to match would have produced
a wrong number instead of an error. Both fixes carry the reasoning in a comment at the
site, because both look like tidying-up when read cold.

### 0008.3 — A `Convolve` needs a static length, and says so

PyTensor will happily build a graph whose convolution length is only known at run time and
then fail somewhere unhelpful. `_static_length` raises a named `ValueError` at compile
time naming the node instead. This is rule 5 in its cheap form: the failure is typed and
attributable at the point where the information to explain it still exists.

## Part 2 — `infer.pymc_backend`: multiple samplers, and what "multiple" is worth

### 0008.4 — Installed is not usable, so the report distinguishes them

PyMC can dispatch NUTS to four implementations over the one `pm.Potential`: its own,
nutpie, numpyro and blackjax. `NUTS_SAMPLERS` names them; `samplers()` reports a
`SamplerStatus` per sampler by checking imports, which is fast and free.

That check is not sufficient, and the evidence arrived unprompted: blackjax 1.6.2 imports
perfectly against pymc 6.3.1 and then dies inside `build_kernel` with `cannot select an
axis to squeeze out`. Confirmed with plain PyMC and no axiom in the picture — it is a
version mismatch between two installed packages, not an axiom bug. A backend that answered
"blackjax: ready" there would be lying.

So `samplers(probe=True)` **runs two draws** from `x ~ N(0, 1)` through each sampler and
reports what came back. `SamplerStatus.usable` means it ran. The failure text is carried
verbatim, including the PyMC version, because that is the fact the user needs.

The catch around the probe is a named tuple, `_MISMATCH`, not `except Exception`
(banned by `tests/contracts/test_no_silent_degradation.py`, and correctly so). Anything
outside that list is a real bug and propagates out of the probe instead of being reported
to the user as somebody else's version skew.

### 0008.5 — The sampler is a constructor argument, not a `fit` keyword

`fit(..., backend=PymcBackend(nuts_sampler="nutpie"))` works because `fit` accepts a
`Backend` *instance* as well as a name. That is the seam for every backend-specific
option, and it is why no PyMC-shaped keyword appears in a generic signature. The
alternative — `fit(..., nuts_sampler=...)` — would put a word only one of three backends
understands into the door all three go through.

### 0008.6 — Per-chain initial values are conditional, and the provenance says which

All backends initialise at the Laplace mode. PyMC's own sampler takes a per-chain list of
initial points, so it gets jittered ones; nutpie refuses a list outright
(`NotImplementedError: nutpie does not support per-chain initvals`) and gets a single
dict. Rather than quietly dropping the jitter, `Posterior.provenance` carries
`init_jitter` and `init_per_chain`, so a run tells you which of the two it got. Rule 4:
the difference between two runs should be readable off the runs.

`cores=1` throughout. Forking a process that has already imported JAX deadlocks, and the
numpyro and blackjax dispatch paths have. Chains are cheap enough here that this is not
worth a platform-conditional.

### 0008.7 — `optimize` and `laplace` are not reimplemented

They delegate to `infer.laplace`. The mode search and the normal approximation are
numpy/scipy over the `value` interpreter and have nothing to do with which sampler is
installed. A backend is a sampler; making it three copies of a Newton solver would be
three places for the same bug.

## Part 3 — What the equivalence gate now asserts

`tests/contracts/test_backend_equivalence.py` (gate 9) compares posterior moments across
backends, and now across samplers: `test_pymc_and_numpyro_sample_the_same_posterior`, and
`test_every_usable_pymc_sampler_gives_the_same_posterior`, which enumerates
`samplers(probe=True)` and holds each usable one to 0.15 posterior sd of PyMC's own with
an sd ratio in `(0.85, 1.18)`. A sampler that is not usable in this environment is skipped
rather than silently passing.

Observed on the conjugate normal–normal at 1000 draws × 2 chains:

| route | disagreement with the closed form |
|---|---|
| pymc / nutpie | 0.094 sd |
| pymc / numpyro | 0.120 sd |
| numpyro backend | 0.050 sd |
| laplace | 0.472 sd |

The last row is not a defect: a normal approximation to a posterior that is not normal
disagrees, and it is the only row of the four where the disagreement is not Monte Carlo
noise. It is in the table because a reader comparing samplers should be able to see the
size of the thing samplers exist to avoid.

## Consequences

- `BACKEND_NAMES` is `("laplace", "numpyro", "pymc")`; `get_backend("pymc")` returns
  `Unsupported` naming the extra when it is not installed. Gate 1 still holds — `import
  axiom` puts neither `pymc` nor `pytensor` in `sys.modules`, and a subprocess test in
  `test_pymc_backend.py` pins that `pytensor_available()` itself does not import pytensor.
- `axiom[pymc]` now carries `nutpie`, and `axiom[all]` carries `axiom[pymc]`, so
  `uv sync --group dev` gives CI the three backends the equivalence gate compares.
  numpyro and blackjax dispatch ride along with `axiom[numpyro]`, which already has jax.
- `nbs/infer/01-backends.ipynb` gained a PyMC section demonstrating `PymcBackend`,
  `samplers`, `SamplerStatus`, `NUTS_SAMPLERS` and `NutsSampler`, including the
  same-posterior-across-samplers table above (gate 12).
