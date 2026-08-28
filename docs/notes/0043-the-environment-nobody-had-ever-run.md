# 0043 — The environment nobody had ever run

*Kind: decision. Opened 2026-08-28. Status: implemented. Closes the CI red that
had stood since [0038](0038-a-dose-is-not-a-switch.md) landed and that every
release since 1.1.3 shipped through.*

CI had been failing on `develop` and on `main` for weeks while `make fast_tests`,
`make gates` and `make lint` were green on every developer machine. The 1.2.0 and
1.3.0 releases both merged red. That gap is the subject of this note: not any one
of the seventeen failures, but the reason a green local run said nothing about CI.

## D43.1 — the lockfile was ignored, so CI resolved a different project

`uv.lock` was in `.gitignore`. Nothing in the repository named a dependency
version, so `uv sync --group dev` in CI re-resolved the whole graph from PyPI on
every run — a resolution no one had executed before and no one could reproduce
afterwards.

The resolution CI landed on, on 2026-08-28:

| | local | CI |
|---|---|---|
| numba | 0.66.0 | **0.63.0b1** |
| numpy | 2.4.6 | **2.5.2** |

numba 0.66.0 requires `numpy<2.5`. numba 0.63.0b1 — a *pre-release* — carries no
upper bound at all. uv is permitted to take a pre-release when no release
satisfies the branch it is exploring, so the resolver walked to numpy 2.5.2 and
then had to reach back for a numba that would tolerate it. numpy 2.5 removed
`np.row_stack`, which pytensor 3.3.0 still calls, so every one of the thirteen
`test_pymc_backend` tests died on `AttributeError` inside a dependency. The
remaining mypy errors were the same cause wearing different clothes: numpy 2.5's
stubs, on a numpy nobody had installed.

**`uv.lock` is committed and every CI job that installs the dev group installs
`--locked`.** The lockfile is a source artifact: it is the only thing that makes
"green on my machine" a claim about CI at all.

## D43.2 — one job still resolves freely, on purpose

The `core` job keeps `uv pip install -e .` with no lock. Its whole reason to
exist is the four-dependency claim — that `pip install axiom` gives a working
toolkit — and a `pip install` a user runs tomorrow *is* a fresh resolution. That
job is the canary. It is the one place drift should be visible, and the note
above is what to read when it goes off.

## D43.3 — the canary had been silent for a different reason

The `core` job had not failed a test in weeks; it had failed to *install* for
weeks. `uv pip install --system` targets the runner's Debian Python, which is
PEP 668 externally-managed, and uv refuses it outright. The job died before
collection, and a job that never runs its tests looks exactly like a job whose
tests pass. It now gets an interpreter of its own via `actions/setup-python`,
the same way the `docs` job already did.

What that exposed, once it could run: `test_funnel_without_a_mode_is_unverified_not_nan`
asserted `find_mode(...).converged is False`, which presumes jax. Without jax the
finite-difference curvature check cannot resolve `a_sd` and `find_mode` returns
`Unverified` — the documented behaviour, and the same verdict by a different
route. The test now accepts either, because "there is no mode here" is the
invariant and which object carries the verdict is an artifact of the install.

## D43.4 — two assertions that were measuring the runner

Independent of any of the above, and worth separating from it:

- `test_evaluate` compared a quadrature sum against `80.0` with `==`. It is a
  sum over nodes; it lands on `80.00000000000001` as readily, and which one you
  get is a property of the numpy build. Every sibling assertion on that key
  already used `approx`. This one now does too.
- `test_import_weight` took a single timing sample under `pytest -n logical`,
  with every core on the runner busy. Scheduler noise there is strictly
  additive, so one unlucky probe can be twice the honest cost — and was
  (1.510s against a 1.313s budget). It now takes the minimum of three probes,
  for the reason `timeit` reports a minimum rather than a mean. The budget is
  unchanged; the measurement is the thing that was wrong.

## What this does not do

The five mypy errors that only appear under numpy 2.5's stubs are not fixed;
they are pinned away from. `np.row_stack` is pytensor's call, not ours, and
pytensor 3.3.0 is the current release — there is no version of this project that
works with numpy 2.5 today. When pytensor ships the fix, unpinning is a
deliberate act with its own note, which is the point of having a lock at all.
