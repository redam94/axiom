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
- `test_import_weight` was wrong twice over. It took a single timing sample
  under `pytest -n logical`, with every core on the runner busy; scheduler
  noise there is strictly additive, so one unlucky probe can be twice the
  honest cost. It now takes the minimum of three, for the reason `timeit`
  reports a minimum rather than a mean. The second fault is the interesting
  one. The budget was `t_pandas + 0.6`, a *constant* allowance whose own
  comment called it "generous for slow CI runners" when it is precisely the
  opposite: a slower runner inflates `t_axiom` while the allowance stays put,
  so the gate tightens exactly where it promised to relax. Three measurements,
  once it was possible to take them all:

  | | pandas | axiom | ratio | difference |
  |---|---|---|---|---|
  | laptop, dev venv | 0.165s | 0.428s | 2.60x | +0.26s |
  | runner, dev venv | 0.615s | 1.384s | 2.25x | +0.77s |
  | runner, core only | 0.179s | 0.546s | 3.06x | +0.37s |
  | runner, core only | 0.260s | 0.833s | 3.20x | +0.57s |

  The last two rows are the same job on two different days. GitHub does not
  hand you the same machine twice, so neither column is reproducible to better
  than about a third of itself, which is the fact the old bound was built
  without.

  Neither form alone covers those four. A constant allowance tightens where the
  machine is slow, and row two failed `+0.6`; a pure ratio tightens where pandas
  is cheap, and rows three and four failed `3.0x`. The cost has a fixed part and
  a part that scales, so the bound is the more generous of the two and both have
  to be exceeded before it fails, at `4.0x` or `+1.0s`. Those margins are wide
  on purpose. This is a coarse regression guard, not a benchmark, and the
  assertion that actually holds the dependency budget is
  `test_no_heavy_modules_imported` beside it -- that one is exact and names the
  offender. A top-level `import plotly` or `import pymc` moves this one by whole
  multiples; nothing else should move it at all.

## D43.5 — three tests that were asserting the platform

Pinning numpy moved the Laplace optimizer onto a different floating-point path
on Linux, and three tests turned out to have been asserting the route rather
than the destination. All three concern the same degenerate funnel, where
`a_sd` is driven to zero and there is no mode to find.

- `scipy.optimize`'s `trust-exact` raised `UnboundLocalError` from inside
  `IterativeSubproblem.solve`. The step `p` is bound only on a *successful*
  Cholesky factorization, so a Hessian indefinite enough that every one of
  `maxiter` iterations lands in the unsuccessful branch falls out of the loop
  and returns a name that was never assigned. `_find_mode` already catches
  `ValueError`, `LinAlgError` and `FloatingPointError` from these solvers,
  under a comment saying scipy raises where it should report failure. This is
  that, wearing the one disguise indistinguishable from a bug. It joins the
  tuple, is logged, and the next optimizer gets its turn.
- `test_funnel_without_a_mode_is_unverified_not_nan` asserted
  `detail["converged"] == "False"`, and then `not find_mode(...).converged`.
  Whether the optimizer stops just short of the neck or settles into it on
  curvature that is positive *semi*-definite and no better is a floating-point
  property of the platform, and `converged` is the field that records which.
  What is true everywhere, and what actually makes the mode unusable, is that
  nowhere in this funnel is the curvature positive definite. Both assertions
  now read `hessian_pd`.
- `test_init_falls_back_to_zeros_when_the_mode_is_unverified` required the
  literal string `"mode search"` in the init note. `_init_z` has three ways to
  say a mode is unusable and only two of them start that way; the funnel took
  the third on Linux. The note must explain itself and name the optimizer. It
  does not have to pick which of the three truths about this mode to tell.

A fourth belongs with them, found once the rest were green.
`test_gradient_below_round_off_resolution_is_unsupported` asserted that the
unresolvable first differences were reported as exactly `"0, 0"`. Whether a
change of ~1e-6 on a value of size 1e12 rounds to zero or to one ulp is decided
by the SIMD kernel numpy dispatches to at runtime, and GitHub's runners are not
all the same CPU: the test passed or failed by which machine picked up the job.
It now asserts what makes those coordinates unresolved in the first place --
each difference sits below `_RESOLUTION` times its own round-off floor -- which
is true whichever way the last bit falls.

None of the four was a wrong number. Each was a test that had written down one
platform's route to a verdict and called it the verdict.

## What this does not do

The five mypy errors that only appear under numpy 2.5's stubs are not fixed;
they are pinned away from. `np.row_stack` is pytensor's call, not ours, and
pytensor 3.3.0 is the current release — there is no version of this project that
works with numpy 2.5 today. When pytensor ships the fix, unpinning is a
deliberate act with its own note, which is the point of having a lock at all.
