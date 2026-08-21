# 05 — Open decisions

Decisions that are *not* made yet, with the shape of each choice and what it
blocks. Record the resolution here when it lands.

## D1 — Licensing and distribution · blocks: Phase 9

`LICENSE` currently says proprietary, inherited from the parent's August 2026
relicensing. That is the conservative default, not a decision.

The porting question is real: material ported from `mmm-framework` *before* the
relicensing was Apache-2.0 and remains available under those terms in
`mmm-framework` ≤ 1.5.0 on PyPI. Material added after is proprietary. Since the
copyright holder is the same person for both, this is a choice rather than a
constraint — but the choice should be made deliberately and recorded, not
inferred from a file that was copied.

Options: proprietary and private (current default) · Apache-2.0 and public ·
source-available with a non-compete (BUSL/PolyForm) · split, with `core`,
`identify`, `design`, and `meta` open and `calibrate` and `surface` held back.

**Blocks:** whether Phase 9 publishes to PyPI, and whether the docs may link the
repository.

## D2 — Default inference backend on non-JAX platforms · blocks: Phase 3

NumPyro is the default. On platforms where `jaxlib` is awkward (some ARM Linux
images, some locked-down environments) that is a hard stop, and the fallback is
`[pymc]` — which needs a C++ toolchain.

Third option worth pricing before Phase 3 closes: ship a small hand-written NUTS
+ Laplace on numpy/scipy for the models `axiom` actually fits, which are low-
dimensional and smooth. Most of the value here is Laplace and SVI, not
1000-parameter NUTS. That would make `axiom` genuinely sampler-free end to end,
at the cost of owning a sampler.

**Recommendation:** do not decide now. Build `infer/backend.py` so a third
implementation is a file, and revisit at the end of Phase 3 with a measurement
of what the surface and meta models actually need.

## D3 — Non-parametric surfaces · blocks: nothing; shapes Phase 3 API

GPs and BART would let `surface` handle shapes the parametric families cannot.
They also change the design math (D-optimality on a GP is not the same object)
and the serialization story.

**Position:** design `surface/kernels.py` around a `ResponseKernel` protocol so a
GP is expressible, ship only parametric families in 1.0, and put the GP question
on the 1.1 list.

## D4 — Does the marketing adapter live in this repo? · blocks: Phase 8

Keeping it here makes the port easier and gives the golden tests a home. Moving
it to a separate `axiom-marketing` package makes the domain-general claim
literal rather than test-enforced.

**Position:** keep it in-repo through 1.0 so the gates have something to check
against, and re-evaluate at 1.1. If it stays, it must never become the primary
documented entry point.

## D5 — How much of `diagnose/backtest.py` survives · blocks: Phase 8

Rolling-origin backtesting is a *predictive* validation tool in a repo whose
charter is causal. It earns its place if the forecast is a decision input; it
does not if it is only ever a credibility exhibit.

**Position:** port the ~500-line harness, mark it clearly as predictive
validation and not causal validation, and let usage decide by 1.1.

## D6 — Unit and numeraire enforcement strictness · blocks: Phase 1

`Dose` carries units. The question is whether mismatched units raise, warn, or
are silently coerced. Strict raises catch real errors and also make every
notebook cell a fight.

**Position:** raise on a *dimension* mismatch (dose vs outcome vs currency),
convert automatically within a dimension when a conversion is registered, and
warn once when a `Dose` has no declared unit at all.

## D7 — Does `axiom` own a fitted-model concept at all? · blocks: Phase 4

Currently no: there is a surface spec, a posterior, and protocols. A user
holding "the fitted thing" holds a tuple. That is clean and slightly awkward.

A thin `Analysis` object bundling `(graph, surface, posterior, evidence,
ledger)` — a container, not a model class — would make the notebook story much
better without reintroducing `BayesianMMM`.

**Position:** add `core/analysis.py` as a frozen container in Phase 4 if the
Phase 3 notebook feels clumsy. Guard it with a rule: it may hold state and
delegate, and it may never contain math.

## D8 — Python floor · blocks: Phase 0

3.12 inherited from the parent. 3.13 is out and 3.14 is near. Nothing here needs
3.13 features; a lower floor of 3.11 would widen reach at the cost of some
typing syntax.

**Position:** stay at 3.12, test on 3.12 and 3.13 in CI.
