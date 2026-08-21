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

## D6 — Unit and numeraire enforcement strictness · **resolved 2026-08-20**

`Dose` carries units. The question was whether mismatched units raise, warn, or
are silently coerced. Strict raises catch real errors and also make every
notebook cell a fight.

**Resolution: three tiers, because three distinct things were being called
"units".**

| | On mismatch |
|---|---|
| **Dimension** — currency vs time vs outcome | raise, always. No flag disables it. |
| **Unit** — USD vs EUR, day vs week | convert automatically when a conversion is registered in the `UnitSystem`, and write a ledger line. Raise when none is registered. |
| **Scope** — this population, this window | never auto-resolved. Requires an explicit assumption through `TransferPlan`. |

An entity with no declared dimension warns once and is treated as dimensionless.
That leniency is for user code only: gate 10 still fails if anything shipped in
`src/axiom/` is undimensioned.

The decision grew in scope while being made. The base dimension set is
**declarable**, not fixed, and dimension checking is abstract interpretation
over `core.expr` at spec-construction time rather than a check at call time —
see `01-architecture.md`. The scope tier is what `00-charter.md` means by
"dimensions are necessary, not sufficient".

## D7 — Does `axiom` own a fitted-model concept at all? · **resolved 2026-08-21: yes**

Resolution: `axiom.io.analysis.Analysis`, a frozen container over
`(specs, panel, posterior, evidence, ledger, provenance)` that delegates and
holds no math. It lives in `io` rather than `core` because it holds a
`Panel` (see `docs/notes/0002-foundation-decisions.md` §2). Original text
kept below for the record.

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

## D9 — The reference quantity for log and logit transforms · blocks: nothing; Phase 1 ships (b) provisionally

`log(outcome)` is not dimensionally well-formed; `log(y / y_ref)` is. People fit
log-space models constantly, and a checker that rejects them is a checker
everyone routes around — which is worse than not having one, because it converts
a hard failure into an unexamined `Opaque` node.

Options: **(a)** raise, and require an explicit `y_ref` on every
`Apply(log, ...)`; **(b)** synthesize `y_ref = 1 [unit of y]`, warn once, and
record the synthesized reference on the spec so it serializes and shows up in
the ledger; **(c)** treat a log-space outcome as its own declared dimension and
convert at the boundary.

**Position:** ship (b) in Phase 1, matching D6's temperament. But this is the
one item in this document that should be settled by trying it rather than by
argument. The failure mode of (b) is that the synthesized reference is
unit-dependent, so a model fit in one currency and the same model fit in another
have different intercepts and nothing complains — precisely the class of error
the dimension system exists to catch, reintroduced by the concession that makes
it usable. Whether that is tolerable depends on how often the intercept is the
quantity being transferred. Revisit at the end of Phase 3 with the surface
notebook in hand.
