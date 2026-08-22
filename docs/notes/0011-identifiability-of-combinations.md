# 0011 — Identifiability of nonlinear parameters: which combination, not whether

*Kind: decision. Opened 2026-08-22. Status: implemented on `feature/dynamics-and-identifiability`; part of the 1.1.0 surface.*

## The problem

axiom already had two identifiability instruments and neither answers the question that
actually comes up.

* `design.structural.fisher_information` / `identifiability_ridge` — the Fisher information
  of a response surface at a design, its flattest eigenvector, and pairwise posterior
  correlations. It answers *how precisely*, presuming each parameter is pinned down at all,
  and it reports the flat direction as an unnamed numerical eigenvector.
* `diagnose.weak_id` — posterior correlation, condition number, prior-to-posterior sd ratio,
  bound saturation. All computed **after** fitting, from draws already in hand.

Neither says the thing a modeller needs to hear before running an experiment: *α alone is
not estimable from this design; α/k is, to within 7 %.* That sentence is a different kind of
statement — about a **combination** — and it needs a different instrument.

## The idea

If scaling several parameters together leaves every prediction unchanged, the likelihood is
flat along that direction and no amount of data moves it. In **log**-parameter coordinates
such a scaling is a *constant* direction, which turns the search for it into linear algebra:

* build the sensitivity matrix `S_ij = ∂f_i / ∂log θ_j` over every observation the design
  would take;
* its null space is the set of scalings that change no prediction — the **symmetries**;
* its row space is what the design **can** see;
* an integer vector `w` in either space names a monomial `∏ θ_j^{w_j}`. `(1, −1)` is `α/k`;
  `(1, 1)` is `α·k`. Those are the things to report, and to put priors on.

This is the scaling-symmetry half of the structural-identifiability literature (Lie-symmetry
methods; Raue et al. 2009 for the practical half), reduced to the part that is a rank
computation and honest about the part that is not.

## The decisions

**D11.1 — an `Observation` is an expression, its data, and its noise.** One object covers a
whole panel's worth of rows for a fitted model's mean, one extra dose you are considering,
*and* a quantity you do not currently measure but could. Everything in the module takes a
sequence of them. This is what makes "what else should I measure?" the same computation as
"what can I estimate?" rather than a second code path.

`Observation` and `SensitivityMatrix` are frozen dataclasses, **not** `Spec`s: they hold the
design's columns and a `rows × parameters` matrix respectively, and specs hold no large
arrays. What survives into a spec is the analysis — `EstimabilityReport` — whose largest
field is a `p`-vector per flat direction.

**D11.2 — log coordinates by default, and positivity is required for them.** "Only the ratio
is identified" is a statement about positive quantities. A non-positive parameter under
`scaling="log"` is `Unsupported` naming it, with `scaling="absolute"` offered — which drops
the requirement and gives linear combinations instead of monomials. Derivatives are central
differences on `θ_j exp(±h)`, which is the log derivative itself rather than a scaled linear
one.

**D11.3 — the rank tolerance *is* the question, and it is a parameter.** A singular value
below `tolerance × σ_max` counts as zero. At `1e-8` you are asking a structural question
("flat to machine precision?"); at `1e-2` a practical one ("a hundred times flatter than the
stiffest direction, so no realistic sample separates it?"). The same design gives different,
both-correct answers at the two settings, and the full spectrum is reported so a reader can
see whether the decision was clear-cut or a judgement call. Making this a hidden constant
would have made every rank statement a lie by omission.

**D11.4 — a named combination is scored statistically, not geometrically.** Requiring an
integer vector to lie *in* the numerical null space fails as soon as the null space is only
approximately flat — which is the interesting case. Instead:

* a **symmetry**'s score is its flatness, `‖Sw‖ / (σ_max‖w‖)` — how much predictions move
  along it relative to the direction they move most. Kept when at or below the tolerance.
* an **estimable** combination's score is its noise amplification,
  `σ_max √(wᵀ(SᵀS)⁻¹w) / ‖w‖` — the standard error of that combination in units of the
  best-determined direction's. One is the best any combination can do; kept at or below
  `1/tolerance`.

Both are scale-free, so neither depends on the units the parameters are measured in, and
they use the same tolerance from opposite ends. On the low-dose saturating design (24 doses
in [0.01, 0.3], `α = 4`, `k = 10`) this gives exactly the right split:

| direction | flatness | noise amplification | verdict at `tolerance = 3e-2` |
|---|---|---|---|
| `α` alone | 0.715 | 239 | neither flat nor estimable |
| `k` alone | 0.699 | 245 | neither flat nor estimable |
| `α · k` | 0.0117 | — | **symmetry** |
| `α / k` | 0.9999 | 3.99 | **estimable** |

The search covers integer vectors with bounded support and coefficients. It is a **search,
not a proof**: when it names nothing, the numerical `null_basis` is still reported and
`search_limits_hit` says so rather than returning an empty list that reads like "no
symmetries".

**D11.5 — `persistent_deficiency`, not `structural_deficiency`.** The field counts directions
flat at *every* parameter point tried. That rules out an accident of the values picked; it
does **not** rule out a property of the design. A saturating response measured only in its
linear regime is flat in `α·k` at every `(α, k)` you try — and the cure is a different dose,
not a different parameterization. The first name claimed more than the computation delivers;
this one says what it measures, and the docstring points at `prescribe_measurements` as the
thing that varies the design.

This distinction was found by a test that asserted `0` and got `1`. The test was wrong; the
name was worse.

**D11.6 — practical identifiability is a simulated experiment, not an eigenvalue.**
`simulated_identifiability` simulates the outcome at a known truth under the model's own
likelihood, refits, and profiles each target (Raue et al. 2009). A target whose profile never
rises by the χ²(0.95)/2 = 1.92 threshold within the search range has an interval that is
**unbounded on that side** — reported as an infinity, never as the edge of the grid. Each
profile point warm-starts from its neighbour, so the profile is a curve rather than a scatter
of local optima.

**D11.7 — combinations can be profiled, and that is the point of the whole module.**
`profile_combination` holds a monomial fixed by *elimination*: the parameter with the largest
exponent is solved for from the others, so every point the optimizer visits satisfies the
constraint exactly rather than approximately through a penalty. On the low-dose design it
gives the sentence this module exists to produce (seed 3, `sigma = 0.02`):

| target | truth | MLE | 95 % profile interval |
|---|---|---|---|
| `alpha` | 4.0 | 3.6 × 10⁵ | (**−∞**, **∞**) |
| `k` | 10.0 | 9.0 × 10⁵ | (**−∞**, **∞**) |
| `alpha / k` | 0.4 | 0.397 | [0.370, 0.425] |

The maximum-likelihood fit runs off along the ridge to `α ≈ 3.6 × 10⁵` while keeping the
ratio at 0.397 — a faithful picture of a flat likelihood, and a reminder that a point
estimate of `α` here is an artifact of where the optimizer stopped. The ratio is pinned to
7 %, and it is the number the report should quote.

**D11.8 — the constructive question gets an answer.** `prescribe_measurements` greedily adds
candidate observations until the design has full rank, reporting which symmetry each addition
breaks. Greedy is not optimal, so the candidates it considered are reported too; when nothing
on the table helps, `complete=False` and `still_flat` names what is left — which is the
useful answer, because it says what would have to be measured instead.

## Relationship to what was already there

Nothing is replaced. `design.structural` still answers precision-per-parameter through
`surface.forward` on a `SupportsForward`; this module answers estimability-of-combinations
over arbitrary expressions, which is what a compiled `DynamicSystem` (note
[0010](0010-dynamic-systems.md)) produces. `diagnose.weak_id` still reads a posterior after
the fact. The three are before-the-fact-per-parameter, before-the-fact-per-combination, and
after-the-fact.

## Bugs this work surfaced

* A `ZeroDivisionError` raising out of a profile objective when the optimizer wandered to a
  parameter value of exactly zero and a negative exponent — the same class as the parent
  repo's bug recorded in `diagnose/weak_id.py`'s docstring. Guarded: the monomial is
  undefined there and the objective returns its sentinel, rather than the exception escaping
  from inside a numerical routine.
* `np.linalg.svd` returns `min(rows, parameters)` singular values but always `parameters`
  right singular vectors. A design with fewer rows than parameters therefore indexed a
  boolean mask of the wrong length. The missing singular values are exact zeros — directions
  the design has no row to see at all — and are now padded in explicitly.

## Evidence

* `tests/unit/test_design_identifiability.py` — 27 tests: the log sensitivity against its
  analytic form, the product model's `a·b`, the low-dose design's `α/k` at a practical
  tolerance and its full rank at machine tolerance, the prescription, the profiles, and
  every refusal.
* `tests/recovery/test_dynamics_recovery.py::test_a_design_with_no_carryover_to_see_cannot_recover_the_decay`
  — a constant inflow makes a carryover model's decay and amplitude one number.
* `nbs/design/08-identifiability.ipynb`.

## Open

* **Vector parameters in combinations.** `profile_combination` requires scalar parameters and
  raises naming any vector ones. Monomials of a vector parameter's elements are well defined
  but the reporting has no good shape for them yet.
* **The rank tolerance has no default that is right.** `1e-8` is reported as the default
  because a structural answer is the one that cannot mislead; a practical answer needs a
  number the analyst chooses. A future version could derive it from the noise and the sample
  size — the quantity that decides it is the ratio of a direction's information to the
  posterior precision the design achieves — but deriving it silently would hide exactly the
  judgement this module is trying to surface.
