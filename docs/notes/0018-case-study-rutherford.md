# 0018 — GEIGER-1911: designing an experiment, and what robustness costs

**Kind**: progress. **Opened**: 2026-08-23. **Status**: landed on `develop`.

A third case study, `nbs/case-studies/rutherford/`, in five notebooks over one shared
synthetic world. It is deliberately a different shape from the two that exist.
[0004](0004-case-study-hypertension.md) is a safety story and
[0009](0009-case-study-tutoring.md) is an allocation story; both *analyse* data. This one
never gets to analyse anything until the fifth notebook, because it is a **design** story:
the deliverable is a set of angles, an aperture geometry, a time allocation and a stopping
rule, all produced before an apparatus exists.

The question is Rutherford's. Is the positive charge of an atom a hard dense centre or
spread through the whole atom, and what experiment tells them apart?

## 0018.1 — Two hypotheses become one parameter, and that is the whole method

Nothing in `axiom.design` scores "hypothesis A against hypothesis B". Everything in it
scores a *parameter* — its Fisher information, its expected posterior sd, its estimability.
So the first move, and the one the rest of the case study rests on, is to find the
parameter the two atoms differ in.

It exists and it is physical. An alpha aimed head-on stops at `D = 2zZe²/4πε₀E`; a
trajectory scattered through `θ` comes no closer than `(D/2)(1 + 1/sin(θ/2))`. A positive
charge of radius `R` therefore kills the scattering past

    sin(θ_cut / 2) = 1 / (2R/D − 1)

and the two atoms are `R = 1.35e-10 m` and `R < 3e-14 m`. The model carries
`lam = log sin(θ_cut/2)` and the hypotheses are two points on that axis, six decades apart.

The payoff is immediate and would not have been available from a two-model comparison: the
denominator goes to zero at `R = D/2`, so **the experiment has a resolving power of 29.6 fm
set by the beam energy alone**, and returns an upper bound rather than a value. That is
notebook 1's most important cell and it is four lines of arithmetic.

## 0018.2 — The Anscombe bridge: Poisson counting through Gaussian design machinery

`design.fisher_information` takes a `noise_sd`. Scattering counts are Poisson with rates
spanning fourteen orders of magnitude across the angle range, so there is no `noise_sd` to
give it.

The surface's `forward` returns `2 sqrt(mu)` instead of `mu`. For any parameter `psi`,

    (d[2 sqrt(mu)] / d psi)² / 1²  ==  (1/mu) (d mu / d psi)²

and the right-hand side is the Poisson Fisher information exactly. So
`fisher_information(surface, ..., noise_sd=1.0)` on this surface *is* the Poisson
information, with no approximation, and every precision claim in the case study is a
statement about counting rather than about a Gaussian approximation to counting. Notebook 1
checks it against a longhand Poisson calculation and agrees to the finite-difference step.

This is worth recording as a **reusable pattern**, not a case-study trick: any count
outcome can be brought inside `axiom.design` this way, and the transform lives in the
expression tree (`Pow(mu, 1/2)`), so rule 3 is not strained — `counts()` is `(forward/2)²`
and the expected count is read back through the same evaluation.

## 0018.3 — A hand-built `SupportsForward` outside `axiom.surface`

The mean is `A u⁻⁴ exp(−(u/s)²) + C exp(−θ²/2w²) + B`, which is not a `SurfaceSpec` — no
kernel in `axiom.surface` is `csc⁴`. It is a hand-assembled `ModelSpec` and a nine-line
dataclass satisfying `SupportsForward`:

* `expr` returns `spec.mean` (matching `Surface.expr`, which is the mean and not the spec —
  worth noting, it is easy to get wrong);
* `forward` is `core.value` on that tree and nothing else;
* `linearize` is `surface.design_matrix(spec, (), ...)` — **no** linear parameters, which is
  the truthful answer since the mean is nonlinear in all five. The whole Jacobian then comes
  from finite differences of `forward`, which is what `design.structural` does when there
  are no linear columns.

Everything in `axiom.design` worked against it unmodified: `fisher_information`,
`expected_posterior_sd`, `identifiability_ridge`, `design_to_identify`,
`sensitivity_matrix`, `estimable_combinations`, `profile_likelihood`. That is the protocol
earning its keep, and it is the first time anything outside `axiom.surface` has implemented
it.

Two mechanical notes for the next person:

* A logarithm needs a dimensionless argument, so a log-amplitude cannot carry the rate's
  dimension. The model declares `RATE_UNIT = Const(1.0, count/time)` and multiplies by it.
  There is no way around this and it is the right shape — the reference scale is named once,
  in the model, rather than hidden in a parameter.
* `Param(prior=Prior(family="fixed", ...))` is how a parameter is held out of a fit.
  `scattering.point_charge_model()` uses it to fix `lam` so that `profile_likelihood` can
  profile the amplitude instead, which is the only answerable question in the world where
  the tail was never seen.

## 0018.4 — The finding: where the information is, and where the *durable* information is

This is the part worth arguing with, because the two criteria disagree and the disagreement
is the case study.

The counter is rate-limited (a human at a scintillation microscope manages ninety flashes a
minute), so every station from 0.5° to 75° collects the same 5,400 counts an hour and what
differs is what the diffuse atom says about them. On that basis:

* the expected log Bayes factor peaks at **five degrees**, 97,000 nats an hour;
* a hundred and fifty degrees is worth 4,900 nats an hour — **twenty times less**.

Now let the diffuse atom widen its multiple-scattering core, which is the only free story
that explains a wide-angle count away, and which is not free because the beam is conserved
(a core `f` times wider is `f²` times lower). Minimizing the evidence over that family:

| angle | nats/h believed | nats/h attacked | kept |
|---|---|---|---|
| 5° | 97,394 | 18 | 0.02% |
| 20° | 67,717 | 1 | 0.00% |
| 90° | 20,659 | 536 | 2.6% |
| **150°** | **4,870** | **4,870** | **100%** |

The angle past which the evidence is untouchable is a computable function of how wide a
core the opposition may claim: 14° against a threefold error, 45° against tenfold, **131°
against thirtyfold** — which is where Geiger and Marsden put their counter.

The consequence for the plan is a lexicographic allocation rather than an optimization: buy
every constraint the argument will need (10 hours of anchor pins the core width well enough
to catch a 10% error with certainty; 10 hours of foil-out pins the background), then spend
the remainder where the evidence cannot be talked out of. That is **110 of 200 hours at
150°**, which the believed-model criterion calls a waste by a factor of twenty.

## 0018.5 — The aperture identity, which is a general lesson about designed measurements

A detector opening covering azimuth `φ` and `θ ± δ` subtends exactly `φ · 2 sinθ sinδ`. The
same counts can be bought by opening in `δ` or in `φ`, and **only `δ` smears**, because the
scattering does not depend on azimuth. So the prescription is: hold the radial half-width at
whatever the workshop can cut and take every additional steradian out in azimuth, opening
radially only once the arc has closed into a full annulus.

At 90° an annular slot and a round hole of the same 0.2 sr are ±0.91° and ±12.87°
respectively, and their smearing errors are 0.025% and 5.2%. Two hundred times, for free.

The smearing law is `2(δ/θ)²` at small angle — verified against the integrated `forward` to
a few per cent — which makes the constraint scale-free and produces the case study's other
concrete instruction: at the workshop's floor of 0.05° the smearing error crosses the
counting error at **0.9°**, so the anchor cannot be moved closer to the beam. Not for want
of counts; for want of brass.

## 0018.6 — What did not work, and one thing that surprised me

**`design.sensitivity.perturb` does not apply here.** It is built on
`optimizer.DesignCandidate` / `evaluate_candidate`, which require an `EconomicInputs` with a
`ValuePerOutcome` in a numeraire. A physics design has no currency and forcing one would
have been dishonest. The robustness sweep in notebook 2 is therefore written directly
against `forward`. Worth considering whether a numeraire-free scoring path is wanted; the
same gap would appear for any non-commercial design.

**`pareto_front` was not used** for the same reason, and the slit trade-off turned out not
to need it: once the aperture identity is noticed, bias and variance stop competing.

**The surprise.** At the hard-centre truth, `fisher_information` reports the information
about `lam` as `1.3e-7` against `2.9e5` at a visible charge, and `expected_posterior_sd`
hands the prior straight back. At the diffuse truth, `log_a` and `lam` are both named as
round-off columns and the matrix is singular. Both are *correct and useful*: the first is
the resolving-power ceiling of 0018.1 arriving in the design math independently, and the
second says the experiment can refute the diffuse atom but can never fit it. A design
calculus that reports "this parameter is not measurable here" as a named finding rather than
as a large number is doing something the parent repo could not.

## 0018.7 — Numbers, for the record

* Resolving power `D/2 = 29.6 fm`; the plan reports `R < 33 fm`; the design math predicted
  32.3 fm before any counts existed. Rutherford published 34 fm in 1911.
* Expected evidence: 4.4 million nats believed, 571,000 attacked, of which the two witness
  stations carry 96%.
* The witness at 150° crosses an O'Brien–Fleming boundary at the **first interim look, five
  hours in**, and would still cross with a source a thousand times weaker — which is roughly
  the source Geiger and Marsden had, and why their answer took months.

## 0018.8 — Follow-ups

1. The single-scattering assumption is `asserted`, not `satisfied`: the design never tested
   it. A foil-thickness series would, and would be a sixth station. Notebook 5's `Verdict` is
   `downgraded` for exactly this reason.
2. The adversary in notebook 2 is unconstrained by the anchor data, so the surviving evidence
   is a lower bound. A properly constrained minimax — the adversary restricted to cores the
   anchor stations cannot reject — would be tighter and is the natural next piece of
   machinery. It is also the general shape of "robust design" and probably belongs in
   `axiom.design` rather than in a notebook.
3. The variance-stabilizing bridge (0018.2) is general. If count outcomes are in scope for
   1.1, it is worth a helper and a gate rather than a case-study docstring.
