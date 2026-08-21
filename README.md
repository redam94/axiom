# axiom

**Bayesian causal decision science.** Declare what you want to know, find out
whether the data can tell you, design the experiment that would, fold the
experiment's answer back into the model, map the response surface it implies,
and pool the evidence across every study you have run.

Four pillars, one vocabulary:

| Pillar | Question it answers | Package |
|---|---|---|
| **Causal modeling** | Is this effect identified, and from what? What would a hidden confounder have to look like to overturn it? | `axiom.identify`, `axiom.estimands`, `axiom.diagnose` |
| **Experimental design & planning** | What is worth measuring next, at what size, by which method, at what cost? | `axiom.design` |
| **Experimental calibration** | How does a randomized result update an observational model, and what did that transfer assume? | `axiom.calibrate` |
| **Response-surface methodology** | What does the dose–response surface look like, where is its optimum, and where should I probe next? | `axiom.surface` |
| **Meta-analysis** | What does the whole body of evidence say, and how heterogeneous is it? | `axiom.meta` |

## Status

**Pre-implementation.** This repository currently holds the architecture and
the implementation plan. No algorithms are implemented yet. Start at
[`docs/plan/00-charter.md`](docs/plan/00-charter.md).

## Design commitments

1. **Domain-general core.** The nouns are `Treatment`, `Dose`, `Unit`,
   `Outcome`, `Covariate`. Marketing (channel / spend / geo / KPI) is *one
   adapter*, in `axiom.adapters.marketing`, and a test gate keeps its vocabulary
   out of the core.
2. **Lightweight by budget, not by aspiration.** `pip install axiom` pulls
   numpy, scipy, pandas, pydantic. Nothing else. The whole design, planning,
   identification, and closed-form meta-analysis surface works there. Samplers
   are extras; a contract test fails the build if `import axiom` drags one in.
3. **Everything is a spec, and every spec is serializable.** Frozen Pydantic
   models with a schema version and a content hash. An analysis round-trips
   through JSON plus a posterior file — never through a pickle.
4. **Numbers carry their provenance.** Every reported interval states which
   interval definition it is and at what mass. Every calibration transfer
   writes an assumption ledger line.
5. **Sampler-agnostic.** Inference sits behind `axiom.infer.Backend`. NumPyro is
   the default implementation; PyMC is an alternate.

## Lineage

`axiom` is a clean-room rewrite that ports selected mathematics from
[`mmm-framework`](../../mmm-framework), a ~177k-LOC Bayesian marketing-mix
platform. The target is ~25k LOC: the causal, design, calibration, surface, and
meta-analysis mathematics, with the modeling platform, agent stack, reporting
engine, and web application left behind. The file-by-file accounting is in
[`docs/plan/02-porting-ledger.md`](docs/plan/02-porting-ledger.md).
