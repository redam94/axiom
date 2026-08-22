# benchmarks/ — real data, published answers

Every other test in this repository checks axiom against a world axiom made up.
That is the right way to ask whether an estimator recovers a truth, and it is
circular in one respect: the data-generating process and the likelihood are the
same code. These benchmarks close that loop from outside — datasets collected by
other people, results computed by other software, published before this library
existed.

```bash
python benchmarks/run_all.py          # every case study whose data is present
python benchmarks/registry.py         # what is here, and on what terms
pytest tests/benchmarks -q            # the assertions
```

## What is checked

| Dataset | Field | Pillar | Checked against | Status |
|---|---|---|---|---|
| **BCG vaccine** — 13 TB trials, 357,347 people | Epidemiology | meta | R `metafor` | vendored |
| **NIST Misra1a** — monomolecular adsorption | Physical chemistry | surface | NIST certified values | vendored |
| **Darfur** — 1,276 refugees surveyed in Chad | Political science | identify + diagnose | Cinelli & Hazlett (2020) | fetch |
| **NSW / LaLonde** — a randomized training programme | Labour economics | identify | Dehejia & Wahba (1999) | fetch |

### Results

**BCG.** All eight published quantities reproduced: pooled log risk ratio
−0.7145, se 0.1798, 95% interval [−1.0669, −0.3622], τ² 0.3132 (REML), Cochran's
Q 152.2330 on 12 df, and metafor's model-based I² 92.22% and H² 12.86. Largest
disagreement anywhere: 4×10⁻³, which is metafor printing H² to two decimals.

**Misra1a.** Both NIST-certified parameters recovered to about one part in ten
thousand, with the certified values inside the 95% intervals, and a residual sum
of squares that sits `+3.3e-06` **above** the certified minimum — as it must,
since the certified value *is* the minimum. A fit reporting a lower one would be
a bug, and that is the class of error a certified benchmark exists to catch.

**Darfur.** Coefficient 0.097316 against a published 0.0973, standard error
0.023257 against 0.0232, residual df 783 exactly — from 1,276 raw rows and 486
village fixed effects. The sensitivity numbers built on it (robustness value
13.9%, 7.6% at α = 0.05, partial R² 2.2%) then match the paper.

This one closes a specific gap. `tests/unit/test_diagnose_sensitivity.py` already
checked the sensitivity *formulas* against those published values — but from
hard-coded summary statistics. Now the summary statistics are checked too.

**NSW.** The experimental benchmark comes back at \$1,794 against a published
\$1,794. Then the demonstration the dataset is famous for: swap the randomized
controls for PSID survey respondents and the same treated people yield −\$15,205
unadjusted and \$752 with eight covariates, against a truth of \$1,794. The
observational estimate is not vague — its standard error is small. It is
confidently wrong, and nothing computed from the observational data alone reveals
that.

## One thing this found

The first BCG run disagreed with metafor on I² by 0.001 — small enough to look
like a rounding error. It is not. I² has two standard definitions:

- **Q-based**, `(Q − df) / Q`, which `axiom.meta.heterogeneity` returns → 92.12%
- **model-based**, `τ² / (τ² + s²)`, which metafor's `rma()` prints → 92.22%

Both are correct; they answer slightly different questions. Reconstructing
metafor's definition from axiom's outputs reproduces 92.22% and H² 12.86 exactly,
which is what the benchmark now asserts. `registry.model_based_i2` implements it,
and `case_bcg.py` prints both.

The wrong fix would have been widening the tolerance until the table went green.

## Licensing, and why half the data is not here

This repository is proprietary. Two of the four datasets carry licences that a
proprietary repository cannot satisfy by redistributing them:

- **Darfur** ships inside the GPL-3 R package `sensemakr`.
- **NSW / PSID** is CC BY-NC — attributable *non-commercial* use.

So they are not vendored. `python benchmarks/fetch.py` downloads them into
`benchmarks/cache/` (git-ignored) on request, prints the licence before doing it,
and records the sha256. Tests over them skip when absent, so a fresh clone runs
10 passed / 4 skipped rather than failing.

The two vendored datasets are safe by a wide margin: NIST Misra1a is a work of
the US federal government and public domain, and the BCG table is thirteen rows
of event counts from a 1994 JAMA paper, reproduced in dozens of textbooks and
packages, with the citation carried in the registry.

```bash
python benchmarks/fetch.py              # list everything and its terms
python benchmarks/fetch.py darfur       # fetch one
python benchmarks/fetch.py --all
```

## Adding one

The bar is a dataset somebody else collected with a result somebody else
published. Without a published number to hit, a benchmark is just another
example — `examples/` is the right home for those.

1. Add a `Dataset` to `registry.py` with its citation, licence, source URL and
   the published values, each with a `_source` saying where it was published.
2. Vendor it under `data/` only if the licence plainly allows redistribution from
   a proprietary repository. If there is any doubt, make it `availability="fetch"`.
3. Write `case_<name>.py` — the framing, the comparison table, and what the
   numbers mean.
4. Add assertions to `tests/benchmarks/test_published_values.py`, with tolerances
   set by the precision of the published value rather than by what passes.
5. Register the case in `run_all.py`.

Where axiom and the reference disagree, find out why before touching a tolerance.
The I² discrepancy above took twenty minutes and turned into the most useful
paragraph in this file.
