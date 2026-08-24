# 0017 — A printed number stops where its uncertainty stops

*Kind: decision. Opened 2026-08-23. Status: implemented on `integrate/develop`; part of the 1.1.0 surface.*

Every result in axiom carries an interval, and every result printed the point
estimate at four or six significant digits regardless of how wide that interval
was. `mean 12.34568` beside `[10.1, 14.6]` is seven digits of which two are
knowledge and five are arithmetic on a finite sample.

That is not a cosmetic complaint either. Trailing digits read as precision: a
reader compares two results on digits that are Monte Carlo noise, quotes a
number to a room at a resolution the data never supported, and treats a mean
printed to five places as though the fifth place means something. It is the
same failure as a skimmed interval (D16.1), one layer down — the interval was
*there*, and the point estimate printed beside it contradicted it.

## D17.1 — one rule, in `core`, applied on the way to a human

`axiom.core.rounding` holds it: keep `DIGITS` (two) significant digits of the
uncertainty, and round the value to that same decimal place. It is the usual
convention in measurement reporting and it needs no configuration to be right.

- `decimals_for(uncertainty)` — the place, negative when the last meaningful
  digit sits left of the point (1234 give or take 500 is known to the ten).
- `round_to(value, uncertainty)` — the numeric form, for a caller that keeps
  computing.
- `format_measured(value, uncertainty)` — the printed form.
- `format_interval(lower, upper)` — both bounds at one place, so the pair reads
  as a pair.

It lives in `core` because `Interval.__str__` needs it and `core` may not import
upward. It depends on `math` alone; the dependency budget does not notice it.

## D17.2 — the uncertainty is whatever the object states about itself

`format_measured` takes the resolution as an argument rather than guessing:
a standard error, a posterior sd, an interval's `half_width` (new property).
With `None`, a zero, or a non-finite uncertainty it falls back to a plain `%g`
— **an unrounded number is the honest answer to "how precise is this?" when the
answer is "no one said"**, and inventing a width would be the same lie in the
other direction.

**D17.2a — a sigma beats a band.** Where an object states both, the standard
error wins: a 95% band is about two sigma wide, so rounding to the band throws
away a digit the sigma supports. `format_interval` takes an `uncertainty=`
override so a card showing `effect 2.50, se 0.76` prints the band as
`[1.01, 3.99]` rather than `[1.0, 4.0]` — one result at one resolution.

**D17.2b — one card, one scale, except where two scales are real.** The generic
renderer resolves a *per-field* uncertainty first (`slope` rounds against
`slope_se`), then falls back to the object's own. A result with two coefficients
has two resolutions and neither is both. Fields not naming a measured quantity —
`p`, `mass`, `alpha`, a count — keep the plain form: a p-value and a coefficient
on one card do not share a scale.

## D17.3 — display only, and the boundary is deliberate

Nothing here touches a stored value. `io` still writes seventeen significant
digits, `Spec` content hashes are unchanged, a failure's `detail` dict still
carries the `repr` that reproduces the bug, and the ledger statements in
`calibrate.transfer` still quote the factors at six digits because those are
audit arithmetic rather than prose. **Round on the way to a human, never on the
way to a file.**

The one ledger site that did change is `meta.priors`: `N(12.3, 1.2)` is a
location and a scale, its `detail` already carries both as `repr`, and six
digits of a pooled sd was the same false precision in prose.

## D17.4 — a report metric derives its precision by default

`report.Metric.precision` was `int = 2` — a fixed two decimals that either
invented digits the interval did not support or dropped digits it did. It is now
`int | None = None`, meaning *take it from the interval*, with the old fixed
behaviour available by setting a number for a house style that wants a column at
one width. Old serialized specs carrying `precision: 2` still load unchanged.

## What changed for a reader

| before | after |
|---|---|
| `[-1, 2] (50% HDI)` | `[-1.0, 2.0] (50% HDI)` |
| `mean 12.34568` · `sd 1.234567` | `mean 12.3` · `sd 1.2` |
| `effect 2.499912` · `se 0.7614238` | `effect 2.50` · `se 0.76` |
| `[1.012, 3.988] (95% WALD)` | `[1.01, 3.99] (95% WALD)` |

Trailing zeros are kept throughout, because they are the claim: `12.30` says the
hundredths digit was resolved and `12.3` says it was not.
