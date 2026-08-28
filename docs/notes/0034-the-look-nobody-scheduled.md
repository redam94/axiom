# 0034 — A spending function cannot price a look nobody scheduled

*Kind: decision. Opened 2026-08-27. Status: implemented on
`feature/always-valid-monitoring`. Closes item 6 of the 1.3 backlog in
[0027](0027-scope-and-the-experiment-lifecycle.md) §D27.6.*

`design.sequential` is a good implementation of the wrong thing for most of the
experiments a company runs. It prices looks that were **fixed in advance** —
`alpha_spending`, `pocock`, `obrien_fleming`, a schedule, a committee — and
that machinery is exactly right for a trial with a protocol.

Inside a company the dashboard is open. The interval refreshes hourly, three
people are watching it, and the decision is taken the first afternoon it looks
good. There is no schedule to spend against and no committee to spend it, and a
nominal 95 % Wald interval read at ten looks excludes the truth **19.4 %** of
the time — measured, in `tests/unit/test_design_anytime.py`, against 4.94 % for
the same interval read once. Not a subtle inflation: four times the advertised
rate.

## D34.1 — a confidence sequence, from the martingale that makes it work

`design.anytime` builds the standard normal-mixture confidence sequence
(Robbins 1970; Howard, Ramdas, McAuliffe and Sekhon 2021) on the canonical
scale `design.sequential` already uses. `B_t = Z_t·√t` is a Brownian motion with
drift `θ`; `exp(λ(B_t − θt) − λ²t/2)` is a martingale for any `λ`; mixing over
`λ ~ N(0, 1/ρ)` has a closed form,

```
M_t(θ) = sqrt(ρ / (t + ρ)) · exp( (B_t − θt)² / (2(t + ρ)) )
```

which starts at 1, so Ville's inequality bounds `P(∃t : M_t ≥ 1/α) ≤ α`.
Inverting gives `u(t) = sqrt((t + ρ)·log((t + ρ)/(ρα²)))` and the interval
`B_t/t ± u(t)/t` at every `t` at once.

The boundary grows like `√(t log t)` where the fixed-sample one grows like `√t`,
and that gap **is** the peeking. Nothing else in the construction is doing any
work.

## D34.2 — the price is stated first, because it is the whole trade

An anytime interval is wider at every single look — about 55 % wider at
`α = 0.05` where it is tuned, and further out on either side of that. The
notebook shows the case that matters: at `Z = 2.9` and full information a Wald
interval on the drift is `[0.9, 4.9]` and excludes zero; the confidence sequence
is `[-0.1, 5.9]` and does not. Fixed-sample `p = 0.0037`, anytime `p = 0.0715`.

Both are correct. They answer different questions, and only one of them is the
question somebody who has been watching a dashboard for six months is entitled
to ask. The module docstring puts this paragraph before the construction rather
than after it, because a reader who takes the width for a defect will use the
wrong one.

## D34.3 — `ρ` is a choice, so it is tuned and recorded

The mixing parameter decides where the sequence is tightest. It cannot be
avoided — every confidence sequence has one — only chosen, and the usual
treatment is a rule of thumb in a footnote.

`tune(alpha, target)` minimizes the half-width at a target information fraction
numerically, over `log ρ`. Tuning at 0.25 buys width early and pays for it late
(6.07 against 6.50 at `t = 0.25`; 3.13 against 3.04 at `t = 1`), which is a real
design decision for a study somebody will read at the halfway point. `ρ` and
`tuned_at` are fields on the result and appear in its ledger line, so the choice
travels with the number.

## D34.4 — `anytime` is a fifth interval definition

`core.IntervalDefinition` gains `"anytime"`, beside `eti`, `hdi`, `wald` and the
`stagewise` added in [0028](0028-the-estimate-a-stopped-study-may-report.md).
An `AnytimeLook` validates that its interval carries it. Gate 6's rule is that
no interval in axiom has its meaning in a variable name, and three
frequentist intervals over the same estimate that differ by 55 % is precisely
the case it exists for.

## D34.5 — the e-value is the point, not a by-product

`M_t(0)` is an e-value: the evidence against the null, starting at 1, a
martingale under it. Its reciprocal is an anytime-valid p-value, and
`AnytimeLook.running_evalue` carries the running maximum because that is the
object Ville's inequality bounds — a decision taken on the maximum is the
decision the guarantee covers, and a decision taken on the current value is not
quite.

Exposing it is deliberate rather than completionist. E-values multiply and
average in ways p-values do not, which is what makes error control across a book
of experiments tractable under arbitrary dependence — and a book of experiments
sharing markets and seasons is dependent in ways nobody can characterise. That
is item 5 of the same backlog and it is next.

## D34.6 — what this does not do

**Nothing chooses between the two.** `design.sequential` and `design.anytime`
sit beside each other with no guidance in code about which a study should use.
The rule is simple — if the looks are in the protocol, spend; if the dashboard
is open, sequence — and it lives in prose in both module docstrings, not in a
function that would have to guess.

**No sub-Gaussian or empirical-Bernstein variants.** The mixture here assumes
the canonical joint distribution, exactly as everything else in
`design.sequential` does, and `CANONICAL` rides on every result. A confidence
sequence for a bounded outcome with unknown variance is a different mixture and
a real thing to want.

**No stopping rule.** `ConfidenceSequence.crossed_at` reports the first look
whose interval excluded the null. It does not tell anybody to stop, and
`design.monitor`'s decision vocabulary is not wired to it. A program that wants
"stop when the sequence clears zero" can read `crossed_at`; a program that wants
that decision priced and ledgered still uses `monitor`.
