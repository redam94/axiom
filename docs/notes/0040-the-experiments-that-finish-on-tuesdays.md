# 0040 — The experiments that finish on Tuesdays

*Kind: decision. Opened 2026-08-28. Status: implemented on
`feature/online-error-control`. Closes the follow-on
[0035](0035-eight-wrong-go-decisions-a-quarter.md) §D35.7 named: e-BH is a
batch procedure and a book whose experiments finish continuously wants an online
one.*

`design.program` sorts a period's readouts, finds a threshold and returns. That
is the right shape for a quarterly review and the wrong shape for what actually
happens: experiments finish one at a time and somebody wants an answer that
afternoon. Waiting until March to learn whether January's test worked is not a
correction anybody accepts, and deciding each one at 5 % as it lands is the
failure 0035 exists to name.

## D40.1 — LORD, and the budget that is repaid by discovery

`design.online` implements LORD (Javanmard and Montanari; Ramdas, Zrnic,
Wainwright and Jordan 2017). Each readout gets a level of its own out of a
budget that starts at `w0` and is **replenished by `alpha` every time something
is rejected**:

```
alpha_t = gamma_t·w0 + (alpha − w0)·gamma_{t−tau_1} + alpha·Σ_{j≥2} gamma_{t−tau_j}
```

Nothing in it needs to know how long the stream is, which is the whole point.

The behaviour that falls out is worth understanding before using it, and the
notebook shows it on a stream of two hundred: the level starts at 0.011, rises
to 0.043 while the first twenty treatments are working, falls to **0.0003**
through a long null run, and recovers to 0.037 after five more discoveries. A
programme that discovers things can afford to keep testing; one that does not,
cannot. That is uncomfortable and it is correct — the budget is being spent
either way, and a run of nulls is exactly the situation in which the next
"finding" is most likely to be noise.

## D40.2 — the gamma sequence is a power law, and the reason is honesty

Any non-negative sequence summing to at most one gives a valid procedure. The
literature's usual choice is `gamma_j ∝ 1/(j·log²j)`, whose normalizing constant
has no closed form; using it means either truncating (and then a stream longer
than the horizon gets level zero) or normalizing over a horizon and quietly
hoping.

`gamma_sequence` uses `gamma_j = j^-decay / zeta(decay)` instead, which sums to
**exactly one** over the infinite stream with a constant `scipy.special.zeta`
computes. `decay` trades early power against late and is on the result.

## D40.3 — arrival order is part of the procedure

The same readouts in a different order give different decisions, because each
level depends on what came before it and nothing that came after. That is not a
defect to be smoothed over — it is what "decide it when it arrives" means — and
a test pins it, because the first instinct on seeing it is to sort the stream
and re-run.

## D40.4 — the trade against `design.program`, stated on both

Two procedures now control the same rate and need different things:

| | needs | waits |
|---|---|---|
| `program_decisions(method="e_bh")` | nothing about dependence | for the batch |
| `online_decisions` | independent readouts | not at all |

`INDEPENDENT_READOUTS` carries that trade on every online result, and its
`challenged_by` names `design.program`'s e-BH route by name. A house that can
wait should batch; a house that cannot should know which of the two conditions
it has started relying on. Neither module chooses.

## D40.5 — what this does not do

**No e-value route.** Online procedures over e-values exist and would inherit
the arbitrary-dependence property, which would collapse the table above into one
row. That is the right eventual answer and is a larger piece than this one.

**No alpha-death recovery.** A programme that goes a very long time without a
discovery has its levels driven toward zero and cannot easily recover, which is
the known cost of LORD and the reason SAFFRON and ADDIS exist. Neither is here.

**Nothing feeds it.** `online_decisions` takes `design.program.Readout`s in
arrival order. Nothing collects those from an `io.ExperimentRun` as it reaches
`read`, which is — again — the `build` entry point deferred since
[0027](0027-scope-and-the-experiment-lifecycle.md).
