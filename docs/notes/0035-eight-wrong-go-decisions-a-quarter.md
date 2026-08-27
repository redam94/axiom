# 0035 — Eight wrong go-decisions a quarter, all individually defensible

*Kind: decision. Opened 2026-08-27. Status: implemented on
`feature/program-error-control`. Closes item 5 of the 1.3 backlog in
[0027](0027-scope-and-the-experiment-lifecycle.md) §D27.6.*

Every error-rate control in the repository is about *an* experiment.
`design.sequential` spends alpha across the looks of one study;
`design.anytime` makes one study's interval valid at any time;
`diagnose.structure` corrects across the implications of one graph. Nothing has
ever looked at the quarter.

Eleven parties, three treatments each, one primary metric and four guardrails
per readout is a hundred and sixty-five go/no-go decisions in a period. At the
conventional 5 %, with **nothing working at all**, `165 × 0.05 = 8.25` of them
are go. Every one of those numbers is individually defensible and individually
documented, and the programme is wrong eight times a quarter.

`design.program` computes the four routes and refuses to choose between them.

## D35.1 — the count comes first

`expected_false_go(n, alpha)` is `n · alpha`, and `ProgramReport` reports it
under **every** method including the one that was used. It is not a bound and
not a worst case: it is the expected number of go-decisions a null book takes
at the level each readout was individually judged at.

Putting it on every report is the design decision. Correcting a programme is a
choice about what it is willing to be wrong about, not a statistical fact, and
nothing here picks the method. What the module can do is make `none` a choice
somebody made rather than a default nobody noticed, and the way to do that is to
print what `none` costs beside what the alternatives buy.

## D35.2 — four routes, and the last one is why `anytime` exposed its e-value

| method | controls | needs |
|---|---|---|
| `none` | nothing across the programme | — |
| `holm` | family-wise: the chance of *any* false go | any dependence |
| `benjamini_hochberg` | false discovery rate | independence or positive regression dependence |
| `e_bh` | the same false discovery rate | **arbitrary** dependence |

On the notebook's book — 165 readouts, eight treatments that really work:

| method | go | real | false |
|---|---|---|---|
| `none` | 15 | 8 | **7** |
| `holm` | 7 | 7 | 0 |
| `benjamini_hochberg` | 8 | 8 | 0 |
| `e_bh` | 5 | 5 | 0 |

Uncorrected takes fifteen and seven are false — close to the 8.25 a null book
would have produced on its own, which is the point. Benjamini-Hochberg finds
all eight and needs positive regression dependence to do it. e-BH (Wang and
Ramdas 2022) finds five and needs **nothing** about how the readouts depend on
each other.

A book of experiments sharing markets, seasons and a macroeconomy is dependent
in ways nobody can characterise. That is exactly the condition BH on p-values
does not cover, and it is why [0034](0034-the-look-nobody-scheduled.md) exposed
the e-value rather than keeping it inside the confidence sequence. The power
e-BH costs is the price of not making a claim you cannot support, and
`ARBITRARY_DEPENDENCE` is the one `Assumption` in this repository whose
`challenged_by` is *"nothing"* — because it is the weakest condition there is,
and the alternative is `positive_dependence`, which the report attaches when
`benjamini_hochberg` is chosen.

## D35.3 — guardrails are decisions

`Readout.metric` defaults to `"primary"` and names a guardrail otherwise, and
guardrails are counted. A programme that checks four of them per experiment is
taking five decisions per experiment, not one, and the multiplicity is real
whether or not anybody was thinking of them as tests. The notebook's 165 is 33
experiments × 5 metrics; the 33 alone would have expected 1.65 false goes rather
than 8.25.

## D35.4 — a missing input is not a no-go

`e_bh` needs an e-value on every readout; the p-value methods need a p-value on
every readout. A readout missing what the method needs raises, naming the
readouts. The tempting alternative — drop it, or treat it as no-go — is a way of
deciding a decision by not making it, and a programme that silently drops its
un-monitored readouts has changed `n`, which changes every threshold.

## D35.5 — the calibrator, and a claim corrected in review

A readout that was not monitored can be calibrated: `from_p_value(p)` is
`p^(-1/2) − 1`, whose expectation under the null is exactly 1. (`1/p` is not an
e-value; its expectation diverges.)

The first draft of the docstring said the calibrated e-value is "markedly weaker
than an e-value computed from the data directly". The test written to pin that
failed, and the truth is more interesting: against a two-sided z the calibrator
is *larger* for weak evidence (1.88× at `z = 2`) and collapses for strong
evidence (0.30× at `z = 4`, 0.007× at `z = 6`). The mixture e-value is paying
for anytime validity in the first regime and its essentially-Gaussian tail is
winning in the second, where the calibrator's is only polynomial.

So the difference is not size. A calibrated e-value **inherits the validity of
the p-value it came from** — one look, fixed sample — while the mixture e-value
is valid at every look. Mixing them in one programme is legitimate (an e-value
is an e-value, and e-BH does not care where they came from) and means the
resulting decisions are only as anytime-valid as their weakest input. The
docstring says that now, and `test_the_calibrator_is_not_the_same_object_as_a_mixture_evalue`
pins the crossover so nobody re-asserts the simpler claim.

## D35.6 — `adjust` moved to `core`

`diagnose.structure` owned the Holm and Benjamini-Hochberg arithmetic;
`diagnose.delivery` imported it from there; `design` could not, because
`design` is *below* `diagnose` in the layering. Rewriting it would have been the
third copy of a twenty-line procedure.

It is now `core.multiplicity` with `core.adjust` and `core.Multiplicity`.
`diagnose.structure.Correction` remains as an alias so that subpackage's public
API is unchanged, and both `diagnose` modules import the function from `core`.
The name changed on the way down — `Multiplicity`, not `Correction` — because
`calibrate.Correction` is a typed evidence-transfer operator and two
`Correction`s in `core`'s import surface would have been a name nobody could
read twice.

## D35.7 — what this does not do

**No power accounting.** The report says what each method controls and how many
goes it takes. It does not say what a correction costs in *power* — the real
question a programme director asks, which is "how much bigger must my
experiments be if I correct across the quarter". That is `design.power` composed
with this, and it is not built.

**No sequential programme.** The decisions are taken together, once, at the end
of a period. A book where experiments finish continuously wants an online
procedure (alpha-investing, LORD) that spends an error budget as readouts
arrive. e-BH is a batch procedure and this module is a batch module.

**Nothing knows which readouts are guardrails.** `metric` is free text, so
`program_decisions` treats a guardrail exactly like a primary. A programme that
wants guardrails held to a different level — or wants a guardrail *failure* to
block a go on the primary — has to say so itself, and the natural place for
that is a decision rule this module does not have.
