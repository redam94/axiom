# 0007 — A surface never travels without its uncertainty, and the reports that carry it

**Kind**: decision. **Opened**: 2026-08-21. **Status**: landed on `develop`; part of the
1.1.0 surface. Adds one layer to the architecture (`report`, layer 8).

Two asks, one principle. A response curve is a posterior quantity and a number that had
an interval should not lose it on the way to a slide.

## Part 1 — `ResponseBand`: the curve as data

### 0007.1 — The failure this fixes was in the notebooks, not the library

`viz.response_curve` already computed a band correctly. The problem was that the band
computation lived *in the figure function*, so anything that wanted a curve as **data** —
the case-study notebooks, and a report — evaluated `forward` at the posterior mean of the
parameters and drew a line. That is a different quantity (the mean of a nonlinear
function is not the function of the mean) and, more importantly, it claims a width of
zero.

`surface.ResponseBand` is a `Spec` holding the dose grid, the posterior mean and median
at every grid point, and the interval at every grid point with **its definition and
mass**. `response_band` builds one for the expected outcome; `marginal_band` for the
derivative. Both push every draw through the same `forward` the likelihood used.

`viz.response_curve` and the new `viz.marginal_curve` now render exactly that object and
nothing else. One implementation of "evaluate the surface over a grid through the draws"
(rule 3), and a figure that cannot be drawn from a band that does not exist.

### 0007.2 — "Always displayed" is a gate, not a convention

`tests/contracts/test_surface_uncertainty.py` asserts both halves:

- every grid evaluation returns a `ResponseBand` whose `interval_at(i)` is an `Interval`
  with the definition and mass that produced it;
- every `viz` figure of a surface has exactly two traces, the first of which is the
  filled band — and **no parameter anywhere in the signature can suppress it**. The test
  inspects the signature for `band`, `show_band`, `uncertainty`, `interval`, `ribbon`
  and fails if one appears.

The same rule is enforced in `report` by construction rather than by convention: a
`Figure` block whose source is a `ResponseBand` is rendered through `viz.response_curve`,
which has no mode that omits the band, and `Theme.band_opacity` is validated `> 0`.

### 0007.3 — What the band is a band *of*

Uncertainty in the **expected** outcome at a dose, averaged over units and the window's
periods — not a predictive interval for one unit. `noise` is not added. The distinction
is real and is recorded in `ResponseBand.kind` rather than left to the reader.

The marginal band matters more than the response band for a dose decision, and it is the
one that was missing: where its interval straddles zero the model does not know whether
the next unit of dose helps, and a line through the posterior mean cannot say so.

## Part 2 — `axiom.report`

### 0007.4 — A template is a Spec that holds no data

Every block that shows something names a **context key**; the data arrives at render
time. That one choice buys everything the ask needed:

- **automated** — building a report is `render(template, context, format)`;
- **repeated** — the same template renders the next database lock, and its content hash
  proves the layout did not move between the two readouts;
- **styled** — `Theme` is a value, so a house style is passed rather than edited, and
  restyling produces a different content hash;
- **checkable** — `missing(template, context)` returns the names it still needs *before*
  anything is drawn.

`tests/unit/test_report.py::test_a_report_holds_no_data_only_the_names_of_data` asserts
the serialized template contains none of the numbers, which is the property the rest
depends on.

A **text** template language (Jinja and friends) was the obvious alternative and was not
taken. A Spec tree round-trips, hashes, diffs, and validates its placeholders when the
template is *built* rather than when it renders — all of which a string does not. The
fluent `ReportBuilder` gives the authoring ergonomics a text language would have, without
giving up any of that.

### 0007.5 — Two kinds of wrong, two kinds of failure

- A **missing context key** is `Unsupported`, naming every absent key at once so a
  scheduled report fails with a list rather than a `KeyError` on the first one.
- A key that is **present but of an unusable type** is a `ValueError` naming the block,
  the key and the type. That is a mistake in the template, not a gap in the data, and
  flattening it into `Unsupported` would send the caller looking in the wrong place.

### 0007.6 — Resolution happens once, for all three formats

`resolve` fills the template and produces renderer-neutral content; the renderers only
lay it out. So HTML, PPTX and PDF cannot disagree about what a number is, and the two
uncertainty rules live in one place instead of three.

### 0007.7 — What each format costs

| format | needs | figures |
|---|---|---|
| HTML | plotly (the `viz` extra) | **interactive**, plotly.js inlined once |
| PPTX | `python-pptx` + `kaleido` | rasterised |
| PDF | `reportlab` + `kaleido` | rasterised |

`kaleido` ships a browser engine to turn a figure into a PNG, which is exactly why the
static formats are behind the `report` extra and HTML is not. A user who wants an
interactive readout pays for plotly and nothing else.

`reportlab` rather than an HTML-to-PDF engine: WeasyPrint wants a cairo/pango stack and
a headless-browser route wants a browser. reportlab is pure Python, and the dependency
budget in `pyproject.toml` is a hard constraint.

Slide packing needs a rule and an implicit one is how decks end up with text off the
bottom. The rule is in the `render.pptx` docstring and is four lines long.

### 0007.8 — Two bugs the repository's own gates caught

Worth recording because both were of the class the gates exist for:

1. `except ImportError` around the *whole* import block in the PDF renderer turned a real
   bug — `reportlab.lib.units` has no `pt`, because reportlab already measures in points
   — into "install the report extra". The guard is now `importlib.util.find_spec` on the
   top-level package only; anything else propagates.
2. `test_no_silent_degradation::test_no_noqa_ble001` rejected a `# noqa: BLE001` on an
   `except Exception` wrapping the kaleido call. It was right: an installed kaleido that
   then fails is a real error, and returning `Unsupported(missing=("report",))` would
   send the caller to install something they already have. The catch is gone.

A third, cosmetic: the report's `Ledger` block collided with `calibrate.Ledger` in
Sphinx's cross-reference index — the same ambiguity class as `rule` in note 0004. Renamed
`LedgerBlock`, which reads better anyway: it renders a ledger, it is not one.

## The case study uses it

`nbs/case-studies/hypertension/06-running-the-trial.ipynb` §11 builds the HYPER-3 readout
from a template and writes HTML, PPTX and PDF. The figure in it is a `ResponseBand` from a
natural cubic spline fitted to the oldest age band, so the reversal arrives with a 95 %
interval; the metrics are `wald` intervals and print as such. Re-running after a second
database lock is the same call with a different context, and the template hash in the
document's `<meta>` is the auditable claim that the layout did not change.

## Left open

- **A figure block that composes several bands.** One figure, one source today; a
  small-multiples block over a mapping of bands is the obvious next one.
- **Cross-references and numbering.** Sections and figures are not numbered and nothing
  can refer to another block.
- **Theme presets.** `Theme()` is the only one shipped; a house style is a value the
  caller writes.
- **HTML page size.** The print stylesheet breaks on sections, which is close to but not
  the same as the PDF's pagination. A report that must match page-for-page across HTML
  and PDF does not yet.
- **`ResponseBand` for a dose *pair*.** The band is one-dimensional; a surface over two
  treatments has no equivalent object, and `viz` has no figure for one.
