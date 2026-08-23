# site/ — the axiom marketing and demonstration site

A static site, hostable on GitHub Pages with no build step at serve time. Eleven
authored pages -- a landing page, one per pillar, the examples index, the
benchmarks, two case studies and a searchable API map -- plus one generated
walkthrough page per example in `examples/`.

The two case studies are deliberately opposite shapes. **HYPER-3** is an analysis
story: a trial runs and a stopping rule fires. **GEIGER-1911** is a design one --
nothing is fitted until the last section, because the deliverable is a set of
angles, an aperture geometry, a time allocation and a stopping rule, all produced
before an apparatus exists.

The rule the site is built on is the library's own fourth rule — **every number
carries its provenance.** Nothing here is typed by hand. Every figure and every
statistic is produced by running axiom, and every chart carries a stamp naming
the call that made it.

## Layout

```
site/
  index.html  identify.html  design.html  calibrate.html  surface.html
  meta.html  examples.html  benchmarks.html              <- generated; committed
  case-study.html  rutherford.html  api.html
  example-01-....html ... example-12-....html            <- one per walkthrough,
                                                            generated from data
  _src/*.html          content fragments (edit these)
  _gen/generate.py     runs axiom, writes assets/data/*.json
  _gen/build.py        wraps _src fragments in the shared shell
  assets/
    axiom.css          the design system
    charts.js          hand-rolled SVG charts (theme-aware, hoverable)
    site.js            theme toggle, copy buttons, API filter
    data/*.json        generated — the numbers behind every page
```

## Rebuilding

```bash
# 1. regenerate the data by actually running axiom (~2 minutes; it fits models)
python site/_gen/generate.py

#    or one section at a time while iterating
python site/_gen/generate.py identify design

# 2. re-render the pages from the fragments
python site/_gen/build.py

# 3. look at it
python -m http.server 8000 --directory site
```

## Where the API page comes from

`api.html` lists every public symbol of every subpackage, and each one opens to
show three things: its signature, the first line of its docstring, and **a real
call**. None of that is written by hand.

The signature and the docstring come from `inspect`. The call comes from the
notebooks: gate 12 already guarantees every public symbol is used in an executed
notebook, so the `api` section of `generate.py` parses that subpackage's
notebooks and lifts out the shortest *executed statement* that uses the symbol,
preferring one that calls it. It walks every statement rather than only the
top-level ones, so when the only construction sits inside a helper the page shows
that line instead of the whole function.

Two rules keep it honest:

- **Imports do not count.** `from axiom.surface import Hill` demonstrates nothing
  about how `Hill` is called, so import statements are skipped. A symbol whose
  only appearance in the notebooks is inside an import gets no snippet and the
  page says so. Five symbols are in that position today; gate 12 passes them
  because an import satisfies it, and this page is where that shows.
- **The output has to be reproducible.** Pydantic renders validator objects and
  memory addresses into an annotation's repr, which would churn the committed
  JSON on every build; `_readable_type` collapses `Annotated[T, ...]` to `T` and
  strips addresses, and a default longer than 40 characters is shown as `...`
  (`design.simulated_power` otherwise carries the entire method registry inline,
  six thousand characters of signature for one parameter). Running the section
  twice produces byte-identical output.

The package list, blurbs and layers live once at the top of `generate.py` as
`ORDER` / `BLURB` / `LAYER`, read by both the `overview` and `api` sections, so
the landing page's symbol count cannot disagree with the map.

## Where the walkthrough pages come from

The `examples` section of `generate.py` runs every script in `examples/` in a
subprocess with `AXIOM_WALKTHROUGH_JSON` set. Each script narrates itself through
`examples/_walkthrough.py`, so that one run produces both the terminal walkthrough
and a structured record: the steps, the reasoning behind each one, the alternative
it rejected, the readouts, the tables and the payload for every chart.

`build.py` renders one page per example from that record. The code shown against a
step is sliced out of the source file by line number -- the code that ran between
that step's heading and the next, with each figure's `title`, `note` and `legend`
elided because the page renders them with the figure a few lines below. The
elision is marked in the code and every page carries the unedited file at the
bottom. So a page cannot describe a step the script no longer takes, and a broken
example fails the generate step rather than shipping stale prose.

Each example's figures are written to their own `assets/data/figures-<stem>.json`,
so a walkthrough page fetches its own charts and not the other eleven examples'.
An example writes its chart options with `@`-prefixed references into its own
payload (`{"rows": "@rows"}`); `build.py` resolves those to real paths and fails the
build if one does not resolve, so a mistyped reference can never reach a reader as
an empty box.

To iterate on one example without re-running the other eleven:

```bash
AXIOM_EXAMPLES_GLOB='07*.py' python site/_gen/generate.py examples
python site/_gen/build.py
```

`build.py` substitutes `{{path.to.value:format}}` tokens in the fragments with
values from `assets/data/*.json`, so a number in the prose is baked in at build
time rather than fetched by script. An unresolvable token is a hard error — the
build fails rather than shipping a blank. `{{@ value lower upper }}` computes
where a point estimate falls inside its own interval, which is how the interval
readouts place their marker honestly instead of centring everything.

Charts are declared in markup and drawn client-side from the same JSON:

```html
<div class="chart" data-chart="band" data-file="surface"
     data-opt='{"key":"band_a","height":300}'></div>
```

Chart types: `sequential`, `band`, `lines`, `intervals`, `funnel`, `bars`,
`heatmap`, `scatter`, `hist`, `dumbbell`.

## Conventions worth keeping

- **Red is reserved.** `--boundary` marks a threshold that was crossed, never a
  data series. The one red mark on the landing page means something.
- **Every figure gets a `.stamp`** naming the call that produced it.
- **Every chart gets a table view.** `attachTable` adds it automatically.
- **Categorical colours are the validated palette**, assigned in fixed slot
  order and never cycled. Light and dark are separately stepped, not inverted.
- **Derived flags belong in the generator, not the prose.** For example the
  "contains truth" column on the surface page is computed in `generate.py`, so
  the table cannot flatter the fit.

## Publishing

`.github/workflows/pages.yml` uploads `site/` to GitHub Pages on pushes to
`main` that touch it. One-time setup: repository **Settings → Pages → Source:
GitHub Actions**.

Because the generated HTML and JSON are committed, the workflow only uploads —
it does not need to install axiom or fit anything.
