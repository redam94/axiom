# site/ — the axiom marketing and demonstration site

A static site, hostable on GitHub Pages with no build step at serve time. Eight
pages: a landing page, one per pillar, the HYPER-3 case study, and a searchable
API map.

The rule the site is built on is the library's own fourth rule — **every number
carries its provenance.** Nothing here is typed by hand. Every figure and every
statistic is produced by running axiom, and every chart carries a stamp naming
the call that made it.

## Layout

```
site/
  index.html  identify.html  design.html  calibrate.html  surface.html
  meta.html  examples.html  benchmarks.html              <- generated; committed
  case-study.html  api.html
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

The `examples` section of `generate.py` executes every script in `examples/` in a
subprocess and stores its source alongside its captured stdout, so the examples
page cannot drift from the scripts and a broken example fails the generate step
rather than shipping stale prose.

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
