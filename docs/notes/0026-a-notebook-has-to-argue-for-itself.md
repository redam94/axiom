# 0026 — A notebook that only demonstrates the API has not made the case for it

*Kind: decision + progress. Opened 2026-08-26. Status: implemented on
`feature/notebook-narrative`.*

Note 0025 made the eighty-four notebooks *render*. They still opened on a
definition. `nbs/design/02` began "One scalar parameter `θ` with a Gaussian
prior of sd `prior_sd`"; `nbs/core/01` began "Every quantity in axiom carries a
`Dimension`". Both sentences are true, and neither says why a reader should
spend an afternoon here. Fifteen of the eighty-four produced a figure at all,
and several of those printed the figure's trace count instead of the figure.

Rule 6 says every public symbol appears in an executed notebook. Gate 12
enforces it, and gate 12 is satisfied by a notebook nobody would read.

## D26.1 — every notebook opens on the failure the subpackage prevents

Not on the type it introduces. The opening is a specific, ordinary way to get a
wrong number, stated in the vocabulary of somebody who has to make a decision:
a per-week rate read as per-day (seven times too fast, nothing raises); a
collider that turns an exact 1.0 into 0.33 with *tighter* standard errors; two
teams reporting "the lift" eight times apart; the parameter whose posterior is
its prior returned unchanged.

Then the machinery, and then a closing paragraph that says what the reader now
has that they did not before. The technical content of each notebook was kept —
this is a reframing and an addition, not a rewrite of what the notebooks
demonstrate.

Where the claim is a number, the number is computed rather than asserted. Nine
draft captions made a claim the data denied and were corrected against the
output rather than the other way round; two of those were pre-existing errors in
the notebooks (see D26.4).

## D26.2 — `nbs/_style.py`: one template, one validated palette, ten builders

Presentation, not library code, and it lives under `nbs/` for that reason.
`axiom.viz` draws *axiom's result types* and every function there is duck-typed
over one of them; `_style` takes arrays and draws the *argument* — a curve
inside its band, ranked bars, a forest, a dumbbell, densities, recovery against
truth, a heat grid, and the marks that point at the number a paragraph is about.

Reach for `axiom.viz` whenever a figure exists for the object at hand. There are
about seventy `viz` calls across the series now, and they come first.

The palette is the data-viz reference instance, validated with its own script
rather than by eye: blue / orange / aqua clear the colour-blindness and
normal-vision separation floors on every pair against this surface, which is why
no builder draws a fourth series without being asked. Two of the three sit below
3:1 contrast, so every series is direct-labelled as well as legended — colour
never carries identity alone.

Ordered quantities get an ordinal ramp instead of categorical hues: HYPER-3's
control-plus-three-doses is a one-hue blue ramp with the control in muted ink,
and its age bands a second ramp in orange. Both were validated `--ordinal`.

The three case-study helper modules (`hyper3`, `tutoring`, `scattering`) had a
palette and a `plotly_white` layout of their own. They now import `_style` and
register the same template, so the eighty-four notebooks are one product rather
than four.

## D26.3 — the notebooks are a `[viz]` surface

`nbs/_style.py` imports plotly unguarded. The notebooks are a development
surface, `make notebooks` installs the whole dev group, and the alternative —
every figure cell behind an `available()` check — buys nothing for a reader and
costs the notebooks their point. `axiom` itself is unaffected: the import-weight
gate still fails if `import axiom` puts plotly in `sys.modules`.

## D26.4 — three claims the notebooks were making that the code denies

Writing a caption against real output is itself a check, and it caught three things.

**`design/03`** said the ghost estimator returns approximately zero on a holdout
panel because the exposure mask carries no effect there. It returns 1.33 against
a truth of 1.0: on that panel the mask *does* track treatment. The switchback
row is the one that returns ~0, and the notebook now says so.

**`meta/04`** drew its "symmetric" corpus at seed 11, where the noise happened to
correlate with the standard errors: Egger rejected symmetry at p = 0.003, on the
corpus whose whole job was to show what symmetry looks like. Reseeded to 0 —
0.58 symmetric, 0.0000 asymmetric.

**`design/08`** said α's profile never reaches the 95% boundary. Its own output
reports a drop of 27.6 at four of six grid points, and 0.008 at the fifth. The
cause is real and worth recording: `profile_likelihood` warm-starts each
constrained fit from the neighbouring one, and on an exactly flat ridge the
optimizer terminates without moving the companion parameter, so the reported
drops are an artifact of where the walk started rather than a profile. The
notebook now says that, and reads its interval from `simulated_identifiability`,
which refits from its own start and correctly reports `[-inf, inf]`.

The third is a defect in `axiom.design.identifiability`, not in the notebook.
It is left as a 1.1 item: a profile over a flat direction should either
re-optimize from a fresh start when the warm start does not move, or report that
it could not profile.

## What is not done

- The figures are not tested. `make notebooks` executes them, so a figure that
  raises fails the build, but nothing checks that a caption's claim matches the
  numbers beneath it. The nine corrections above were found by reading.
- `_style` is not type-checked: mypy's `files` is `src/axiom`, and `nbs/` is
  outside it as `tests/` and `examples/` are.
