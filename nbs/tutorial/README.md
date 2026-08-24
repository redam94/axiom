# The tutorial — one question, carried all the way

[`01-the-whole-loop.ipynb`](01-the-whole-loop.ipynb) is the front door to `axiom`. It is
one continuous piece of work rather than a tour of the subpackages, and it crosses every
phase a decision actually has to go through:

> A logistics operator runs 40 distribution depots in one region and 500 nationally.
> **Should the standing weekly maintenance schedule go from nothing to 60 hours per depot?**

| section | what it decides | subpackages |
|---|---|---|
| 1. The question | the estimand the decision needs, and the break-even it has to clear | `core`, `estimands` |
| 2. Identification | whether the panel on hand can produce that number — it cannot | `identify` |
| 3. The belief we start from | the observational fit, inflated and stated as such | `sim`, `surface` |
| 4. Design | the effect to power for, the size, the value of the answer, three costed designs, and whether the estimator is calibrated on that panel shape | `design` |
| 5. Measurement | the experiment as one typed `Measurement`, and whether the old model agrees with it | `calibrate` |
| 6. Calibration | refitting under the measurement, and carrying it to the decision's estimand across a ledger | `calibrate`, `surface` |
| 7. The follow-up | whether to repeat it, when it goes stale, and which dose to test next | `design`, `estimands` |
| 8. The report | the document, written from the evidence record | `axiom_dossier` |

The world is synthetic (`axiom.sim`), so every estimate can be checked against a known
truth — and the punchline is that the experiment **reverses** the decision the
observational model would have made.

The last section uses [`axiom-dossier`](../../packages/axiom-dossier/README.md), the
report add-on. It is a separate distribution: `axiom` never imports it and none of the
twelve contract gates can see it. `uv sync --group dev` installs it, which is what lets
`make notebooks` execute this notebook.

Runs in about fifteen seconds:

```bash
uv run pytest --nbmake nbs/tutorial/
```

## Where to go next

- `nbs/end-to-end/` — the same phases, each taken further on its own.
- `nbs/case-studies/` — three full studies: a sequential dose-finding trial, a tutoring
  programme with a budget, and a physics experiment.
- `nbs/<subpackage>/` — every public symbol of every subpackage, executed.
