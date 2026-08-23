"""Conservation ecology — does a result from one reserve apply to another?

The question
    A grazing-exclusion trial ran in one reserve and worked. A second reserve wants
    to know what to expect. The two differ in rainfall, soil, and the composition of
    the herbivore community. Is the first reserve's number the second reserve's
    number, and if not, what would make it so?

Why this is not a statistics question
    No amount of data from the source reserve tells you whether it transports. That
    depends on *what differs* between the two populations and where those
    differences sit relative to the causal path. axiom takes a selection diagram --
    the graph, annotated with where the populations differ -- and reads off whether
    the effect transports, and what you would have to measure in the target to
    license it.

Pillars: identify (transportability)
"""

import numpy as np
from _walkthrough import Walkthrough

from axiom.identify import (
    directly_transportable,
    identify,
    ols,
    s_admissible_sets,
    selection_diagram,
    transport_verdict,
    trivially_transportable,
)
from axiom.sim import transport_pair

N, SEED = 6_000, 0

w = Walkthrough(
    field="Conservation ecology",
    title="Borrowing an answer from a reserve you did not study",
    question="""A grazing-exclusion trial worked in one reserve. A second reserve,
        with different rainfall and a different herbivore community, wants to know
        what to expect. Does the first reserve's number apply, and if not, what has
        to be measured before it can?""",
)

# ----------------------------------------------------------------------------------

w.step(
    "Say where the two populations differ, not just that they do",
    why="""A selection diagram is the ordinary causal graph with one extra piece of
        information: which mechanisms differ between the populations. Here the S-node
        points into Z — the reserves have different herbivore communities, but
        grazing exclusion works the same way in both. That distinction, between
        differing in who is in the population and differing in how the world works,
        is the entire question, and no dataset contains it.""",
    instead="""Pooling the two reserves and adding a site indicator. That estimates a
        blend of two populations, which is not the target reserve's answer and is not
        the source reserve's either. It also cannot represent the case that matters
        most: the mechanism itself differing.""",
)
source, target = transport_pair()
w.out(f"source reserve : {source.graph.to_text()}")
w.out(f"target reserve : {target.graph.to_text()}")
w.out(f"S-node on      : {list(source.graph.selection)}")
w.out("")
diagram = selection_diagram(source.graph)
w.out(f"selection diagram : {diagram.to_text()}")
w.out("  S[Z] is a population indicator, not a variable anyone records --")
w.out("  which is why it is marked unmeasured and never appears in a fit.")

source_frame = source.observed(source.simulate(N, seed=SEED))
target_frame = target.observed(target.simulate(N, seed=SEED + 17))
w.figure(
    "populations",
    kind="dumbbell",
    data={
        "rows": [
            {
                "label": f"mean {col}",
                "a": float(source_frame[col].mean()),
                "b": float(target_frame[col].mean()),
            }
            for col in ("Z", "X", "Y")
        ]
    },
    opt={
        "rows": "@rows",
        "xLabel": "population mean",
        "aLabel": "source reserve",
        "bLabel": "target reserve",
        "labelWidth": 120,
    },
    title="What actually differs between the two reserves",
    note="""Z moved, and X and Y moved with it because Z causes them both. Nothing
        about the mechanism changed — which is precisely the claim the S-node makes,
        and precisely what makes the effect borrowable.""",
    legend=(("ink-2", "source reserve"), ("accent", "target reserve")),
)

edges = np.linspace(-4.5, 6.0, 41)
centres = ((edges[:-1] + edges[1:]) / 2).tolist()
w.figure(
    "shift",
    kind="lines",
    data={
        "z": centres,
        "source": (np.histogram(source_frame["Z"], bins=edges, density=True)[0]).tolist(),
        "target": (np.histogram(target_frame["Z"], bins=edges, density=True)[0]).tolist(),
    },
    opt={
        "series": [
            {"label": "source", "x": "@z", "y": "@source", "colour": "ink-2"},
            {"label": "target", "x": "@z", "y": "@target", "colour": "accent"},
        ],
        "xLabel": "Z — herbivore community composition",
        "yLabel": "density",
        "height": 250,
        "rightPad": 70,
    },
    title="The same mechanism, a different population inside it",
    note="""Two reserves drawn from different parts of the same covariate space. If
        the effect of exclusion varied with Z, this shift alone would make the source
        number wrong for the target — which is why the answer depends on the shape of
        the graph and not on how far apart these two curves are.""",
    legend=(("ink-2", "source reserve"), ("accent", "target reserve")),
)

# ----------------------------------------------------------------------------------

w.step(
    "Ask the three transport questions, before collecting anything",
    why="""They are increasingly demanding and each has a different consequence.
        Directly transportable would mean the source number is the target number
        untouched. Trivially transportable would mean the target's own data suffices
        and the source is irrelevant. The general verdict is the interesting middle:
        transportable, but only after re-weighting on a specific set.""",
    instead="""Running the analysis in the source, then arguing about external
        validity in the discussion section. By then the target's data collection has
        already happened, and the covariates that would have licensed the transfer
        were not on the form.""",
)
w.table(
    ["question", "answer", "what it would have meant"],
    [
        [
            "directly transportable",
            str(directly_transportable(source.graph, "X", "Y")),
            "the source estimate is the target estimate, untouched",
        ],
        [
            "trivially transportable",
            str(trivially_transportable(source.graph, "X", "Y")),
            "the target's own data is enough; ignore the source",
        ],
    ],
)
verdict = transport_verdict(source.graph, "X", "Y")
w.out(f"transport verdict : {verdict.status}")
for f in ("route", "reason"):
    value = getattr(verdict, f, None)
    if value:
        w.out(f"  {f:8s}: {value}")
w.say(f"""Read the middle row carefully: trivially transportable is
    {trivially_transportable(source.graph, "X", "Y")}, which says the target could
    answer this from its own data — if it ran its own trial. It has not, and that is
    the whole reason the question is being asked. The verdict's route is
    '{verdict.route}': borrow the source, re-weighted on a set the target must
    measure. Transport is what you use when the target has covariates but no
    experiment.""")

# ----------------------------------------------------------------------------------

w.step(
    "Turn the verdict into a shopping list for the second reserve",
    why="""The S-admissible sets are the useful output of this entire example. They
        say which covariates must be measured in the TARGET so the source effect can
        be re-weighted onto it. That is a decision about a field protocol, made
        before anyone flies out, and it is the thing a discussion-section paragraph
        about generalisability never delivers.""",
    instead="""Measuring everything. Field time in a remote reserve is the binding
        constraint, and 'collect all covariates' is how the two that mattered end up
        recorded badly alongside forty that did not.""",
)
sets = s_admissible_sets(source.graph, "X", "Y")
w.out(f"S-admissible sets : {[sorted(s) for s in sets]}")
w.out("")
w.out("Measure these in the target and the source effect can be re-weighted onto")
w.out("the target population. Anything else is optional.")

# ----------------------------------------------------------------------------------

w.step(
    "Check the borrowed number against the target's own",
    why="""The transfer is licensed by the diagram, but this is a synthetic pair, so
        the claim can also be checked: estimate in the source, estimate in the target,
        and compare both to the target's true effect. In a real study only the first
        of those three exists, which is why the diagram had to carry the argument.""",
    instead="""Treating agreement here as the evidence that transport works. It is
        not — it is a check that the implementation matches the theorem. The evidence
        that transport works is the S-node claim, and that is an ecological
        judgement.""",
)
v = identify(source.graph, "X", "Y")
source_est = ols(source_frame, "Y", "X", covariates=v.adjustment_set)
target_est = ols(target_frame, "Y", "X", covariates=v.adjustment_set)
target_truth = target.total_effect("X", "Y")
naive_source = ols(source_frame, "Y", "X")

rows = []
for label, est in (
    ("source, unadjusted", naive_source),
    ("source, adjusted", source_est),
    ("target, adjusted", target_est),
):
    ci = est.ci(0.95)
    w.out(f"{label:20s} {est.estimate:6.3f} +/- {est.se:.3f}  [{ci.lower:6.3f}, {ci.upper:6.3f}]")
    rows.append(
        {
            "label": label,
            "estimate": est.estimate,
            "lower": ci.lower,
            "upper": ci.upper,
            "note": f"error against the target truth {est.estimate - target_truth:+.3f}",
            "bad": abs(est.estimate - target_truth) > 2 * est.se,
        }
    )
w.out(f"{'target truth':20s} {target_truth:6.3f}")
w.figure(
    "transfer",
    kind="intervals",
    data={"rows": rows},
    opt={
        "rows": "@rows",
        "truth": target_truth,
        "truthLabel": "the target's true effect",
        "xLabel": "effect of grazing exclusion",
        "labelWidth": 155,
        "rowHeight": 38,
    },
    title="The source reserve's adjusted answer is the target reserve's answer",
    note="""Conditioning on the S-admissible set is what carries it across. The
        unadjusted source estimate is wrong in both reserves, for the ordinary
        confounding reason rather than for a transport reason — worth keeping on the
        chart so the two failures are not confused.""",
)

# ----------------------------------------------------------------------------------

w.step(
    "Move the S-node onto the mechanism and watch the route change",
    why="""Everything above rests on where the populations differ. Put the S-node on
        Y as well — the reserves now respond to exclusion differently, rather than
        merely containing different herbivores — and the same call on the same data
        comes back by a different route. The answer tracks the ecology, not the
        sample size.""",
    instead="""Assuming a difference in outcomes implies a difference in mechanism.
        The first chart in this walkthrough shows Y differing between the reserves
        while the mechanism is identical. Outcome differences are what a population
        shift looks like; they are not evidence about the mechanism either way.""",
)
mechanism_differs = source.graph.with_selection("Y")
harder = transport_verdict(mechanism_differs, "X", "Y")
w.out(f"S-node on          : {list(mechanism_differs.selection)}")
w.out(f"transport verdict  : {harder.status}")
for f in ("route", "reason"):
    value = getattr(harder, f, None)
    if value:
        w.out(f"  {f:8s}: {value}")
w.out(f"directly transportable : {directly_transportable(mechanism_differs, 'X', 'Y')}")
w.say(f"""The status is still '{harder.status}', which is easy to misread as 'nothing
    changed'. The route is what changed: from '{verdict.route}' to '{harder.route}'.
    'Trivial' means the effect is identifiable from the target's own data and the
    source contributes nothing — which is the correct answer, and an expensive one.
    Once the mechanism differs, the first reserve's trial is no longer evidence about
    the second, and the second has to run its own.""")

# ----------------------------------------------------------------------------------

w.finding(f"""The source reserve's adjusted estimate was {source_est.estimate:.3f} and the
    target's true effect was {target_truth:.3f}. The transfer worked, and it worked
    because the mechanism is shared and only Z's distribution differs — which is a
    claim about ecology that the diagram records and no amount of data from either
    reserve could establish.""")
w.finding(f"""The useful output is not that number. It is the list {[sorted(s) for s in sets]}:
    measure that in the target and the borrowed effect is licensed. Decided before
    the field season, it is a line on a protocol. Decided afterwards, it is a
    limitation.""")
w.finding(f"""And the call is not automatic. Adding an S-node on the mechanism moved the
    route from '{verdict.route}' to '{harder.route}' with no change to the data at
    all — from 'borrow the source, re-weighted' to 'the source tells you nothing, run
    your own trial'. The same three lines of code give opposite operational advice
    depending on a claim an ecologist makes about the two reserves, which is exactly
    where that decision belongs.""")
