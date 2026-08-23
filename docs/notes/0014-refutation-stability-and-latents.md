# 0014 — Refuting the graph, resampling the search, and dropping sufficiency

*Kind: decision. Opened 2026-08-22. Status: implemented on `feature/dynamics-and-identifiability`; part of the 1.1.0 surface.*

Note [0013](0013-discovery-and-granularity.md) gave axiom a way to *learn* a
graph. This note covers the three things that have to be true before anyone
should act on one: that the graph you drew has been tested against the data,
that the graph a search returned is not an artifact of the rows it happened to
see, and that neither result quietly assumed you measured every common cause.

They were built in that order deliberately. Refutation applies to the graph a
user already has, which is the common case; stability qualifies discovery's
answer; FCI removes the last standing assumption. Each is useful without the
next.

## D14.1 — `diagnose.structure`: the graph is falsifiable, so falsify it

Every identification result in axiom already carries an `Assumption` named
`graph_is_correct` in state `unverified`, and `diagnose.refute` then works hard
on the *estimate* — placebos, permutations, subsets, added noise — while the
diagram that licensed the estimate goes unexamined. That is backwards: the
graph is the one input with a free, sharp test attached, because a DAG implies
conditional independencies and the data either shows them or does not.

`refute_structure` enumerates the implications (every non-adjacent pair, given
the parents of one of them), tests each, and returns a `StructureRefutation`.

**D14.1a — the verdict can only ever be negative.** A test that fails to reject
is not evidence of independence: it may be underpowered, or the dependence may
be nonlinear and invisible to a partial correlation. So a graph that survives
comes back `downgraded`, with `graph_is_correct` still `unverified` — never
`identified`. A graph that fails comes back `blocked` with that assumption
`violated`. There is no path through this module that upgrades a graph.

This asymmetry is the whole design. It is easy to write a checker that says
"passed", and every such checker is wrong.

**D14.1b — effect size decides, not p alone.** With 200 rows nothing rejects
and a graph passes by being untested; with a million rows everything rejects,
because every graph is an idealization and idealizations are detectably false at
scale. Both `alpha` and `effect_threshold` are parameters, and every check
carries its partial correlation so a reader can see which regime they are in.
Multiplicity is handled with Holm (default) or Benjamini-Hochberg — the
implications of even a small graph number in the dozens, and uncorrected that
guarantees a false refutation.

**D14.1c — a refutation names its repair.** Each implication tested is the
absence of one edge, so a failure points at exactly the edge whose absence the
data denies. `implicated` lists them worst-first. The output is not "your graph
is wrong" but "these edges are missing, this one most of all" — which makes
refutation a *step* in the loop rather than a verdict at the end of it:

    draw -> refute the structure -> repair -> identify -> estimate ->
    refute the estimate -> decide

## D14.2 — `discover.stability`: one graph is not a finding

A structure search returns exactly one graph and says nothing about how much of
it the data determined and how much the last few rows did. `edge_stability`
resamples the rows, re-runs the search, and counts how often each edge came
back. The output is per-edge rather than per-graph, which is the useful
granularity: a discovered graph is rarely all right or all wrong, it is a stable
core with a fringe.

**D14.2a — two ways to be unstable, kept apart.** `adjacent` is how often the
pair was connected at all; `forward` / `backward` / `undirected` split that by
direction. A pair with `adjacent = 1.0` and `undirected = 0.95` is *not* an
unstable edge — it is a **stable edge whose direction observation cannot
settle**, and more rows will not change it. That calls for an intervention,
and `orientation_gain` prices which one. `contested` and `undecided_direction`
are separate accessors because they imply different next actions.

**D14.2b — the resample keeps the regime.** Rows are resampled with their
intervention labels attached, so an interventional row stays interventional and
`gies` is still the right search on the resample. Choosing the search from the
data (`gies` when there are targets, `ges` when there are not) means the
stability of an experimentally-oriented edge is measured under the experiment.

**D14.2c — what the bootstrap cannot see.** It measures sampling variability
and only that. A confounder missing from the data is missing from every
resample and shows up as a beautifully stable wrong edge. The module docstring
says this, because a stability number is exactly the kind of output that reads
as a confidence when it is not one — which is the argument for D14.3.

## D14.3 — `discover.fci`: dropping causal sufficiency

`ges` and `gies` assume no unmeasured variable drives two measured ones. That
is the assumption ADMGs exist to drop; dropping it is why `identify_effect` can
return a hedge instead of a false estimand. Discovery was the last place in
axiom still making it.

FCI's output is a PAG, whose edges carry a mark at each end: an arrowhead means
*not an ancestor*, a tail means *is an ancestor*, a circle means the data does
not determine which. So `x <-> y` says a latent common cause — neither causes
the other — and `x o-> y` says `y` does not cause `x` while leaving open
whether `x` causes `y` or something hidden drives both. A CPDAG has no way to
write either sentence, which is why the vocabulary is the deliverable and not
just the algorithm.

**D14.3a — sound, deliberately not complete.** The adjacency search, the
Possible-D-SEP refinement that separates FCI from PC, v-structure orientation,
and Zhang's rules R1–R3 are implemented. **R4 (discriminating paths) and R5–R10
are not.** Leaving rules out leaves circles where a complete implementation
would place marks: the result is sound but not maximally informative. Every
mark placed is right; fewer may be placed than could be. Under-orientation is
the safe direction to be incomplete in, and `PAG.limits_hit` says so on every
result rather than in a docstring nobody reads.

**D14.3b — tested against an oracle first, data second.**
`oracle_independence` answers conditional-independence queries from a known
graph by d-separation. Running FCI against it separates the algorithm from the
statistics: with a perfect test any disagreement is a bug, not a sample. The
soundness property is then checked as a property rather than on examples — over
random four-variable graphs with a latent, every arrowhead is confirmed against
the truth's ancestor relation and every tail likewise. Not one unsound mark.
The data-driven tests check that a real test with enough rows reaches the same
place.

## D14.4 — one Spec-design change the gates forced

Gate 4 mutates a Spec's fields and re-hashes. `PagEdge` and `EdgeSupport` both
store a pair of variables and had validators that *rejected* an unsorted pair.
Under mutation that is a crash rather than a finding, and the fix is the better
design anyway: **normalize rather than refuse.** Writing `PagEdge(a="y",
b="x", mark_a="arrow", mark_b="tail")` now yields the identical object as
`PagEdge(a="x", b="y", mark_a="tail", mark_b="arrow")` — the marks travel with
their own ends, as do `forward` / `backward` on `EdgeSupport`. Canonical form
is enforced without making callers responsible for producing it, and equality
and content-hashing on these Specs now mean what a reader expects.

## What is still open

* **R4 and R5–R10.** The completeness rules. Under-orientation is safe but it
  is not free — a discriminating path can settle a mark that currently comes
  back a circle.
* **Nonlinear independence testing.** Both refutation and FCI use partial
  correlation, so a dependence with zero linear component is invisible to them.
  A kernel or distance-correlation test would widen what can be refuted; the
  `PartialCorrelation` protocol boundary is where it would plug in.
* **Stability for PAGs.** `edge_stability` resamples GES/GIES. The same loop
  over FCI would give per-mark support, which is arguably the more honest
  output when latents are on the table.
