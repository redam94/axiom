# 0012 — Beyond DAGs: which alternative frameworks axiom implements, and which it does not

*Kind: decision. Opened 2026-08-22. Status: implemented on `feature/dynamics-and-identifiability`; part of the 1.1.0 surface.*

## The premise

A DAG is not one assumption, it is four:

1. **causal sufficiency** — nothing unmeasured drives two variables at once;
2. **acyclicity** — no quantity is determined together with another, and nothing feeds back;
3. **the estimand is obvious from the picture** — the diagram displays what you will estimate;
4. **one population, complete data, one granularity**.

Each has a graphical framework that repairs it, and the frameworks form a
hierarchy rather than a menu (`DAGs ⊂ ADMGs ⊂ segregated graphs`; chain graphs
between; MAGs and PAGs the equivalence-class reading). Knowing where a problem
sits in that lattice tells you which identification theory applies.

This note records which repairs axiom now performs, which it deliberately does
not, and why. Note [0010](0010-dynamic-systems.md) covers the system *language*
that feeds these; this one is about what can then be **asked** of the graph.

## What landed

### D12.1 — ADMGs and the ID algorithm: a search became a proof

`identify()` searches a menu — back-door, front-door, instrument — and reports
the first route that works. Under latent confounding, drawn honestly, that
search is incomplete, and its silence is not evidence. Concretely, on

```
M -> Y, W -> M, W -> X, W -> Y, X -> M, M <-> W, X <-> Y
```

`identify` returns **blocked**, naming all three routes it tried, while the
effect is in fact identified by

```
sum_{M, W} P(W) P(M | W, X) [sum_{X} P(X | W) P(Y | M, W, X)]
```

`identify_effect` (the ID algorithm — Tian & Pearl 2002; Shpitser & Pearl 2006;
Huang & Valtorta 2006) is sound **and complete**: an estimand when one exists,
and a **hedge** — a pair of nested C-components — when none does. The hedge is
the part that changes behaviour. "The search found nothing" and "no estimand
exists" have opposite consequences: the first says look harder, the second says
stop and run an experiment. Only the second is a fact about the world.

`identify_conditional_effect` is IDC (Shpitser & Pearl 2008) for
`P(y | do(x), z)`.

**D12.1a — the estimand is an object.** ID returns a `Formula` — `Density`,
`Product`, `Marginal`, `Ratio` — not a sentence and not just a Boolean. It
renders to text and LaTeX for a report, and it *evaluates* against a discrete
joint (`JointTable`). That last part is not a convenience: it is what makes
`tests/recovery/test_id_recovery.py` possible, where every estimand is checked
against the true `P(y | do(x))` of a simulated world to floating-point
equality. An identification algorithm nobody can check is a claim, not a tool.

**D12.1b — both spellings of a latent are the same graph.** A bidirected edge
is a projected latent; a node in `unmeasured` is one waiting to be projected.
`latent_projection` runs first, so the two cannot disagree — and a test holds
them to giving an identical formula.

**D12.1c — identification is `downgraded`, never `identified`.** The algorithm
identifies *given the diagram*. The diagram itself, and positivity, are
assumptions it cannot check, so they ride along named and unverified. Under
`Verdict`'s own invariants that forces the status down, which is right.

### D12.2 — Cycles: sigma-separation, and a bug it exposed

`CausalGraph` refuses cycles, and it should: **d-separation is unsound on a
cyclic graph**. Conditioning on a variable inside a feedback loop does not block
the loop, because the loop's members are functionally intertwined.

`MixedGraph` drops the acyclicity requirement and answers with sigma-separation
(Forré & Mooij 2017), which differs from d-separation in exactly one place: *a
non-collider blocks only if it points to a node in a different strongly
connected component*. On an acyclic graph every component is a singleton, so
the criterion collapses to d-separation — the acyclic case falls out rather
than being special-cased.

**D12.2a — computed twice, on purpose.** `sigma_separated` walks the graph
directly; `acyclify` builds the acyclification (Mooij & Claassen 2020, Def. 3),
whose d-separation equals the original's sigma-separation by their Proposition
2. The rule is subtle enough that one implementation is a hypothesis. Two,
written from different definitions and agreeing over tens of thousands of
random queries on dense cyclic graphs, is evidence.

**D12.2b — this found a real error in note 0010's reduced form.** The graph
`unrolled_graph` emitted for a simultaneous block gave its members the block's
parents but **no bidirected edge between them**. That says two variables
determined together become independent once you condition on everything feeding
the block — which is false whenever the structural equations have disturbances,
i.e. always. The acyclification is the correct object and now *is* the reduced
form: shared parents *and* a bidirected clique. A test pins the consequence
(`quantity.t0` and `price.t0` are dependent given `cost.t0, income.t0`), and
another checks the acyclification's directed part still equals the reduced-form
edges `dynamics` computes independently.

`unrolled_mixed_graph` returns the cyclic graph as written, for looking at.

### D12.3 — SWIGs: the estimand on the picture

A DAG does not display potential outcomes, and reading one as an NPSEM-IE
smuggles in *cross-world* independences — relations between `Y(0)` and `Y(1)` —
that no experiment can check and that identification does not need. `swig`
(Richardson & Robins 2013) splits each intervened node into a random half that
keeps its parents and a fixed half that emits its children; d-separation on the
result reads off the single-world (FFRCISTG) independences.

**D12.3a — conditioning on a relabelled node is refused.** If an adjustment
variable becomes `M(x)`, it is a variable in the hypothetical world and data
cannot condition on it. Getting this wrong made `single_world_ignorability`
disagree with the back-door criterion on 45 % of queries; getting it right made
the two agree on all 10,544. That agreement is a theorem, and it is now a test.

**D12.3b — the sequential SWIG audits note 0010's g-formula.** The per-stage
adjustment sets `sequential_plan` derives from the sequential back-door
criterion are confirmed sequentially ignorable by the SWIG construction, which
shares no code with it. Two derivations of the same condition agreeing is worth
more than either alone.

## What was deliberately not built

The survey lists roughly twenty frameworks. Six clusters are worth naming here,
with the reason each was left out — a gap stated is worth more than a gap
implied.

| framework | what it repairs | why not now |
|---|---|---|
| **MAGs / PAGs, FCI** | you cannot recover a graph from data, only an equivalence class | axiom has **no causal discovery at all**; adding PAG discovery means adding conditional-independence testing, faithfulness assumptions and a whole verdict vocabulary for "orientable / not orientable". That is a phase, not a module. |
| **CPDAGs, PC/GES/GIES** | which edges observational data can orient | same reason. GIES is the principled way to say *what an experiment buys*, which fits axiom's design pillar well; it is the strongest candidate for next. |
| **Twin networks, ID\*/IDC\*** | layer-3 counterfactuals (probability of necessity) | needs the counterfactual graph and a partial-identification story (most such estimands are only bounded). The formula algebra built here is the prerequisite, and it is now in place. |
| **Chain graphs, segregated graphs** | interference and symmetric dependence *with* latent confounding | the right home for spillover between units. Segregated graphs need a third edge type and their own separation and identification theory. Large, and axiom has no interference vocabulary yet to attach it to. |
| **m-graphs (missingness)** | whether missing data can be recovered at all | small enough to be tempting, and genuinely useful next to `data.Completeness`. Left out only for scope; no complete algorithm exists for MNAR, so the honest version reports recoverable / not-known-recoverable. |
| **Cluster-DAGs, causal abstraction** | reasoning at a coarser granularity than the micro-model | directly relevant to `adapters`, where aggregate variables stand for many micro ones. C-DAG identification reduces to ID on the cluster graph, so it is now cheap; the hard part is the *abstraction* side — when an aggregate is causally coherent at all. |

Two more from the survey are recorded as **not worth productizing yet**:
category-theoretic / string-diagram formulations, and hypergraph causality.
Both are foundational rather than applied, and neither has settled
identification theory to implement.

Cyclic *identification* theory — as opposed to the cyclic Markov property
implemented here — is also far less developed than the acyclic case. axiom now
answers separation questions on cyclic graphs soundly; it does not claim to
identify effects in them beyond what the acyclification licenses.

## What this changes about using axiom

The recommendation, in the survey's words, is to "replace *I found a back-door
set* with *the ID algorithm returns this estimand, or a hedge proving
non-identification*". That is now a one-line change at the call site, and the
old route search remains worth running first: it is faster, and when it
succeeds it names a route a reader recognizes. The two are complementary, and
the notebook shows them side by side.

## Evidence

* `tests/unit/test_identify_cyclic.py` — the two sigma-separation
  implementations against each other on cyclic and acyclic graphs; the
  acyclification against `dynamics`' reduced form; the dependence the missing
  bidirected edge used to hide.
* `tests/unit/test_identify_id_algorithm.py` — back-door, front-door, napkin,
  bow arc, the graph the route menu refuses, IDC, and the formula algebra.
* `tests/unit/test_identify_swig.py` — SWIG ignorability against the back-door
  criterion over random graphs; the sequential SWIG against `sequential_plan`.
* `tests/recovery/test_id_recovery.py` — every estimand evaluated against the
  true interventional distribution of a simulated discrete world, and a pair of
  worlds that are observationally identical with different effects, which is
  what the bow arc's hedge asserts.
* `nbs/identify/05-beyond-dags.ipynb`.
