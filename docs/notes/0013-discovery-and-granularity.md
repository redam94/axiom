# 0013 — What the data can orient, and at what granularity

*Kind: decision. Opened 2026-08-22. Status: implemented on `feature/dynamics-and-identifiability`; part of the 1.1.0 surface.*

Two of the gaps note [0012](0012-beyond-dags.md) recorded, closed. Both were
picked for the same reason: each turns a question axiom could only answer with
an assumption into one it can answer with a computation.

## D13.1 — `axiom.discover`: observation identifies a *class*, and that is the finding

Before this, axiom took a graph as given. Where the graph came from was the
user's problem, and the honest position was that a hand-drawn DAG presented as
known is an assumption wearing a diagram's clothes.

The first thing structure learning has to say is a negative one: **observational
data cannot identify a DAG.** Markov-equivalent DAGs entail exactly the same
independencies, so nothing in the data separates them. What is identified is
the equivalence class, and `EssentialGraph` is it — directed where every member
agrees, undirected where they do not.

That is not a caveat bolted onto the output; it *is* the output. A chain
`a -> b -> c` comes back with both edges undirected, because that is the truth
about what the data knows.

**D13.1a — checked against the definition, not the theorem.** `cpdag` computes
the class with v-structures plus Meek's rules — a closure algorithm. The test
enumerates all 543 DAGs on four nodes, groups them by the *set of
d-separations they entail* (the definition of Markov equivalence), and checks
that the computed CPDAG equals the union of each group. 185 classes, and all
543 agree. Verma & Pearl's skeleton-and-v-structures characterization is then
checked against the same enumeration rather than assumed.

**D13.1b — the design question is the one worth exposing.** `orientation_gain`
prices a proposed experiment *in edges*, before it is run:

```python
orientation_gain(graph, [["a"]])   # (('a','b'), ('a','d'))
orientation_gain(graph, [["b"]])   # (('a','b'),)
```

Randomizing a variable severs its incoming edges, so an edge with exactly one
endpoint in the target has its direction revealed. An edge with *both* endpoints
in the target does not — both ends were randomized — so the biggest experiment
is not the most informative one, and the function says so. An experiment that
returns `()` teaches nothing you could not already have derived.

This is the piece that connects discovery to axiom's design pillar, and it needs
no data at all: it is a graph computation about a hypothesis.

**D13.1c — GES and GIES, with all three phases.** `ges` and `gies` are greedy
search over (interventional) essential graphs with a decomposable Gaussian BIC.
The interventional part is one line and it is the point: a row where a variable
was randomized carries no information about *that* variable's parents and full
information about everything else's, so it is excluded from one local term and
no others.

Candidates are applied, extended to a DAG, re-reduced to an essential graph and
**rescored in full** rather than by the incremental delta formulas. Slower;
nothing to get wrong.

**The turning phase is not optional, and this is why.** An interventional
essential graph orients a cut edge the moment the edge is inserted, so a forward
step that guesses a direction wrongly is locked in, and no single insert or
delete escapes it. On the four-variable test world with three intervention
targets, forward-and-backward alone walks to the *complete graph* — 1,668 nats
below the truth — because step 2 inserted `c -> d` where the world has
`d -> c`. Adding turning recovers the true DAG exactly. That trap is now a test,
with the gap size asserted, so a future change that drops turning fails loudly
rather than quietly returning a denser graph.

**D13.1d — the penalty is a dial, not a monotone one.** Raising the sparsity
penalty can produce a graph with *more* edges: at `penalty=20` on the test world
the search lands in a five-edge local optimum where `penalty=5` finds the true
four-edge class. Greedy search follows a path, and changing the score changes
the path as well as the destination. Documented on the field, and pinned by a
test, because the obvious reading of the knob is wrong.

**D13.1e — faithfulness is stated, not assumed silently.** Every method here
needs the data's independencies to be exactly the graph's, with no coincidental
cancellation. That assumption is not testable from the data it is assumed about.
The subpackage docstring says so, and `DiscoveryResult.steps` records every move
so a suspicious answer can be read back rather than guessed at.

## D13.2 — `identify.cluster`: the granularity you actually know

You rarely know the micro-level DAG; you often know how *blocks* relate. A
cluster-DAG (Anand, Ribeiro, Tian & Bareinboim 2023) is that partial knowledge
as a graph over clusters. Its semantics is entirely about absence: a missing
edge between two clusters says no variable in one is adjacent to any variable in
the other; within a cluster, anything goes.

Both transfers were verified by enumerating every compatible micro-level DAG:

* **separation** — every cluster-level d-separation holds in all compatible
  DAGs, and every statement holding in all of them is one the C-DAG makes;
* **identification** — `identify_cluster_effect` runs ID on the cluster graph,
  and its yes/no matches "identified in every compatible DAG".

The discriminating case is the cluster-level bow arc: the C-DAG says *not*
identified, and exactly 52 of the 96 compatible ADMGs identify it — so
"identified in all" is false, and the two agree for a reason rather than by
luck.

**D13.2a — a variable-level question is refused.** Asking for the effect of one
variable inside a multi-member cluster raises, naming the cluster and its size.
The C-DAG genuinely does not contain the within-cluster structure such an answer
needs, and quietly answering the coarse question instead of the one asked is the
failure the refusal exists to prevent.

This is the framework earning its keep beyond the inference. axiom's `adapters`
layer exists because aggregate variables stand in for many micro ones; an
aggregate that behaves like a variable is a *claim*, and a C-DAG is where the
claim gets written down.

## Still open

From note 0012's list: PAG/CPDAG discovery under **latent confounding** (FCI and
its variants) is untouched — everything here assumes causal sufficiency, which
is a large assumption and the one an ADMG exists to drop. Counterfactual
identification (ID\*), segregated graphs for interference, and missingness
graphs remain as recorded.

Two new items this work opens:

* **Discovery under latent confounding.** `EssentialGraph` has no `∘` edge mark,
  so it cannot represent a PAG. That is the natural next structure, and FCI
  would then produce it.
* **Refutation.** A discovered graph is a hypothesis, and axiom's `diagnose`
  layer already refutes *estimates*. Refuting a *structure* — testing the
  independencies a graph implies against the data that produced it — is the
  obvious pairing, and does not exist yet.

## Evidence

* `tests/unit/test_discover_essential.py` — the CPDAG against the definitional
  union over all 543 four-node DAGs; Meek closure; what an experiment orients.
* `tests/recovery/test_discover_recovery.py` — GES recovering the true CPDAG,
  GIES matching the true I-essential graph, the turning-phase trap with its gap
  asserted, and the non-monotone penalty.
* `tests/unit/test_identify_cluster.py` — both C-DAG theorems by enumeration.
* `nbs/discover/01-what-the-data-can-orient.ipynb`, and the cluster section of
  `nbs/identify/05-beyond-dags.ipynb`.
