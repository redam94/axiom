# 0010 — Systems that are not DAGs: simultaneity, time, and the compiler that removes both

*Kind: decision. Opened 2026-08-22. Status: implemented on `feature/dynamics-and-identifiability`; part of the 1.1.0 surface.*

## The problem

A causal DAG is a **solved** model. It presumes the variables can be ordered so that every
arrow points forward. Two ordinary modelling situations break that presumption without
being ill-posed:

* **Simultaneity.** Price and quantity are set together. A dose responds to the outcome it
  moves, within the same accounting period. Written honestly the equations form a cycle,
  and the object is not a DAG.
* **Time structure.** Yesterday's outcome enters today's equation. The summary graph over
  `{treatment, outcome}` has a loop; the graph over `{treatment@t, outcome@t}` for every `t`
  does not.

Before this note, axiom's only response to either was `CausalGraph.feedback`: a boolean that
says "the static summary graph hides feedback, so static adjustment is insufficient" and
downgrades the verdict. That flag is a refusal, not an answer. `identify.verdict` even
carries a named assumption for it (`no_time_varying_confounding`) whose `challenged_by` line
reads *"g-methods or an unrolled graph are needed"*. This is that unrolled graph.

## The decision: compile, do not extend

**D10.1 — Simultaneity and time are removed by a compiler, not by new evaluator nodes.**

The tempting design is a node — `Solve`, or a real `ODESystem` evaluator — that every
interpreter then has to implement. That would put a solver inside `value`, inside the jax
interpreter, inside the PyTensor interpreter, and inside anything added later; the four
would have to agree, and rule 3 ("one `forward()`") would be enforced by hope.

Instead `axiom.dynamics` is a *front end*. A `DynamicSystem` is a declarative `Spec`; the
compiler turns it into ordinary `axiom.core.expr` trees with no cycles and no time shifts.
Everything downstream — `value`, jax, PyTensor, `dimension`, `latex`, `linearize`, the
design math, the identifiability analysis — works on the result with no special case, and
gate 9 (numpy vs jax agreement) covers it for free.

`axiom.core.expr` is unchanged by this work. Not one node was added.

**D10.2 — Layer 3.** `dynamics` imports `axiom.core` and nothing else from axiom, so
`identify` (4), `design` (5) and `diagnose` (6) can all read a compiled system.
`identify.dynamic` holds the bridge to `CausalGraph`, because identification of a dynamic
system is identification, and belongs with the rest of it.

## The language

Variables carry a dimension and a role; equations are ordinary expression trees whose
`Data` leaves are *time-shifted references*. A variable name is a plain identifier, so two
suffixes are unambiguous:

| form | means | written by |
|---|---|---|
| `y` | `y` at `t` | an equation |
| `y.l2` | `y` at `t-2` | an equation |
| `y.t3` | `y` at absolute period 3 | the unroller |

**D10.3 — a surface syntax for the equations only.** The declarations stay in Python, where
they are checked; only the equations get a syntax, because that is the part that is
unreadable as a tree:

```python
system = parse_system(
    """
    quantity = a - b * price + c * income
    price    = d + e * quantity + cost
    """,
    variables=(...),
    parameters=(...),
)
```

It has no distributions, no priors, no `do` operator, no plate notation. Everything else in
axiom is a `Spec`, and a second way to declare priors would be a second thing to keep in
sync. `y[t-1]` and `y.l1` are the same reference; `y[t+1]` parses and is refused, naming
what would be needed (a rational-expectations solution, which is out of scope — see below).

## Blocks

The strongly connected components of the contemporaneous dependency graph (Tarjan 1972) are
exactly the sets that must be solved together — the block-recursive form of Fisher (1966,
ch. 4). Lagged edges never enter a block, which is why unrolling gives a DAG.

The decomposition also bounds the work: forty equations with two two-variable cycles is
thirty-eight substitutions and two 2×2 solves, not one 40×40 solve.

## Solving a block

**D10.4 — exact for a linear block; a declared and measurable approximation otherwise.**

* `affine_split` decides *structurally* whether each right-hand side is affine in the
  block's unknowns and extracts the coefficient expressions. If it is, Gauss-Jordan
  elimination over those expressions gives the reduced form — exact for every parameter
  value, and differentiable, so the design math sees the true derivative. This is the
  econometric reduced form built by the same algebra a textbook uses.
* A nonlinear block is compiled as `sweeps` Gauss-Seidel passes from the variable's declared
  `initial` value. `BlockSolution.exact` is `False` and `residuals` carries `rhs(v) - v` per
  unknown, so the error you are actually running is *evaluable on your data* rather than
  promised. A block with no `sweeps` is `Unsupported`, not a guess.

Two properties worth knowing before reading a residual: the unknown updated **last** in a
Gauss-Seidel sweep satisfies its own equation exactly, so its residual is identically zero —
read the largest residual over the block. And the tree grows *geometrically* in the sweeps,
so useful counts are single digits; the node budget is checked after every pass, which turns
an optimistic `sweeps=25` into an `Unsupported` in a second instead of an out-of-memory.

A block that is symbolically singular — no equation determines a variable once the others
are eliminated — is `Unsupported` naming the variable. That is an identification statement
about the model, not a numerical accident, and it deserves to be said out loud. Detecting it
needs exact recognition of expressions like `1 - 1`, which is why `algebra.constant_value`
evaluates constant subtrees rather than folding constants into the tree.

## Two unrollings, because there are two meanings

**D10.5 — the conditional and the marginal form are both first class, and named.**

| | conditional (`conditional_form`) | marginal (`unroll`) |
|---|---|---|
| shape | one generic period | one expression per node per period |
| lags | real panel columns (`y.l1`) | substituted away |
| reads | `x`, `y.l1`, … | `x.t0 … x.tT` only |
| needs | every lagged endogenous variable **measured** | nothing measured |
| is | the ordinary dynamic regression | the trajectory, and what identification reads |

They agree numerically when the lagged columns are the model's own values and disagree
otherwise; the difference is exactly the difference between a one-step-ahead conditional
mean and a marginal one. Conflating them is how dynamic models get fit wrong, so the
compiler will not let you: a conditional form that would need an *unobserved* lag comes back
`Unsupported` naming the latent variable and pointing at the marginal form.

`prepare_panel` materializes the lagged columns per unit in time order, filling a lag that
reaches before a unit's first period with the declared `initial`.

## Identification on the unrolled graph

`identify.unrolled_graph` emits the time-indexed `CausalGraph`; `identify.sequential_plan`
runs the sequential back-door criterion stage by stage (Pearl 2009 §4.4.3; the g-formula is
Robins 1986). At stage *k* of treatments *A₁…Aₙ* it checks that

1. every covariate chosen so far is measured and is not a descendant of *Aₖ…Aₙ*, and
2. *Aₖ* is d-separated from the outcome given the history, in the graph with every edge
   *out of* *Aₖ…Aₙ* deleted.

On the canonical example — a dose that responds to the last outcome — it returns the
textbook answer: adjust for nothing at stage 1, for `outcome.t0` at stage 2, for
`outcome.t1` at stage 3. `static_adjustment_fails` records the thing worth knowing even when
a plan is found: whether one time-invariant set could have done the job. When it is true, a
static analysis of that system is *wrong*, not merely less efficient.

**D10.6 — a sequential plan is `downgraded`, never `identified`.** Positivity is not a
graphical property and cannot be checked from a graph, so it rides along as a named,
unverified assumption. Under `Verdict`'s own invariants that forces the status down, which
is the correct outcome.

**D10.7 — the reduced form is what makes a simultaneous system have a causal graph at all.**
A contemporaneous cycle is not a DAG and never will be. Solve the block and every member
depends on the block's parents and on nothing inside the block: acyclic, and a faithful
statement that *no ordering of the simultaneous variables is causal within the period*.
`unrolled_graph(..., reduced=False)` emits the structural edges as written and raises
`GraphError` if they are cyclic — the honest outcome, since no DAG algorithm can answer a
question about them.

## What this deliberately does not do

* **Leads / rational expectations.** `y[t+1]` parses and is refused with the reason. Solving
  a forward-looking system needs a saddle-path solution (Blanchard–Kahn and friends); that
  is a different piece of machinery and it is not pretended here.
* **A multi-output mean.** `ModelSpec.mean` is one expression, so fitting an entire *marginal*
  trajectory at once has no home: the marginal form gives one expression per observed node
  and `ModelSpec` takes one. Conditional form covers the practical fitting case, and a single
  marginal node can be wrapped with `to_model_spec`. A `Stack` node (or a multi-output mean)
  is the 1.2 item that would close this; it is recorded here rather than worked around.
* **Continuous time.** `ODESystem` remains dimension-check-only (decision C1). A discretized
  ODE *is* expressible as a `DynamicSystem` and unrolls, which covers the common case
  without an integrator inside the expression language.
* **Certifying a fixed point.** Fixed points are not always unique and Gauss-Seidel does not
  always converge to the one you meant. The module reports; it does not certify.

## Evidence

* `tests/unit/test_dynamics_language.py` — parsing, validation, blocks, algebra (26 tests).
* `tests/unit/test_dynamics_unroll.py` — the unrolled recursion against the loop it compiles,
  the exact reduced form against its closed form, the swept block against its fixed point,
  and every refusal (20 tests).
* `tests/unit/test_identify_dynamic.py` — the unrolled graph, the sequential criterion, the
  blocked case and the measured case (13 tests).
* `tests/recovery/test_dynamics_recovery.py` — simulate through the marginal form, fit the
  conditional form, recover `decay` and `beta` within 15 %.
* `nbs/dynamics/01-systems-and-blocks.ipynb`, `nbs/dynamics/02-unrolling-and-fitting.ipynb`,
  `nbs/identify/04-dynamic-systems.ipynb`.

## Related

Identifiability of the compiled system — which parameters, or which *combinations* of them,
a design can actually estimate — is note [0011](0011-identifiability-of-combinations.md).
