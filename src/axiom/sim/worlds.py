"""Named worlds with known ground truth.

Each function returns a deterministic ``LinearSCM`` built for one
identification route, and its docstring states the number an estimator is
supposed to recover and the number a naive estimator gets instead. Recovery
tests (``tests/recovery``) and the ``identify`` notebooks use these rather
than inventing a data-generating process per test, so the truth lives in one
place.

Throughout: ``X`` is the treatment, ``Y`` the outcome, ``Z`` a covariate or
instrument, ``M`` a mediator. Noise is standard normal unless stated.
"""

from __future__ import annotations

from axiom.sim.scm import LinearSCM

__all__ = [
    "confounded_world",
    "feedback_world",
    "frontdoor_world",
    "hidden_confounder_world",
    "iv_world",
    "mediator_world",
    "transport_pair",
]


def confounded_world() -> LinearSCM:
    """``Z -> X`` (0.8), ``Z -> Y`` (1.5), ``X -> Y`` (2.0); everything measured.

    Truth: the effect of ``X`` on ``Y`` is 2.0. ``{Z}`` satisfies the
    back-door criterion (Pearl 2009, Def. 3.3.1), so OLS of ``Y`` on ``X``
    and ``Z`` recovers 2.0. OLS of ``Y`` on ``X`` alone recovers
    ``2.0 + 1.5 * Cov(Z, X) / Var(X) = 2.0 + 1.5 * 0.8 / 1.64 ~= 2.73``.
    """
    return LinearSCM.from_text("Z -> X: 0.8, Z -> Y: 1.5, X -> Y: 2.0", name="confounded")


def hidden_confounder_world() -> LinearSCM:
    """The confounded world with ``Z`` unmeasured.

    Truth is still 2.0, but no measured set blocks the back-door path
    ``X <- Z -> Y``: the verdict is a downgrade, not an identification, and
    every estimator on the observed columns is biased by the same ~0.73.
    """
    return LinearSCM.from_text(
        "Z -> X: 0.8, Z -> Y: 1.5, X -> Y: 2.0", unmeasured=["Z"], name="hidden_confounder"
    )


def iv_world() -> LinearSCM:
    """``Z -> X`` (1.0), ``X -> Y`` (2.0), ``X <-> Y`` (latent scale 1.0).

    Truth: 2.0. ``Z`` is a valid instrument — relevant (``Z -> X``), and
    excluded (no path to ``Y`` except through ``X``) and unconfounded with
    ``Y`` (Pearl 2009, §7.4.2). Two-stage least squares recovers 2.0; OLS is
    biased upward by ``Cov(U, X) / Var(X) = 1 / 3 ~= 0.33`` because the
    latent common cause enters both ``X`` and ``Y`` with unit loading.
    """
    return LinearSCM.from_text("Z -> X: 1.0, X -> Y: 2.0, X <-> Y: 1.0", name="iv")


def frontdoor_world() -> LinearSCM:
    """``X -> M`` (1.2), ``M -> Y`` (1.5), ``X <-> Y`` (latent scale 1.0).

    Truth: ``1.2 * 1.5 = 1.8``. ``{M}`` satisfies the front-door criterion
    (Pearl 2009, Def. 3.3.3): it intercepts every directed path from ``X``
    to ``Y``, no back-door path reaches ``M`` from ``X``, and ``X`` blocks
    every back-door path from ``M`` to ``Y``. The linear front-door estimate
    is the product of the ``M``-on-``X`` slope and the ``Y``-on-``M``-given-``X``
    slope. OLS of ``Y`` on ``X`` recovers ``1.8 + 1 / 2 = 2.3``.
    """
    return LinearSCM.from_text("X -> M: 1.2, M -> Y: 1.5, X <-> Y: 1.0", name="frontdoor")


def mediator_world() -> LinearSCM:
    """``X -> M`` (1.0), ``M -> Y`` (0.5), ``X -> Y`` (1.0); everything measured, no confounding.

    Truth: the total effect is ``1.0 + 1.0 * 0.5 = 1.5``; the direct effect
    is 1.0. OLS of ``Y`` on ``X`` recovers the total effect; adding ``M`` as
    a covariate recovers the direct effect instead — the classic case where
    "controlling for everything" answers a different question.
    """
    return LinearSCM.from_text("X -> M: 1.0, M -> Y: 0.5, X -> Y: 1.0", name="mediator")


def feedback_world() -> LinearSCM:
    """The confounded world's summary graph, flagged ``feedback=True``.

    The flag is an assertion by the modeller that the static graph
    ``Z -> X, Z -> Y, X -> Y`` summarises a process in which ``Y`` also feeds
    back into later ``X``. The structural equations here are the same as
    ``confounded_world`` — a ``LinearSCM`` is acyclic by construction — so
    the data do not contain the feedback; the flag exists to make the verdict
    machinery say that static adjustment is insufficient, and this world
    documents that the flag is an assertion about the world, not a property
    the simulation can exhibit.
    """
    return LinearSCM.from_text(
        "Z -> X: 0.8, Z -> Y: 1.5, X -> Y: 2.0", feedback=True, name="feedback"
    )


def transport_pair() -> tuple[LinearSCM, LinearSCM]:
    """A source and a target population that differ only in ``Z``'s distribution.

    Both share ``Z -> X`` (0.8), ``Z -> Y`` (1.5), ``X -> Y`` (2.0) and
    ``selection=["Z"]``: the selection diagram has an S-node into ``Z``
    (Bareinboim & Pearl 2014, Def. 3). In the source ``Z ~ N(0, 1)``; in the
    target ``Z ~ N(1.5, 1)``. Every structural equation other than ``Z``'s is
    identical, which is what the S-node says.

    What is invariant and what transport recovers:

    * The coefficient on ``X -> Y`` is 2.0 in both populations, and so is
      ``d E[Y | do(x)] / dx``. In a linear model with no ``X``–``Z``
      interaction the per-unit effect does not need transporting.
    * The interventional level ``E*[Y | do(x)]`` does differ:
      ``E[Y | do(x)] = 2.0 x + 1.5 E[Z]`` is ``2.0 x`` in the source and
      ``2.0 x + 2.25`` in the target. Read it off the source's
      ``E[Y | do(x)]`` and you are off by ``1.5 * 1.5 = 2.25``.
    * ``{Z}`` is S-admissible — it d-separates the S-node from ``Y`` in the
      mutilated graph — so the transport formula
      ``P*(y | do(x)) = sum_z P(y | do(x), z) P*(z)`` (Bareinboim & Pearl
      2014, Thm. 2) applies: fit ``E[Y | x, z]`` in the source and average
      over the target's ``Z``. That recovers ``2.0 x + 2.25``.

    ``LinearSCM.interventional_mean("Y", {"X": x})`` on each world gives the
    two truths exactly.
    """
    edges = "Z -> X: 0.8, Z -> Y: 1.5, X -> Y: 2.0"
    source = LinearSCM.from_text(edges, selection=["Z"], name="transport_source")
    target = LinearSCM.from_text(
        edges, selection=["Z"], intercepts={"Z": 1.5}, name="transport_target"
    )
    return source, target
