"""Provenance bias: is the model-read offset ``delta`` identified in this corpus?

A corpus mixes two kinds of read of the same quantity: a fitted model's
read and a randomized experiment's read. The pool carries one offset
``delta`` per family, added to every model read, so that the pooled ``mu``
is on the *experimental* scale. ``delta`` is a within-effect contrast: it is
identified only by a contributor that reports **both** reads of the same
effect in the same family — then ``y_model − y_experiment = delta + noise``
with the shared study effect cancelling. With no dual-read contributor the
only route left is exchangeability between model-only and experiment-only
contributors, which would let a difference between *who* runs experiments
and *who* runs models masquerade as a provenance bias. The pool refuses that
route: ``delta_identification`` returns ``blocked``, ``pool`` keeps ``delta``
out of the mean and reports its prior as the posterior, flagged.

Ported by specification from the parent's ``benchmarks/meta_model.py``
(``delta_m`` identified by dual-read contributors; ledger row
``meta/pool.py``).
"""

from __future__ import annotations

from axiom.core import Verdict
from axiom.meta.schema import Corpus

__all__ = ["NO_DUAL_READ", "delta_identification"]

NO_DUAL_READ = "no contributor reports both reads"


def delta_identification(corpus: Corpus, family: str | None = None) -> Verdict:
    """``identified`` when at least one contributor in the family reports both a model
    and an experimental read; otherwise ``blocked`` with the reason
    ``"no contributor reports both reads"``.

    With ``family=None`` the whole corpus is inspected and the verdict is
    ``identified`` only if *every* family present has a dual-read
    contributor (the per-family verdict is what ``pool`` uses). The
    ``route`` names the contributors that carry the identification. What the
    verdict does *not* check is that a contributor's two reads target the
    same effect (population, window, dose); that is the contributor's claim
    and ``pool`` records it in ``detail``.
    """
    families = (family,) if family is not None else corpus.families()
    if not families:
        return Verdict(status="blocked", reason="the corpus has no records", route="")
    lacking = [f for f in families if not corpus.dual_read_contributors(f)]
    if lacking:
        where = ", ".join(lacking)
        return Verdict(
            status="blocked",
            reason=f"{NO_DUAL_READ} in family {where}",
            route="",
        )
    carriers = sorted({c for f in families for c in corpus.dual_read_contributors(f)})
    return Verdict(status="identified", route="dual-read contributors: " + ", ".join(carriers))
