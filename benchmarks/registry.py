"""Real datasets with published results, and where each one came from.

Everything axiom is tested against elsewhere is synthetic: a world with a known
truth, so recovery is checkable. That answers "does the arithmetic work". It does
not answer "does this library, run end to end on data somebody else collected,
land on the number somebody else published". This module is for that question.

Each dataset carries its provenance as data rather than as a comment: source,
citation, licence, and the published values axiom is expected to reproduce. That
is the same commitment the library makes about its own numbers.

Two kinds of dataset live here:

**Vendored.** Small tables of published facts whose redistribution is
unproblematic, committed under ``benchmarks/data/``. They work offline and the
tests over them run in CI.

**Fetched.** Datasets whose upstream licence does not permit redistribution from
this repository — which is proprietary, so a non-commercial or copyleft licence is
a real conflict, not a formality. These are downloaded on request into
``benchmarks/cache/`` (git-ignored) by ``python benchmarks/fetch.py``, under the
upstream terms, by whoever wants to run them. Tests over them skip when the file
is absent.

    from benchmarks.registry import DATASETS, load

    frame = load("bcg")
    print(DATASETS["bcg"].licence, DATASETS["bcg"].published)
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

npt_ArrayLike = npt.ArrayLike

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
CACHE = HERE / "cache"


@dataclass(frozen=True)
class Dataset:
    """One real dataset, its terms of use, and what has been published about it."""

    name: str
    title: str
    field: str
    pillar: str
    availability: Literal["vendored", "fetch"]
    citation: str
    source: str
    licence: str
    #: what axiom is expected to reproduce, and where each number was published
    published: dict[str, Any] = field(default_factory=dict)
    #: only for fetch datasets
    url: str = ""
    filename: str = ""
    note: str = ""

    @property
    def path(self) -> Path:
        if self.availability == "vendored":
            return DATA / f"{self.name}.csv"
        return CACHE / (self.filename or f"{self.name}.csv")

    @property
    def available(self) -> bool:
        return self.path.exists()

    def sha256(self) -> str:
        return hashlib.sha256(self.path.read_bytes()).hexdigest()


DATASETS: dict[str, Dataset] = {
    "bcg": Dataset(
        name="bcg",
        title="BCG vaccine against tuberculosis: 13 randomized trials",
        field="Epidemiology / public health",
        pillar="meta",
        availability="vendored",
        citation=(
            "Colditz, G. A., Brewer, T. F., Berkey, C. S., Wilson, M. E., "
            "Burdick, E., Fineberg, H. V., & Mosteller, F. (1994). Efficacy of BCG "
            "vaccine in the prevention of tuberculosis: meta-analysis of the "
            "published literature. JAMA, 271(9), 698-702."
        ),
        source="https://wviechtb.github.io/metadat/reference/dat.bcg.html",
        licence=(
            "Thirteen rows of event counts published in a 1994 JAMA meta-analysis; "
            "redistributed in many textbooks and packages. Attribution above."
        ),
        published={
            # metafor: escalc(measure="RR", ai=tpos, bi=tneg, ci=cpos, di=cneg)
            # then rma(yi, vi) -- REML, the package default.
            "estimate": -0.7145,
            "se": 0.1798,
            "ci_lower": -1.0669,
            "ci_upper": -0.3622,
            "tau2_reml": 0.3132,
            "tau2_reml_se": 0.1664,
            "q": 152.2330,
            "q_df": 12,
            # metafor's rma() prints the MODEL-BASED I^2 and H^2, which are not the
            # Q-based ones axiom's heterogeneity() returns. See i2_model_based below.
            "i2_model_based": 0.9222,
            "h2_model_based": 12.86,
            "_source": (
                "R metafor, rma(yi, vi) on log risk ratios; values quoted in the "
                "metafor documentation and reproduced widely."
            ),
        },
        note=(
            "Two things bite here. Sign convention: the log risk ratio below is "
            "vaccinated-versus-control, so NEGATIVE means the vaccine helped; some "
            "metafor examples reverse the arms and print +0.7145 for the same data. "
            "And I-squared has two standard definitions that differ by about a tenth "
            "of a percentage point on this dataset -- axiom reports the Q-based one, "
            "metafor's rma() prints the model-based one."
        ),
    ),
    "misra1a": Dataset(
        name="misra1a",
        title="NIST StRD Misra1a: monomolecular adsorption, volume against pressure",
        field="Physical chemistry / dental research",
        pillar="surface",
        availability="vendored",
        citation=(
            "Misra, D. (1978). Dental Research Monomolecular Adsorption Study. "
            "National Institute of Standards and Technology, Statistical Reference "
            "Datasets (StRD), Nonlinear Regression."
        ),
        source="https://www.itl.nist.gov/div898/strd/nls/data/misra1a.shtml",
        licence=(
            "Work of the US federal government: public domain in the United States "
            "(17 U.S.C. 105). NIST asks that it be cited, not that permission be "
            "sought."
        ),
        published={
            # y = b1 * (1 - exp(-b2 * x)), certified to 11 significant digits.
            "b1": 2.3894212918e02,
            "b1_sd": 2.7070075241e00,
            "b2": 5.5015643181e-04,
            "b2_sd": 7.2668688436e-06,
            "residual_sum_of_squares": 1.2455138894e-01,
            "residual_sd": 1.0187876330e-01,
            "df": 12,
            "n": 14,
            # axiom's ExponentialKernel is f(u) = 1 - exp(-u), u = dose / k, so
            # beta maps to b1 and k maps to 1 / b2.
            "axiom_beta": 2.3894212918e02,
            "axiom_k": 1.0 / 5.5015643181e-04,
            "_source": "NIST StRD certified values, computed to 500 digits and rounded.",
        },
        note=(
            "The certified values are least squares. axiom fits a posterior, so the "
            "test asks whether the posterior concentrates on the certified values, "
            "not whether it equals them to 11 digits."
        ),
    ),
    "darfur": Dataset(
        name="darfur",
        title="Darfurian refugees in eastern Chad: violence and attitudes to peace",
        field="Political science / conflict",
        pillar="identify + diagnose",
        availability="fetch",
        citation=(
            "Cinelli, C., & Hazlett, C. (2020). Making sense of sensitivity: "
            "extending omitted variable bias. Journal of the Royal Statistical "
            "Society B, 82(1), 39-67. Data collected by Hazlett (2019)."
        ),
        source="https://github.com/carloscinelli/sensemakr",
        licence=(
            "Ships inside the GPL-3 R package sensemakr. Not redistributed here: "
            "fetched on request, under the upstream package's terms."
        ),
        url=(
            "https://raw.githubusercontent.com/nlapier2/PySensemakr/main/"
            "sensemakr/data/darfur.csv"
        ),
        filename="darfur.csv",
        published={
            # Cinelli & Hazlett (2020), Table 1 and the sensemakr vignette.
            "coefficient": 0.0973,
            "se": 0.0232,
            "df": 783,
            "n": 1276,
            "robustness_value": 0.139,
            "robustness_value_alpha": 0.076,
            "partial_r2": 0.022,
            "_source": (
                "Cinelli & Hazlett (2020) Table 1; the regression is peacefactor on "
                "directlyharmed with age, farmer_dar, herder_dar, pastvoted, hhsize_darfur, "
                "female and village fixed effects."
            ),
        },
        note=(
            "axiom already tests the sensitivity FORMULAS against these published "
            "numbers using hard-coded summary statistics. What the raw data adds is "
            "the step before: reproducing 0.0973 and 0.0232 from 1,276 rows."
        ),
    ),
    "lalonde_nsw": Dataset(
        name="lalonde_nsw",
        title="National Supported Work: a randomized job training programme",
        field="Labour economics",
        pillar="identify + design",
        availability="fetch",
        citation=(
            "LaLonde, R. (1986). Evaluating the econometric evaluations of training "
            "programs. American Economic Review, 76(4), 604-620. Subsample and "
            "reanalysis: Dehejia, R., & Wahba, S. (1999). JASA, 94(448), 1053-1062."
        ),
        source="https://users.nber.org/~rdehejia/nswdata2.html",
        licence=(
            "CC BY-NC: attributable NON-COMMERCIAL use. This repository is "
            "proprietary, so the data is not vendored here -- fetch it yourself and "
            "observe the licence."
        ),
        url="https://users.nber.org/~rdehejia/data/nsw_dw.dta",
        filename="nsw_dw.dta",
        published={
            "experimental_ate": 1794.0,
            "n": 445,
            "n_treated": 185,
            "n_control": 260,
            "_source": (
                "Dehejia & Wahba (1999): the experimental benchmark on their 445-unit "
                "subsample, against which observational estimators are judged."
            ),
        },
        note=(
            "The point of this dataset is that randomization gives a benchmark, and "
            "the observational comparison groups miss it badly. It is the standard "
            "way to show that an estimator's plausibility is not evidence."
        ),
    ),
    "psid_controls": Dataset(
        name="psid_controls",
        title="PSID comparison group for the National Supported Work study",
        field="Labour economics",
        pillar="identify",
        availability="fetch",
        citation=(
            "Panel Study of Income Dynamics, as extracted by LaLonde (1986) and "
            "redistributed by Dehejia & Wahba (1999)."
        ),
        source="https://users.nber.org/~rdehejia/nswdata2.html",
        licence="CC BY-NC, as above. Not vendored.",
        url="https://users.nber.org/~rdehejia/data/psid_controls.dta",
        filename="psid_controls.dta",
        published={
            "n": 2490,
            "_source": "Dehejia & Wahba (1999): the PSID-1 comparison group.",
        },
        note=(
            "Survey respondents who were never in the programme. Swapping them in "
            "for the randomized controls is what turns a +$1,794 programme into an "
            "apparently harmful one."
        ),
    ),
}


# ----------------------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------------------


class NotFetched(RuntimeError):
    """Raised when a fetch-only dataset has not been downloaded yet."""


def load(name: str) -> pd.DataFrame:
    """The dataset as a frame, or a typed failure telling you how to get it."""
    spec = DATASETS[name]
    if not spec.available:
        raise NotFetched(
            f"{name!r} is not present at {spec.path}.\n"
            f"  licence: {spec.licence}\n"
            f"  get it:  python benchmarks/fetch.py {name}"
        )
    if spec.path.suffix == ".dta":
        return pd.read_stata(spec.path)
    return pd.read_csv(spec.path)


def log_risk_ratios(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Log risk ratio and its standard error for each 2x2 trial in the BCG data.

    This is the standard escalc(measure="RR") computation, done here rather than
    in axiom: axiom's meta layer pools (estimate, se) pairs and takes no view on
    how a 2x2 table becomes one. Vaccinated arm over control arm, so negative is
    protective.
    """
    ai = frame["tpos"].to_numpy(float)  # vaccinated, developed TB
    bi = frame["tneg"].to_numpy(float)  # vaccinated, did not
    ci = frame["cpos"].to_numpy(float)  # control, developed TB
    di = frame["cneg"].to_numpy(float)  # control, did not
    y = np.log(ai / (ai + bi)) - np.log(ci / (ci + di))
    var = 1.0 / ai - 1.0 / (ai + bi) + 1.0 / ci - 1.0 / (ci + di)
    return y, np.sqrt(var)


def model_based_i2(tau2: float, se: npt_ArrayLike) -> tuple[float, float]:
    """metafor's ``I²`` and ``H²`` for a random-effects fit, from axiom's outputs.

    ``I²`` has two standard definitions and they do not agree. The Q-based one,
    which :func:`axiom.meta.heterogeneity` returns, is ``(Q − df) / Q``. The
    model-based one, which metafor's ``rma()`` prints, is ``τ² / (τ² + s²)`` with
    ``s²`` the "typical" within-study variance of Higgins & Thompson (2002):

        s² = (k − 1) · Σw / ((Σw)² − Σw²),   w = 1 / se²

    Neither is wrong. They answer "what share of the observed dispersion exceeds
    what sampling alone predicts" and "what share of total variance is between
    studies", and on this dataset they differ by 0.1 of a percentage point. This
    function exists so the benchmark can check axiom against metafor's number on
    metafor's definition rather than pretending the two are the same.
    """
    w = 1.0 / np.asarray(se, dtype=float) ** 2
    k = w.size
    s2 = (k - 1) * w.sum() / (w.sum() ** 2 - (w**2).sum())
    return float(tau2 / (tau2 + s2)), float((tau2 + s2) / s2)


def summary() -> pd.DataFrame:
    """One row per dataset: what it is, where it came from, whether it is here."""
    return pd.DataFrame(
        [
            {
                "name": d.name,
                "field": d.field,
                "pillar": d.pillar,
                "availability": d.availability,
                "available": d.available,
                "licence": d.licence.split(":")[0][:38],
            }
            for d in DATASETS.values()
        ]
    )


if __name__ == "__main__":
    pd.set_option("display.width", 160)
    print(summary().to_string(index=False))
    print()
    for d in DATASETS.values():
        mark = "present" if d.available else "not fetched"
        print(f"{d.name:14s} {mark:12s} {d.title}")
        if d.available:
            print(f"{'':14s} sha256 {d.sha256()[:16]}...")
