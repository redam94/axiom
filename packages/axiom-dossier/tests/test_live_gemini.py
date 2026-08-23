"""Against the real API. Deselected by default: ``pytest -m live`` to run.

These are the tests that would have caught the two things unit tests cannot: a
model id that does not exist, and a response shape that changed. They cost money
and need ``GEMINI_API_KEY``, so they are opt-in.
"""

from __future__ import annotations

import os

import pytest
from axiom.core import Assumption, Interval, Unsupported

from axiom_dossier import (
    LIGHT_MODEL,
    PROSE_MODEL,
    EvidenceBuilder,
    Gemini,
    Narration,
    Narrator,
    build,
    unverified,
)

pytestmark = pytest.mark.live

HAS_KEY = bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
needs_key = pytest.mark.skipif(not HAS_KEY, reason="set GEMINI_API_KEY to run the live tests")


@pytest.fixture
def evidence():
    return (
        EvidenceBuilder("HYPER-3", "Does the 40 mg arm lower systolic pressure?")
        .step("design", "Design", what="Two arms of 300 participants.", why="Powered at 80 %.")
        .finding(
            "contrast",
            Interval(lower=-16.7, upper=-8.1, definition="eti", mass=0.9),
            label="40 mg vs control",
            unit="mmHg",
            precision=1,
        )
        .diagnostic("coverage", 0.94, label="Interval coverage")
        .assume(
            Assumption(
                name="no_unmeasured_confounding",
                facet="population",
                statement="age is the only common cause of dose and pressure",
                challenged_by="a sensitivity analysis",
                state="unverified",
            )
        )
        .build()
    )


@needs_key
@pytest.mark.parametrize("model_id", [PROSE_MODEL, LIGHT_MODEL])
def test_both_default_models_exist_and_answer(model_id: str) -> None:
    """The defaults are real model ids on a real key — not a guess in a constant."""
    model = Gemini(model_id)
    assert model.available() is True
    out = model.generate("Reply with the single word: ready.", temperature=0.0)
    assert not isinstance(out, Unsupported), out
    assert "ready" in out.lower()


@needs_key
def test_a_real_narration_keeps_every_number_traceable(evidence) -> None:
    """The property the whole design rests on, against the actual model."""
    narrator = Narrator(prose=Gemini(PROSE_MODEL), light=Gemini(LIGHT_MODEL))
    built = build(evidence, narrator=narrator)

    assert built.missing() == ()
    narrated = [n for n in built.narrations if n.narrated]
    assert narrated, "nothing was narrated"
    for n in narrated:
        assert n.verified, f"{n.key} invented {n.rejected}"
        assert unverified(n.text, evidence) == ()


@needs_key
def test_the_lite_model_writes_an_abstract_that_checks_out(evidence) -> None:
    out = Narrator(light=Gemini(LIGHT_MODEL)).abstract(evidence)
    assert isinstance(out, Narration), out
    assert out.verified, out.rejected
    assert len(out.text.split()) > 20


def test_a_missing_key_is_reported_rather_than_raised(monkeypatch) -> None:
    """Runs without a key on purpose — the no-credential path is the common one."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    ready = Gemini().available()
    assert isinstance(ready, Unsupported)
    assert "GEMINI_API_KEY" in ready.reason
