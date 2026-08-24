"""The language-model seam: one small protocol, and a Gemini implementation behind it.

Three properties are deliberate.

**It is a protocol, not a base class.** Anything with a ``name`` and a
``generate`` is a model here. That keeps the narration testable without a
network — ``Offline`` below is a real implementation, not a mock — and keeps the
package from being a Gemini wrapper wearing a general name.

**The key is never stored.** ``Gemini`` holds a model id and nothing else; the
credential is read from the environment at the moment of the call. So a
``Gemini`` can be constructed, hashed, printed and serialized without ever
carrying a secret, which matters because the objects around it are ``Spec`` s
that serialize themselves into report provenance.

**A missing dependency is a typed failure, not an ImportError.** ``google-genai``
is imported inside ``generate``. Importing this module, building a dossier and
rendering every deterministic section all work with the extra absent; only the
narration itself comes back as ``Unsupported`` naming what to install.

The same holds for a call that starts and does not finish. A timeout or a
transport error is reported as ``Unsupported`` rather than raised, because a
document that is renderable from its generated prose must not be taken down by
a network. The exception class and message travel in ``detail`` so the cause is
still there to read.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from axiom.core import Unsupported

__all__ = ["Gemini", "LanguageModel", "Offline", "PROSE_MODEL", "LIGHT_MODEL"]

PROSE_MODEL = "gemini-3.7-flash"
"""Default for prose: the sections a reader actually reads."""

LIGHT_MODEL = "gemini-3.5-flash-lite"
"""Default for the cheap mechanical passes — titles, captions, compression."""

_KEY_VARIABLES = ("GEMINI_API_KEY", "GOOGLE_API_KEY")


@runtime_checkable
class LanguageModel(Protocol):
    """Anything that can turn a prompt into text, or explain why it cannot."""

    @property
    def name(self) -> str:
        """The model identifier, recorded in the report's provenance."""
        ...

    def generate(
        self, prompt: str, *, system: str = "", temperature: float = 0.2
    ) -> str | Unsupported:
        """Text, or ``Unsupported`` naming what is missing."""
        ...


@dataclass(frozen=True)
class Offline:
    """A deterministic stand-in that needs no network and no key.

    It returns the prompt's evidence block unchanged, which makes it useless for
    prose and ideal for everything else: the notebooks execute in CI against it,
    the tests assert narration plumbing against it, and because it never invents
    a number it always passes the provenance check.
    """

    label: str = "offline"

    @property
    def name(self) -> str:
        return f"offline/{self.label}"

    def generate(
        self, prompt: str, *, system: str = "", temperature: float = 0.2
    ) -> str | Unsupported:
        marker = "<<facts>>"
        if marker in prompt:
            body = prompt.split(marker, 1)[1].split("<<end>>", 1)[0]
            return body.strip()
        return prompt.strip()


@dataclass(frozen=True)
class Gemini:
    """Google's Gemini through ``google-genai``, imported when first called.

    ``api_key_variable`` names an environment variable rather than holding a
    key, so this object stays safe to serialize into a provenance appendix.
    """

    model: str = PROSE_MODEL
    api_key_variable: str = ""
    timeout_seconds: float = 180.0
    max_output_tokens: int = 4096
    _client: list[object] = field(default_factory=list, repr=False, compare=False)

    @property
    def name(self) -> str:
        return self.model

    def _key(self) -> str | None:
        names = (self.api_key_variable,) if self.api_key_variable else _KEY_VARIABLES
        for var in names:
            value = os.environ.get(var)
            if value:
                return value
        return None

    def available(self) -> bool | Unsupported:
        """Whether a call would get as far as the network. Cheap; makes no request."""
        try:
            import google.genai  # noqa: F401
        except ModuleNotFoundError:
            return Unsupported(
                reason="google-genai is not installed; narration needs it",
                detail={"install": 'pip install "axiom-dossier[gemini]"'},
                missing=("google-genai",),
            )
        if self._key() is None:
            wanted = self.api_key_variable or " or ".join(_KEY_VARIABLES)
            return Unsupported(
                reason=f"no API key: set {wanted}",
                detail={"model": self.model},
                missing=(wanted,),
            )
        return True

    def generate(
        self, prompt: str, *, system: str = "", temperature: float = 0.2
    ) -> str | Unsupported:
        ready = self.available()
        if isinstance(ready, Unsupported):
            return ready
        from google import genai
        from google.genai import types

        if not self._client:
            self._client.append(genai.Client(api_key=self._key()))
        client = self._client[0]

        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=self.max_output_tokens,
            system_instruction=system or None,
            http_options=types.HttpOptions(timeout=int(self.timeout_seconds * 1000)),
        )
        try:
            response = client.models.generate_content(  # type: ignore[attr-defined]
                model=self.model, contents=prompt, config=config
            )
        except Exception as exc:  # noqa: BLE001 - returned as a typed failure
            # A timeout or a transport error is the same kind of event as a
            # missing key: the model could not be reached. Letting it propagate
            # would take down a document that is perfectly renderable from its
            # generated prose, which is the one thing narration must never do.
            # The class and message are kept so the cause is not lost.
            return Unsupported(
                reason=f"{self.model} could not be reached: {type(exc).__name__}",
                detail={"error": str(exc)[:300], "timeout_seconds": str(self.timeout_seconds)},
            )
        text = getattr(response, "text", None)
        if not text:
            # An empty body is a real outcome (a safety block, an exhausted token
            # budget), not an exception. Say which, rather than returning "".
            feedback = getattr(response, "prompt_feedback", None)
            candidates = getattr(response, "candidates", None) or []
            reason = getattr(candidates[0], "finish_reason", None) if candidates else None
            return Unsupported(
                reason=f"{self.model} returned no text",
                detail={
                    "finish_reason": str(reason),
                    "prompt_feedback": str(feedback),
                },
            )
        return str(text)
