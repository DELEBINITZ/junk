"""Guardrail primitives: detectors return verdicts; pipelines compose them.

================================ THE GUARDRAIL MODEL ======================
"Guardrails" are the safety checks wrapped around the LLM. The design here is
deliberately tiny and composable:

  * a DETECTOR looks at some text and returns a VERDICT — one of three actions:
        ALLOW  (fine, leave it),
        REDACT (return cleaned text and keep going),
        BLOCK  (stop the turn; the caller substitutes a safe canned response).
  * a PIPELINE (see pipeline.py) just runs a list of detectors in order and
    folds their verdicts into one final result.

This file only defines the SHAPES (the vocabulary). The actual logic lives in
detectors.py (what to look for) and pipeline.py (how to combine them). Splitting
"what a verdict is" from "how we decide it" keeps detectors swappable — a regex
detector and a hosted-model detector are interchangeable as long as both return
a GuardrailVerdict.
===========================================================================
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Protocol

from pydantic import BaseModel, Field


class Action(str, Enum):
    """The three things a detector can decide. Severity increases down the list;
    a pipeline treats BLOCK as terminal and REDACT as "rewrite then continue"."""
    ALLOW = "allow"
    REDACT = "redact"   # transform text, continue
    BLOCK = "block"     # stop; caller returns a safe response


class GuardrailVerdict(BaseModel):
    """One detector's decision about one piece of text. When ``action == REDACT``
    the cleaned text is carried in ``text`` (the pipeline swaps it in); ``reason``
    and ``metadata`` exist for logging/observability — e.g. WHICH secret types
    were found — without ever putting the sensitive value itself in the reason."""
    detector: str
    action: Action = Action.ALLOW
    reason: str = ""
    text: str | None = None  # transformed text when action == REDACT
    metadata: dict[str, Any] = Field(default_factory=dict)


class GuardrailResult(BaseModel):
    """The pipeline's combined outcome handed back to a guardrail node. ``blocked``
    tells the caller whether to short-circuit; ``text`` is the (possibly redacted
    or replaced-with-a-safe-message) text to use; ``verdicts``/``reasons``/``flags``
    are the audit trail of what each detector decided."""
    blocked: bool = False
    text: str = ""                      # possibly redacted/replaced
    reasons: list[str] = Field(default_factory=list)
    verdicts: list[GuardrailVerdict] = Field(default_factory=list)
    flags: dict[str, Any] = Field(default_factory=dict)


class Detector(Protocol):
    """The contract every detector implements: a ``name`` and an async ``check``
    that returns a verdict. A Protocol (duck-typed) so heuristic detectors and
    model-backed ones drop in interchangeably. ``check`` is async because some
    detectors call out to a hosted classifier over the network."""
    name: str

    async def check(self, text: str, ctx: Any) -> GuardrailVerdict: ...


__all__ = ["Action", "GuardrailVerdict", "GuardrailResult", "Detector"]
