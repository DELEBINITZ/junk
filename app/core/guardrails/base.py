"""Guardrail primitives: detectors return verdicts; pipelines compose them."""

from __future__ import annotations

from enum import Enum
from typing import Any, Protocol

from pydantic import BaseModel, Field


class Action(str, Enum):
    ALLOW = "allow"
    REDACT = "redact"   # transform text, continue
    BLOCK = "block"     # stop; caller returns a safe response


class GuardrailVerdict(BaseModel):
    detector: str
    action: Action = Action.ALLOW
    reason: str = ""
    text: str | None = None  # transformed text when action == REDACT
    metadata: dict[str, Any] = Field(default_factory=dict)


class GuardrailResult(BaseModel):
    blocked: bool = False
    text: str = ""                      # possibly redacted/replaced
    reasons: list[str] = Field(default_factory=list)
    verdicts: list[GuardrailVerdict] = Field(default_factory=list)
    flags: dict[str, Any] = Field(default_factory=dict)


class Detector(Protocol):
    name: str

    async def check(self, text: str, ctx: Any) -> GuardrailVerdict: ...


__all__ = ["Action", "GuardrailVerdict", "GuardrailResult", "Detector"]
