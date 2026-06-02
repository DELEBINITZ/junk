"""Guardrail verdict + interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(slots=True)
class GuardrailVerdict:
    allowed: bool
    category: str = "safe"
    reason: str = ""
    redacted_text: str | None = None


class InputGuardrail(Protocol):
    def run(self, text: str, ctx=None) -> GuardrailVerdict: ...


class OutputGuardrail(Protocol):
    def run(self, answer: str, citations: list[str], ctx=None) -> GuardrailVerdict: ...
