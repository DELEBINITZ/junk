"""Input and output guardrail pipelines (the safety spine).

Input:  secrets redaction -> injection screen -> topic safety.
Output: PII-leak redaction -> groundedness/citation verification.

Policy: hard-block on injection/harm and on *hallucinated citations*; flag (not
block) merely-weak groundedness, per the "flag-not-block on borderline" rule.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from app.core.contracts import Chunk
from app.core.guardrails.base import Action, Detector, GuardrailResult, GuardrailVerdict
from app.core.guardrails.detectors import (
    PIIRedactor,
    PromptInjectionDetector,
    SecretRedactor,
    TopicSafetyDetector,
)
from app.core.llm.base import NO_CONTEXT_REFUSAL
from app.core.rag.citations import verify_groundedness

HALLUCINATION_FALLBACK = (
    "I can't fully ground that answer in the retrieved sources, so I'm holding it "
    "back to avoid stating something unverified. Try rephrasing, or confirm the "
    "relevant report is ingested."
)
BLOCK_REPLY = (
    "I can't help with that request. If you believe this is a mistake, rephrase "
    "your security question and I'll try again."
)


class InputGuardrailPipeline:
    def __init__(self, detectors: Sequence[Detector]) -> None:
        self.detectors = list(detectors)

    async def run(self, text: str, ctx: Any = None) -> GuardrailResult:
        cur = text
        verdicts: list[GuardrailVerdict] = []
        reasons: list[str] = []
        for d in self.detectors:
            v = await d.check(cur, ctx)
            verdicts.append(v)
            if v.action == Action.BLOCK:
                return GuardrailResult(blocked=True, text=BLOCK_REPLY, reasons=[v.reason], verdicts=verdicts)
            if v.action == Action.REDACT and v.text is not None:
                cur = v.text
                reasons.append(v.reason)
        return GuardrailResult(blocked=False, text=cur, reasons=reasons, verdicts=verdicts)


class OutputGuardrailPipeline:
    def __init__(
        self,
        detectors: Sequence[Detector],
        *,
        groundedness: bool = True,
        refusal_markers: Sequence[str] = (),
    ) -> None:
        self.detectors = list(detectors)
        self.groundedness = groundedness
        self.refusal_markers = tuple(refusal_markers)

    async def run(self, answer: str, chunks: Sequence[Chunk], ctx: Any = None) -> GuardrailResult:
        cur = answer
        verdicts: list[GuardrailVerdict] = []
        reasons: list[str] = []
        flags: dict[str, Any] = {}
        for d in self.detectors:
            v = await d.check(cur, ctx)
            verdicts.append(v)
            if v.action == Action.REDACT and v.text is not None:
                cur = v.text
                reasons.append(v.reason)

        if self.groundedness:
            rep = verify_groundedness(cur, chunks, refusal_markers=self.refusal_markers)
            flags["groundedness"] = rep.model_dump()
            if rep.invalid_indices:  # cited a source that doesn't exist -> hallucination
                return GuardrailResult(
                    blocked=True, text=HALLUCINATION_FALLBACK, reasons=[rep.reason],
                    verdicts=verdicts, flags=flags,
                )
            if not rep.grounded:
                flags["ungrounded"] = True
                reasons.append("weak groundedness (flagged)")

        return GuardrailResult(blocked=False, text=cur, reasons=reasons, verdicts=verdicts, flags=flags)


def build_input_guardrails(settings) -> InputGuardrailPipeline:
    if not settings.guardrails_enabled:
        return InputGuardrailPipeline([])
    detectors: list[Detector] = [SecretRedactor()]
    if settings.injection_detection:
        detectors.append(PromptInjectionDetector(settings.prompt_guard_url))
    if settings.topic_safety:
        detectors.append(TopicSafetyDetector(settings.llama_guard_url))
    return InputGuardrailPipeline(detectors)


def build_output_guardrails(settings) -> OutputGuardrailPipeline:
    detectors: list[Detector] = []
    if settings.guardrails_enabled and settings.pii_redaction:
        detectors.append(PIIRedactor())
    return OutputGuardrailPipeline(
        detectors,
        groundedness=settings.groundedness_check and settings.guardrails_enabled,
        refusal_markers=(NO_CONTEXT_REFUSAL,),
    )


__all__ = [
    "InputGuardrailPipeline",
    "OutputGuardrailPipeline",
    "build_input_guardrails",
    "build_output_guardrails",
    "HALLUCINATION_FALLBACK",
    "BLOCK_REPLY",
]
