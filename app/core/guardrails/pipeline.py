"""Input and output guardrail pipelines (the safety spine).

================================ THE SAFETY SPINE =========================
A chat turn is wrapped on BOTH ends by a guardrail pipeline (these are the very
first and very last nodes in the agent graph — see agent/nodes.py):

  INPUT  (before any reasoning): secrets redaction -> injection screen -> topic safety
  OUTPUT (before the answer ships): PII-leak redaction -> groundedness/citation check

A pipeline is just "run these detectors in order and fold their verdicts". The
ordering matters: a REDACT verdict rewrites the text that the NEXT detector sees,
and the FIRST detector that says BLOCK short-circuits the rest.

Policy (the deliberate strict-vs-lenient split):
  * HARD BLOCK on prompt injection, genuine harm, and *hallucinated citations*
    (the answer cited a source that doesn't exist — a verifiable lie).
  * FLAG, don't block, on merely-WEAK groundedness — borderline cases are
    surfaced for review rather than refused, so we don't nuke usable answers.
This "block the clearly-wrong, flag the borderline" stance is what keeps the
guardrails safe without making the assistant uselessly trigger-happy.
===========================================================================
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

# The safe canned replies substituted when a guardrail blocks. We never surface
# the model's blocked text or the internal reason to the user — just a neutral
# message — so a block can't itself become an information leak.
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
    """Runs the INPUT detectors over the user's question. Folds verdicts with two
    rules: the first BLOCK wins (return immediately with the safe reply), and each
    REDACT rewrites the running text so later detectors see the cleaned version."""

    def __init__(self, detectors: Sequence[Detector]) -> None:
        self.detectors = list(detectors)

    async def run(self, text: str, ctx: Any = None) -> GuardrailResult:
        cur = text                                  # the running (progressively redacted) text
        verdicts: list[GuardrailVerdict] = []
        reasons: list[str] = []
        for d in self.detectors:
            v = await d.check(cur, ctx)
            verdicts.append(v)
            if v.action == Action.BLOCK:
                # Stop the whole turn now; the caller (input_guardrail_node) routes
                # straight to END and returns BLOCK_REPLY instead of an answer.
                return GuardrailResult(blocked=True, text=BLOCK_REPLY, reasons=[v.reason], verdicts=verdicts)
            if v.action == Action.REDACT and v.text is not None:
                cur = v.text                        # carry the cleaned text into the next detector
                reasons.append(v.reason)
        # Nothing blocked: ``cur`` is the (possibly redacted) safe question to use.
        return GuardrailResult(blocked=False, text=cur, reasons=reasons, verdicts=verdicts)


class OutputGuardrailPipeline:
    """Runs the OUTPUT detectors over the generated answer, then (optionally)
    verifies GROUNDEDNESS — that the answer's [n] citations actually point at the
    retrieved chunks. This is the anti-hallucination backstop: the answer node
    already instructs "cite only the context", and this re-checks it independently."""

    def __init__(
        self,
        detectors: Sequence[Detector],
        *,
        groundedness: bool = True,
        # Phrases the model uses to legitimately REFUSE ("I don't have enough
        # grounded info"). A refusal is correct behavior, so groundedness must not
        # penalize it — these markers tell the verifier to treat it as fine.
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
        # First pass: the redaction detectors (e.g. PII). Same fold as input —
        # REDACT rewrites the answer text; these don't block.
        for d in self.detectors:
            v = await d.check(cur, ctx)
            verdicts.append(v)
            if v.action == Action.REDACT and v.text is not None:
                cur = v.text
                reasons.append(v.reason)

        if self.groundedness:
            # Verify the answer against the chunks it was supposed to draw from.
            rep = verify_groundedness(cur, chunks, refusal_markers=self.refusal_markers)
            flags["groundedness"] = rep.model_dump()
            if rep.invalid_indices:  # cited a source that doesn't exist -> hallucination
                # A citation pointing at a non-existent source is a verifiable
                # fabrication -> HARD BLOCK and replace with the safe fallback.
                return GuardrailResult(
                    blocked=True, text=HALLUCINATION_FALLBACK, reasons=[rep.reason],
                    verdicts=verdicts, flags=flags,
                )
            if not rep.grounded:
                # Weak but not fabricated: FLAG only (per the borderline policy),
                # let the answer through with a marker for observability/review.
                flags["ungrounded"] = True
                reasons.append("weak groundedness (flagged)")

        return GuardrailResult(blocked=False, text=cur, reasons=reasons, verdicts=verdicts, flags=flags)


def build_input_guardrails(settings) -> InputGuardrailPipeline:
    """Assemble the input pipeline FROM CONFIG. Secret redaction is the always-on
    floor whenever guardrails are enabled; injection screening and topic safety
    are each opt-in (and each can be backed by a model URL). Guardrails fully off
    => an empty pipeline that passes everything through unchanged."""
    if not settings.guardrails_enabled:
        return InputGuardrailPipeline([])
    detectors: list[Detector] = [SecretRedactor()]   # always first: scrub secrets before anything
    if settings.injection_detection:
        detectors.append(PromptInjectionDetector(settings.prompt_guard_url))
    if settings.topic_safety:
        detectors.append(TopicSafetyDetector(settings.llama_guard_url))
    return InputGuardrailPipeline(detectors)


def build_output_guardrails(settings) -> OutputGuardrailPipeline:
    """Assemble the output pipeline FROM CONFIG. PII redaction is opt-in;
    groundedness verification runs when enabled. ``NO_CONTEXT_REFUSAL`` is passed
    as a refusal marker so a model that correctly says "I don't have enough
    grounded info" is never flagged as ungrounded."""
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
