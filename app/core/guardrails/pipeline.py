"""Input and output guardrail pipelines (plan §10).

INPUT  : redact PII -> prompt-injection screen -> topic/safety screen.
OUTPUT : redact PII leak (citation/groundedness + toxicity hooks land next).
The graph's guardrail nodes call these; refusal happens when allowed=False.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.core.guardrails.base import GuardrailVerdict
from app.core.guardrails.detectors import TopicSafety, detect_injection, redact


@dataclass
class InputGuardrailPipeline:
    extra_patterns: list[str] = field(default_factory=list)
    topic: TopicSafety = field(default_factory=TopicSafety)

    def run(self, text: str, ctx=None) -> GuardrailVerdict:
        redacted = redact(text)
        if detect_injection(redacted, self.extra_patterns):
            return GuardrailVerdict(
                allowed=False, category="prompt_injection",
                reason="blocked instruction detected", redacted_text=redacted,
            )
        unsafe, category = self.topic.is_unsafe(redacted)
        if unsafe:
            return GuardrailVerdict(
                allowed=False, category=category or "unsafe",
                reason="unsafe topic", redacted_text=redacted,
            )
        return GuardrailVerdict(allowed=True, category="safe", redacted_text=redacted)


@dataclass
class OutputGuardrailPipeline:
    def run(self, answer: str, citations: list[str], ctx=None) -> GuardrailVerdict:
        # Second line of PII defense on the way out; citation/groundedness +
        # toxicity scoring slot in here next (plan §10).
        redacted = redact(answer)
        return GuardrailVerdict(allowed=True, category="safe", redacted_text=redacted)
