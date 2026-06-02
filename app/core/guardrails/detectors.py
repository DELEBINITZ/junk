"""Individual guardrail detectors.

Defaults are the heuristics already in the codebase (regex/Presidio PII redaction,
substring/base64 prompt-injection). Model adapters (Prompt Guard 2, Llama Guard 3)
are config-gated, lazy-loaded, and fall back to the heuristic if the model/lib is
unavailable — so the default path needs no extra dependency. See plan §10.
"""

from __future__ import annotations

import logging
import os

import httpx

from app.config import settings
from app.guardrails.pii import redact_pii
from app.guardrails.prompt_injection import is_prompt_injection


logger = logging.getLogger(__name__)


def redact(text: str) -> str:
    """PII redaction. `redact_pii` already prefers Presidio and falls back to
    regex, so this is the single redaction entry point for the chassis."""

    return redact_pii(text)


def detect_injection(text: str, extra_patterns: list[str] | None = None) -> bool:
    """Heuristic screen always runs (cheap backstop). When
    INJECTION_PROVIDER=prompt_guard + an endpoint is set, also consult the model;
    failures fall back to the heuristic verdict."""

    if is_prompt_injection(text, extra_patterns):
        return True
    if settings.injection_provider.lower() == "prompt_guard" and settings.prompt_guard_url:
        try:
            return _prompt_guard_flags(text)
        except Exception as exc:  # pragma: no cover - env-dependent
            logger.warning("guardrail.prompt_guard_unavailable", extra={"error": str(exc)})
    return False


def _prompt_guard_flags(text: str) -> bool:
    response = httpx.post(settings.prompt_guard_url, json={"text": text}, timeout=5.0)
    response.raise_for_status()
    data = response.json()
    if isinstance(data, dict):
        if "injection" in data:
            return bool(data["injection"])
        label = str(data.get("label", "")).lower()
        return "inject" in label or "jailbreak" in label
    return False


class TopicSafety:
    """Topic/safety classification. Default 'off' (the legacy keyword moderation
    is false-positive-prone). Set TOPIC_SAFETY_PROVIDER=llama_guard to use the
    guard model once its endpoint exists; failures fall back to 'allow'."""

    def __init__(self, provider: str | None = None):
        self.provider = (
            provider or os.getenv("TOPIC_SAFETY_PROVIDER", settings.topic_safety_provider)
        ).lower()

    def is_unsafe(self, text: str) -> tuple[bool, str]:
        if self.provider == "llama_guard":
            try:
                return self._llama_guard(text)
            except Exception as exc:  # pragma: no cover - env-dependent
                logger.warning("guardrail.llama_guard_unavailable", extra={"error": str(exc)})
                return (False, "")
        return (False, "")

    def _llama_guard(self, text: str) -> tuple[bool, str]:
        if not settings.llama_guard_url:
            raise RuntimeError("llama_guard endpoint not configured")
        response = httpx.post(settings.llama_guard_url, json={"text": text}, timeout=5.0)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict):
            if "unsafe" in data:
                return (bool(data["unsafe"]), str(data.get("category", "")))
            label = str(data.get("label", "")).lower()
            return ("unsafe" in label, str(data.get("category", "")))
        return (False, "")
