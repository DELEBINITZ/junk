"""Concrete detectors.

Heuristics are the always-on floor; optional model endpoints (Prompt Guard 2,
Llama Guard 3, Presidio) layer on top via config. Domain-aware: this is a
security product, so asset domains/IPs and exploit/malware discussion are NOT
treated as PII or unsafe — only genuine secrets and high-sensitivity PII are
redacted, and only egregiously harmful requests are blocked.
"""

from __future__ import annotations

import re
from typing import Any

from app.core.guardrails.base import Action, GuardrailVerdict

# --- secrets (always redact before LLM/logs) --------------------------------
_SECRET_PATTERNS = [
    ("aws_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("private_key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
    ("bearer", re.compile(r"\b[Bb]earer\s+[A-Za-z0-9\-._~+/]{20,}=*")),
    ("api_key_kv", re.compile(r"(?i)\b(?:api[_-]?key|secret|token|password)\s*[:=]\s*\S{6,}")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b")),
]

# --- PII (redact in OUTPUT to avoid leaking customer PII in answers) ---------
_PII_PATTERNS = [
    ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("credit_card", re.compile(r"\b(?:\d[ -]?){13,16}\b")),
    ("phone", re.compile(r"\b(?:\+?\d{1,3}[ -]?)?\(?\d{3}\)?[ -]?\d{3}[ -]?\d{4}\b")),
]

# --- prompt injection (user input screen) -----------------------------------
_INJECTION_PATTERNS = [
    re.compile(r"(?i)ignore (?:all |the |your )?(?:previous|prior|above) (?:instructions|prompts?)"),
    re.compile(r"(?i)disregard (?:all|the|your|previous|above)"),
    re.compile(r"(?i)you are now (?:a|an|in|dan|developer mode)"),
    re.compile(r"(?i)(?:reveal|print|show|repeat) (?:your |the )?(?:system )?(?:prompt|instructions)"),
    re.compile(r"(?i)pretend (?:to be|you are)"),
    re.compile(r"(?i)\bjailbreak\b|\bDAN mode\b"),
    re.compile(r"(?i)exfiltrate|leak (?:all )?(?:data|secrets|the database)"),
]

# --- egregiously harmful (kept tiny; Llama Guard does real classification) ---
_HARM_PATTERNS = [
    re.compile(r"(?i)how (?:do|to) (?:i |you )?(?:make|build|synthesize) (?:a )?(?:bomb|explosive|nerve agent|bioweapon)"),
    re.compile(r"(?i)(?:child|minor)\s+(?:porn|sexual)"),
    re.compile(r"(?i)how to (?:kill|murder|poison) (?:a |my )?(?:person|someone|people)"),
]


class SecretRedactor:
    name = "secret_redactor"

    async def check(self, text: str, ctx: Any) -> GuardrailVerdict:
        redacted = text
        hits: list[str] = []
        for label, pat in _SECRET_PATTERNS:
            if pat.search(redacted):
                hits.append(label)
                redacted = pat.sub(f"[REDACTED_{label.upper()}]", redacted)
        if hits:
            return GuardrailVerdict(detector=self.name, action=Action.REDACT, text=redacted,
                                    reason="secrets redacted", metadata={"types": hits})
        return GuardrailVerdict(detector=self.name)


class PIIRedactor:
    """Used in OUTPUT to prevent leaking PII not present in the user's own query."""

    name = "pii_redactor"

    def __init__(self, enabled_types: tuple[str, ...] = ("ssn", "credit_card")) -> None:
        self.enabled = set(enabled_types)

    async def check(self, text: str, ctx: Any) -> GuardrailVerdict:
        redacted = text
        hits: list[str] = []
        for label, pat in _PII_PATTERNS:
            if label not in self.enabled:
                continue
            if pat.search(redacted):
                hits.append(label)
                redacted = pat.sub(f"[REDACTED_{label.upper()}]", redacted)
        if hits:
            return GuardrailVerdict(detector=self.name, action=Action.REDACT, text=redacted,
                                    reason="PII redacted", metadata={"types": hits})
        return GuardrailVerdict(detector=self.name)


class PromptInjectionDetector:
    name = "prompt_injection"

    def __init__(self, model_url: str = "") -> None:
        self.model_url = model_url

    async def _model_flags(self, text: str) -> bool:
        if not self.model_url:
            return False
        try:
            import httpx

            async with httpx.AsyncClient(timeout=10.0) as c:
                r = await c.post(f"{self.model_url.rstrip('/')}/classify", json={"inputs": text})
                r.raise_for_status()
                data = r.json()
            # Prompt Guard returns label scores; treat JAILBREAK/INJECTION > 0.8 as flag
            top = data[0] if isinstance(data, list) else data
            label = str(top.get("label", "")).upper()
            score = float(top.get("score", 0))
            return label in {"JAILBREAK", "INJECTION"} and score >= 0.8
        except Exception:
            return False

    async def check(self, text: str, ctx: Any) -> GuardrailVerdict:
        if any(p.search(text) for p in _INJECTION_PATTERNS) or await self._model_flags(text):
            return GuardrailVerdict(detector=self.name, action=Action.BLOCK,
                                    reason="possible prompt injection / jailbreak attempt")
        return GuardrailVerdict(detector=self.name)


class TopicSafetyDetector:
    """Lenient by design — security analysts legitimately discuss exploits,
    malware, and attacks. Only egregiously harmful requests are blocked by the
    heuristic; nuanced classification is delegated to Llama Guard when set."""

    name = "topic_safety"

    def __init__(self, model_url: str = "") -> None:
        self.model_url = model_url

    async def _model_blocks(self, text: str) -> bool:
        if not self.model_url:
            return False
        try:
            import httpx

            async with httpx.AsyncClient(timeout=10.0) as c:
                r = await c.post(f"{self.model_url.rstrip('/')}/classify", json={"inputs": text})
                r.raise_for_status()
                out = r.json()
            verdict = (out[0] if isinstance(out, list) else out).get("label", "safe")
            return str(verdict).lower().startswith("unsafe")
        except Exception:
            return False

    async def check(self, text: str, ctx: Any) -> GuardrailVerdict:
        if any(p.search(text) for p in _HARM_PATTERNS) or await self._model_blocks(text):
            return GuardrailVerdict(detector=self.name, action=Action.BLOCK,
                                    reason="request appears to seek harm unrelated to defense")
        return GuardrailVerdict(detector=self.name)


__all__ = ["SecretRedactor", "PIIRedactor", "PromptInjectionDetector", "TopicSafetyDetector"]
