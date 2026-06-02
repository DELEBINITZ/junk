"""Concrete detectors.

================================ DESIGN: TWO LAYERS =======================
Each concern (secrets, PII, injection, topic-safety) has a CHEAP heuristic layer
(regex, always on, no infra) and an OPTIONAL strong layer (a hosted classifier —
Prompt Guard 2 for injection, Llama Guard 3 for topic safety, Presidio for PII).
The regex is the FLOOR that always runs; the model, when a URL is configured,
adds nuance on top. Heuristics catch the obvious cases for free; the model is
there for the subtle ones in production.

================================ DOMAIN AWARENESS (important) =============
This is a SECURITY product, so the policy is deliberately tuned, not generic:
  * Asset domains and IP addresses are the subject matter — they are NOT PII and
    are never redacted. (A generic PII filter that scrubbed IPs would gut the
    product.)
  * Talking about exploits, malware, and attacks is a defender's daily job, so
    topic-safety is LENIENT — only egregiously harmful requests are blocked.
What we DO scrub/stop: real secrets (keys, tokens) before they reach the LLM or
logs, high-sensitivity PII (SSN, card numbers) leaking into answers, and prompt-
injection / jailbreak attempts.
===========================================================================
"""

from __future__ import annotations

import re
from typing import Any

from app.core.guardrails.base import Action, GuardrailVerdict

# --- secrets (always redact before LLM/logs) --------------------------------
# Why redact secrets at the INPUT boundary: anything we send to the LLM may be
# logged, cached, or forwarded to a model provider. Stripping credentials BEFORE
# that point keeps them from ever leaving our trust boundary. Each entry is
# (label, pattern); the label names the kind of secret in the redaction marker.
_SECRET_PATTERNS = [
    ("aws_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),                      # AWS access key id
    ("private_key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),  # PEM private key header
    ("bearer", re.compile(r"\b[Bb]earer\s+[A-Za-z0-9\-._~+/]{20,}=*")),    # "Bearer <token>"
    ("api_key_kv", re.compile(r"(?i)\b(?:api[_-]?key|secret|token|password)\s*[:=]\s*\S{6,}")),  # key=value secrets
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b")),  # a JWT (3 dot-parts)
]

# --- PII (redact in OUTPUT to avoid leaking customer PII in answers) ---------
# Note these are checked on the ANSWER, not the question: the goal is to stop the
# model from emitting sensitive personal data into a reply. ``email``/``phone``
# patterns exist but are NOT enabled by default (see PIIRedactor) — only the
# high-sensitivity ones (SSN, card numbers) redact, to avoid over-scrubbing.
_PII_PATTERNS = [
    ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("credit_card", re.compile(r"\b(?:\d[ -]?){13,16}\b")),
    ("phone", re.compile(r"\b(?:\+?\d{1,3}[ -]?)?\(?\d{3}\)?[ -]?\d{3}[ -]?\d{4}\b")),
]

# --- prompt injection (user input screen) -----------------------------------
# Prompt injection = text in the USER'S input that tries to override the system's
# instructions ("ignore previous instructions", "reveal your prompt", "you are now
# DAN"). These patterns catch the classic phrasings. Detecting injection lets us
# BLOCK the turn — and, combined with the human action gate, is why a hijacked
# prompt still cannot make the agent perform a side-effecting action on its own.
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
# Intentionally a SHORT list of unambiguous harms (weapons of mass destruction,
# CSAM, direct violence). Real topic classification is the model's job (Llama
# Guard); this heuristic only exists so something blocks the worst cases even
# with no model wired. Kept narrow precisely so it doesn't false-positive on the
# legitimate offensive-security discussion this product is FOR.
_HARM_PATTERNS = [
    re.compile(r"(?i)how (?:do|to) (?:i |you )?(?:make|build|synthesize) (?:a )?(?:bomb|explosive|nerve agent|bioweapon)"),
    re.compile(r"(?i)(?:child|minor)\s+(?:porn|sexual)"),
    re.compile(r"(?i)how to (?:kill|murder|poison) (?:a |my )?(?:person|someone|people)"),
]


class SecretRedactor:
    """INPUT detector. Replaces any detected secret with a ``[REDACTED_*]`` marker
    so credentials never reach the LLM or logs. Returns REDACT (not BLOCK): the
    user's question is still answerable once the secret is masked out."""

    name = "secret_redactor"

    async def check(self, text: str, ctx: Any) -> GuardrailVerdict:
        redacted = text
        hits: list[str] = []
        # Run every secret pattern; each match is substituted with a labelled
        # placeholder. We accumulate the TYPES found (not the values) for the audit.
        for label, pat in _SECRET_PATTERNS:
            if pat.search(redacted):
                hits.append(label)
                redacted = pat.sub(f"[REDACTED_{label.upper()}]", redacted)
        if hits:
            return GuardrailVerdict(detector=self.name, action=Action.REDACT, text=redacted,
                                    reason="secrets redacted", metadata={"types": hits})
        return GuardrailVerdict(detector=self.name)     # default verdict == ALLOW


class PIIRedactor:
    """OUTPUT detector — the last-line "don't leak PII in the answer" check. Only
    the ``enabled_types`` are scrubbed; the default is the high-sensitivity set
    (SSN, card numbers) ONLY. Emails/phones are deliberately left in by default
    because in a security context they're often legitimate findings, not leaks."""

    name = "pii_redactor"

    def __init__(self, enabled_types: tuple[str, ...] = ("ssn", "credit_card")) -> None:
        self.enabled = set(enabled_types)           # which PII categories to actually redact

    async def check(self, text: str, ctx: Any) -> GuardrailVerdict:
        redacted = text
        hits: list[str] = []
        for label, pat in _PII_PATTERNS:
            if label not in self.enabled:
                continue                            # skip categories not turned on
            if pat.search(redacted):
                hits.append(label)
                redacted = pat.sub(f"[REDACTED_{label.upper()}]", redacted)
        if hits:
            return GuardrailVerdict(detector=self.name, action=Action.REDACT, text=redacted,
                                    reason="PII redacted", metadata={"types": hits})
        return GuardrailVerdict(detector=self.name)


class PromptInjectionDetector:
    """INPUT detector that BLOCKS likely prompt-injection / jailbreak attempts.
    Two-layer: the regex floor plus an optional Prompt Guard model. Blocking here
    means the turn never reaches routing/retrieval/answer."""

    name = "prompt_injection"

    def __init__(self, model_url: str = "") -> None:
        self.model_url = model_url                  # empty => heuristic-only (no model layer)

    async def _model_flags(self, text: str) -> bool:
        # Optional strong layer: ask a hosted Prompt Guard classifier. If no URL is
        # configured we skip it entirely and rely on the regex floor.
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
            # FAIL OPEN on a model error (network/timeout): the regex floor still
            # ran, so we don't hard-fail the whole request if the classifier is down.
            return False

    async def check(self, text: str, ctx: Any) -> GuardrailVerdict:
        # Flag if EITHER the regex floor OR the model trips. BLOCK is correct here:
        # an injection attempt isn't something we want to "clean and continue".
        if any(p.search(text) for p in _INJECTION_PATTERNS) or await self._model_flags(text):
            return GuardrailVerdict(detector=self.name, action=Action.BLOCK,
                                    reason="possible prompt injection / jailbreak attempt")
        return GuardrailVerdict(detector=self.name)


class TopicSafetyDetector:
    """INPUT detector for genuinely harmful requests. Lenient BY DESIGN — security
    analysts legitimately discuss exploits, malware, and attacks. Only egregiously
    harmful requests are blocked by the heuristic; nuanced classification is
    delegated to Llama Guard when set. The leniency is the whole point: a strict
    generic safety filter would refuse the product's normal use cases."""

    name = "topic_safety"

    def __init__(self, model_url: str = "") -> None:
        self.model_url = model_url                  # empty => heuristic-only

    async def _model_blocks(self, text: str) -> bool:
        # Optional strong layer: a Llama Guard endpoint returns safe/unsafe. Skipped
        # when no URL is set.
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
            return False                            # fail open on classifier error

    async def check(self, text: str, ctx: Any) -> GuardrailVerdict:
        # Block only on the narrow harm list OR an explicit "unsafe" from the model.
        if any(p.search(text) for p in _HARM_PATTERNS) or await self._model_blocks(text):
            return GuardrailVerdict(detector=self.name, action=Action.BLOCK,
                                    reason="request appears to seek harm unrelated to defense")
        return GuardrailVerdict(detector=self.name)


__all__ = ["SecretRedactor", "PIIRedactor", "PromptInjectionDetector", "TopicSafetyDetector"]
