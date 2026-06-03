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
import unicodedata
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

# --- obfuscation normalization (defeat homoglyph / zero-width evasion) -------
# Attackers bypass naive regex with zero-width characters ("i<zwsp>gnore"),
# unicode homoglyphs, or fullwidth forms. Normalize to NFKC and strip zero-width /
# bidi / soft-hyphen characters BEFORE matching, so the regex floor can't be
# trivially evaded. (The model-backed layer, when configured, is the real defense;
# this just makes the free floor much harder to slip past.)
_ZERO_WIDTH = re.compile(r"[​-‏‪-‮⁠﻿­]")


def _normalize(text: str) -> str:
    return _ZERO_WIDTH.sub("", unicodedata.normalize("NFKC", text))


# --- exfiltration vectors (neutralize in OUTPUT) ----------------------------
# A successful injection's payoff is often data EXFILTRATION via the rendered
# answer: an auto-loading markdown image fires a GET to an attacker URL with the
# stolen data in the query string (zero click), or a javascript:/data: link runs
# on click. We neutralize those. Bare http(s) links are LEFT — asset URLs are
# legitimate findings in a security product.
_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_BAD_SCHEME_LINK = re.compile(r"\[[^\]]*\]\(\s*(?:javascript|data|vbscript):[^)]*\)", re.IGNORECASE)


def neutralize_injection(text: str) -> tuple[str, list[str]]:
    """Make UNTRUSTED content (retrieved docs, tool outputs) safe to put in the
    prompt by REDACTING any injection-instruction span in place. We deliberately
    do NOT block the turn — an attacker could otherwise plant a trigger phrase in
    a document purely to deny service. Returns (cleaned_text, hit_labels); the
    original text is returned unchanged when nothing matched."""
    norm = _normalize(text)
    hits = [pat.pattern[:24] for pat in _INJECTION_PATTERNS if pat.search(norm)]
    if not hits:
        return text, []
    cleaned = norm
    for pat in _INJECTION_PATTERNS:
        cleaned = pat.sub("[neutralized-instruction]", cleaned)
    return cleaned, hits


# Labels a classifier (Prompt Guard 2 / HF text-classification) uses for a
# malicious prompt. We normalize across naming conventions.
_INJECTION_LABELS = {"injection", "jailbreak", "malicious", "unsafe", "label_1"}


def _flag_injection(payload: Any, threshold: float) -> bool:
    """Normalize a classifier's response (which may be ``[{label,score}]``,
    ``[[{label,score}]]``, or ``{label,score}``) and return True if any
    malicious/jailbreak label scores at or above ``threshold``. Kept pure so the
    parsing is unit-testable without a live model."""
    items = payload
    while isinstance(items, list) and items and isinstance(items[0], list):
        items = items[0]            # unwrap [[...]] -> [...]
    if isinstance(items, dict):
        items = [items]
    if not isinstance(items, list):
        return False
    for it in items:
        if not isinstance(it, dict):
            continue
        label = str(it.get("label", "")).strip().lower()
        try:
            score = float(it.get("score", 0) or 0)
        except (TypeError, ValueError):
            score = 0.0
        if label in _INJECTION_LABELS and score >= threshold:
            return True
    return False


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

    def __init__(self, model_url: str = "", threshold: float = 0.8, fail_closed: bool = False) -> None:
        self.model_url = model_url                  # empty => heuristic-only (no model layer)
        self.threshold = threshold                  # min malicious-label score to flag
        self.fail_closed = fail_closed              # on model error: block (True) vs allow (False)

    async def _model_flags(self, text: str) -> bool:
        # Optional strong layer: Prompt Guard 2 (a hosted text-classifier). If no URL
        # is configured we skip it and rely on the regex floor.
        if not self.model_url:
            return False
        try:
            import httpx

            async with httpx.AsyncClient(timeout=10.0) as c:
                r = await c.post(f"{self.model_url.rstrip('/')}/classify", json={"inputs": text})
                r.raise_for_status()
                data = r.json()
            return _flag_injection(data, self.threshold)
        except Exception:
            # On a model error (network/timeout): fail CLOSED (block) if configured
            # for high-security, else fail OPEN — the regex floor already ran.
            return self.fail_closed

    async def check(self, text: str, ctx: Any) -> GuardrailVerdict:
        # Match on NORMALIZED text so zero-width / homoglyph obfuscation can't slip
        # past the floor. Flag if EITHER the regex floor OR the model trips. BLOCK is
        # correct here: an injection attempt isn't something we "clean and continue".
        if any(p.search(_normalize(text)) for p in _INJECTION_PATTERNS) or await self._model_flags(text):
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
        # Normalize first so obfuscation can't dodge the floor.
        if any(p.search(_normalize(text)) for p in _HARM_PATTERNS) or await self._model_blocks(text):
            return GuardrailVerdict(detector=self.name, action=Action.BLOCK,
                                    reason="request appears to seek harm unrelated to defense")
        return GuardrailVerdict(detector=self.name)


class OutputExfiltrationGuard:
    """OUTPUT detector. Neutralizes data-exfiltration vectors a successful prompt
    injection could plant in the ANSWER: auto-loading markdown images (a zero-click
    pixel that GETs an attacker URL with stolen data in the query string) and
    javascript:/data: links. Bare http(s) links are LEFT untouched — asset URLs are
    legitimate findings in this product. REDACT (defang), never block."""

    name = "output_exfiltration"

    async def check(self, text: str, ctx: Any) -> GuardrailVerdict:
        cleaned = _MD_IMAGE.sub("[external image removed]", text)
        cleaned = _BAD_SCHEME_LINK.sub("[link removed]", cleaned)
        if cleaned != text:
            return GuardrailVerdict(detector=self.name, action=Action.REDACT, text=cleaned,
                                    reason="neutralized exfiltration vector (auto-loading image / script link)")
        return GuardrailVerdict(detector=self.name)


class LlamaGuardDetector:
    """INPUT detector backed by Llama Guard 3 (self-hosted on your GPU). Unlike the
    simple classify-endpoint seam, Llama Guard is a CHAT model: we send the text as
    a user turn to an OpenAI-compatible endpoint and it replies "safe" or
    "unsafe\\nS<category>". It runs ALONGSIDE the narrow harm-regex floor
    (defense in depth), and BLOCKS on "unsafe". Fail policy is configurable."""

    name = "llama_guard"

    def __init__(self, base_url: str, model: str, *, fail_closed: bool = False, timeout: float = 10.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.fail_closed = fail_closed
        self.timeout = timeout

    async def check(self, text: str, ctx: Any) -> GuardrailVerdict:
        try:
            import httpx

            async with httpx.AsyncClient(timeout=self.timeout) as c:
                r = await c.post(
                    f"{self.base_url}/chat/completions",
                    json={"model": self.model, "messages": [{"role": "user", "content": text}],
                          "max_tokens": 16, "temperature": 0},
                )
                r.raise_for_status()
                content = str(r.json()["choices"][0]["message"]["content"]).strip().lower()
            unsafe = content.startswith("unsafe")
        except Exception:
            unsafe = self.fail_closed              # block on error only if fail-closed
        if unsafe:
            return GuardrailVerdict(detector=self.name, action=Action.BLOCK,
                                    reason="content classified unsafe (llama guard)")
        return GuardrailVerdict(detector=self.name)


class PresidioPIIRedactor:
    """OUTPUT detector backed by Microsoft Presidio (self-hosted, in-process). Far
    stronger than the regex floor: NER + context + checksums catch names, emails,
    phones, SSNs, cards, IBANs, etc. ``entities`` is the SECURITY-TUNED set — note
    it deliberately omits IP_ADDRESS/URL/DOMAIN, which are the product's subject
    matter, not PII. Presidio (and its spaCy model) is an optional dependency;
    build_output_guardrails falls back to the regex redactor if it isn't installed.
    Uses the same ``pii_redactor`` name so it slots in interchangeably."""

    name = "pii_redactor"

    def __init__(self, entities: tuple[str, ...] = (), threshold: float = 0.5) -> None:
        self._entities = list(entities) or None     # None => all of Presidio's recognizers
        self._threshold = threshold
        self._analyzer = None
        self._anonymizer = None

    def _engines(self):
        # Lazily build the (heavy) analyzer/anonymizer once and reuse them.
        if self._analyzer is None:
            from presidio_analyzer import AnalyzerEngine
            from presidio_anonymizer import AnonymizerEngine

            self._analyzer = AnalyzerEngine()
            self._anonymizer = AnonymizerEngine()
        return self._analyzer, self._anonymizer

    async def check(self, text: str, ctx: Any) -> GuardrailVerdict:
        import asyncio

        try:
            analyzer, anonymizer = self._engines()
            # analyze() runs spaCy NER (CPU-bound) — push it off the event loop.
            results = await asyncio.to_thread(
                analyzer.analyze, text=text, entities=self._entities, language="en"
            )
            results = [r for r in results if r.score >= self._threshold]
            if not results:
                return GuardrailVerdict(detector=self.name)
            from presidio_anonymizer.entities import OperatorConfig

            anon = anonymizer.anonymize(
                text=text, analyzer_results=results,
                operators={"DEFAULT": OperatorConfig("replace", {"new_value": "[REDACTED_PII]"})},
            )
            types = sorted({r.entity_type for r in results})
            return GuardrailVerdict(detector=self.name, action=Action.REDACT, text=anon.text,
                                    reason="PII redacted (presidio)", metadata={"types": types})
        except Exception:
            # Engine/model error -> don't break the turn; emit no redaction. (Build
            # already chose Presidio only when importable; this guards runtime faults.)
            return GuardrailVerdict(detector=self.name)


__all__ = ["SecretRedactor", "PIIRedactor", "PromptInjectionDetector", "TopicSafetyDetector",
           "OutputExfiltrationGuard", "LlamaGuardDetector", "PresidioPIIRedactor",
           "neutralize_injection"]
