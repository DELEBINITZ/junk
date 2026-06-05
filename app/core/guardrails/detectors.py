"""Concrete guardrail detectors.

================================ DESIGN: LIBRARIES + GAPS =================
The heavy lifting is done by established LIBRARIES/MODELS, not hand-rolled rules:
  * PII detection/redaction  -> Microsoft **Presidio** (NER + context + checksums).
  * prompt injection / jailbreak (user input) -> a hosted injection classifier
    (default **protectai/deberta-v3-base-prompt-injection-v2** — Apache-2.0, ungated).
  * content / topic safety   -> a hosted safety chat model (default
    **Qwen/Qwen3Guard-Gen-8B** — Apache-2.0, ungated; Llama Guard 3 also works).

This file only hand-codes the security pieces those libraries do NOT provide:
  * ``SecretRedactor`` — strip credentials (API keys, AWS keys, PEM private keys,
    JWTs, bearer tokens) from the INPUT before it reaches the LLM or logs. General
    PII libraries don't reliably catch machine secrets; this is the cheap floor that
    keeps them inside our trust boundary.
  * ``neutralize_injection`` — defang injection-instruction spans found INSIDE
    retrieved documents / tool outputs, IN PLACE, so the chunk can still be used as
    evidence. This is RAG-specific (a classifier blocks; it can't surgically
    neutralize a span and keep the rest), so we keep a small pattern set for it.
  * ``OutputExfiltrationGuard`` — neutralize data-exfiltration vectors a successful
    injection could plant in the ANSWER (auto-loading markdown images, javascript:/
    data: links). LLM-output-specific; no library covers it.

================================ DOMAIN AWARENESS (important) =============
This is a SECURITY product, so the policy is deliberately tuned, not generic:
  * Asset domains and IP addresses are the subject matter — they are NOT PII and are
    never redacted. (``pii_entities`` in config deliberately omits IP_ADDRESS/URL/
    DOMAIN so Presidio doesn't gut the product.)
  * Talking about exploits/malware/attacks is a defender's daily job, so content
    safety is delegated to the safety model (tuned), not a strict generic filter.
===========================================================================
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from app.core.guardrails.base import Action, GuardrailVerdict

# --- secrets (always redact before LLM/logs) — NO library covers this inline -----
# Anything we send to the LLM may be logged, cached, or forwarded; strip credentials
# BEFORE that point. Each entry is (label, pattern); the label names the kind of
# secret in the redaction marker. (For repo-scale secret scanning use detect-secrets
# / gitleaks out of band; this is the inline chat-input floor.)
_SECRET_PATTERNS = [
    ("aws_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),                      # AWS access key id
    ("private_key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),  # PEM private key header
    ("bearer", re.compile(r"\b[Bb]earer\s+[A-Za-z0-9\-._~+/]{20,}=*")),    # "Bearer <token>"
    ("api_key_kv", re.compile(r"(?i)\b(?:api[_-]?key|secret|token|password)\s*[:=]\s*\S{6,}")),  # key=value secrets
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b")),  # a JWT (3 dot-parts)
]

# --- injection patterns — used ONLY by neutralize_injection (indirect defense) ----
# These are NOT used to screen user input (that's Prompt Guard's job). They exist
# solely to locate + defang injection instructions embedded in RETRIEVED content, in
# place — a surgical neutralization a block/allow classifier can't do.
_INJECTION_PATTERNS = [
    re.compile(r"(?i)ignore (?:all |the |your )?(?:previous|prior|above) (?:instructions|prompts?)"),
    re.compile(r"(?i)disregard (?:all|the|your|previous|above)"),
    re.compile(r"(?i)you are now (?:a|an|in|dan|developer mode)"),
    re.compile(r"(?i)(?:reveal|print|show|repeat) (?:your |the )?(?:system )?(?:prompt|instructions)"),
    re.compile(r"(?i)pretend (?:to be|you are)"),
    re.compile(r"(?i)\bjailbreak\b|\bDAN mode\b"),
    re.compile(r"(?i)exfiltrate|leak (?:all )?(?:data|secrets|the database)"),
]

# --- obfuscation normalization (defeat homoglyph / zero-width evasion) -------
# Attackers bypass naive matching with zero-width characters ("i<zwsp>gnore"),
# unicode homoglyphs, or fullwidth forms. Normalize to NFKC and strip zero-width /
# bidi / soft-hyphen characters BEFORE matching, so neutralize_injection can't be
# trivially evaded.
_ZERO_WIDTH = re.compile(r"[​-‏‪-‮⁠﻿­]")


def _normalize(text: str) -> str:
    return _ZERO_WIDTH.sub("", unicodedata.normalize("NFKC", text))


# --- exfiltration vectors (neutralize in OUTPUT) — LLM-output-specific, no lib ----
# A successful injection's payoff is often data EXFILTRATION via the rendered answer:
# an auto-loading markdown image fires a GET to an attacker URL with the stolen data
# in the query string (zero click), or a javascript:/data: link runs on click. We
# neutralize those. Bare http(s) links are LEFT — asset URLs are legitimate findings.
_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_BAD_SCHEME_LINK = re.compile(r"\[[^\]]*\]\(\s*(?:javascript|data|vbscript):[^)]*\)", re.IGNORECASE)


def neutralize_injection(text: str) -> tuple[str, list[str]]:
    """Make UNTRUSTED content (retrieved docs, tool outputs) safe to put in the
    prompt by REDACTING any injection-instruction span in place. We deliberately do
    NOT block the turn — an attacker could otherwise plant a trigger phrase in a
    document purely to deny service. Returns (cleaned_text, hit_labels); the original
    text is returned unchanged when nothing matched."""
    norm = _normalize(text)
    hits = [pat.pattern[:24] for pat in _INJECTION_PATTERNS if pat.search(norm)]
    if not hits:
        return text, []
    cleaned = norm
    for pat in _INJECTION_PATTERNS:
        cleaned = pat.sub("[neutralized-instruction]", cleaned)
    return cleaned, hits


# Malicious-prompt labels across HF text-classification injection models, normalized:
# ProtectAI deberta-v3 v2 emits SAFE/INJECTION; Prompt-Guard-style models emit
# LABEL_0/LABEL_1; others say jailbreak/malicious/unsafe.
_INJECTION_LABELS = {"injection", "jailbreak", "malicious", "unsafe", "label_1"}


def _flag_injection(payload: Any, threshold: float) -> bool:
    """Normalize a classifier's response (which may be ``[{label,score}]``,
    ``[[{label,score}]]``, or ``{label,score}``) and return True if any malicious/
    jailbreak label scores at or above ``threshold``. Kept pure so the parsing is
    unit-testable without a live model."""
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
    """INPUT detector (custom — no library covers inline). Replaces any detected
    secret with a ``[REDACTED_*]`` marker so credentials never reach the LLM or logs.
    Returns REDACT (not BLOCK): the question is still answerable once the secret is
    masked out."""

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


class PromptInjectionDetector:
    """INPUT detector backed by a self-hosted prompt-injection text classifier
    (default **protectai/deberta-v3-base-prompt-injection-v2**; any HF
    text-classification injection model fits — labels are normalized). BLOCKS likely
    prompt-injection / jailbreak attempts — the turn never reaches
    routing/retrieval/answer.

    Library-based: there is no hand-rolled regex screen here (that was removed). With
    no ``model_url`` the detector is a NO-OP (a dev convenience); production REQUIRES
    the URL — the prod config guard refuses to boot without it."""

    name = "prompt_injection"

    def __init__(self, model_url: str = "", threshold: float = 0.8, fail_closed: bool = True) -> None:
        self.model_url = model_url                  # empty => no-op (dev); required in prod
        self.threshold = threshold                  # min malicious-label score to flag
        self.fail_closed = fail_closed              # on model error: block (True) vs allow (False)

    async def check(self, text: str, ctx: Any) -> GuardrailVerdict:
        if not self.model_url:
            return GuardrailVerdict(detector=self.name)   # no classifier wired -> pass through (dev)
        try:
            import httpx

            async with httpx.AsyncClient(timeout=10.0) as c:
                r = await c.post(f"{self.model_url.rstrip('/')}/classify", json={"inputs": text})
                r.raise_for_status()
                flagged = _flag_injection(r.json(), self.threshold)
        except Exception:
            # Model error (network/timeout): fail CLOSED (block) for high-security, or
            # OPEN if explicitly configured for availability.
            flagged = self.fail_closed
        if flagged:
            return GuardrailVerdict(detector=self.name, action=Action.BLOCK,
                                    reason="possible prompt injection / jailbreak (prompt guard)")
        return GuardrailVerdict(detector=self.name)


class OutputExfiltrationGuard:
    """OUTPUT detector (custom — LLM-output-specific, no library). Neutralizes data-
    exfiltration vectors a successful prompt injection could plant in the ANSWER:
    auto-loading markdown images (a zero-click pixel that GETs an attacker URL with
    stolen data in the query string) and javascript:/data: links. Bare http(s) links
    are LEFT untouched — asset URLs are legitimate findings. REDACT (defang), never
    block."""

    name = "output_exfiltration"

    async def check(self, text: str, ctx: Any) -> GuardrailVerdict:
        cleaned = _MD_IMAGE.sub("[external image removed]", text)
        cleaned = _BAD_SCHEME_LINK.sub("[link removed]", cleaned)
        if cleaned != text:
            return GuardrailVerdict(detector=self.name, action=Action.REDACT, text=cleaned,
                                    reason="neutralized exfiltration vector (auto-loading image / script link)")
        return GuardrailVerdict(detector=self.name)


class LlamaGuardDetector:
    """INPUT detector backed by a self-hosted content/topic-safety CHAT model —
    default **Qwen3Guard-Gen** (Apache-2.0, ungated); Llama Guard also works. We send
    the text as a user turn to an OpenAI-compatible endpoint; the reply's FIRST line
    carries the verdict — "unsafe\\nS<category>" (Llama Guard) or
    "Safety: Safe|Controversial|Unsafe" (Qwen3Guard) — and we BLOCK only on unsafe.
    Qwen3Guard's middle tier "Controversial" is deliberately ALLOWED: this is a
    security product and offensive-security discussion is a defender's daily job.
    This REPLACES the old hand-rolled harm-regex floor. Fail policy is configurable."""

    name = "llama_guard"

    def __init__(self, base_url: str, model: str, *, fail_closed: bool = True, timeout: float = 10.0) -> None:
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
            # Verdict is on the FIRST line: "unsafe\nS2" (Llama Guard) or
            # "safety: unsafe" / "safety: controversial" (Qwen3Guard). Only an
            # explicit unsafe blocks; "controversial" passes (see class docstring).
            first_line = content.splitlines()[0] if content else ""
            unsafe = "unsafe" in first_line
        except Exception:
            unsafe = self.fail_closed              # block on error only if fail-closed
        if unsafe:
            return GuardrailVerdict(detector=self.name, action=Action.BLOCK,
                                    reason="content classified unsafe (llama guard)")
        return GuardrailVerdict(detector=self.name)


class PresidioPIIRedactor:
    """OUTPUT detector backed by **Microsoft Presidio** (self-hosted, in-process) —
    THE PII redactor. NER + context + checksums catch names, emails, phones, SSNs,
    cards, IBANs, etc. ``entities`` is the SECURITY-TUNED set (config ``pii_entities``)
    — it deliberately omits IP_ADDRESS/URL/DOMAIN, which are the product's subject
    matter, not PII. Requires the ``pii`` extra (presidio + a spaCy model); the
    pipeline builder fails fast if it isn't installed."""

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
            # Runtime engine/model fault -> don't break the turn; emit no redaction.
            # (Build already verified Presidio is importable; this guards live faults.)
            return GuardrailVerdict(detector=self.name)


__all__ = ["SecretRedactor", "PromptInjectionDetector", "OutputExfiltrationGuard",
           "LlamaGuardDetector", "PresidioPIIRedactor", "neutralize_injection"]
