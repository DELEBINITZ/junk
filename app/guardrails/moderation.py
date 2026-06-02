"""Simple harmful legal request classifier for the PoC."""

from __future__ import annotations

from app.guardrails.prompt_injection import normalize_text


HARMFUL_LEGAL_PATTERNS = [
    "unfair",
    "screws over",
    "deceptive",
    "illegal",
    "fraud",
    "hide liability",
    "discriminate",
]


def is_harmful_legal_request(text: str) -> bool:
    """Block requests that ask for deceptive, abusive, or illegal drafting."""

    normalized = normalize_text(text)
    return any(normalize_text(pattern) in normalized for pattern in HARMFUL_LEGAL_PATTERNS)
