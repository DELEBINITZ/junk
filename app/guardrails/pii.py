"""PII detection and redaction helpers.

Presidio is used when available, but regex fallback keeps the PoC dependable in
minimal local environments and Docker builds.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any


PII_PATTERNS: list[tuple[str, re.Pattern[str], str]] = [
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN_REDACTED]"),
    (
        "email",
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
        "[EMAIL_REDACTED]",
    ),
    ("phone", re.compile(r"\b(?:\+?1[-.\s]?)?\d{3}[-.\s]\d{4}\b"), "[PHONE_REDACTED]"),
    (
        "credit_card",
        re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
        "[CREDIT_CARD_REDACTED]",
    ),
    ("passport", re.compile(r"\b[A-Z][0-9]{8}\b"), "[PASSPORT_REDACTED]"),
]


def detect_pii(text: str) -> list[dict[str, object]]:
    """Return detected PII spans from Presidio plus local regex recognizers."""

    matches: list[dict[str, object]] = []
    matches.extend(_detect_with_presidio(text))
    for entity_type, pattern, _replacement in PII_PATTERNS:
        for match in pattern.finditer(text):
            matches.append(
                {
                    "entity_type": entity_type,
                    "start": match.start(),
                    "end": match.end(),
                    "text": match.group(0),
                }
            )
    return _deduplicate_matches(matches)


def redact_pii(text: str) -> str:
    """Redact supported PII types using Presidio first and regex as fallback."""

    presidio_redacted = _redact_with_presidio(text)
    if presidio_redacted is not None:
        text = presidio_redacted
    redacted = text
    for _entity_type, pattern, replacement in PII_PATTERNS:
        redacted = pattern.sub(replacement, redacted)
    return redacted


@lru_cache(maxsize=1)
def _presidio_analyzer() -> Any | None:
    try:
        from presidio_analyzer import AnalyzerEngine

        return AnalyzerEngine()
    except Exception:
        return None


@lru_cache(maxsize=1)
def _presidio_anonymizer() -> Any | None:
    try:
        from presidio_anonymizer import AnonymizerEngine

        return AnonymizerEngine()
    except Exception:
        return None


def _detect_with_presidio(text: str) -> list[dict[str, object]]:
    analyzer = _presidio_analyzer()
    if analyzer is None:
        return []
    try:
        results = analyzer.analyze(text=text, language="en")
    except Exception:
        return []
    return [
        {
            "entity_type": str(result.entity_type).lower(),
            "start": int(result.start),
            "end": int(result.end),
            "text": text[int(result.start) : int(result.end)],
            "score": float(getattr(result, "score", 0.0)),
        }
        for result in results
    ]


def _redact_with_presidio(text: str) -> str | None:
    analyzer = _presidio_analyzer()
    anonymizer = _presidio_anonymizer()
    if analyzer is None or anonymizer is None:
        return None
    try:
        results = analyzer.analyze(text=text, language="en")
        anonymized = anonymizer.anonymize(text=text, analyzer_results=results)
    except Exception:
        return None
    redacted = anonymized.text
    replacements = {
        "<EMAIL_ADDRESS>": "[EMAIL_REDACTED]",
        "<PHONE_NUMBER>": "[PHONE_REDACTED]",
        "<US_SSN>": "[SSN_REDACTED]",
        "<CREDIT_CARD>": "[CREDIT_CARD_REDACTED]",
        "<US_PASSPORT>": "[PASSPORT_REDACTED]",
    }
    for marker, replacement in replacements.items():
        redacted = redacted.replace(marker, replacement)
    return redacted


def _deduplicate_matches(matches: list[dict[str, object]]) -> list[dict[str, object]]:
    seen: set[tuple[int, int, str]] = set()
    unique: list[dict[str, object]] = []
    for match in sorted(matches, key=lambda item: (int(item["start"]), int(item["end"]))):
        key = (int(match["start"]), int(match["end"]), str(match["entity_type"]))
        if key in seen:
            continue
        seen.add(key)
        unique.append(match)
    return unique
