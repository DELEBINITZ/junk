"""Static, dependency-free threat-detection rules and matchers.

Separated from guardrails.py (which imports Presidio/LLM clients) so the static
detection layer can be unit-tested fast and in CI without heavy dependencies.
This is the fast pre-filter; the dynamic LLM classifier in guardrails.py is the
semantic safety net for novel attacks.
"""

import re

# ---------------------------------------------------------------------------
# Injection / jailbreak / extraction patterns
# ---------------------------------------------------------------------------

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(the\s+)?(previous|prior|above|earlier)\s+(instructions?|messages?|context)",
    r"you\s+are\s+now\s+",
    r"system\s*:\s*",
    r"<\s*system\s*>",
    # "forget everything/all the chat history/context/instructions/rules/prompt"
    r"forget\s+(all\s+|everything\s+|about\s+)?(the\s+|your\s+|our\s+)?(chat\s+|conversation\s+)?(history|context|instructions?|rules?|everything|prompt)",
    r"pretend\s+you\s+are",
    r"act\s+as\s+if",
    r"new\s+instructions?\s*:",
    r"disregard\s+(all|any|your|the)\s+(previous|prior|above)",
    r"override\s+(system|safety|security)",
    r"\bDAN\b.*mode",
    r"developer\s+mode\s+(enabled|on|activated)",
    r"do\s+anything\s+now",
    r"jailbreak",
    r"bypass\s+(filter|safety|restriction|guardrail)",
    r"roleplay\s+as\s+(a\s+)?(malicious|evil|unrestricted)",
    r"sudo\s+mode",
    r"ignore\s+(content\s+)?policy",
    r"base64\s+decode",
    r"execute\s+(this|following)\s+(code|script|command)",
    # --- System prompt / guardrail extraction ---
    # Object must be tied to the ASSISTANT so generic product talk is NOT flagged.
    # "your"/"your system" clearly targets the assistant -> full object list.
    r"(reveal|show|tell|give|print|repeat|expose|share|disclose|list|display|output|dump)\s+(me\s+|us\s+)?your(\s+system)?\s+(prompt|instructions?|guard\s*rails?|rules?|configuration|config|directives?|policies|policy)",
    # "the system" is AMBIGUOUS (a product legitimately has "the system configuration",
    # "the system rules"), so restrict it to objects that are unambiguously the
    # assistant's own internals — avoids cold-blocking benign product questions.
    r"(reveal|show|tell|give|print|repeat|expose|share|disclose|list|display|output|dump)\s+(me\s+|us\s+)?the\s+system\s+(prompt|instructions?|guard\s*rails?|directives?)",
    r"what\s+(are|were|is)\s+your(\s+system)?\s+(instructions?|rules?|guard\s*rails?|prompt|directives?|configuration|config)",
    r"what\s+(are|were|is)\s+the\s+system\s+(instructions?|guard\s*rails?|prompt|directives?)",
    r"(repeat|print|output|echo|say)\s+(the\s+)?(text|words|everything|prompt)\s+(above|before|verbatim)",
    r"your\s+(initial|original|first|underlying|exact)\s+(prompt|instructions?|message|directive)",
]

INJECTION_RE = re.compile("|".join(INJECTION_PATTERNS), re.IGNORECASE)

OBFUSCATION_PATTERNS = [
    r"[iI1l]\s*[gG9]\s*[nN]\s*[oO0]\s*[rR]\s*[eE3]",  # i g n o r e (spaced)
    r"(?:[^\w]*\w){5,}(?:instructions|rules|policy)",  # char-separated words
    r"[​‌‍﻿]",  # zero-width chars (injection hiding)
    r"\\u[0-9a-fA-F]{4}",  # unicode escapes in user text
]

OBFUSCATION_RE = re.compile("|".join(OBFUSCATION_PATTERNS), re.IGNORECASE)

# ---------------------------------------------------------------------------
# Output prompt-leak markers (backstop)
# ---------------------------------------------------------------------------

# Distinctive phrases that only appear if the model regurgitated its own system
# prompt / guardrails. Backstop against prompt-extraction that slipped through.
# Kept domain-neutral where possible so the backstop follows the DERIVED persona
# (see agents/identity.py + prompts/orchestrator.py) rather than a fixed identity.
PROMPT_LEAK_MARKERS = (
    # legacy security-persona anchors (still valid targets to catch)
    "you are an expert security intelligence assistant",
    "you are the security intelligence assistant synthesizing",
    # stable structural anchors present in the current dynamic templates. These lines
    # are LITERAL in the templates (they don't vary with the agent set), so they anchor
    # the backstop to the derived persona even when only its top block is regurgitated.
    "you are the intelligent gateway",
    "you are the strategic planner",
    "you assist only within your capabilities",
    "you are synthesizing your agents",
    "your personality:",
    "what you can help with:",
    "respond with exactly one json object",
    "boundaries (non-negotiable)",
    "disclosure rules (important)",
    "disclosure & security boundaries",
    "available agents:",
    'action": "simple"',
    'action": "complex"',
)


def contains_prompt_leak(text: str) -> bool:
    """True if output appears to leak the system prompt / guardrails."""
    low = text.lower()
    return any(marker in low for marker in PROMPT_LEAK_MARKERS)
