"""Citation + groundedness verification.

WHAT "GROUNDEDNESS" MEANS (and why a RAG system must check it): the whole point
of RAG is that the answer is backed by retrieved evidence, not invented. A
"grounded" answer is one whose claims are actually supported by the chunks we
retrieved, with each claim tagged by an inline ``[n]`` citation marker pointing at
the chunk that supports it. The classic failure mode is an LLM that writes a
confident answer and cites ``[3]`` for a fact that source never stated — a
hallucinated citation.

This module is the deterministic floor that catches that: with NO model it checks
two things —
  1. every ``[n]`` marker the answer used points at a chunk that was REALLY
     retrieved (no out-of-range / invented citations), and
  2. the cited chunks share meaningful word overlap with the answer text (a cheap
     proxy for "this source actually supports what was written").
The output guardrail calls this to flag hallucinated citations and unsupported
answers before they reach the user. A real NLI ("does premise entail hypothesis?")
model can be layered on top in prod (``groundedness_check``); this remains the floor.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from pydantic import BaseModel

from app.core.contracts import Chunk

_CITE = re.compile(r"\[(\d+)\]")     # matches citation markers like [1], [12]
_WORD = re.compile(r"[a-z0-9]+")     # crude word tokenizer for overlap scoring


class GroundednessReport(BaseModel):
    # The structured verdict the output guardrail acts on.
    grounded: bool                    # overall: is the answer adequately supported?
    has_citations: bool
    cited_indices: list[int] = []
    invalid_indices: list[int] = []   # cited a source that wasn't retrieved (hallucinated marker)
    coverage: float = 0.0             # share of cited chunks with real overlap
    reason: str = ""                  # human-readable explanation when not grounded


def extract_citation_indices(text: str) -> list[int]:
    # Pull the integers out of every [n] marker, de-duplicated and sorted. Also
    # used by answer_node to map markers back to their source chunks.
    return sorted({int(m) for m in _CITE.findall(text)})


def _overlap(a: str, b: str) -> float:
    # Asymmetric word-set overlap: fraction of A's unique words that also appear
    # in B. Here A is the answer and B a cited chunk, so this asks "how much of the
    # answer's vocabulary is backed by this source?" — a fast, model-free support
    # signal (not a true entailment check, just the floor).
    ta, tb = set(_WORD.findall(a.lower())), set(_WORD.findall(b.lower()))
    if not ta:
        return 0.0
    return len(ta & tb) / len(ta)


def verify_groundedness(
    answer: str,
    chunks: Sequence[Chunk],
    *,
    refusal_markers: Sequence[str] = (),
    min_overlap: float = 0.08,
) -> GroundednessReport:
    # An explicit refusal ("I don't have enough grounded information") makes NO
    # factual claim, so there is nothing to support — treat it as grounded and let
    # it through. This is why the system prompt tells the model to refuse rather
    # than guess: a refusal always passes this gate.
    low = answer.lower()
    if any(mk.lower() in low for mk in refusal_markers):
        return GroundednessReport(grounded=True, has_citations=False, reason="refusal")

    # An answer that makes claims but cites NOTHING is ungrounded by definition —
    # we can't verify a single statement, so reject it.
    cited = extract_citation_indices(answer)
    if not cited:
        return GroundednessReport(
            grounded=False, has_citations=False, coverage=0.0,
            reason="answer makes claims without any citation",
        )

    # Split the cited markers into hallucinated vs. real: a valid index must point
    # at a chunk that was actually retrieved (1..len(chunks)). Anything outside
    # that range is a fabricated citation.
    invalid = [i for i in cited if i < 1 or i > len(chunks)]
    valid = [i for i in cited if 1 <= i <= len(chunks)]
    # Of the real citations, count how many actually overlap the answer text — i.e.
    # plausibly support it rather than being decorative.
    supported = 0
    for i in valid:
        if _overlap(answer, chunks[i - 1].text) >= min_overlap:
            supported += 1
    # coverage = fraction of ALL cited markers that are both real and supportive.
    coverage = supported / len(cited) if cited else 0.0
    # Grounded only if NO citation was fabricated AND at least half the citations
    # genuinely back the text. Both conditions matter: a single invented source is
    # disqualifying on its own.
    grounded = not invalid and coverage >= 0.5
    reason = ""
    if invalid:
        reason = f"cites non-existent source(s): {invalid}"
    elif not grounded:
        reason = "cited sources do not support the answer text"
    return GroundednessReport(
        grounded=grounded, has_citations=True, cited_indices=cited,
        invalid_indices=invalid, coverage=round(coverage, 3), reason=reason,
    )


__all__ = ["GroundednessReport", "extract_citation_indices", "verify_groundedness"]
