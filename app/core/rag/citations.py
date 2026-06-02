"""Citation + groundedness verification.

A deterministic backstop (no model needed): it checks that an answer's inline
``[n]`` markers reference *real* retrieved chunks and that the cited chunks share
meaningful lexical overlap with the answer. The output guardrail uses this to
flag hallucinated citations and unsupported answers. A NLI model endpoint can be
layered on top in prod (``groundedness_check``); this remains the floor.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from pydantic import BaseModel

from app.core.contracts import Chunk

_CITE = re.compile(r"\[(\d+)\]")
_WORD = re.compile(r"[a-z0-9]+")


class GroundednessReport(BaseModel):
    grounded: bool
    has_citations: bool
    cited_indices: list[int] = []
    invalid_indices: list[int] = []   # cited a source that wasn't retrieved
    coverage: float = 0.0             # share of cited chunks with real overlap
    reason: str = ""


def extract_citation_indices(text: str) -> list[int]:
    return sorted({int(m) for m in _CITE.findall(text)})


def _overlap(a: str, b: str) -> float:
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
    # An explicit refusal is "grounded" (it makes no factual claim).
    low = answer.lower()
    if any(mk.lower() in low for mk in refusal_markers):
        return GroundednessReport(grounded=True, has_citations=False, reason="refusal")

    cited = extract_citation_indices(answer)
    if not cited:
        return GroundednessReport(
            grounded=False, has_citations=False, coverage=0.0,
            reason="answer makes claims without any citation",
        )

    invalid = [i for i in cited if i < 1 or i > len(chunks)]
    valid = [i for i in cited if 1 <= i <= len(chunks)]
    supported = 0
    for i in valid:
        if _overlap(answer, chunks[i - 1].text) >= min_overlap:
            supported += 1
    coverage = supported / len(cited) if cited else 0.0
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
