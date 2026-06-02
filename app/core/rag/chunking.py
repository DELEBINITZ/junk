"""Section-aware chunker for the local ingest helper.

In production the external cron embeds and pushes vectors, so the platform need
not chunk on the hot path. But the ``/ingest`` endpoint and dev seeders use this
to produce clause/section-precise chunks (better citations) with bounded size
and small overlap, never splitting mid-sentence when avoidable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_HEADING = re.compile(
    r"^\s*(#{1,6}\s+.+|\d+(?:\.\d+)*\.?\s+[A-Z].{0,80}|[A-Z][A-Z0-9 \-]{3,80})\s*$"
)
_SENT = re.compile(r"(?<=[.!?])\s+")


@dataclass
class ChunkPiece:
    text: str
    section: str
    ordinal: int


def _approx_tokens(text: str) -> int:
    return max(1, len(text.split()))


def _split_sections(text: str) -> list[tuple[str, str]]:
    """Return [(section_title, body)]. Falls back to one untitled section."""
    lines = text.splitlines()
    sections: list[tuple[str, list[str]]] = []
    current_title = ""
    current: list[str] = []
    for line in lines:
        if _HEADING.match(line) and len(line.strip()) < 90:
            if current:
                sections.append((current_title, current))
            current_title = line.strip().lstrip("#").strip()
            current = []
        else:
            current.append(line)
    if current or not sections:
        sections.append((current_title, current))
    return [(t, "\n".join(b).strip()) for t, b in sections if "\n".join(b).strip() or t]


def chunk_document(
    text: str,
    *,
    target_tokens: int = 400,
    overlap_tokens: int = 60,
    max_tokens: int = 600,
) -> list[ChunkPiece]:
    pieces: list[ChunkPiece] = []
    ordinal = 0
    for title, body in _split_sections(text):
        sentences = [s for s in _SENT.split(body) if s.strip()]
        if not sentences:
            continue
        buf: list[str] = []
        count = 0
        for sent in sentences:
            st = _approx_tokens(sent)
            if count + st > max_tokens and buf:
                pieces.append(ChunkPiece(" ".join(buf).strip(), title, ordinal))
                ordinal += 1
                # carry overlap
                carry, ccount = [], 0
                for s in reversed(buf):
                    carry.insert(0, s)
                    ccount += _approx_tokens(s)
                    if ccount >= overlap_tokens:
                        break
                buf, count = carry, ccount
            buf.append(sent)
            count += st
            if count >= target_tokens:
                pieces.append(ChunkPiece(" ".join(buf).strip(), title, ordinal))
                ordinal += 1
                carry, ccount = [], 0
                for s in reversed(buf):
                    carry.insert(0, s)
                    ccount += _approx_tokens(s)
                    if ccount >= overlap_tokens:
                        break
                buf, count = carry, ccount
        if buf:
            pieces.append(ChunkPiece(" ".join(buf).strip(), title, ordinal))
            ordinal += 1
    return pieces


__all__ = ["ChunkPiece", "chunk_document", "_split_sections"]
