"""Chunk contracts without destroying clause boundaries."""

from __future__ import annotations

from app.documents.parser import ParsedSection


def section_chunks(sections: list[ParsedSection], max_words: int = 900, overlap_words: int = 80):
    """Yield chunk dictionaries while preserving section boundaries.

    Most contract sections fit in a single chunk. If one is too large, the split
    happens inside that section with overlap, so retrieval can still cite the
    original clause instead of an arbitrary global token window.
    """

    for section in sections:
        words = section.text.split()
        if len(words) <= max_words:
            yield {
                "section_number": section.section_number,
                "section_title": section.section_title,
                "text": section.text,
                "chunk_index": 0,
            }
            continue

        start = 0
        chunk_index = 0
        while start < len(words):
            end = min(start + max_words, len(words))
            yield {
                "section_number": section.section_number,
                "section_title": section.section_title,
                "text": " ".join(words[start:end]),
                "chunk_index": chunk_index,
            }
            if end == len(words):
                break
            start = max(end - overlap_words, start + 1)
            chunk_index += 1
