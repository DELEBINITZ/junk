"""Section-aware chunker for the local ingest helper.

WHY CHUNKING EXISTS (the RAG concept): a retriever can't embed a whole 40-page
contract as one vector — you'd lose precision and blow past the embedding model's
input limit. So before indexing we split each document into "chunks": small,
self-contained passages. Each chunk is embedded and stored separately, and at
query time we retrieve the few chunks most relevant to the question. Smaller,
clause-precise chunks => sharper citations ("this clause says…") and tighter
context for the LLM.

Mental model: cut along the document's own SECTIONS first (headings, numbered
clauses), then pack whole SENTENCES into size-bounded chunks, carrying a little
OVERLAP from the end of one chunk into the start of the next so a fact that
straddles a chunk boundary still appears intact in at least one chunk.

In production the external cron embeds and pushes vectors, so the platform need
not chunk on the hot path. But the ``/ingest`` endpoint and dev seeders use this
to produce clause/section-precise chunks (better citations) with bounded size
and small overlap, never splitting mid-sentence when avoidable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Recognises a heading line three ways: Markdown "## ...", a numbered clause like
# "3.2 Termination", or an ALL-CAPS title. Headings become section boundaries.
_HEADING = re.compile(
    r"^\s*(#{1,6}\s+.+|\d+(?:\.\d+)*\.?\s+[A-Z].{0,80}|[A-Z][A-Z0-9 \-]{3,80})\s*$"
)
# Sentence splitter: break after . ! or ? followed by whitespace. The lookbehind
# keeps the punctuation attached to the sentence it ends.
_SENT = re.compile(r"(?<=[.!?])\s+")


@dataclass
class ChunkPiece:
    # One produced chunk: its text, the section heading it came from (carried into
    # the citation so the reader sees WHERE in the doc it is), and its position.
    text: str
    section: str
    ordinal: int       # 0-based order of this chunk within the whole document


def _approx_tokens(text: str) -> int:
    # Cheap token proxy: count whitespace-separated words. Good enough to BOUND
    # chunk size without importing a real tokenizer; the LLM's true token count is
    # close enough for sizing decisions here.
    return max(1, len(text.split()))


def _split_sections(text: str) -> list[tuple[str, str]]:
    """Return [(section_title, body)]. Falls back to one untitled section.

    First pass of chunking: walk the document line by line and START A NEW
    SECTION every time a heading line appears. Splitting on the document's own
    structure (rather than blindly every N tokens) keeps related clauses together
    and gives every chunk a meaningful section label for citations.
    """
    lines = text.splitlines()
    sections: list[tuple[str, list[str]]] = []
    current_title = ""
    current: list[str] = []
    for line in lines:
        # A heading line closes the section being accumulated and opens a new one.
        if _HEADING.match(line) and len(line.strip()) < 90:
            if current:
                sections.append((current_title, current))
            current_title = line.strip().lstrip("#").strip()   # strip Markdown "#"
            current = []
        else:
            current.append(line)
    # Flush the trailing section (and guarantee at least one section for a doc
    # with no headings at all).
    if current or not sections:
        sections.append((current_title, current))
    # Drop sections that are empty after trimming, unless they carry a title.
    return [(t, "\n".join(b).strip()) for t, b in sections if "\n".join(b).strip() or t]


def chunk_document(
    text: str,
    *,
    target_tokens: int = 400,    # soft target: emit a chunk once it reaches this size
    overlap_tokens: int = 60,    # how much tail to repeat at the start of the next chunk
    max_tokens: int = 600,       # hard ceiling: never let a chunk grow past this
) -> list[ChunkPiece]:
    """Second pass of chunking: within each section, pack whole sentences into
    size-bounded, overlapping chunks. ``target_tokens`` is the comfortable size,
    ``max_tokens`` the hard cap, and ``overlap_tokens`` the slice carried forward
    so context spanning a boundary survives in both neighbours."""
    pieces: list[ChunkPiece] = []
    ordinal = 0
    for title, body in _split_sections(text):
        # Split the section body into sentences so we never cut mid-sentence.
        sentences = [s for s in _SENT.split(body) if s.strip()]
        if not sentences:
            continue
        buf: list[str] = []      # sentences accumulated for the chunk in progress
        count = 0                # running token count of ``buf``
        for sent in sentences:
            st = _approx_tokens(sent)
            # Adding this sentence would breach the HARD cap: flush what we have
            # first, so no chunk ever exceeds ``max_tokens``.
            if count + st > max_tokens and buf:
                pieces.append(ChunkPiece(" ".join(buf).strip(), title, ordinal))
                ordinal += 1
                # Carry overlap: seed the next chunk with the LAST few sentences of
                # this one (walking backwards until we've collected ~overlap_tokens).
                # This repetition is deliberate — it's what stops a fact split across
                # the boundary from being lost to retrieval.
                carry, ccount = [], 0
                for s in reversed(buf):
                    carry.insert(0, s)
                    ccount += _approx_tokens(s)
                    if ccount >= overlap_tokens:
                        break
                buf, count = carry, ccount
            buf.append(sent)
            count += st
            # Reached the SOFT target: emit the chunk and again carry overlap into
            # the next one. (Same carry logic as above.)
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
        # Flush whatever's left in this section as a final (smaller) chunk.
        if buf:
            pieces.append(ChunkPiece(" ".join(buf).strip(), title, ordinal))
            ordinal += 1
    return pieces


__all__ = ["ChunkPiece", "chunk_document", "_split_sections"]
