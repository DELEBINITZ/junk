"""Document parsers for the local ingest path (PDF/DOCX/text/markdown/html).

FIRST STEP OF INGESTION: turn raw uploaded BYTES into plain TEXT. Ingestion is a
pipeline — parse -> chunk -> index — and this file is the "parse" stage. The text
it returns is what gets chunked and embedded into the corpus.

Two deliberate choices: parsers for heavy formats LAZY-IMPORT their library
(pypdf / python-docx) so those deps are only required when someone actually
uploads that type; and in production the external cron usually delivers pre-
extracted text, so these mainly serve direct uploads via ``/ingest``.
"""

from __future__ import annotations

import re


def parse_text(data: bytes | str) -> str:
    """The base case: plain/markdown text. Decodes bytes as UTF-8, replacing
    undecodable bytes instead of raising, so a slightly malformed file still
    ingests rather than failing the whole upload."""
    if isinstance(data, bytes):
        return data.decode("utf-8", errors="replace")
    return data


def parse_html(data: bytes | str) -> str:
    """Crudely strip HTML to its visible text: drop <script>/<style> blocks
    entirely, remove all remaining tags, then collapse whitespace. Good enough to
    feed RAG (we want the prose, not the markup); not a full HTML parser."""
    text = parse_text(data)
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", text)   # remove non-content blocks
    text = re.sub(r"(?s)<[^>]+>", " ", text)                        # strip every tag
    return re.sub(r"\s+", " ", text).strip()                        # normalize whitespace


def parse_pdf(data: bytes) -> str:
    """Extract text from a PDF, page by page, joined with blank lines. ``BytesIO``
    wraps the in-memory bytes as a file-like object for pypdf. ``or ""`` guards
    pages that yield no extractable text (e.g. scanned images)."""
    from io import BytesIO

    from pypdf import PdfReader  # lazy

    reader = PdfReader(BytesIO(data))
    return "\n\n".join((page.extract_text() or "") for page in reader.pages)


def parse_docx(data: bytes) -> str:
    """Extract text from a Word .docx by concatenating its paragraphs."""
    from io import BytesIO

    import docx  # python-docx, lazy

    document = docx.Document(BytesIO(data))
    return "\n".join(p.text for p in document.paragraphs)


def parse(data: bytes, *, content_type: str = "", filename: str = "") -> str:
    """The dispatcher: pick the right parser from the filename extension OR the
    MIME content-type (whichever is available), defaulting to plain text. This is
    the single entry point ingestion calls — callers don't choose a parser
    themselves."""
    name = (filename or "").lower()
    ct = (content_type or "").lower()
    if name.endswith(".pdf") or "pdf" in ct:
        return parse_pdf(data)
    if name.endswith(".docx") or "wordprocessingml" in ct:
        return parse_docx(data)
    if name.endswith((".html", ".htm")) or "html" in ct:
        return parse_html(data)
    return parse_text(data)


__all__ = ["parse", "parse_text", "parse_html", "parse_pdf", "parse_docx"]
