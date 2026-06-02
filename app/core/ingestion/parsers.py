"""Document parsers for the local ingest path (PDF/DOCX/text/markdown/html).

Lazy-imports pypdf / python-docx so they're only needed when actually parsing
those types. In production the external cron typically delivers pre-extracted
text; these support direct uploads via ``/ingest``.
"""

from __future__ import annotations

import re


def parse_text(data: bytes | str) -> str:
    if isinstance(data, bytes):
        return data.decode("utf-8", errors="replace")
    return data


def parse_html(data: bytes | str) -> str:
    text = parse_text(data)
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_pdf(data: bytes) -> str:
    from io import BytesIO

    from pypdf import PdfReader  # lazy

    reader = PdfReader(BytesIO(data))
    return "\n\n".join((page.extract_text() or "") for page in reader.pages)


def parse_docx(data: bytes) -> str:
    from io import BytesIO

    import docx  # python-docx, lazy

    document = docx.Document(BytesIO(data))
    return "\n".join(p.text for p in document.paragraphs)


def parse(data: bytes, *, content_type: str = "", filename: str = "") -> str:
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
