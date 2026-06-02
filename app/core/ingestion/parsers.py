"""Multi-format text extraction for ingestion (plan §7.1).

PDF via pypdf, DOCX via python-docx (both lazy, in the `prod` extra), everything
else read as UTF-8 text. Returns plain text that the indexer then chunks +
embeds. Unstructured/Surya OCR can replace this later behind the same signature.
"""

from __future__ import annotations

from pathlib import Path


def extract_text(path: str | Path) -> str:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _pdf_text(path)
    if suffix in (".docx", ".doc"):
        return _docx_text(path)
    return path.read_text(encoding="utf-8", errors="ignore")


def _pdf_text(path: Path) -> str:
    from pypdf import PdfReader  # lazy

    reader = PdfReader(str(path))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def _docx_text(path: Path) -> str:
    import docx  # lazy (python-docx)

    document = docx.Document(str(path))
    return "\n".join(paragraph.text for paragraph in document.paragraphs)
