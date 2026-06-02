"""Citation parsing and evidence checks for grounded contract answers."""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.db.repository import DataStore
from app.domain import User
from app.rbac.permissions import can_read_document


CITATION_PATTERN = re.compile(r"\[([A-Z]{2}-\d{4}),\s*Section\s+([0-9]+(?:\.[0-9]+)?)\]")


@dataclass(slots=True)
class Citation:
    contract_id: str
    section_number: str
    raw: str


@dataclass(slots=True)
class CitationVerification:
    citation: Citation
    valid: bool
    reason: str
    confidence: float


def parse_citations(text: str) -> list[Citation]:
    """Extract citations in the required `[TC-1001, Section 2.1]` format."""

    return [
        Citation(contract_id=match.group(1), section_number=match.group(2), raw=match.group(0))
        for match in CITATION_PATTERN.finditer(text)
    ]


def verify_citation(
    citation: Citation,
    user: User,
    store: DataStore,
    expected_terms: list[str] | None = None,
) -> CitationVerification:
    """Verify one citation against document access, section existence, and terms."""

    document = store.document_by_contract_id(citation.contract_id)
    if document is None:
        return CitationVerification(citation, False, "document_not_found", 0.0)
    if not can_read_document(user, document, store):
        return CitationVerification(citation, False, "document_access_denied", 0.0)

    top_level_section = citation.section_number.split(".")[0]
    section = store.section_by_number(document.id, top_level_section)
    if section is None:
        return CitationVerification(citation, False, "section_not_found", 0.0)

    if not expected_terms:
        return CitationVerification(citation, True, "verified", 0.85)

    section_text = section.text.lower()
    matched_terms = [term for term in expected_terms if term.lower() in section_text]
    # Confidence increases as more expected evidence terms are present in the cited section.
    confidence = 0.55 + 0.45 * (len(matched_terms) / max(len(expected_terms), 1))
    return CitationVerification(
        citation=citation,
        valid=bool(matched_terms),
        reason="verified" if matched_terms else "terms_not_supported",
        confidence=confidence if matched_terms else 0.35,
    )


def verify_answer_citations(
    answer: str,
    user: User,
    store: DataStore,
    expected_terms: list[str] | None = None,
    support_terms_by_citation: dict[str, list[str]] | None = None,
) -> list[CitationVerification]:
    """Verify every citation in an answer using optional per-citation terms."""

    return [
        verify_citation(
            citation,
            user,
            store,
            expected_terms=_terms_for_citation(citation, expected_terms, support_terms_by_citation),
        )
        for citation in parse_citations(answer)
    ]


def _terms_for_citation(
    citation: Citation,
    expected_terms: list[str] | None,
    support_terms_by_citation: dict[str, list[str]] | None,
) -> list[str] | None:
    """Resolve the most specific support terms available for a citation."""

    if not support_terms_by_citation:
        return expected_terms
    key = f"{citation.contract_id}|{citation.section_number}"
    return (
        support_terms_by_citation.get(citation.raw)
        or support_terms_by_citation.get(key)
        or support_terms_by_citation.get(citation.contract_id)
        or expected_terms
    )
