"""Final answer validation before anything leaves the backend."""

from __future__ import annotations

import logging

from app.db.repository import DataStore
from app.domain import User
from app.guardrails.pii import redact_pii
from app.observability.logging import safe_extra
from app.rag.citation_verifier import verify_answer_citations


logger = logging.getLogger(__name__)


def validate_and_redact_output(
    answer: str,
    user: User,
    store: DataStore,
    expected_terms: list[str] | None = None,
    support_terms_by_citation: dict[str, list[str]] | None = None,
) -> dict[str, object]:
    """Verify citations, enforce confidence threshold, and redact final PII."""

    redacted_answer = redact_pii(answer)
    verifications = verify_answer_citations(
        redacted_answer,
        user,
        store,
        expected_terms,
        support_terms_by_citation=support_terms_by_citation,
    )
    require_citations = store.guardrail_configs[user.organization_id].require_citations
    threshold = store.guardrail_configs[user.organization_id].hallucination_confidence_threshold

    if require_citations and not verifications:
        # If an answer contains no verifiable citations and citations are
        # required for the tenant, return a safe low-confidence response.
        logger.warning(
            "guardrail.output.unsupported_no_citations",
            extra=safe_extra(user_id=user.id, organization_id=user.organization_id),
        )
        return {
            "answer": "I am not fully confident in this answer. Please verify against the contract.",
            "citations": [],
            "confidence": 0.0,
            "status": "unsupported",
        }

    valid = [item for item in verifications if item.valid]
    confidence = min((item.confidence for item in valid), default=0.0)
    if require_citations and (len(valid) != len(verifications) or confidence < threshold):
        logger.warning(
            "guardrail.output.low_confidence",
            extra=safe_extra(
                user_id=user.id,
                organization_id=user.organization_id,
                valid_citations=len(valid),
                total_citations=len(verifications),
                confidence=confidence,
                threshold=threshold,
            ),
        )
        return {
            "answer": "I am not fully confident in this answer. Please verify against the contract.",
            "citations": [item.citation.raw for item in valid],
            "confidence": confidence,
            "status": "low_confidence",
        }

    logger.debug(
        "guardrail.output.verified",
        extra=safe_extra(
            user_id=user.id,
            organization_id=user.organization_id,
            citation_count=len(valid),
            confidence=confidence if valid else 1.0,
        ),
    )
    return {
        "answer": redacted_answer,
        "citations": [item.citation.raw for item in valid],
        "confidence": confidence if valid else 1.0,
        "status": "verified",
    }
