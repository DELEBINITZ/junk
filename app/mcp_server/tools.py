"""Typed contract-analysis tools exposed through MCP.

Every tool receives the authenticated `User` from backend context and performs
its own authorization check. The agent can request a tool call, but it cannot
override tenant or role decisions.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
import logging
from typing import Any

from app.db.repository import DataStore
from app.domain import Document, User
from app.guardrails.pii import redact_pii
from app.observability.logging import safe_extra
from app.rag.retrieval import search_chunks
from app.rbac.permissions import can_query_document, can_read_document, queryable_documents


logger = logging.getLogger(__name__)


class ToolError(Exception):
    """Tool-level error that maps cleanly into an MCP JSON-RPC error response."""

    def __init__(self, message: str, code: int = 403):
        super().__init__(message)
        self.code = code


CLAUSE_SECTION = {
    "termination": "8",
    "payment": "3",
    "liability": "7",
    "confidentiality": "4",
    "force_majeure": "9",
    "warranty": "6",
}

TOOL_ORDER = [
    "search_contracts",
    "extract_clause",
    "compare_clauses",
    "extract_metadata",
    "calculate_risk_score",
    "find_expiring_contracts",
]


def tool_definitions() -> list[dict[str, Any]]:
    """Return MCP tool schemas for `tools/list` discovery."""

    return [
        {
            "name": "search_contracts",
            "description": "Semantic search across contracts authorized for the current user.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": 5},
                    "filter_tags": {"type": "array", "items": {"type": "string"}},
                    "organization_id": {"type": "string", "description": "Optional; cross-checked against JWT."},
                },
                "required": ["query"],
            },
        },
        {
            "name": "extract_clause",
            "description": "Extract an exact clause from one authorized contract.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "contract_id": {"type": "string"},
                    "clause_type": {
                        "type": "string",
                        "enum": sorted(CLAUSE_SECTION.keys()),
                    },
                },
                "required": ["contract_id", "clause_type"],
            },
        },
        {
            "name": "compare_clauses",
            "description": "Compare one clause type across up to five authorized contracts.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "contract_ids": {"type": "array", "items": {"type": "string"}, "maxItems": 5},
                    "clause_type": {"type": "string", "enum": sorted(CLAUSE_SECTION.keys())},
                },
                "required": ["contract_ids", "clause_type"],
            },
        },
        {
            "name": "extract_metadata",
            "description": "Return structured metadata extracted from an authorized contract.",
            "inputSchema": {
                "type": "object",
                "properties": {"contract_id": {"type": "string"}},
                "required": ["contract_id"],
            },
        },
        {
            "name": "calculate_risk_score",
            "description": "Calculate rule-based contract risk score with cited factors.",
            "inputSchema": {
                "type": "object",
                "properties": {"contract_id": {"type": "string"}},
                "required": ["contract_id"],
            },
        },
        {
            "name": "find_expiring_contracts",
            "description": "Find authorized contracts expiring in a date range.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "start_date": {"type": "string", "format": "date"},
                    "end_date": {"type": "string", "format": "date"},
                    "auto_renewal_only": {"type": "boolean", "default": False},
                    "organization_id": {"type": "string", "description": "Optional; cross-checked against JWT."},
                },
                "required": ["start_date", "end_date"],
            },
        },
    ]


def call_tool(name: str, arguments: dict[str, Any], user: User, store: DataStore) -> dict[str, Any]:
    """Dispatch a validated MCP tool name to its implementation."""

    if name == "search_contracts":
        return search_contracts(user, store, **arguments)
    if name == "extract_clause":
        return extract_clause(user, store, **arguments)
    if name == "compare_clauses":
        return compare_clauses(user, store, **arguments)
    if name == "extract_metadata":
        return extract_metadata_tool(user, store, **arguments)
    if name == "calculate_risk_score":
        return calculate_risk_score(user, store, **arguments)
    if name == "find_expiring_contracts":
        return find_expiring_contracts(user, store, **arguments)
    raise ToolError(f"Unknown tool: {name}", code=404)


def search_contracts(
    user: User,
    store: DataStore,
    query: str,
    top_k: int = 5,
    filter_tags: list[str] | None = None,
    organization_id: str | None = None,
) -> dict[str, Any]:
    """Semantic search across the caller's authorized contract chunks.

    This is one of the most important security boundaries in the PoC. The MCP
    client/agent may ask to search contracts, but it never gets to decide which
    tenant or document set is searchable. The authenticated `user` drives that
    scope, and this tool reduces the corpus before vector ranking happens.
    """

    # Never trust tenant context supplied by the client or model. It is allowed
    # only as a convenience hint, and must match the organization from the JWT.
    _assert_same_org_arg(user, organization_id)

    # RBAC happens before retrieval. This prevents a vector search from scoring
    # or even seeing chunks from cross-tenant or non-shared documents.
    documents = queryable_documents(user, store)

    # Tags are a secondary filter inside the already-authorized set. They narrow
    # a search to categories such as "healthcare" or "software"; they never
    # grant access to documents the user could not otherwise query.
    if filter_tags:
        wanted = set(filter_tags)
        documents = [document for document in documents if wanted.intersection(document.tags)]

    # The retrieval layer needs document metadata to build cited search hits.
    # Keeping this as a map also guarantees chunks with unknown document IDs are
    # ignored by `search_chunks`.
    document_map = {document.id: document for document in documents}
    chunks = store.chunks_for_documents(document_map.keys(), organization_id=user.organization_id)

    # Clamp top_k so a caller cannot request an unbounded result set. For the PoC
    # this protects response size; in production it also protects latency/cost.
    safe_top_k = max(1, min(top_k, 20))
    hits = search_chunks(query, chunks, document_map, store.embedder, top_k=safe_top_k)
    logger.info(
        "mcp.tool.search_contracts.complete",
        extra=safe_extra(
            user_id=user.id,
            organization_id=user.organization_id,
            authorized_documents=len(documents),
            searched_chunks=len(chunks),
            requested_top_k=top_k,
            safe_top_k=safe_top_k,
            hit_count=len(hits),
            filter_tags=filter_tags or [],
        ),
    )

    return {
        "matches": [
            {
                "contract_id": hit.document.contract_id,
                "title": hit.document.title,
                "score": round(hit.score, 4),
                "section_number": hit.chunk.metadata["section_number"],
                "section_title": hit.chunk.metadata["section_title"],
                # Chunks are stored with PII redacted during ingestion. The
                # snippet is capped so tool output remains readable and safe to
                # include in an LLM prompt.
                "snippet": hit.chunk.text[:1200],
                # The agent and output validator both rely on this exact format
                # to verify grounded answers before returning them to the user.
                "citation": f"[{hit.document.contract_id}, Section {hit.chunk.metadata['section_number']}]",
            }
            for hit in hits
        ]
    }


def extract_clause(
    user: User,
    store: DataStore,
    contract_id: str,
    clause_type: str,
) -> dict[str, Any]:
    """Return exact clause text from a mapped section in one authorized contract."""

    document = _require_contract(user, store, contract_id, require_query=True)
    section_number = CLAUSE_SECTION.get(clause_type)
    if section_number is None:
        logger.warning(
            "mcp.tool.extract_clause.unsupported_type",
            extra=safe_extra(user_id=user.id, contract_id=contract_id, clause_type=clause_type),
        )
        raise ToolError(f"Unsupported clause_type: {clause_type}", code=400)
    section = store.section_by_number(document.id, section_number, organization_id=user.organization_id)
    if section is None:
        logger.info(
            "mcp.tool.extract_clause.not_found",
            extra=safe_extra(user_id=user.id, contract_id=contract_id, clause_type=clause_type),
        )
        return {
            "contract_id": contract_id,
            "clause_type": clause_type,
            "found": False,
            "text": "Not found",
            "citation": None,
        }
    citation_section = _preferred_subsection(section.text, section_number)
    logger.info(
        "mcp.tool.extract_clause.complete",
        extra=safe_extra(
            user_id=user.id,
            contract_id=contract_id,
            clause_type=clause_type,
            section_number=section.section_number,
        ),
    )
    return {
        "contract_id": contract_id,
        "title": document.title,
        "clause_type": clause_type,
        "found": True,
        "section_number": section.section_number,
        "section_title": section.section_title,
        "text": redact_pii(section.text),
        "citation": f"[{contract_id}, Section {citation_section}]",
        "line_start": section.line_start,
        "line_end": section.line_end,
    }


def compare_clauses(
    user: User,
    store: DataStore,
    contract_ids: list[str],
    clause_type: str,
) -> dict[str, Any]:
    """Compare deterministic clause extracts across up to five contracts."""

    if not contract_ids or len(contract_ids) > 5:
        raise ToolError("compare_clauses requires 1 to 5 contract_ids", code=400)
    clauses = [extract_clause(user, store, contract_id, clause_type) for contract_id in contract_ids]
    comparisons = []
    for clause in clauses:
        text = str(clause.get("text", ""))
        comparisons.append(
            {
                "contract_id": clause["contract_id"],
                "title": clause.get("title"),
                "citation": clause.get("citation"),
                "key_terms": _extract_clause_terms(text),
                "risk_level": _clause_risk_level(text),
            }
        )
    return {"clause_type": clause_type, "comparisons": comparisons, "summary": _comparison_summary(comparisons)}


def extract_metadata_tool(user: User, store: DataStore, contract_id: str) -> dict[str, Any]:
    """Return parsed metadata for one authorized contract with source citations."""

    document = _require_contract(user, store, contract_id, require_query=True)
    metadata = dict(document.metadata)
    metadata["contract_id"] = document.contract_id
    metadata["title"] = document.title
    metadata["citations"] = {
        "effective_date": f"[{contract_id}, Section 2.1]",
        "expiration_date": f"[{contract_id}, Section 2.1]",
        "contract_value": f"[{contract_id}, Section 3.1]",
        "notice_period_days": f"[{contract_id}, Section 8.1]",
        "auto_renewal": f"[{contract_id}, Section 2.2]",
        "governing_law": f"[{contract_id}, Section 10.1]",
    }
    return metadata


def calculate_risk_score(user: User, store: DataStore, contract_id: str) -> dict[str, Any]:
    """Calculate explainable rule-based risk factors for one contract."""

    document = _require_contract(user, store, contract_id, require_query=True)
    metadata = document.metadata
    risk_points = 0
    factors: list[dict[str, str]] = []
    recommendations: list[str] = []

    notice_days = metadata.get("notice_period_days")
    if isinstance(notice_days, int) and notice_days >= 180:
        risk_points += 3
        factors.append({"level": "high", "text": f"Long {notice_days}-day notice period.", "citation": f"[{contract_id}, Section 8.1]"})
    elif isinstance(notice_days, int) and notice_days >= 60:
        risk_points += 2
        factors.append({"level": "medium", "text": f"{notice_days}-day termination notice period.", "citation": f"[{contract_id}, Section 8.1]"})

    if metadata.get("auto_renewal"):
        risk_points += 2
        factors.append({"level": "medium", "text": "Auto-renewal requires deadline tracking.", "citation": f"[{contract_id}, Section 2.2]"})
        if metadata.get("action_required_by"):
            recommendations.append(f"Track non-renewal deadline: {metadata['action_required_by']}.")

    liability = extract_clause(user, store, contract_id, "liability")
    liability_text = str(liability.get("text", "")).lower()
    if "not be capped" in liability_text or "not capped" in liability_text:
        risk_points += 3
        factors.append({"level": "high", "text": "Uncapped liability appears in the liability section.", "citation": str(liability.get("citation"))})
    elif "twelve (12) months" in liability_text or "12 months" in liability_text:
        risk_points += 2
        factors.append({"level": "medium", "text": "Liability is capped at prior 12 months of fees.", "citation": str(liability.get("citation"))})

    force_majeure = extract_clause(user, store, contract_id, "force_majeure")
    fm_text = str(force_majeure.get("text", "")).lower()
    if "internet" in fm_text or "component shortages" in fm_text:
        risk_points += 1
        factors.append({"level": "medium", "text": "Force majeure language may be broad for operational continuity.", "citation": str(force_majeure.get("citation"))})

    if not recommendations:
        recommendations.append("Review highlighted clauses before renewal or amendment.")

    overall = "low"
    if risk_points >= 5:
        overall = "high"
    elif risk_points >= 2:
        overall = "medium"

    return {
        "contract_id": document.contract_id,
        "title": document.title,
        "overall_risk": overall,
        "risk_points": risk_points,
        "risk_factors": factors,
        "recommendations": recommendations,
    }


def find_expiring_contracts(
    user: User,
    store: DataStore,
    start_date: str,
    end_date: str,
    auto_renewal_only: bool = False,
    organization_id: str | None = None,
) -> dict[str, Any]:
    """Find authorized contracts expiring in a date range and sort by value."""

    _assert_same_org_arg(user, organization_id)
    start = _parse_date(start_date)
    end = _parse_date(end_date)
    if start > end:
        raise ToolError("start_date must be before end_date", code=400)

    results = []
    for document in queryable_documents(user, store):
        metadata = document.metadata
        expires = metadata.get("expiration_date")
        if not isinstance(expires, str):
            continue
        expiration_date = _parse_date(expires)
        if not (start <= expiration_date <= end):
            continue
        if auto_renewal_only and not metadata.get("auto_renewal"):
            continue
        results.append(
            {
                "contract_id": document.contract_id,
                "title": document.title,
                "expires": expires,
                "notice_period_days": metadata.get("notice_period_days"),
                "action_required_by": metadata.get("action_required_by"),
                "value": metadata.get("contract_value"),
                "contract_value_numeric": metadata.get("contract_value_numeric"),
                "auto_renewal": bool(metadata.get("auto_renewal")),
                "citations": {
                    "expiration": f"[{document.contract_id}, Section 2.1]",
                    "value": f"[{document.contract_id}, Section 3.1]",
                    "notice": f"[{document.contract_id}, Section 8.1]",
                    "renewal": f"[{document.contract_id}, Section 2.2]",
                },
            }
        )
    results.sort(key=lambda item: float(item.get("contract_value_numeric") or 0), reverse=True)
    logger.info(
        "mcp.tool.find_expiring_contracts.complete",
        extra=safe_extra(
            user_id=user.id,
            organization_id=user.organization_id,
            start_date=start_date,
            end_date=end_date,
            auto_renewal_only=auto_renewal_only,
            result_count=len(results),
            contract_ids=[item["contract_id"] for item in results],
        ),
    )
    return {"contracts": results}


def _require_contract(
    user: User,
    store: DataStore,
    contract_id: str,
    require_query: bool = False,
) -> Document:
    """Load a contract and enforce read/query access before returning it."""

    document = store.document_by_contract_id(contract_id, organization_id=user.organization_id)
    if document is None:
        logger.warning(
            "mcp.contract.not_found",
            extra=safe_extra(user_id=user.id, contract_id=contract_id),
        )
        raise ToolError("Contract not found", code=404)
    allowed = can_query_document(user, document, store) if require_query else can_read_document(user, document, store)
    if not allowed:
        logger.warning(
            "mcp.contract.access_denied",
            extra=safe_extra(
                user_id=user.id,
                organization_id=user.organization_id,
                contract_id=contract_id,
                document_organization_id=document.organization_id,
                require_query=require_query,
            ),
        )
        raise ToolError("Contract access denied", code=403)
    return document


def _assert_same_org_arg(user: User, organization_id: str | None) -> None:
    """Reject client/model-supplied tenant context that conflicts with the JWT."""

    if organization_id is not None and organization_id != user.organization_id:
        logger.warning(
            "mcp.organization_mismatch",
            extra=safe_extra(
                user_id=user.id,
                authenticated_organization_id=user.organization_id,
                supplied_organization_id=organization_id,
            ),
        )
        raise ToolError("organization_id does not match authenticated user", code=403)


def _parse_date(value: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ToolError(f"Invalid date: {value}", code=400) from exc


def _preferred_subsection(section_text: str, section_number: str) -> str:
    match = next(
        (line for line in section_text.splitlines() if line.strip().startswith(f"{section_number}.1")),
        None,
    )
    return f"{section_number}.1" if match else section_number


def _extract_clause_terms(text: str) -> dict[str, Any]:
    lowered = text.lower()
    terms: dict[str, Any] = {
        "notice_period_days": None,
        "liability_cap": None,
        "exceptions": [],
        "auto_renewal": "auto" in lowered and "renew" in lowered,
        "termination_fee": "termination fee" in lowered,
    }
    notice = next((days for days in [30, 60, 90, 180] if f"{days} days" in lowered), None)
    terms["notice_period_days"] = notice
    if "twelve (12) months" in lowered or "12 months" in lowered:
        terms["liability_cap"] = "prior 12 months fees"
    elif "purchase price" in lowered:
        terms["liability_cap"] = "purchase price"
    elif "not be capped" in lowered or "not capped" in lowered:
        terms["liability_cap"] = "uncapped"
    for word in ["confidentiality", "intellectual property", "hipaa", "gross negligence", "willful misconduct"]:
        if word in lowered:
            terms["exceptions"].append(word)
    return terms


def _clause_risk_level(text: str) -> str:
    lowered = text.lower()
    if "180 days" in lowered or "not be capped" in lowered or "not capped" in lowered:
        return "high"
    if "90 days" in lowered or "60 days" in lowered or "twelve (12) months" in lowered:
        return "medium"
    return "low"


def _comparison_summary(comparisons: list[dict[str, Any]]) -> str:
    high = [item["contract_id"] for item in comparisons if item["risk_level"] == "high"]
    if high:
        return f"Highest risk clauses: {', '.join(high)}."
    return "No high-risk clause differences identified."
