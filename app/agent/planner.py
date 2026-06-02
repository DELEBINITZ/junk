"""Rule-based planner for the controlled contract-analysis agent.

The planner is intentionally deterministic. For this assignment, predictability
and auditability matter more than a creative general-purpose planner that might
invent tools or skip evidence requirements.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class PlanStep:
    step_id: str
    tool: str
    args: dict[str, Any]
    depends_on: list[str]


@dataclass(slots=True)
class AgentPlan:
    intent: str
    steps: list[PlanStep]
    refusal: str | None = None


def build_plan(query: str) -> AgentPlan:
    """Classify a user query into a small, auditable MCP tool plan."""

    lowered = query.lower()
    if "draft" in lowered and "new contract" in lowered:
        return AgentPlan(
            intent="out_of_scope_drafting",
            steps=[],
            refusal=(
                "I can analyze existing contracts, compare clauses, extract metadata, and identify "
                "risks. I cannot draft a completely new contract from scratch in this PoC because "
                "that is outside the allowed contract-analysis tool scope."
            ),
        )

    contract_id = _find_contract_id(query)
    if contract_id is None and not _looks_like_contract_analysis(lowered):
        # Avoid sending social or unrelated text into vector search. Retrieval
        # will always find a nearest neighbor, even when no good answer exists.
        return AgentPlan(
            intent="out_of_scope_non_contract_query",
            steps=[],
            refusal=(
                "I can help analyze contracts, compare clauses, extract metadata, identify risks, "
                "test guardrails, and explain MCP tool results. Please ask a contract-analysis question."
            ),
        )

    if "q2 2025" in lowered or ("renew" in lowered and "2025" in lowered):
        return AgentPlan(
            intent="renewal_action_plan",
            steps=[
                PlanStep(
                    step_id="s1",
                    tool="find_expiring_contracts",
                    args={"start_date": "2025-04-01", "end_date": "2025-06-30"},
                    depends_on=[],
                )
            ],
        )

    if contract_id and "notice" in lowered:
        return AgentPlan(
            intent="notice_period_lookup",
            steps=[
                PlanStep(
                    step_id="s1",
                    tool="extract_metadata",
                    args={"contract_id": contract_id},
                    depends_on=[],
                )
            ],
        )

    if contract_id and ("value" in lowered or "amount" in lowered or "cost" in lowered):
        return AgentPlan(
            intent="contract_value_lookup",
            steps=[
                PlanStep(
                    step_id="s1",
                    tool="extract_metadata",
                    args={"contract_id": contract_id},
                    depends_on=[],
                )
            ],
        )

    if contract_id and ("contact" in lowered or "ssn" in lowered or "email" in lowered):
        return AgentPlan(
            intent="pii_contact_lookup",
            steps=[
                PlanStep(
                    step_id="s1",
                    tool="search_contracts",
                    args={"query": f"{contract_id} contact representative SSN email phone", "top_k": 5},
                    depends_on=[],
                )
            ],
        )

    if contract_id and "risk" in lowered:
        return AgentPlan(
            intent="risk_lookup",
            steps=[
                PlanStep(
                    step_id="s1",
                    tool="calculate_risk_score",
                    args={"contract_id": contract_id},
                    depends_on=[],
                )
            ],
        )

    return AgentPlan(
        intent="semantic_search",
        steps=[
            PlanStep(
                step_id="s1",
                tool="search_contracts",
                args={"query": query, "top_k": 5},
                depends_on=[],
            )
        ],
    )


def _find_contract_id(query: str) -> str | None:
    """Find corpus-style contract IDs such as TC-1001 or MC-2033."""

    match = re.search(r"\b[A-Z]{2}-\d{4}\b", query.upper())
    return match.group(0) if match else None


def _looks_like_contract_analysis(lowered_query: str) -> bool:
    """Heuristic gate for deciding whether fallback semantic search is appropriate."""

    contract_terms = {
        "agreement",
        "auto-renewal",
        "business associate",
        "citation",
        "clause",
        "compare",
        "confidential",
        "contract",
        "deadline",
        "document",
        "effective date",
        "email",
        "expire",
        "expiration",
        "force majeure",
        "governing law",
        "guardrail",
        "hipaa",
        "indemnification",
        "lease",
        "liability",
        "mcp",
        "metadata",
        "nda",
        "notice",
        "party",
        "payor",
        "payment",
        "phi",
        "phone",
        "pii",
        "renew",
        "risk",
        "section",
        "ssn",
        "termination",
        "value",
        "warranty",
    }
    return any(term in lowered_query for term in contract_terms)
