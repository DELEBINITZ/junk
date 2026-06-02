"""End-to-end AI query workflow.

This module is the main place to explain the assignment's AI safety posture:
guard the input, plan deterministically, call authorized MCP tools, compose only
from tool evidence, verify citations, redact output, and audit the result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import re
from typing import Any

from app.agent.executor import execute_plan
from app.agent.planner import AgentPlan, build_plan
from app.audit.logger import log_event
from app.db.repository import DataStore
from app.domain import QueryRecord, User, new_id
from app.guardrails.moderation import is_harmful_legal_request
from app.guardrails.output_validator import validate_and_redact_output
from app.guardrails.pii import redact_pii
from app.guardrails.prompt_injection import is_prompt_injection
from app.llm.client import estimate_tokens, get_llm_client
from app.llm.prompts import (
    ANSWER_POLISHING_SYSTEM_PROMPT,
    RAG_SYSTEM_PROMPT,
    build_polishing_user_prompt,
    build_rag_user_prompt,
)
from app.observability.logging import safe_extra
from app.rag.citation_verifier import parse_citations


PROMPT_INJECTION_ERROR = "Your query appears to contain prohibited patterns. Please rephrase."
MIN_SEMANTIC_MATCH_SCORE = 0.12
logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ComposedAnswer:
    """Answer text plus the evidence terms required for citation verification."""

    text: str
    support_terms_by_citation: dict[str, list[str]] = field(default_factory=dict)
    llm_system_prompt: str | None = None
    llm_user_prompt: str | None = None
    llm_mode: str = "answer_polishing"


@dataclass(slots=True)
class LLMTrace:
    """Usage metadata returned with every AI query response."""

    provider: str
    model: str
    mode: str
    used: bool
    prompt_tokens_estimate: int = 0
    completion_tokens_estimate: int = 0
    total_tokens_estimate: int = 0
    fallback_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "mode": self.mode,
            "used": self.used,
            "prompt_tokens_estimate": self.prompt_tokens_estimate,
            "completion_tokens_estimate": self.completion_tokens_estimate,
            "total_tokens_estimate": self.total_tokens_estimate,
            "fallback_reason": self.fallback_reason,
        }


def run_agent_query(query: str, user: User, store: DataStore) -> dict[str, Any]:
    """Run the guarded plan-execute-verify loop for one user query."""

    redacted_query = redact_pii(query)
    logger.info(
        "ai.query.start",
        extra=safe_extra(
            user_id=user.id,
            organization_id=user.organization_id,
            role=user.role,
            query_chars=len(redacted_query),
        ),
    )
    config = store.guardrail_configs[user.organization_id]
    if is_prompt_injection(query, config.blocked_keywords):
        logger.warning(
            "guardrail.prompt_injection.blocked",
            extra=safe_extra(user_id=user.id, organization_id=user.organization_id, query=redacted_query),
        )
        log_event(
            store,
            user,
            action="ai.query",
            resource_type="query",
            resource_id=None,
            outcome="blocked",
            details={"reason": "prompt_injection", "query": redacted_query},
        )
        return {"error": PROMPT_INJECTION_ERROR, "guardrails": {"prompt_injection": "blocked"}}

    if is_harmful_legal_request(query):
        logger.warning(
            "guardrail.moderation.blocked",
            extra=safe_extra(user_id=user.id, organization_id=user.organization_id, query=redacted_query),
        )
        log_event(
            store,
            user,
            action="ai.query",
            resource_type="query",
            resource_id=None,
            outcome="blocked",
            details={"reason": "harmful_legal_request", "query": redacted_query},
        )
        return {
            "error": (
                "I can analyze existing clauses, but I will not generate harmful, deceptive, "
                "or abusive legal language."
            ),
            "guardrails": {"moderation": "blocked"},
        }

    plan = build_plan(redacted_query)
    query_id = new_id()
    logger.info(
        "ai.plan.created",
        extra=safe_extra(
            query_id=query_id,
            user_id=user.id,
            organization_id=user.organization_id,
            intent=plan.intent,
            step_count=len(plan.steps),
            refused=bool(plan.refusal),
        ),
    )
    if plan.refusal:
        # Refusals are successful policy outcomes: no tool calls, no retrieval,
        # and a stored audit/query record explaining why the agent stopped.
        result = {
            "query_id": query_id,
            "status": "refused",
            "intent": plan.intent,
            "plan": _serialize_plan(plan),
            "answer": plan.refusal,
            "tool_results": [],
            "llm": _not_used_llm_trace("refusal").as_dict(),
            "guardrails": {"output": "not_applicable"},
        }
        _record_query(store, user, query_id, redacted_query, result)
        logger.info(
            "ai.query.refused",
            extra=safe_extra(query_id=query_id, intent=plan.intent, user_id=user.id),
        )
        return result

    tool_results = execute_plan(plan, user, store)
    logger.info(
        "ai.tools.executed",
        extra=safe_extra(
            query_id=query_id,
            intent=plan.intent,
            tool_count=len(tool_results),
            failed_steps=[
                {"step_id": item.get("step_id"), "tool": item.get("tool"), "error": item.get("error")}
                for item in tool_results
                if item.get("status") != "success"
            ],
        ),
    )
    composed = _compose_answer(redacted_query, plan, tool_results, store)
    # LLM usage is optional and only for wording. The composed answer already
    # contains facts and citations from authorized tool output.
    final_answer, llm_trace = _generate_or_polish_with_llm(composed)
    logger.info("ai.llm.trace", extra=safe_extra(query_id=query_id, **llm_trace.as_dict()))
    validated = validate_and_redact_output(
        final_answer,
        user,
        store,
        support_terms_by_citation=composed.support_terms_by_citation,
    )
    result = {
        "query_id": query_id,
        "status": validated["status"],
        "intent": plan.intent,
        "plan": _serialize_plan(plan, executed_results=tool_results),
        "answer": validated["answer"],
        "citations": validated["citations"],
        "confidence": validated["confidence"],
        "tool_results": tool_results,
        "llm": llm_trace.as_dict(),
        "guardrails": {"output": validated["status"]},
    }
    _record_query(store, user, query_id, redacted_query, result)
    logger.info(
        "ai.query.complete",
        extra=safe_extra(
            query_id=query_id,
            status=result["status"],
            intent=plan.intent,
            citation_count=len(result["citations"]),
            confidence=result["confidence"],
        ),
    )
    return result


def _compose_answer(query: str, plan: AgentPlan, tool_results: list[dict[str, Any]], store: DataStore) -> ComposedAnswer:
    """Build a grounded answer from MCP tool results, not from model memory."""

    if plan.intent == "renewal_action_plan":
        return _compose_renewal_plan(tool_results)
    if plan.intent == "notice_period_lookup":
        first = _first_successful_result(tool_results)
        if first is None:
            return ComposedAnswer("I could not complete the notice-period lookup with the available tools.")
        metadata = first["result"]
        contract_id = metadata["contract_id"]
        days = metadata.get("notice_period_days")
        if days is None:
            return ComposedAnswer(f"I could not find a termination notice period for {contract_id}.")
        citation = metadata["citations"]["notice_period_days"]
        return (
            ComposedAnswer(
                text=f"The termination notice period is {days} days {citation}.",
                support_terms_by_citation={citation: [f"{days} days", "Termination Notice Period"]},
            )
        )
    if plan.intent == "contract_value_lookup":
        first = _first_successful_result(tool_results)
        if first is None:
            return ComposedAnswer("I could not complete the contract-value lookup with the available tools.")
        metadata = first["result"]
        contract_id = metadata["contract_id"]
        value = metadata.get("contract_value")
        if not value:
            return ComposedAnswer(f"I could not find a contract value for {contract_id}.")
        citation = metadata["citations"]["contract_value"]
        return ComposedAnswer(
            text=f"The contract value for {contract_id} is {value} {citation}.",
            support_terms_by_citation={citation: [str(value).split()[0], "Total Contract Value"]},
        )
    if plan.intent == "risk_lookup":
        first = _first_successful_result(tool_results)
        if first is None:
            return ComposedAnswer("I could not complete the risk lookup with the available tools.")
        risk = first["result"]
        factors = " ".join(
            f"{item['text']} {item['citation']}" for item in risk.get("risk_factors", [])
        )
        support = {
            item["citation"]: _risk_support_terms(item)
            for item in risk.get("risk_factors", [])
            if item.get("citation")
        }
        return ComposedAnswer(
            text=(
                f"{risk['contract_id']} has {risk['overall_risk']} overall risk. "
                f"{factors or 'No high-risk factors were identified.'}"
            ),
            support_terms_by_citation=support,
        )
    if plan.intent == "pii_contact_lookup":
        first = _first_successful_result(tool_results)
        if first is None:
            return ComposedAnswer("I could not complete the contact lookup with the available tools.")
        matches = first["result"].get("matches", [])
        header_matches = [match for match in matches if match.get("section_number") == "0"]
        if header_matches:
            best = header_matches[0]
            contact_lines = [
                line.strip()
                for line in str(best["snippet"]).splitlines()
                if "Representative:" in line
            ]
            contact_text = " ".join(contact_lines) if contact_lines else str(best["snippet"])[:500]
            citation = best["citation"]
            return ComposedAnswer(
                text=f"The redacted contact information is: {contact_text} {citation}.",
                support_terms_by_citation={citation: ["Contact Information", "Representative"]},
            )
        return ComposedAnswer("I could not find sectioned contact information with enough confidence.")

    first = _first_successful_result(tool_results)
    if first is None:
        return ComposedAnswer("I could not complete the semantic search with the available tools.")
    matches = first["result"].get("matches", [])
    if not matches:
        return ComposedAnswer("I could not find relevant authorized contract sections.")
    best = matches[0]
    if float(best.get("score") or 0) < MIN_SEMANTIC_MATCH_SCORE:
        # Nearest-neighbor retrieval always returns something. The score guard
        # prevents low-relevance matches from becoming confident-looking answers.
        return ComposedAnswer(
            "I could not find a relevant authorized contract section for that question. "
            "Please ask about a contract ID, clause type, metadata field, renewal, risk, or guardrail scenario."
        )
    return _compose_semantic_rag_answer(query, matches)


def _compose_semantic_rag_answer(query: str, matches: list[dict[str, Any]]) -> ComposedAnswer:
    """Prepare a grounded RAG answer and LLM prompt from authorized search hits."""

    context_matches = matches[:4]
    context_blocks = []
    support: dict[str, list[str]] = {}
    for index, match in enumerate(context_matches, start=1):
        citation = str(match["citation"])
        support[citation] = [str(match["section_title"]).split()[0]]
        context_blocks.append(
            "\n".join(
                [
                    f"Context {index}: {citation}",
                    f"Contract: {match['contract_id']} - {match['title']}",
                    f"Section: {match['section_title']}",
                    f"Text: {match['snippet']}",
                ]
            )
        )

    best = context_matches[0]
    fallback_answer = _concise_semantic_fallback(query, best)
    context_text = "\n\n".join(context_blocks)
    return ComposedAnswer(
        text=fallback_answer,
        support_terms_by_citation=support,
        llm_system_prompt=RAG_SYSTEM_PROMPT,
        llm_user_prompt=build_rag_user_prompt(query, context_text),
        llm_mode="rag_generation",
    )


def _concise_semantic_fallback(query: str, match: dict[str, Any]) -> str:
    """Summarize the top retrieved section without dumping the raw chunk.

    Retrieval snippets are intentionally long enough to support citations, but
    chat answers should be shaped around the user's question. This fallback is
    used whenever the local LLM is disabled or its answer fails validation, so it
    must be concise and citation-safe on its own.
    """

    contract_id = str(match["contract_id"])
    section_title = str(match["section_title"])
    citation = str(match["citation"])
    snippet = str(match["snippet"])

    clause_units = _numbered_clause_units(snippet)
    selected_units = _select_relevant_units(query, section_title, clause_units)
    summary = " ".join(selected_units) if selected_units else _first_relevant_sentence(snippet)
    summary = _compact_text(summary)
    return f"{contract_id} {section_title}: {summary} {citation}."


def _numbered_clause_units(text: str) -> list[str]:
    """Return subsection-sized units such as `4.2 Obligations: ...`."""

    units: list[str] = []
    current: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if re.match(r"^\d+\.\d+\s+", line):
            if current:
                units.append(_compact_text(" ".join(current)))
            current = [line]
            continue
        if current and not re.match(r"^\d+\.\s+[A-Z]", line):
            current.append(line)
    if current:
        units.append(_compact_text(" ".join(current)))
    return units


def _select_relevant_units(query: str, section_title: str, units: list[str]) -> list[str]:
    """Pick the most relevant subsection summaries for a semantic answer."""

    if not units:
        return []
    query_terms = _semantic_terms(f"{query} {section_title}")
    scored: list[tuple[int, int, str]] = []
    for index, unit in enumerate(units):
        unit_terms = _semantic_terms(unit)
        overlap = len(query_terms.intersection(unit_terms))
        if "obligation" in query_terms and "obligation" in unit_terms:
            overlap += 2
        if "duration" in unit_terms or "survive" in unit_terms:
            overlap += 1
        scored.append((overlap, -index, unit))

    best = [unit for score, _index, unit in sorted(scored, reverse=True) if score > 0][:2]
    if not best:
        best = [units[0]]
    return [_strip_clause_prefix(unit) for unit in best]


def _first_relevant_sentence(text: str) -> str:
    sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", _compact_text(text))]
    return next((sentence for sentence in sentences if sentence), _compact_text(text)[:500])


def _semantic_terms(text: str) -> set[str]:
    stop_words = {
        "a",
        "an",
        "and",
        "for",
        "from",
        "in",
        "is",
        "of",
        "or",
        "search",
        "section",
        "the",
        "to",
        "with",
    }
    return {
        token.rstrip("s")
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]+", text.lower())
        if token not in stop_words and len(token) > 2
    }


def _strip_clause_prefix(text: str) -> str:
    return re.sub(r"^\d+\.\d+\s+[^:]+:\s*", "", text).strip()


def _compact_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _compose_renewal_plan(tool_results: list[dict[str, Any]]) -> ComposedAnswer:
    """Compose the Q2 renewal action plan from expiration, clause, and risk tools."""

    expiring = next(
        (
            result["result"]["contracts"]
            for result in tool_results
            if result["tool"] == "find_expiring_contracts" and result.get("status") == "success"
        ),
        [],
    )
    if not expiring:
        return ComposedAnswer("I found no authorized contracts expiring in Q2 2025.")

    lines = ["RENEWAL ACTION PLAN", "", "HIGH PRIORITY", ""]
    support: dict[str, list[str]] = {}
    for index, contract in enumerate(expiring, start=1):
        risk_result = _find_result(tool_results, f"risk_{contract['contract_id']}")
        risk = (
            risk_result["result"]
            if risk_result and risk_result.get("status") == "success"
            else {"overall_risk": "medium", "risk_factors": []}
        )
        citations = contract["citations"]
        support[citations["expiration"]] = [_display_date(contract["expires"])]
        support[citations["value"]] = [str(contract["value"]).split()[0]]
        support[citations["notice"]] = [f"{contract['notice_period_days']} days"]
        support[citations["renewal"]] = ["automatically renew"] if contract["auto_renewal"] else ["No automatic renewal"]
        lines.extend(
            [
                f"{index}. {contract['contract_id']} - {contract['title']}",
                f"  - Expires: {_display_date(contract['expires'])} {citations['expiration']}",
                f"  - Value: {contract['value']} {citations['value']}",
                f"  - Termination notice: {contract['notice_period_days']} days {citations['notice']}",
                (
                    "  - Auto-renewal: Yes, unless notice is provided before term end "
                    f"{citations['renewal']}"
                    if contract["auto_renewal"]
                    else f"  - Auto-renewal: No {citations['renewal']}"
                ),
                f"  - Action required by: {_display_date(contract['action_required_by'])}",
                f"  - Risk: {risk['overall_risk']}",
                "  - Recommendation: Review renewal decision before the action date and negotiate liability/SLA improvements if renewing.",
                "",
            ]
        )
    return ComposedAnswer("\n".join(lines).strip(), support_terms_by_citation=support)


def _find_result(tool_results: list[dict[str, Any]], step_id: str) -> dict[str, Any] | None:
    return next((result for result in tool_results if result["step_id"] == step_id), None)


def _first_successful_result(tool_results: list[dict[str, Any]]) -> dict[str, Any] | None:
    return next((result for result in tool_results if result.get("status") == "success"), None)


def _risk_support_terms(factor: dict[str, str]) -> list[str]:
    text = factor.get("text", "").lower()
    if "auto-renewal" in text:
        return ["automatically renew"]
    if "notice" in text:
        return ["Termination Notice Period"]
    if "liability" in text:
        return ["Liability"]
    if "force majeure" in text:
        return ["Force Majeure"]
    return [factor.get("text", "").split()[0]]


def _display_date(value: str | None) -> str:
    if not value:
        return "Not found"
    year, month, day = value.split("-")
    names = {
        "01": "January",
        "02": "February",
        "03": "March",
        "04": "April",
        "05": "May",
        "06": "June",
        "07": "July",
        "08": "August",
        "09": "September",
        "10": "October",
        "11": "November",
        "12": "December",
    }
    return f"{names[month]} {int(day)}, {year}"


def _serialize_plan(plan: AgentPlan, executed_results: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    serialized = [
        {
            "step_id": step.step_id,
            "tool": step.tool,
            "args": step.args,
            "depends_on": step.depends_on,
        }
        for step in plan.steps
    ]
    for result in executed_results or []:
        if not any(step["step_id"] == result["step_id"] for step in serialized):
            serialized.append(
                {
                    "step_id": result["step_id"],
                    "tool": result["tool"],
                    "args": result["args"],
                    "depends_on": ["s1"],
                }
            )
    return serialized


def _record_query(store: DataStore, user: User, query_id: str, redacted_query: str, result: dict[str, Any]) -> None:
    """Persist the query trace and audit event with PII-redacted query text."""

    store.add_query_record(
        QueryRecord(
            id=query_id,
            organization_id=user.organization_id,
            user_id=user.id,
            query=redacted_query,
            status=str(result.get("status", "complete")),
            plan=result.get("plan", []),
            result=result,
        )
    )
    log_event(
        store,
        user,
        action="ai.query",
        resource_type="query",
        resource_id=query_id,
        outcome=str(result.get("status", "complete")),
        details={"query": redacted_query, "intent": result.get("intent")},
    )


def _generate_or_polish_with_llm(composed: ComposedAnswer) -> tuple[str, LLMTrace]:
    """Use a configured local LLM for grounded RAG generation or polishing.

    Deterministic mode does not call a model. External-model failures and unsafe
    model outputs fall back to the deterministic composed answer so the demo
    remains reliable and citation verification still has evidence to inspect.
    """

    client = get_llm_client()
    if client.__class__.__name__ == "DeterministicLLMClient":
        logger.debug(
            "llm.skipped.deterministic_provider",
            extra=safe_extra(provider=client.provider_name, mode=composed.llm_mode),
        )
        return composed.text, LLMTrace(
            provider=client.provider_name,
            model=client.model_name,
            mode=composed.llm_mode,
            used=False,
            fallback_reason="deterministic_provider",
        )

    system_prompt = composed.llm_system_prompt or ANSWER_POLISHING_SYSTEM_PROMPT
    user_prompt = composed.llm_user_prompt or build_polishing_user_prompt(composed.text)
    prompt_tokens = estimate_tokens(system_prompt) + estimate_tokens(user_prompt)
    try:
        logger.info(
            "llm.invoke.start",
            extra=safe_extra(
                provider=client.provider_name,
                model=client.model_name,
                mode=composed.llm_mode,
                system_prompt_name=(
                    "RAG_SYSTEM_PROMPT"
                    if composed.llm_system_prompt == RAG_SYSTEM_PROMPT
                    else "ANSWER_POLISHING_SYSTEM_PROMPT"
                ),
                prompt_tokens_estimate=prompt_tokens,
            ),
        )
        generated = client.invoke(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        ).strip()
    except Exception as exc:
        logger.exception(
            "llm.call.failed",
            extra=safe_extra(
                provider=client.provider_name,
                model=client.model_name,
                mode=composed.llm_mode,
            ),
        )
        return composed.text, LLMTrace(
            provider=client.provider_name,
            model=client.model_name,
            mode=composed.llm_mode,
            used=False,
            prompt_tokens_estimate=prompt_tokens,
            fallback_reason=f"llm_error:{exc.__class__.__name__}",
        )

    if not generated:
        logger.warning(
            "llm.response.empty",
            extra=safe_extra(
                provider=client.provider_name,
                model=client.model_name,
                mode=composed.llm_mode,
            ),
        )
        return composed.text, LLMTrace(
            provider=client.provider_name,
            model=client.model_name,
            mode=composed.llm_mode,
            used=False,
            prompt_tokens_estimate=prompt_tokens,
            fallback_reason="empty_llm_response",
        )

    # The model must preserve citations. If it does not, the deterministic tool
    # answer is safer than returning unsupported prose.
    if parse_citations(composed.text) and not parse_citations(generated):
        logger.warning(
            "llm.response.rejected_missing_citations",
            extra=safe_extra(
                provider=client.provider_name,
                model=client.model_name,
                mode=composed.llm_mode,
            ),
        )
        return composed.text, LLMTrace(
            provider=client.provider_name,
            model=client.model_name,
            mode=composed.llm_mode,
            used=False,
            prompt_tokens_estimate=prompt_tokens,
            completion_tokens_estimate=estimate_tokens(generated),
            total_tokens_estimate=prompt_tokens + estimate_tokens(generated),
            fallback_reason="llm_response_missing_citations",
        )

    completion_tokens = estimate_tokens(generated)
    return generated, LLMTrace(
        provider=client.provider_name,
        model=client.model_name,
        mode=composed.llm_mode,
        used=True,
        prompt_tokens_estimate=prompt_tokens,
        completion_tokens_estimate=completion_tokens,
        total_tokens_estimate=prompt_tokens + completion_tokens,
    )


def _not_used_llm_trace(mode: str) -> LLMTrace:
    client = get_llm_client()
    return LLMTrace(
        provider=client.provider_name,
        model=client.model_name,
        mode=mode,
        used=False,
        fallback_reason="not_applicable",
    )
