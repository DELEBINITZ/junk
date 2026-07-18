"""API routes — all LangGraph-powered. No direct LLM calls here."""

import time
from uuid import uuid4

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse, PlainTextResponse
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from security_intel.api.auth import require_user, optional_user
from security_intel.api.schemas import (
    ChatRequest, ChatResponse, SessionInfo, SessionListResponse,
)
from security_intel.api.streaming import stream_agent_events
from security_intel.security.rbac import SecurityContext
from security_intel.observability.logging import get_logger, new_trace_id, set_trace_context
from security_intel.observability.tracing import traced_config
from security_intel.observability.metrics import (
    REGISTRY, RequestMetrics, record_request, attribute_cost,
)
from security_intel.observability.eval_scoring import score_answer

logger = get_logger("api")
router = APIRouter(prefix="/v1")


@router.get("/metrics")
async def metrics_endpoint():
    """Prometheus scrape endpoint — latency/token/cost histograms, per-agent and
    per-tenant counters, routing-action and answer-flag counts."""
    return PlainTextResponse(
        REGISTRY.render_prometheus(), media_type="text/plain; version=0.0.4; charset=utf-8"
    )


def _routing_action(result: dict, agents_used: list[str]) -> str:
    """Coarse routing label for the metrics counter."""
    if agents_used:
        return "COMPLEX" if result.get("is_complex") else "SIMPLE"
    if result.get("is_chitchat") or result.get("direct_response"):
        return "DIRECT"
    if result.get("needs_clarification"):
        return "CLARIFY"
    return "OTHER"


def _emit_chat_metrics(result, agents_used, org_id, latency_ms, usage_cb):
    """Emit SLO metrics + sample answer quality online. Never raises (best-effort)."""
    final = result.get("final_answer", "") or ""
    findings = [{"text": r.get("findings", "")} for r in result.get("agent_results", [])]
    # Online answer-quality sampling: only when there were findings to ground against
    # (a DIRECT greeting has no sources, so grounding it is meaningless).
    flags = score_answer(final, findings).flags if (agents_used and final) else []
    usage = getattr(usage_cb, "usage_metadata", {}) or {}
    m = RequestMetrics(
        tenant=org_id or "unknown",
        latency_ms=latency_ms,
        agents_used=agents_used,
        routing_action=_routing_action(result, agents_used),
        cost=attribute_cost(usage),
        answer_flags=flags,
        outcome="ok" if final else "empty",
    )
    record_request(m)


@router.get("/health")
async def health(request: Request):
    """Health check — verifies critical service connectivity."""
    checks = {"status": "ok", "version": "2.0.0", "services": {}}

    # Check database
    db = getattr(request.app.state, "db", None)
    if db:
        try:
            async with db.connection() as conn:
                await conn.execute("SELECT 1")
            checks["services"]["postgres"] = "ok"
        except Exception as e:
            checks["services"]["postgres"] = f"error: {e}"
            checks["status"] = "degraded"
    else:
        checks["services"]["postgres"] = "disabled"

    # Check agents
    registry = getattr(request.app.state, "registry", None)
    if registry:
        checks["services"]["agents"] = registry.agent_ids
    else:
        checks["services"]["agents"] = []
        checks["status"] = "degraded"

    return checks


@router.get("/meta")
async def meta(request: Request):
    """Derived assistant identity — so clients render the persona that matches the
    ENABLED agents instead of hardcoding one. Powers dynamic title/blurb/suggestions.
    """
    profile = getattr(request.app.state, "profile", None)
    registry = getattr(request.app.state, "registry", None)
    if not profile:
        return {"name": "Assistant", "tagline": "", "domains": "", "agents": []}

    # User-facing capability areas the master advertises. Specialist names (Atlas,
    # Sentinel, Aura, …) are INTERNAL and deliberately not surfaced here — the master is one
    # voice. `id` is kept for clients that need a stable key.
    capabilities = []
    if registry:
        for aid in registry.agent_ids:
            spec = registry.get_spec(aid)
            if spec:
                capabilities.append({
                    "id": aid,
                    "name": spec.domain_label or spec.display_name,
                    "description": " ".join((spec.description or "").split()),
                    "capabilities": spec.capabilities,
                })

    return {
        "name": profile.name,
        "tagline": profile.tagline,
        "domains": profile.domains,
        "scope": profile.scope,
        "capabilities": capabilities,
    }


@router.post("/chat", response_model=ChatResponse)
async def chat(
    body: ChatRequest,
    request: Request,
    sc: SecurityContext = Depends(optional_user),
):
    """Non-streaming chat. Runs full LangGraph orchestrator pipeline."""
    orchestrator = request.app.state.orchestrator
    session_id = body.session_id or f"sess_{uuid4().hex[:12]}"
    trace_id = new_trace_id()
    set_trace_context(trace_id=trace_id, org_id=sc.org_id)

    config = RunnableConfig(
        configurable={
            "thread_id": session_id,
            "org_id": sc.org_id,
            "user_id": sc.user_id,
            "roles": list(sc.roles),
        }
    )

    # Add Langfuse tracing if configured
    langfuse = getattr(request.app.state, "langfuse_handler", None)
    if langfuse:
        config = traced_config(
            config, langfuse,
            trace_name="chat",
            user_id=sc.user_id,
            session_id=session_id,
            metadata={"org_id": sc.org_id, "trace_id": trace_id},
        )

    input_state = {
        "messages": [HumanMessage(content=body.message)],
        "history": [],
        "summary": "",
        "user_query": body.message,
        "org_id": sc.org_id,
        "user_id": sc.user_id,
        "roles": list(sc.roles),
        "session_id": session_id,
        "is_complex": False,
        "is_chitchat": False,
        "needs_clarification": False,
        "direct_response": "",
        "plan": None,
        "agent_results": [],
        "final_answer": "",
        "citations": [],
        "blocked": False,
        "block_reason": "",
    }

    logger.info(f"Chat request: session={session_id}", extra={"extra_data": {
        "query_len": len(body.message), "session_id": session_id,
    }})

    # Attach a per-request token-usage accumulator so cost can be attributed per
    # model without threading usage through every node.
    usage_cb = UsageMetadataCallbackHandler()
    _cbs = config.get("callbacks") or []
    if not isinstance(_cbs, list):
        _cbs = [_cbs]
    config["callbacks"] = [*_cbs, usage_cb]

    _t0 = time.perf_counter()
    result = await orchestrator.ainvoke(input_state, config=config)
    _latency_ms = (time.perf_counter() - _t0) * 1000

    agents_used = [r["agent_id"] for r in result.get("agent_results", [])]

    try:
        _emit_chat_metrics(result, agents_used, sc.org_id, _latency_ms, usage_cb)
    except Exception as e:  # noqa: BLE001 — metrics must never break the response
        logger.warning(f"chat metrics emit failed (non-fatal): {e}")

    return ChatResponse(
        answer=result.get("final_answer", ""),
        session_id=session_id,
        trace_id=trace_id,
        citations=[],
        agents_used=agents_used,
        blocked=result.get("blocked", False),
        block_reason=result.get("block_reason", ""),
        is_complex=result.get("is_complex", False),
    )


@router.get("/chat/stream")
async def chat_stream(
    message: str,
    request: Request,
    session_id: str | None = None,
    sc: SecurityContext = Depends(optional_user),
):
    """SSE streaming endpoint. Streams events from LangGraph's astream_events."""
    orchestrator = request.app.state.orchestrator
    sid = session_id or f"sess_{uuid4().hex[:12]}"

    config = RunnableConfig(
        configurable={
            "thread_id": sid,
            "org_id": sc.org_id,
            "user_id": sc.user_id,
            "roles": list(sc.roles),
        }
    )

    input_state = {
        "messages": [HumanMessage(content=message)],
        "history": [],
        "summary": "",
        "user_query": message,
        "org_id": sc.org_id,
        "user_id": sc.user_id,
        "roles": list(sc.roles),
        "session_id": sid,
        "is_complex": False,
        "is_chitchat": False,
        "needs_clarification": False,
        "direct_response": "",
        "plan": None,
        "agent_results": [],
        "final_answer": "",
        "citations": [],
        "blocked": False,
        "block_reason": "",
    }

    return StreamingResponse(
        stream_agent_events(orchestrator, input_state, config),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Session-ID": sid},
    )


# ---------------------------------------------------------------------------
# Session management (ChatGPT-like sidebar)
# ---------------------------------------------------------------------------

@router.get("/sessions", response_model=SessionListResponse)
async def list_sessions(
    request: Request,
    limit: int = 50,
    offset: int = 0,
    sc: SecurityContext = Depends(optional_user),
):
    """List user's chat sessions (most recent first)."""
    conversations = getattr(request.app.state, "conversations", None)
    if not conversations:
        return SessionListResponse(sessions=[])

    sessions = await conversations.list_sessions(sc.org_id, sc.user_id, limit, offset)
    return SessionListResponse(
        sessions=[
            SessionInfo(
                session_id=s.id,
                title=s.title,
                message_count=s.message_count,
                created_at=s.created_at.isoformat() if s.created_at else None,
                updated_at=s.updated_at.isoformat() if s.updated_at else None,
            )
            for s in sessions
        ]
    )


@router.get("/sessions/{session_id}/messages")
async def get_session_messages(
    session_id: str,
    request: Request,
    limit: int = 50,
    sc: SecurityContext = Depends(optional_user),
):
    """Get messages for a specific session."""
    conversations = getattr(request.app.state, "conversations", None)
    if not conversations:
        return {"messages": []}

    messages = await conversations.get_messages(sc.org_id, session_id, limit=limit)
    return {
        "messages": [
            {
                "id": m.id,
                "role": m.role,
                "content": m.content,
                "citations": m.citations,
                "created_at": m.created_at.isoformat() if m.created_at else None,
            }
            for m in messages
        ]
    }


@router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str,
    request: Request,
    sc: SecurityContext = Depends(optional_user),
):
    """Delete a chat session and all its messages."""
    conversations = getattr(request.app.state, "conversations", None)
    if not conversations:
        return {"ok": True}

    await conversations.delete_session(sc.org_id, session_id)
    return {"ok": True}
