"""API routes — all LangGraph-powered. No direct LLM calls here."""

from uuid import uuid4

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
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

logger = get_logger("api")
router = APIRouter(prefix="/v1")


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
        "user_query": body.message,
        "org_id": sc.org_id,
        "user_id": sc.user_id,
        "roles": list(sc.roles),
        "session_id": session_id,
        "is_complex": False,
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

    result = await orchestrator.ainvoke(input_state, config=config)

    agents_used = [r["agent_id"] for r in result.get("agent_results", [])]

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
        "user_query": message,
        "org_id": sc.org_id,
        "user_id": sc.user_id,
        "roles": list(sc.roles),
        "session_id": sid,
        "is_complex": False,
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
