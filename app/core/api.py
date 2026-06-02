"""Chassis API — the new agentic surface, mounted alongside the existing routes.

These endpoints exercise the registry + router + orchestrator + streaming +
chat persistence end-to-end through the real auth path. The legacy /ai and /mcp
routes are untouched; this is the path the production system grows on. See plan
§6, §9, §11.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel

from app.auth.dependencies import require_role, require_user
from app.core.agent.orchestrator import Orchestrator
from app.core.observability.metrics import metrics
from app.core.registry import get_registry
from app.core.router import Router
from app.core.streaming.sse import stream_turn
from app.db.repository import DataStore, get_store
from app.domain import User


router = APIRouter(tags=["capabilities"])


class RoutePreviewIn(BaseModel):
    query: str


class AgentQueryIn(BaseModel):
    message: str
    session_id: str | None = None


class IngestDocIn(BaseModel):
    contract_id: str
    title: str
    text: str
    tags: list[str] = []
    doc_type: str | None = None
    metadata: dict = {}


class IngestIn(BaseModel):
    documents: list[IngestDocIn]


@router.get("/capabilities")
def list_capabilities(user: User = Depends(require_user)):
    """Modules and tools this user's organization + role are entitled to."""

    registry = get_registry()
    return {
        "user": {"organization_id": user.organization_id, "role": user.role},
        "modules": [
            {
                "id": m.id,
                "display_name": m.display_name,
                "version": m.version,
                "autonomy": m.default_autonomy.value,
                "tools": [t.name for t in m.tools],
            }
            for m in registry.modules_for_user(user)
        ],
        "tools": registry.list_tool_definitions(user),
    }


@router.post("/route/preview")
def preview_route(body: RoutePreviewIn, user: User = Depends(require_user)):
    """Show which module(s)/tools the router would expose for a query — the
    bounded-context guarantee, without executing anything."""

    registry = get_registry()
    decision = Router(registry).route(body.query, user)
    return {
        "query": body.query,
        "module_ids": decision.module_ids,
        "tool_names": decision.tool_names,
        "lane": decision.lane,
        "scores": decision.scores,
    }


@router.post("/agent/query")
def agent_query(
    body: AgentQueryIn,
    user: User = Depends(require_user),
    store: DataStore = Depends(get_store),
):
    """Run one orchestrated turn (guardrails -> route -> tools -> grounded answer)
    and persist it to the chat session."""

    metrics.incr("agent.query.total")
    turn = Orchestrator(user, store).run_query(body.message, session_id=body.session_id)
    metrics.incr(f"agent.query.{turn.status}")
    return turn.to_dict()


@router.post("/agent/stream")
def agent_stream(
    body: AgentQueryIn,
    user: User = Depends(require_user),
    store: DataStore = Depends(get_store),
):
    """Stream one orchestrated turn as SSE typed events (status/tool_call/
    tool_result/token/citation/done)."""

    metrics.incr("agent.stream.total")
    orchestrator = Orchestrator(user, store)
    return StreamingResponse(
        stream_turn(orchestrator, body.message, session_id=body.session_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/sessions")
def list_sessions(user: User = Depends(require_user)):
    """List the calling user's own chat sessions (org + user scoped)."""

    from app.core.memory.conversations import get_conversation_store

    sessions = get_conversation_store().list_sessions(user)
    return {"sessions": [s.to_dict() for s in sessions]}


@router.get("/sessions/{session_id}/messages")
def session_messages(session_id: str, user: User = Depends(require_user)):
    """Render a session's messages for the UI. Returns 404 if not owned."""

    from app.core.memory.conversations import get_conversation_store

    store = get_conversation_store()
    if not any(s.id == session_id for s in store.list_sessions(user)):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    return {"messages": [m.to_dict() for m in store.get_messages(user, session_id)]}


@router.get("/metrics")
def get_metrics(fmt: str = "json", user: User = Depends(require_user)):
    """In-process counters. `?fmt=prometheus` returns the exposition format for a
    Prometheus scrape; default JSON."""

    if fmt == "prometheus":
        return PlainTextResponse(metrics.render_prometheus())
    return metrics.snapshot()


@router.post("/ingest/reports")
def ingest_reports(body: IngestIn, user: User = Depends(require_role("admin", "analyst"))):
    """Embed documents and upsert them into the org's vector index for RAG. The
    organization is taken from the caller's JWT (the mandatory tenant tag), never
    from the payload. Requires RETRIEVAL_BACKEND=qdrant + a reachable Qdrant/TEI."""

    from app.core.ingestion.indexer import QdrantIngestionService

    metrics.incr("ingest.reports")
    stats = QdrantIngestionService().index_documents(
        user.organization_id, [doc.model_dump() for doc in body.documents]
    )
    return {"documents": stats.documents, "chunks": stats.chunks, "organization_id": user.organization_id}
