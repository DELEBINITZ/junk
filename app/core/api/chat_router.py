"""Chat, streaming, sessions, cross-session recall, route preview, capabilities."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import StreamingResponse

from app.core.api.deps import get_services, require_user
from app.core.api.schemas import (
    CapabilitiesResponse,
    CapabilityInfo,
    ChatRequest,
    ChatResponse,
    CreateSessionRequest,
    MessageInfo,
    RoutePreviewRequest,
    RoutePreviewResponse,
    SessionDetail,
    SessionInfo,
    ToolInfo,
    UpdateSessionRequest,
)
from app.core.errors import NotFound
from app.core.security.context import SecurityContext
from app.core.streaming import sse_from_events

router = APIRouter(prefix="/v1", tags=["chat"])


def _session_info(s) -> SessionInfo:
    return SessionInfo(id=s.id, title=s.title, summary=s.summary, message_count=s.message_count,
                       created_at=s.created_at, updated_at=s.updated_at)


def _message_info(m) -> MessageInfo:
    return MessageInfo(id=m.id, role=m.role, content=m.content, citations=m.citations, created_at=m.created_at)


# ---- chat ----
@router.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest, request: Request, sc: SecurityContext = Depends(require_user)) -> ChatResponse:
    services = get_services(request)
    with services.metrics.timer("chat_turn_ms"):
        r = await services.orchestrator.run_turn(sc, question=body.message, session_id=body.session_id)
    services.metrics.inc("chat_turns_total")
    return ChatResponse(
        answer=r.answer, citations=r.citations, session_id=r.session_id, message_id=r.message_id,
        route=r.route_modules, flags=r.output_flags, trace_id=r.trace_id,
    )


@router.get("/chat/stream")
async def chat_stream(
    request: Request,
    message: str = Query(min_length=1),
    session_id: str | None = None,
    sc: SecurityContext = Depends(require_user),
) -> StreamingResponse:
    """SSE token stream (EventSource-compatible: token via ?access_token=)."""
    services = get_services(request)
    services.metrics.inc("chat_streams_total")
    gen = services.orchestrator.stream_turn(sc, question=message, session_id=session_id)
    return StreamingResponse(
        sse_from_events(gen), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


# ---- sessions ----
@router.get("/sessions", response_model=list[SessionInfo])
async def list_sessions(request: Request, sc: SecurityContext = Depends(require_user),
                        limit: int = 50, offset: int = 0) -> list[SessionInfo]:
    services = get_services(request)
    rows = await services.conversations.list_sessions(sc.org_id, sc.user_id, limit=limit, offset=offset)
    return [_session_info(s) for s in rows]


@router.post("/sessions", response_model=SessionInfo)
async def create_session(body: CreateSessionRequest, request: Request,
                         sc: SecurityContext = Depends(require_user)) -> SessionInfo:
    services = get_services(request)
    s = await services.conversations.create_session(sc.org_id, sc.user_id, body.title)
    return _session_info(s)


@router.get("/sessions/search", response_model=list[MessageInfo])
async def search_sessions(request: Request, q: str = Query(min_length=1),
                          sc: SecurityContext = Depends(require_user), limit: int = 20) -> list[MessageInfo]:
    """Cross-session recall over the user's past chats."""
    services = get_services(request)
    msgs = await services.conversations.search_messages(sc.org_id, sc.user_id, q, limit=limit)
    return [_message_info(m) for m in msgs]


@router.get("/sessions/{session_id}", response_model=SessionDetail)
async def get_session(session_id: str, request: Request,
                      sc: SecurityContext = Depends(require_user)) -> SessionDetail:
    services = get_services(request)
    s = await services.conversations.get_session(sc.org_id, session_id)
    if not s:
        raise NotFound("session not found")
    msgs = await services.conversations.get_messages(sc.org_id, session_id)
    return SessionDetail(**_session_info(s).model_dump(), messages=[_message_info(m) for m in msgs])


@router.patch("/sessions/{session_id}", response_model=SessionInfo)
async def update_session(session_id: str, body: UpdateSessionRequest, request: Request,
                         sc: SecurityContext = Depends(require_user)) -> SessionInfo:
    services = get_services(request)
    s = await services.conversations.update_session(sc.org_id, session_id, title=body.title)
    if not s:
        raise NotFound("session not found")
    return _session_info(s)


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str, request: Request,
                         sc: SecurityContext = Depends(require_user)) -> dict:
    services = get_services(request)
    ok = await services.conversations.delete_session(sc.org_id, session_id)
    if not ok:
        raise NotFound("session not found")
    return {"status": "deleted", "session_id": session_id}


# ---- routing / capabilities ----
@router.post("/route/preview", response_model=RoutePreviewResponse)
async def route_preview(body: RoutePreviewRequest, request: Request,
                        sc: SecurityContext = Depends(require_user)) -> RoutePreviewResponse:
    services = get_services(request)
    rr = await services.orchestrator.preview_route(sc, body.message)
    return RoutePreviewResponse(modules=rr.modules, scores=rr.scores, mode=rr.mode, fallback=rr.fallback)


@router.get("/capabilities", response_model=CapabilitiesResponse)
async def capabilities(request: Request, sc: SecurityContext = Depends(require_user)) -> CapabilitiesResponse:
    services = get_services(request)
    view = services.registry.capability_view(sc)
    visible = set(view.module_ids)
    callable_by_module: dict[str, list[str]] = {}
    for t in view.tools:
        callable_by_module.setdefault(t.module_id, []).append(t.name)
    modules = []
    for m in services.registry.modules():
        if m.id not in visible:
            continue
        modules.append(CapabilityInfo(
            id=m.id, display_name=m.manifest.display_name, description=m.manifest.description,
            enabled=m.enabled, autonomy=m.manifest.default_autonomy.value,
            tools=callable_by_module.get(m.id, []),
        ))
    tools = [ToolInfo(name=t.name, description=t.description, side_effecting=t.side_effecting,
                      rbac_role=t.rbac_role, module=t.module_id) for t in view.tools]
    return CapabilitiesResponse(modules=modules, tools=tools)


__all__ = ["router"]
