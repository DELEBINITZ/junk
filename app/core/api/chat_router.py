"""Chat, streaming, sessions, cross-session recall, route preview, capabilities.

This is the main user-facing router — the HTTP front door to the agent. It maps
each endpoint to the orchestrator/stores and keeps the handlers THIN: validate +
authenticate at the edge, delegate the real work, shape the response. Two things
worth internalizing while reading:

  * EVERY endpoint depends on ``require_user``, which verifies the token and
    yields a SecurityContext (sc). The org/user on ``sc`` come from that verified
    token, and every store/orchestrator call is scoped by ``sc.org_id`` — so a
    caller only ever touches their own tenant's data.
  * There are two chat shapes: POST ``/chat`` returns the whole answer at once;
    GET ``/chat/stream`` streams tokens live over Server-Sent Events (SSE).
"""

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
    FeedbackRequest,
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
from app.core.security.deps import resolve_identity
from app.core.streaming import sse_from_events

router = APIRouter(prefix="/v1", tags=["chat"])


# _session_info / _message_info: project the internal Session/Message models down
# to their API schemas, dropping internal fields (org_id, user_id, tool_calls, ...)
# the client shouldn't see. One place to control exactly what leaves the service.
def _session_info(s) -> SessionInfo:
    return SessionInfo(id=s.id, title=s.title, summary=s.summary, message_count=s.message_count,
                       created_at=s.created_at, updated_at=s.updated_at)


def _message_info(m) -> MessageInfo:
    return MessageInfo(id=m.id, role=m.role, content=m.content, citations=m.citations,
                       created_at=m.created_at, feedback=getattr(m, "feedback", 0))


# ---- chat ----
@router.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest, request: Request) -> ChatResponse:
    """Run ONE non-streaming chat turn and return the complete answer.

    Identity is resolved via ``resolve_identity``: from the verified token
    (local/oidc), or — in apikey mode — from the API key + the body's org_id/
    user_id/roles. Then the whole turn goes to the orchestrator (the agent graph:
    guard -> triage -> route -> gather -> answer -> guard) scoped to that caller.
    The response carries everything the UI needs: answer, citations, the session +
    message it persisted to, the routed modules, guardrail flags, and a trace id."""
    sc = resolve_identity(request, org_id=body.org_id, user_id=body.user_id, roles=body.roles)
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
    """SSE token stream (EventSource-compatible: token via ?access_token=).

    Streams the answer token-by-token as Server-Sent Events so the user watches it
    appear live. Two design points worth knowing:
      * It's a GET (not POST) because the browser's EventSource API only does GET.
      * EventSource CANNOT send an Authorization header, so this path authenticates
        via an ``access_token`` query param (or cookie) — that's why require_user
        accepts the token there. The handler is otherwise like /chat but yields an
        event stream instead of one JSON body.
    The headers disable buffering/caching so tokens flush immediately end-to-end
    (e.g. ``X-Accel-Buffering: no`` stops nginx from holding the stream)."""
    services = get_services(request)
    services.metrics.inc("chat_streams_total")
    gen = services.orchestrator.stream_turn(sc, question=message, session_id=session_id)
    return StreamingResponse(
        sse_from_events(gen), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


# ---- sessions ----
# These endpoints are the chat-history sidebar's backend (list / create / open /
# rename / delete). Each one passes sc.org_id (and often sc.user_id) into the
# conversation store, which is how tenant + per-user scoping is enforced on the
# store side. A missing/foreign session resolves to None -> NotFound (404), so a
# caller can never confirm or touch another tenant's session by guessing its id.
@router.get("/sessions", response_model=list[SessionInfo])
async def list_sessions(request: Request, sc: SecurityContext = Depends(require_user),
                        limit: int = 50, offset: int = 0) -> list[SessionInfo]:
    """List the caller's own sessions, newest first, paginated by limit/offset."""
    services = get_services(request)
    rows = await services.conversations.list_sessions(sc.org_id, sc.user_id, limit=limit, offset=offset)
    return [_session_info(s) for s in rows]


@router.post("/sessions", response_model=SessionInfo)
async def create_session(body: CreateSessionRequest, request: Request,
                         sc: SecurityContext = Depends(require_user)) -> SessionInfo:
    """Start a new, empty conversation owned by the caller (org+user from sc)."""
    services = get_services(request)
    s = await services.conversations.create_session(sc.org_id, sc.user_id, body.title)
    return _session_info(s)


@router.get("/sessions/search", response_model=list[MessageInfo])
async def search_sessions(request: Request, q: str = Query(min_length=1),
                          sc: SecurityContext = Depends(require_user), limit: int = 20) -> list[MessageInfo]:
    """Cross-session recall over the user's past chats.

    Searches across ALL of this user's conversations for messages matching ``q``
    (full-text search on the Postgres backend, word-overlap in memory). Declared
    BEFORE ``/sessions/{session_id}`` on purpose: FastAPI matches routes in order,
    so the literal ``/sessions/search`` must win over the ``{session_id}`` wildcard."""
    services = get_services(request)
    msgs = await services.conversations.search_messages(sc.org_id, sc.user_id, q, limit=limit)
    return [_message_info(m) for m in msgs]


@router.get("/sessions/{session_id}", response_model=SessionDetail)
async def get_session(session_id: str, request: Request,
                      sc: SecurityContext = Depends(require_user)) -> SessionDetail:
    """Open one session: its metadata plus the full message transcript. The org
    guard lives in the store; here a non-owned/unknown id simply comes back empty
    -> 404."""
    services = get_services(request)
    s = await services.conversations.get_session(sc.org_id, session_id)
    if not s:
        raise NotFound("session not found")
    msgs = await services.conversations.get_messages(sc.org_id, session_id)
    return SessionDetail(**_session_info(s).model_dump(), messages=[_message_info(m) for m in msgs])


@router.patch("/sessions/{session_id}", response_model=SessionInfo)
async def update_session(session_id: str, body: UpdateSessionRequest, request: Request,
                         sc: SecurityContext = Depends(require_user)) -> SessionInfo:
    """Rename a session (PATCH = partial update). 404 if it isn't the caller's."""
    services = get_services(request)
    s = await services.conversations.update_session(sc.org_id, session_id, title=body.title)
    if not s:
        raise NotFound("session not found")
    return _session_info(s)


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str, request: Request,
                         sc: SecurityContext = Depends(require_user)) -> dict:
    """Delete a session (and its messages). 404 if it isn't the caller's."""
    services = get_services(request)
    ok = await services.conversations.delete_session(sc.org_id, session_id)
    if not ok:
        raise NotFound("session not found")
    return {"status": "deleted", "session_id": session_id}


# ---- message feedback (thumbs up/down) ----
@router.post("/messages/{message_id}/feedback")
async def message_feedback(message_id: str, body: FeedbackRequest, request: Request,
                           sc: SecurityContext = Depends(require_user)) -> dict:
    """Rate an assistant message (-1/0/1). Org-scoped: a caller can only rate its
    own org's messages (RLS / ownership check in the store)."""
    services = get_services(request)
    ok = await services.conversations.set_message_feedback(sc.org_id, message_id, body.value)
    if not ok:
        raise NotFound("message not found")
    return {"status": "ok", "message_id": message_id, "feedback": body.value}


# ---- routing / capabilities ----
@router.post("/route/preview", response_model=RoutePreviewResponse)
async def route_preview(body: RoutePreviewRequest, request: Request,
                        sc: SecurityContext = Depends(require_user)) -> RoutePreviewResponse:
    """Show where a question WOULD route without answering it. Runs only the
    supervisor's classification step and returns the chosen modules + their
    scores — a debugging/transparency window into the router."""
    services = get_services(request)
    rr = await services.orchestrator.preview_route(sc, body.message)
    return RoutePreviewResponse(modules=rr.modules, scores=rr.scores, mode=rr.mode, fallback=rr.fallback)


@router.get("/capabilities", response_model=CapabilitiesResponse)
async def capabilities(request: Request, sc: SecurityContext = Depends(require_user)) -> CapabilitiesResponse:
    """Report what THIS caller can do: the modules and tools visible to them.

    The capability VIEW is RBAC- and license-filtered for ``sc``, so two users in
    different roles get different answers (e.g. a viewer won't see analyst-only
    tools). We invert the view's tool list into tools-per-module, then emit one
    CapabilityInfo per visible module plus the flat ToolInfo list. This is what
    lets a client render a capability-aware UI and makes the agent's surface
    introspectable rather than hidden."""
    services = get_services(request)
    view = services.registry.capability_view(sc)
    visible = set(view.module_ids)
    # Group the caller's callable tools by their owning module for the per-module view.
    callable_by_module: dict[str, list[str]] = {}
    for t in view.tools:
        callable_by_module.setdefault(t.module_id, []).append(t.name)
    modules = []
    for m in services.registry.modules():
        if m.id not in visible:                  # skip modules this caller can't see
            continue
        modules.append(CapabilityInfo(
            id=m.id, display_name=m.manifest.display_name, description=m.manifest.description,
            enabled=m.enabled, autonomy=m.manifest.default_autonomy.value,
            tools=callable_by_module.get(m.id, []),
        ))
    # Flat list of every tool the caller may call, with its safety metadata.
    tools = [ToolInfo(name=t.name, description=t.description, side_effecting=t.side_effecting,
                      rbac_role=t.rbac_role, module=t.module_id) for t in view.tools]
    return CapabilitiesResponse(modules=modules, tools=tools)


__all__ = ["router"]
