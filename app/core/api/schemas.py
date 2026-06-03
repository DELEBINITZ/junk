"""API request/response models.

These are the PYDANTIC SCHEMAS that define the HTTP contract of the service: the
exact JSON shape of every request body and response. FastAPI uses them to (1)
parse + VALIDATE incoming JSON (a bad body becomes an automatic 422, before any
handler runs), (2) SERIALIZE handler return values to JSON, and (3) generate the
OpenAPI docs. Keeping them in one file makes the whole API surface readable at a
glance. They are grouped by feature with ``# ---- section ----`` banners that
mirror the routers (auth / chat / sessions / routing / approvals).

A note on tenancy: notice none of these request models carry ``org_id``. That is
deliberate — the tenant is taken from the verified token (SecurityContext) in the
routers, never accepted from the client body, so a caller cannot ask for another
org's data by editing JSON.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ---- auth ----
class UserInfo(BaseModel):
    # The non-secret identity echoed by /v1/auth/me, derived from the verified JWT.
    id: str
    email: str
    org_id: str
    roles: list[str]


# ---- chat ----
class ChatRequest(BaseModel):
    # One user turn. ``min_length=1`` rejects empty messages at the edge (422).
    # ``session_id`` is optional: omit it to start a brand-new conversation.
    # Note: NO identity fields — the tenant/user/roles always come from the verified
    # JWT (SecurityContext), never from the request body.
    message: str = Field(min_length=1)
    session_id: str | None = None


class ChatResponse(BaseModel):
    # The full result of a non-streaming turn: the answer plus everything the UI
    # needs to render it — citations (sources), which session/message it became,
    # which modules ``route``d, guardrail ``flags``, and a ``trace_id`` for
    # correlating logs/observability with this exact turn.
    answer: str
    citations: list[dict[str, Any]] = []
    session_id: str
    message_id: str
    route: list[str] = []
    flags: dict[str, Any] = {}
    trace_id: str


# ---- sessions ----
class CreateSessionRequest(BaseModel):
    title: str = "New chat"


class UpdateSessionRequest(BaseModel):
    title: str | None = None


class SessionInfo(BaseModel):
    # The session-list view (sidebar item): metadata only, no messages. This is
    # the API projection of the internal Session model — note org_id/user_id are
    # intentionally NOT exposed to clients.
    id: str
    title: str
    summary: str = ""
    message_count: int = 0
    created_at: str
    updated_at: str


class MessageInfo(BaseModel):
    # A single message as shown in the transcript. ``citations`` carry the sources
    # behind an assistant turn; internal fields (org_id, tool_calls, meta) are omitted.
    id: str
    role: str
    content: str
    citations: list[dict[str, Any]] = []
    created_at: str
    feedback: int = 0               # user rating: -1 down, 0 none, 1 up


class FeedbackRequest(BaseModel):
    # Rate an assistant message. -1 = thumbs down, 0 = clear, 1 = thumbs up.
    value: int = Field(ge=-1, le=1)


class SessionDetail(SessionInfo):
    # Opening a session = its metadata (inherited from SessionInfo) PLUS the full
    # message list. Inheriting keeps the two views guaranteed-consistent.
    messages: list[MessageInfo] = []


# ---- routing / capabilities ----
class RoutePreviewRequest(BaseModel):
    # Ask "where WOULD this question route?" without actually answering it — a
    # debugging/UX affordance backed by the supervisor.
    message: str


class RoutePreviewResponse(BaseModel):
    # The supervisor's decision, exposed: chosen ``modules``, the per-module
    # ``scores`` it ranked on, the routing ``mode``, and whether it had to
    # ``fallback`` (no module scored high enough).
    modules: list[str]
    scores: dict[str, float]
    mode: str
    fallback: bool


class ToolInfo(BaseModel):
    # One tool as advertised to the client. ``side_effecting`` + ``rbac_role`` let
    # the UI show which tools are gated/privileged (the safety metadata from the
    # Tool contract).
    name: str
    description: str
    side_effecting: bool
    rbac_role: str
    module: str


class CapabilityInfo(BaseModel):
    # One capability MODULE as seen by THIS caller — only the tools their role can
    # actually call are listed (the capability view is RBAC-filtered in the router).
    id: str
    display_name: str
    description: str
    enabled: bool
    autonomy: str
    tools: list[str]


class CapabilitiesResponse(BaseModel):
    # "What can I do here?" — the modules + tools visible to the caller. Drives a
    # capability-aware UI and makes the agent's surface introspectable.
    modules: list[CapabilityInfo]
    tools: list[ToolInfo]


# ---- approvals ----
class ApprovalInfo(BaseModel):
    # One pending side-effecting action awaiting human approval (the action-gate
    # inbox item). Shows what tool/action wants to run, with which arguments.
    id: str
    action_type: str
    tool: str
    module: str
    arguments: dict[str, Any]
    status: str
    created_at: str


# Export every public (capitalized) name defined above — i.e. all the schema
# classes — so ``from app.core.api.schemas import *`` pulls them in. Computed
# dynamically so adding a model here needs no edit to this list.
__all__ = [name for name in dir() if name[0].isupper()]
