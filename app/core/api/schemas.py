"""API request/response models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ---- auth ----
class LoginRequest(BaseModel):
    email: str
    password: str


class UserInfo(BaseModel):
    id: str
    email: str
    org_id: str
    roles: list[str]


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserInfo


class RefreshRequest(BaseModel):
    refresh_token: str


# ---- chat ----
class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    session_id: str | None = None


class ChatResponse(BaseModel):
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
    id: str
    title: str
    summary: str = ""
    message_count: int = 0
    created_at: str
    updated_at: str


class MessageInfo(BaseModel):
    id: str
    role: str
    content: str
    citations: list[dict[str, Any]] = []
    created_at: str


class SessionDetail(SessionInfo):
    messages: list[MessageInfo] = []


# ---- routing / capabilities ----
class RoutePreviewRequest(BaseModel):
    message: str


class RoutePreviewResponse(BaseModel):
    modules: list[str]
    scores: dict[str, float]
    mode: str
    fallback: bool


class ToolInfo(BaseModel):
    name: str
    description: str
    side_effecting: bool
    rbac_role: str
    module: str


class CapabilityInfo(BaseModel):
    id: str
    display_name: str
    description: str
    enabled: bool
    autonomy: str
    tools: list[str]


class CapabilitiesResponse(BaseModel):
    modules: list[CapabilityInfo]
    tools: list[ToolInfo]


# ---- ingestion ----
class IngestDocumentIn(BaseModel):
    doc_id: str
    title: str = ""
    text: str
    source: str = "reports"
    published_at: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestRequest(BaseModel):
    collection: str = "reports_kb"
    documents: list[IngestDocumentIn]
    chunk: bool = True


class IngestResponse(BaseModel):
    documents: int
    chunks: int


# ---- approvals ----
class ApprovalInfo(BaseModel):
    id: str
    action_type: str
    tool: str
    module: str
    arguments: dict[str, Any]
    status: str
    created_at: str


__all__ = [name for name in dir() if name[0].isupper()]
