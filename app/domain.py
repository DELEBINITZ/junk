"""Domain objects shared across API, MCP, RAG, and agent layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4


Role = Literal["admin", "analyst", "viewer"]
AccessLevel = Literal["read", "query", "edit"]


def new_id() -> str:
    return str(uuid4())


def now_utc() -> datetime:
    return datetime.now(UTC)


@dataclass(slots=True)
class Organization:
    id: str
    name: str


@dataclass(slots=True)
class User:
    id: str
    organization_id: str
    email: str
    name: str
    role: Role
    password_hash: str
    is_active: bool = True


@dataclass(slots=True)
class Document:
    """Contract document plus tenant ownership and extracted metadata."""

    id: str
    organization_id: str
    contract_id: str
    title: str
    filename: str
    uploaded_by: str
    tags: list[str] = field(default_factory=list)
    raw_text: str = ""
    redacted_text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    deleted_at: datetime | None = None
    created_at: datetime = field(default_factory=now_utc)


@dataclass(slots=True)
class DocumentShare:
    document_id: str
    user_id: str
    access_level: AccessLevel


@dataclass(slots=True)
class Section:
    id: str
    document_id: str
    organization_id: str
    section_number: str
    section_title: str
    text: str
    line_start: int
    line_end: int
    page_number: int | None = None


@dataclass(slots=True)
class Chunk:
    """Vector-searchable text unit with enough metadata to cite a section."""

    id: str
    document_id: str
    organization_id: str
    section_id: str
    chunk_index: int
    text: str
    embedding: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GuardrailConfig:
    organization_id: str
    hallucination_confidence_threshold: float = 0.7
    pii_redaction_enabled: bool = True
    blocked_keywords: list[str] = field(default_factory=list)
    require_citations: bool = True
    toxicity_threshold: float = 0.8


@dataclass(slots=True)
class AuditEvent:
    id: str
    organization_id: str | None
    user_id: str | None
    action: str
    resource_type: str
    resource_id: str | None
    outcome: str
    details: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=now_utc)


@dataclass(slots=True)
class Report:
    id: str
    organization_id: str
    created_by: str
    title: str
    query: str
    result: dict[str, Any]
    created_at: datetime = field(default_factory=now_utc)


@dataclass(slots=True)
class QueryRecord:
    """Stored trace of an AI query for audit and replay/debugging."""

    id: str
    organization_id: str
    user_id: str
    query: str
    status: str
    plan: list[dict[str, Any]]
    result: dict[str, Any]
    created_at: datetime = field(default_factory=now_utc)
