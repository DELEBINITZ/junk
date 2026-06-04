"""Chat persistence: sessions + messages, with cross-session recall.

================================ MENTAL MODEL =============================
This is the "chat session / message store" — the same idea as ChatGPT/Claude
keeping your conversations in a sidebar. A SESSION is one conversation thread; a
MESSAGE is one turn (user/assistant/...) inside it. The store gives the app:
durable sessions per (org, user), full message history, auto-titles, a rolling
SUMMARY of older turns (so we can keep context bounded — see summarizer.py), and
"cross-session recall" = search across ALL of a user's past chats.

This file defines the INTERFACE (``ConversationStore``) plus the shared row models.
The backing implementation is Postgres (``pg_conversations.py``).

SECURITY — read this carefully: every method is **org-scoped**. The first
argument is always ``org_id``, and that value comes from the verified token
(SecurityContext), NEVER from the request body. The Postgres store enforces this
at the database via Row-Level Security (RLS). One tenant can never read another
tenant's sessions or messages.
===========================================================================
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any, Protocol

from pydantic import BaseModel, Field


# Timestamps are stored as ISO-8601 strings (UTC) so the model serializes to JSON
# directly and sorts lexicographically the same way it sorts chronologically.
def _now() -> str:
    return datetime.now(UTC).isoformat()


# Generate a readable, prefixed unique id, e.g. "sess_ab12..." / "msg_cd34...".
# The prefix makes ids self-describing in logs; the random hex avoids collisions.
def _uid(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:24]}"


class Message(BaseModel):
    """One turn in a conversation. ``org_id`` is stored on every message (not just
    the parent session) so the tenant tag travels with the row itself — the same
    "provenance + isolation on every atom" pattern used for RAG Chunks."""

    id: str
    session_id: str
    org_id: str
    role: str                       # user | assistant | system | tool
    content: str = ""
    created_at: str = Field(default_factory=_now)
    citations: list[dict[str, Any]] = Field(default_factory=list)  # sources backing an assistant turn
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)  # tools the agent invoked this turn
    meta: dict[str, Any] = Field(default_factory=dict)
    feedback: int = 0               # user rating of an assistant turn: -1 down, 0 none, 1 up


class Session(BaseModel):
    """One conversation thread, owned by a (org_id, user_id) pair. ``summary`` is
    the rolling summary of older turns (kept here so a long chat stays bounded);
    ``message_count`` and ``updated_at`` are maintained on every append so the
    session list can sort by recency without scanning messages."""

    id: str
    org_id: str
    user_id: str
    title: str = "New chat"         # first user message auto-becomes the title (see append_message)
    created_at: str = Field(default_factory=_now)
    updated_at: str = Field(default_factory=_now)
    summary: str = ""               # rolling summary of older turns (bounds context)
    message_count: int = 0
    # WATERMARK: how many of the OLDEST messages are already folded into ``summary``.
    # The summarizer compacts only the turns evicted from the live window that sit
    # above this mark, so nothing is lost as the chat grows and nothing is re-summarized.
    summarized_upto: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConversationStore(Protocol):
    """The STORE INTERFACE the Postgres backend implements (structural typing — a
    class just has to provide these methods, no base class to inherit). Note that
    EVERY method takes ``org_id`` first: tenant scoping is part of the contract, not
    an afterthought. ``aclose`` lets the backend release resources (e.g. the PG pool)
    on shutdown."""

    async def create_session(self, org_id: str, user_id: str, title: str = "New chat") -> Session: ...
    async def get_session(self, org_id: str, session_id: str) -> Session | None: ...
    async def list_sessions(self, org_id: str, user_id: str, *, limit: int = 50, offset: int = 0) -> list[Session]: ...
    async def update_session(self, org_id: str, session_id: str, *, title: str | None = None, summary: str | None = None, summarized_upto: int | None = None) -> Session | None: ...
    async def delete_session(self, org_id: str, session_id: str) -> bool: ...
    async def append_message(self, org_id: str, session_id: str, role: str, content: str, *, citations=None, tool_calls=None, meta=None) -> Message: ...
    async def get_messages(self, org_id: str, session_id: str, *, limit: int | None = None, offset: int = 0) -> list[Message]: ...
    async def search_messages(self, org_id: str, user_id: str, query: str, *, limit: int = 20) -> list[Message]: ...
    async def set_message_feedback(self, org_id: str, message_id: str, value: int) -> bool: ...
    async def aclose(self) -> None: ...


__all__ = ["Message", "Session", "ConversationStore"]
