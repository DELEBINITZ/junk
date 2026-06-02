"""Chat persistence: sessions + messages, with cross-session recall.

================================ MENTAL MODEL =============================
This is the "chat session / message store" — the same idea as ChatGPT/Claude
keeping your conversations in a sidebar. A SESSION is one conversation thread; a
MESSAGE is one turn (user/assistant/...) inside it. The store gives the app:
durable sessions per (org, user), full message history, auto-titles, a rolling
SUMMARY of older turns (so we can keep context bounded — see summarizer.py), and
"cross-session recall" = search across ALL of a user's past chats.

WHY a Protocol + two implementations? The platform is config-driven: this file
defines the INTERFACE (``ConversationStore``) plus a zero-infra in-memory backend
used by default and in tests. The Postgres backend (``pg_conversations.py``)
implements the very same interface for production durability.

SECURITY — read this carefully: every method is **org-scoped**. The first
argument is always ``org_id``, and that value comes from the verified token
(SecurityContext), NEVER from the request body. The in-memory store enforces this
by checking ``row.org_id == org_id`` on every access; the Postgres store enforces
it even harder, at the database, via Row-Level Security (RLS). One tenant can
never read another tenant's sessions or messages.
===========================================================================
"""

from __future__ import annotations

import re
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
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConversationStore(Protocol):
    """The STORE INTERFACE both backends implement (structural typing — a class
    just has to provide these methods, no base class to inherit). Note that EVERY
    method takes ``org_id`` first: tenant scoping is part of the contract, not an
    afterthought. ``aclose`` lets a backend release resources (e.g. the PG pool)
    on shutdown; the in-memory store has nothing to close."""

    async def create_session(self, org_id: str, user_id: str, title: str = "New chat") -> Session: ...
    async def get_session(self, org_id: str, session_id: str) -> Session | None: ...
    async def list_sessions(self, org_id: str, user_id: str, *, limit: int = 50, offset: int = 0) -> list[Session]: ...
    async def update_session(self, org_id: str, session_id: str, *, title: str | None = None, summary: str | None = None) -> Session | None: ...
    async def delete_session(self, org_id: str, session_id: str) -> bool: ...
    async def append_message(self, org_id: str, session_id: str, role: str, content: str, *, citations=None, tool_calls=None, meta=None) -> Message: ...
    async def get_messages(self, org_id: str, session_id: str, *, limit: int | None = None) -> list[Message]: ...
    async def search_messages(self, org_id: str, user_id: str, query: str, *, limit: int = 20) -> list[Message]: ...
    async def aclose(self) -> None: ...


_WORD = re.compile(r"[a-z0-9]+")     # crude tokenizer for the in-memory recall search


class InMemoryConversationStore:
    """Default, zero-infrastructure backend: everything lives in process dicts.
    Perfect for local dev and tests, lost on restart. The Postgres backend swaps
    in for durability without changing any caller, because both satisfy
    ``ConversationStore``. ``backend`` is a label used in /health and metrics."""

    backend = "memory"

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}          # session_id -> Session
        self._messages: dict[str, list[Message]] = {}    # session_id -> ordered messages

    def _own(self, org_id: str, session_id: str) -> Session | None:
        """The tenant-isolation guard for this backend: return the session ONLY if
        it belongs to ``org_id``. Every public method funnels through here (or the
        equivalent org filter), so a caller can never reach across orgs even if it
        guesses another tenant's session id. This is the in-memory analogue of the
        Postgres RLS policy."""
        s = self._sessions.get(session_id)
        if s and s.org_id == org_id:
            return s
        return None

    async def create_session(self, org_id: str, user_id: str, title: str = "New chat") -> Session:
        s = Session(id=_uid("sess"), org_id=org_id, user_id=user_id, title=title)
        self._sessions[s.id] = s
        self._messages[s.id] = []
        return s

    async def get_session(self, org_id: str, session_id: str) -> Session | None:
        return self._own(org_id, session_id)

    async def list_sessions(self, org_id: str, user_id: str, *, limit: int = 50, offset: int = 0) -> list[Session]:
        rows = [s for s in self._sessions.values() if s.org_id == org_id and s.user_id == user_id]
        rows.sort(key=lambda s: s.updated_at, reverse=True)
        return rows[offset: offset + limit]

    async def update_session(self, org_id: str, session_id: str, *, title=None, summary=None) -> Session | None:
        # Partial update: only the provided fields change. ``summary`` is set here
        # by the summarizer when it compresses older turns; ``title`` by a rename.
        s = self._own(org_id, session_id)
        if not s:
            return None
        if title is not None:
            s.title = title
        if summary is not None:
            s.summary = summary
        s.updated_at = _now()
        return s

    async def delete_session(self, org_id: str, session_id: str) -> bool:
        s = self._own(org_id, session_id)
        if not s:
            return False
        self._sessions.pop(session_id, None)
        self._messages.pop(session_id, None)
        return True

    async def append_message(self, org_id, session_id, role, content, *, citations=None, tool_calls=None, meta=None) -> Message:
        # Appending requires an owned session; reaching another org's session here
        # raises (a write must never silently land in the wrong tenant).
        s = self._own(org_id, session_id)
        if not s:
            raise KeyError("session not found for org")
        m = Message(
            id=_uid("msg"), session_id=session_id, org_id=org_id, role=role, content=content,
            citations=citations or [], tool_calls=tool_calls or [], meta=meta or {},
        )
        self._messages[session_id].append(m)
        # Maintain the session's denormalized counters so list_sessions stays cheap.
        s.message_count += 1
        s.updated_at = _now()
        # Auto-title an untouched session from its first user message (the ChatGPT
        # behavior where a new chat names itself after what you first asked).
        if s.title == "New chat" and role == "user" and content.strip():
            s.title = content.strip()[:60]
        return m

    async def get_messages(self, org_id, session_id, *, limit=None) -> list[Message]:
        # Org guard first: a non-owned session reads as empty, never another org's.
        if not self._own(org_id, session_id):
            return []
        msgs = self._messages.get(session_id, [])
        # ``limit`` returns the most RECENT N (the tail) — what a chat turn needs
        # to rebuild short-term context without loading the entire history.
        return msgs[-limit:] if limit else list(msgs)

    async def search_messages(self, org_id, user_id, query, *, limit=20) -> list[Message]:
        """Cross-session recall: search ALL of this user's chats for messages that
        share words with ``query``. This is a naive lexical scorer (count of
        overlapping words) — fine for dev. The Postgres backend does the real
        thing with full-text search (FTS). Scoping is doubly enforced: only this
        (org, user)'s sessions are even considered."""
        q = set(_WORD.findall(query.lower()))
        if not q:
            return []
        owned = {sid for sid, s in self._sessions.items() if s.org_id == org_id and s.user_id == user_id}
        scored: list[tuple[int, Message]] = []
        for sid in owned:
            for m in self._messages.get(sid, []):
                overlap = len(q & set(_WORD.findall(m.content.lower())))
                if overlap:
                    scored.append((overlap, m))
        # Best overlap first; ties broken by recency (newer created_at wins).
        scored.sort(key=lambda t: (t[0], t[1].created_at), reverse=True)
        return [m for _s, m in scored[:limit]]

    async def aclose(self) -> None:
        # Nothing to release for an in-process store; present to satisfy the protocol.
        return None


__all__ = ["Message", "Session", "ConversationStore", "InMemoryConversationStore"]
