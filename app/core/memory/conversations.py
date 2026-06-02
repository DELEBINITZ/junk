"""Chat persistence: sessions + messages, with cross-session recall.

This is the ChatGPT/Claude-style memory: durable sessions per (org, user),
full message history, titles, rolling summaries, and a search across a user's
past chats. Everything is **org-scoped** — every method takes ``org_id`` and the
store refuses to return another org's rows. In-memory by default; the Postgres
backend (``pg_conversations``) enforces the same isolation via RLS.
"""

from __future__ import annotations

import re
import uuid
from datetime import UTC, datetime
from typing import Any, Protocol

from pydantic import BaseModel, Field


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _uid(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:24]}"


class Message(BaseModel):
    id: str
    session_id: str
    org_id: str
    role: str                       # user | assistant | system | tool
    content: str = ""
    created_at: str = Field(default_factory=_now)
    citations: list[dict[str, Any]] = Field(default_factory=list)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)


class Session(BaseModel):
    id: str
    org_id: str
    user_id: str
    title: str = "New chat"
    created_at: str = Field(default_factory=_now)
    updated_at: str = Field(default_factory=_now)
    summary: str = ""
    message_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConversationStore(Protocol):
    async def create_session(self, org_id: str, user_id: str, title: str = "New chat") -> Session: ...
    async def get_session(self, org_id: str, session_id: str) -> Session | None: ...
    async def list_sessions(self, org_id: str, user_id: str, *, limit: int = 50, offset: int = 0) -> list[Session]: ...
    async def update_session(self, org_id: str, session_id: str, *, title: str | None = None, summary: str | None = None) -> Session | None: ...
    async def delete_session(self, org_id: str, session_id: str) -> bool: ...
    async def append_message(self, org_id: str, session_id: str, role: str, content: str, *, citations=None, tool_calls=None, meta=None) -> Message: ...
    async def get_messages(self, org_id: str, session_id: str, *, limit: int | None = None) -> list[Message]: ...
    async def search_messages(self, org_id: str, user_id: str, query: str, *, limit: int = 20) -> list[Message]: ...
    async def aclose(self) -> None: ...


_WORD = re.compile(r"[a-z0-9]+")


class InMemoryConversationStore:
    backend = "memory"

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._messages: dict[str, list[Message]] = {}

    def _own(self, org_id: str, session_id: str) -> Session | None:
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
        s = self._own(org_id, session_id)
        if not s:
            raise KeyError("session not found for org")
        m = Message(
            id=_uid("msg"), session_id=session_id, org_id=org_id, role=role, content=content,
            citations=citations or [], tool_calls=tool_calls or [], meta=meta or {},
        )
        self._messages[session_id].append(m)
        s.message_count += 1
        s.updated_at = _now()
        if s.title == "New chat" and role == "user" and content.strip():
            s.title = content.strip()[:60]
        return m

    async def get_messages(self, org_id, session_id, *, limit=None) -> list[Message]:
        if not self._own(org_id, session_id):
            return []
        msgs = self._messages.get(session_id, [])
        return msgs[-limit:] if limit else list(msgs)

    async def search_messages(self, org_id, user_id, query, *, limit=20) -> list[Message]:
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
        scored.sort(key=lambda t: (t[0], t[1].created_at), reverse=True)
        return [m for _s, m in scored[:limit]]

    async def aclose(self) -> None:
        return None


__all__ = ["Message", "Session", "ConversationStore", "InMemoryConversationStore"]
