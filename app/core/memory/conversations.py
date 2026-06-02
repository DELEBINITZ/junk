"""Chat persistence: sessions + messages (source of truth for the UI) and
cross-session recall.

Increment 1 keeps this in-memory; the production target is PostgreSQL
(sessions/messages) as source of truth plus a per-user Qdrant collection
(`conversations_kb`) for semantic cross-session search. Every read is scoped by
org_id AND user_id — chat history is PRIVATE to its owner (plan §9.1-§9.2).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from threading import RLock
from uuid import uuid4

from app.config import settings
from app.domain import User


def _now() -> datetime:
    return datetime.now(UTC)


@dataclass
class ChatMessage:
    id: str
    session_id: str
    org_id: str
    user_id: str
    role: str  # "user" | "assistant" | "tool"
    content: str
    citations: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=_now)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "role": self.role,
            "content": self.content,
            "citations": self.citations,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class ChatSession:
    id: str
    org_id: str
    user_id: str
    title: str
    created_at: datetime = field(default_factory=_now)
    rolling_summary: str = ""
    messages: list[ChatMessage] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at.isoformat(),
            "message_count": len(self.messages),
        }


class ConversationStore:
    def __init__(self):
        self._lock = RLock()
        self._sessions: dict[str, ChatSession] = {}

    def _owned(self, session: ChatSession, user: User) -> bool:
        return session.org_id == user.organization_id and session.user_id == user.id

    def start_session(self, user: User, title: str = "New chat") -> ChatSession:
        with self._lock:
            session = ChatSession(
                id=str(uuid4()), org_id=user.organization_id, user_id=user.id, title=title,
            )
            self._sessions[session.id] = session
            return session

    def append(
        self,
        user: User,
        session_id: str | None,
        role: str,
        content: str,
        citations: list[str] | None = None,
    ) -> ChatMessage:
        with self._lock:
            session = self._sessions.get(session_id) if session_id else None
            if session is None:
                session = ChatSession(
                    id=session_id or str(uuid4()),
                    org_id=user.organization_id,
                    user_id=user.id,
                    title=(content[:40] or "New chat"),
                )
                self._sessions[session.id] = session
            elif not self._owned(session, user):
                raise PermissionError("session does not belong to the requesting user")
            message = ChatMessage(
                id=str(uuid4()), session_id=session.id, org_id=user.organization_id,
                user_id=user.id, role=role, content=content, citations=list(citations or []),
            )
            session.messages.append(message)
            return message

    def get_messages(self, user: User, session_id: str) -> list[ChatMessage]:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None or not self._owned(session, user):
                return []
            return list(session.messages)

    def recent_history(self, user: User, session_id: str, k: int = 6) -> list[ChatMessage]:
        return self.get_messages(user, session_id)[-k:]

    def list_sessions(self, user: User) -> list[ChatSession]:
        with self._lock:
            return [s for s in self._sessions.values() if self._owned(s, user)]

    def search_past(self, user: User, query: str, since: str | None = None) -> list[dict]:
        """Cross-session recall over the user's OWN chats. Naive substring match
        today; the production path embeds turns into a per-user Qdrant collection
        and does semantic search with an org_id+user_id payload filter."""

        needle = (query or "").lower()
        out: list[dict] = []
        with self._lock:
            for session in self._sessions.values():
                if not self._owned(session, user):
                    continue
                for message in session.messages:
                    if needle and needle in message.content.lower():
                        out.append(
                            {
                                "session_id": session.id,
                                "role": message.role,
                                "content": message.content[:300],
                                "created_at": message.created_at.isoformat(),
                            }
                        )
        return out[:20]


_store: ConversationStore | None = None


def get_conversation_store():
    """Return the chat store for the configured backend. Default "memory" (no
    driver needed); "postgres" returns the RLS-backed store. The Postgres module
    is imported lazily so the default path never requires psycopg."""

    global _store
    if _store is None:
        backend = os.getenv("STORE_BACKEND", settings.store_backend).lower()
        if backend == "postgres":
            from app.core.memory.pg_conversations import PostgresConversationStore

            _store = PostgresConversationStore()
        else:
            _store = ConversationStore()
    return _store


def reset_conversation_store() -> None:
    global _store
    _store = None
