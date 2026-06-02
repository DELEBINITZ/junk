"""Postgres-backed chat persistence.

Same interface as the in-memory ConversationStore, but durable and org-isolated
at the database layer via RLS (every read/write runs inside org_transaction).
Reads are additionally scoped by user_id — chat history is private to its owner
(plan §9.1-§9.2). Selected when STORE_BACKEND=postgres.
"""

from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

from app.core.db.postgres import PostgresDatabase, get_database
from app.core.memory.conversations import ChatMessage, ChatSession
from app.domain import User


def _loads(value: Any) -> list[str]:
    if isinstance(value, list):
        return value
    if isinstance(value, (str, bytes)):
        try:
            return list(json.loads(value))
        except (ValueError, TypeError):
            return []
    return []


class PostgresConversationStore:
    def __init__(self, database: PostgresDatabase | None = None):
        self._db = database or get_database()

    def start_session(self, user: User, title: str = "New chat") -> ChatSession:
        session_id = str(uuid4())
        with self._db.org_transaction(user.organization_id) as conn:
            conn.execute(
                "INSERT INTO sessions (id, organization_id, user_id, title) VALUES (%s, %s, %s, %s)",
                (session_id, user.organization_id, user.id, title),
            )
        return ChatSession(id=session_id, org_id=user.organization_id, user_id=user.id, title=title)

    def append(
        self,
        user: User,
        session_id: str | None,
        role: str,
        content: str,
        citations: list[str] | None = None,
    ) -> ChatMessage:
        cites = list(citations or [])
        with self._db.org_transaction(user.organization_id) as conn:
            if session_id:
                owner = conn.execute(
                    "SELECT user_id FROM sessions WHERE id = %s", (session_id,)
                ).fetchone()
                if owner is None:
                    conn.execute(
                        "INSERT INTO sessions (id, organization_id, user_id, title) "
                        "VALUES (%s, %s, %s, %s)",
                        (session_id, user.organization_id, user.id, content[:40] or "New chat"),
                    )
                elif owner[0] != user.id:
                    raise PermissionError("session does not belong to the requesting user")
            else:
                session_id = str(uuid4())
                conn.execute(
                    "INSERT INTO sessions (id, organization_id, user_id, title) "
                    "VALUES (%s, %s, %s, %s)",
                    (session_id, user.organization_id, user.id, content[:40] or "New chat"),
                )
            message_id = str(uuid4())
            conn.execute(
                "INSERT INTO messages (id, session_id, organization_id, user_id, role, content, citations) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (message_id, session_id, user.organization_id, user.id, role, content, json.dumps(cites)),
            )
        return ChatMessage(
            id=message_id, session_id=session_id, org_id=user.organization_id,
            user_id=user.id, role=role, content=content, citations=cites,
        )

    def get_messages(self, user: User, session_id: str) -> list[ChatMessage]:
        with self._db.org_transaction(user.organization_id) as conn:
            owner = conn.execute(
                "SELECT user_id FROM sessions WHERE id = %s", (session_id,)
            ).fetchone()
            if owner is None or owner[0] != user.id:
                return []
            rows = conn.execute(
                "SELECT id, role, content, citations FROM messages "
                "WHERE session_id = %s AND user_id = %s ORDER BY created_at",
                (session_id, user.id),
            ).fetchall()
        return [
            ChatMessage(
                id=str(r[0]), session_id=session_id, org_id=user.organization_id,
                user_id=user.id, role=r[1], content=r[2], citations=_loads(r[3]),
            )
            for r in rows
        ]

    def recent_history(self, user: User, session_id: str, k: int = 6) -> list[ChatMessage]:
        return self.get_messages(user, session_id)[-k:]

    def list_sessions(self, user: User) -> list[ChatSession]:
        with self._db.org_transaction(user.organization_id) as conn:
            rows = conn.execute(
                "SELECT id, title FROM sessions "
                "WHERE user_id = %s AND soft_deleted_at IS NULL ORDER BY created_at DESC",
                (user.id,),
            ).fetchall()
        return [
            ChatSession(id=str(r[0]), org_id=user.organization_id, user_id=user.id, title=r[1])
            for r in rows
        ]

    def search_past(self, user: User, query: str, since: str | None = None) -> list[dict]:
        like = f"%{query}%"
        sql = (
            "SELECT session_id, role, content, created_at FROM messages "
            "WHERE user_id = %s AND content ILIKE %s "
        )
        params: list[Any] = [user.id, like]
        if since:
            sql += "AND created_at >= %s "
            params.append(since)
        sql += "ORDER BY created_at DESC LIMIT 20"
        with self._db.org_transaction(user.organization_id) as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
        return [
            {"session_id": str(r[0]), "role": r[1], "content": str(r[2])[:300],
             "created_at": r[3].isoformat() if hasattr(r[3], "isoformat") else str(r[3])}
            for r in rows
        ]
