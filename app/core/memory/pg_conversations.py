"""PostgreSQL-backed conversation store (``store_backend=postgres``).

Identical interface to the in-memory store, but every operation runs inside an
``org_transaction`` so Postgres RLS enforces tenant isolation at the database —
the strongest place to enforce it. Cross-session recall uses Postgres full-text
search over the user's own messages.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from app.core.memory.conversations import Message, Session


def _uid(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:24]}"


def _iso(v: Any) -> str:
    return v.isoformat() if hasattr(v, "isoformat") else str(v)


class PostgresConversationStore:
    backend = "postgres"

    def __init__(self, db, settings) -> None:
        self.db = db
        self.settings = settings

    def _session(self, row: dict) -> Session:
        return Session(
            id=row["id"], org_id=row["org_id"], user_id=row["user_id"], title=row["title"],
            summary=row["summary"], message_count=row["message_count"],
            metadata=row.get("metadata") or {},
            created_at=_iso(row["created_at"]), updated_at=_iso(row["updated_at"]),
        )

    def _message(self, row: dict) -> Message:
        return Message(
            id=row["id"], session_id=row["session_id"], org_id=row["org_id"], role=row["role"],
            content=row["content"], citations=row.get("citations") or [],
            tool_calls=row.get("tool_calls") or [], meta=row.get("meta") or {},
            created_at=_iso(row["created_at"]),
        )

    async def _cur(self, conn):
        from psycopg.rows import dict_row

        return conn.cursor(row_factory=dict_row)

    async def create_session(self, org_id: str, user_id: str, title: str = "New chat") -> Session:
        sid = _uid("sess")
        async with self.db.org_transaction(org_id) as conn:
            async with await self._cur(conn) as cur:
                await cur.execute(
                    "INSERT INTO chat_sessions (id, org_id, user_id, title) VALUES (%s,%s,%s,%s) "
                    "RETURNING *", (sid, org_id, user_id, title),
                )
                return self._session(await cur.fetchone())

    async def get_session(self, org_id: str, session_id: str) -> Session | None:
        async with self.db.org_transaction(org_id) as conn:
            async with await self._cur(conn) as cur:
                await cur.execute("SELECT * FROM chat_sessions WHERE id=%s", (session_id,))
                row = await cur.fetchone()
                return self._session(row) if row else None

    async def list_sessions(self, org_id: str, user_id: str, *, limit: int = 50, offset: int = 0) -> list[Session]:
        async with self.db.org_transaction(org_id) as conn:
            async with await self._cur(conn) as cur:
                await cur.execute(
                    "SELECT * FROM chat_sessions WHERE user_id=%s ORDER BY updated_at DESC "
                    "LIMIT %s OFFSET %s", (user_id, limit, offset),
                )
                return [self._session(r) for r in await cur.fetchall()]

    async def update_session(self, org_id: str, session_id: str, *, title=None, summary=None) -> Session | None:
        sets, params = [], []
        if title is not None:
            sets.append("title=%s")
            params.append(title)
        if summary is not None:
            sets.append("summary=%s")
            params.append(summary)
        if not sets:
            return await self.get_session(org_id, session_id)
        sets.append("updated_at=now()")
        params.append(session_id)
        async with self.db.org_transaction(org_id) as conn:
            async with await self._cur(conn) as cur:
                await cur.execute(
                    f"UPDATE chat_sessions SET {', '.join(sets)} WHERE id=%s RETURNING *", params
                )
                row = await cur.fetchone()
                return self._session(row) if row else None

    async def delete_session(self, org_id: str, session_id: str) -> bool:
        async with self.db.org_transaction(org_id) as conn:
            async with await self._cur(conn) as cur:
                await cur.execute("DELETE FROM chat_sessions WHERE id=%s", (session_id,))
                return cur.rowcount > 0

    async def append_message(self, org_id, session_id, role, content, *, citations=None, tool_calls=None, meta=None) -> Message:
        mid = _uid("msg")
        async with self.db.org_transaction(org_id) as conn:
            async with await self._cur(conn) as cur:
                await cur.execute(
                    "INSERT INTO chat_messages (id, session_id, org_id, role, content, citations, tool_calls, meta) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s,%s) RETURNING *",
                    (mid, session_id, org_id, role, content, json.dumps(citations or []),
                     json.dumps(tool_calls or []), json.dumps(meta or {})),
                )
                msg = self._message(await cur.fetchone())
                await cur.execute(
                    "UPDATE chat_sessions SET message_count = message_count + 1, updated_at = now(), "
                    "title = CASE WHEN title = 'New chat' AND %s = 'user' THEN left(%s, 60) ELSE title END "
                    "WHERE id=%s", (role, content, session_id),
                )
                return msg

    async def get_messages(self, org_id, session_id, *, limit=None) -> list[Message]:
        q = "SELECT * FROM chat_messages WHERE session_id=%s ORDER BY created_at"
        params: list = [session_id]
        if limit:
            q = ("SELECT * FROM (SELECT * FROM chat_messages WHERE session_id=%s "
                 "ORDER BY created_at DESC LIMIT %s) t ORDER BY created_at")
            params = [session_id, limit]
        async with self.db.org_transaction(org_id) as conn:
            async with await self._cur(conn) as cur:
                await cur.execute(q, params)
                return [self._message(r) for r in await cur.fetchall()]

    async def search_messages(self, org_id, user_id, query, *, limit=20) -> list[Message]:
        async with self.db.org_transaction(org_id) as conn:
            async with await self._cur(conn) as cur:
                await cur.execute(
                    "SELECT m.* FROM chat_messages m JOIN chat_sessions s ON s.id = m.session_id "
                    "WHERE s.user_id=%s AND to_tsvector('english', m.content) @@ plainto_tsquery('english', %s) "
                    "ORDER BY m.created_at DESC LIMIT %s", (user_id, query, limit),
                )
                return [self._message(r) for r in await cur.fetchall()]

    async def aclose(self) -> None:
        return None


__all__ = ["PostgresConversationStore"]
