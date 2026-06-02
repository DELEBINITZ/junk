"""PostgreSQL-backed conversation store (``store_backend=postgres``).

================================ MENTAL MODEL =============================
This is the PRODUCTION twin of InMemoryConversationStore. It implements the exact
same ``ConversationStore`` interface, so callers (the chat service, the routers)
are completely unaware which backend is live — only ``bootstrap`` decides, from
``store_backend``. The difference is durability AND how tenant isolation is
enforced.

THE KEY IDEA — defense at the database via RLS: every method below runs its SQL
inside ``self.db.org_transaction(org_id)``. That context manager opens a Postgres
transaction and sets a per-transaction org GUC (a session/transaction variable);
each table has a Row-Level Security (RLS) policy that filters rows by that GUC. So
the SQL here often DOESN'T even say ``WHERE org_id = ...`` for SELECTs — Postgres
itself transparently restricts every query to the current tenant's rows. Even a
bug that forgot a filter cannot leak another org's data. (See db/postgres.py for
the GUC mechanics.) Crucially, ``org_id`` is passed in from the verified token
(SecurityContext), never from the request body.

Cross-session recall (``search_messages``) uses Postgres full-text search (FTS):
``to_tsvector``/``plainto_tsquery`` — the real version of the in-memory store's
toy word-overlap scorer.
===========================================================================
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from app.core.memory.conversations import Message, Session


# Same id scheme as the in-memory store so ids look identical across backends.
def _uid(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:24]}"


def _iso(v: Any) -> str:
    # Postgres returns timestamps as datetime objects; the pydantic models store
    # ISO strings. Normalize either shape to a string so rows map cleanly.
    return v.isoformat() if hasattr(v, "isoformat") else str(v)


class PostgresConversationStore:
    """Durable conversation store. Holds the PG connection pool wrapper (``db``)
    and settings; ``backend`` labels it for /health and metrics. All state lives
    in Postgres, so this object is just a thin, stateless mapper over SQL."""

    backend = "postgres"

    def __init__(self, db, settings) -> None:
        self.db = db              # app.core.db.postgres.PostgresDatabase (owns the pool + RLS)
        self.settings = settings

    # _session / _message: map a raw DB row (dict) into the shared pydantic model,
    # so the rest of the app handles identical Session/Message objects regardless
    # of which backend produced them.
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
        # Open a cursor whose rows come back as dicts (column name -> value), which
        # is what _session/_message expect. psycopg is lazy-imported so the
        # in-memory default never requires the driver to be installed.
        from psycopg.rows import dict_row

        return conn.cursor(row_factory=dict_row)

    async def create_session(self, org_id: str, user_id: str, title: str = "New chat") -> Session:
        sid = _uid("sess")
        # org_transaction sets the RLS org GUC for this transaction; the INSERT
        # writes a row tagged with this org, and RLS will scope all later reads.
        async with self.db.org_transaction(org_id) as conn:
            async with await self._cur(conn) as cur:
                await cur.execute(
                    "INSERT INTO chat_sessions (id, org_id, user_id, title) VALUES (%s,%s,%s,%s) "
                    "RETURNING *", (sid, org_id, user_id, title),
                )
                return self._session(await cur.fetchone())

    async def get_session(self, org_id: str, session_id: str) -> Session | None:
        # Note there is NO "AND org_id=%s" here — RLS adds the tenant filter for us.
        # If this session belongs to another org, the policy hides it and we get
        # back None, exactly as if it didn't exist. That is RLS doing the isolation.
        async with self.db.org_transaction(org_id) as conn:
            async with await self._cur(conn) as cur:
                await cur.execute("SELECT * FROM chat_sessions WHERE id=%s", (session_id,))
                row = await cur.fetchone()
                return self._session(row) if row else None

    async def list_sessions(self, org_id: str, user_id: str, *, limit: int = 50, offset: int = 0) -> list[Session]:
        # The WHERE filters by user; RLS independently filters by org. Most-recent
        # first, paginated by limit/offset for the sidebar's session list.
        async with self.db.org_transaction(org_id) as conn:
            async with await self._cur(conn) as cur:
                await cur.execute(
                    "SELECT * FROM chat_sessions WHERE user_id=%s ORDER BY updated_at DESC "
                    "LIMIT %s OFFSET %s", (user_id, limit, offset),
                )
                return [self._session(r) for r in await cur.fetchall()]

    async def update_session(self, org_id: str, session_id: str, *, title=None, summary=None) -> Session | None:
        # Build a dynamic SET clause from only the fields that were provided. The
        # column names are hardcoded (not user input) and values go through bound
        # %s params, so this stays injection-safe despite the f-string assembly.
        sets, params = [], []
        if title is not None:
            sets.append("title=%s")
            params.append(title)
        if summary is not None:
            sets.append("summary=%s")
            params.append(summary)
        if not sets:                               # nothing to change -> just read it back
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
        # RLS makes this safe to issue without an org filter: the DELETE can only
        # affect a row the current org owns. rowcount tells us if anything matched.
        async with self.db.org_transaction(org_id) as conn:
            async with await self._cur(conn) as cur:
                await cur.execute("DELETE FROM chat_sessions WHERE id=%s", (session_id,))
                return cur.rowcount > 0

    async def append_message(self, org_id, session_id, role, content, *, citations=None, tool_calls=None, meta=None) -> Message:
        mid = _uid("msg")
        # Both statements run in ONE transaction (and one tenant scope): insert the
        # message, then bump the session's counters. Doing them atomically keeps
        # message_count and the message rows from ever drifting apart. The JSONB
        # columns (citations/tool_calls/meta) are passed as JSON-encoded strings.
        async with self.db.org_transaction(org_id) as conn:
            async with await self._cur(conn) as cur:
                await cur.execute(
                    "INSERT INTO chat_messages (id, session_id, org_id, role, content, citations, tool_calls, meta) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s,%s) RETURNING *",
                    (mid, session_id, org_id, role, content, json.dumps(citations or []),
                     json.dumps(tool_calls or []), json.dumps(meta or {})),
                )
                msg = self._message(await cur.fetchone())
                # Same denormalized-counter + auto-title logic as the in-memory
                # store, but expressed in SQL: the CASE auto-titles an untouched
                # session from the first user message.
                await cur.execute(
                    "UPDATE chat_sessions SET message_count = message_count + 1, updated_at = now(), "
                    "title = CASE WHEN title = 'New chat' AND %s = 'user' THEN left(%s, 60) ELSE title END "
                    "WHERE id=%s", (role, content, session_id),
                )
                return msg

    async def get_messages(self, org_id, session_id, *, limit=None) -> list[Message]:
        # Default: full history in chronological order. With ``limit`` we want the
        # most RECENT N but still returned oldest->newest, so we grab the newest N
        # in a subquery (ORDER BY ... DESC LIMIT) then re-sort ascending outside.
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
        # Cross-session recall via Postgres full-text search (FTS): to_tsvector
        # turns each message into searchable lexemes, plainto_tsquery parses the
        # user's words, and @@ matches them. Scoped by user (WHERE) and by org
        # (RLS on both joined tables) — recall never crosses tenant or user lines.
        async with self.db.org_transaction(org_id) as conn:
            async with await self._cur(conn) as cur:
                await cur.execute(
                    "SELECT m.* FROM chat_messages m JOIN chat_sessions s ON s.id = m.session_id "
                    "WHERE s.user_id=%s AND to_tsvector('english', m.content) @@ plainto_tsquery('english', %s) "
                    "ORDER BY m.created_at DESC LIMIT %s", (user_id, query, limit),
                )
                return [self._message(r) for r in await cur.fetchall()]

    async def aclose(self) -> None:
        # The shared PostgresDatabase owns the pool's lifecycle, so this store has
        # nothing of its own to close; present only to satisfy the protocol.
        return None


__all__ = ["PostgresConversationStore"]
