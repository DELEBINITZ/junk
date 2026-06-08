"""Chat session and message persistence (Postgres with explicit org_id filtering)."""

import json
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4

from security_intel.db.postgres import Database


@dataclass
class ChatMessage:
    id: str
    session_id: str
    role: str
    content: str
    citations: list[dict] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)
    meta: dict = field(default_factory=dict)
    created_at: datetime | None = None


@dataclass
class ChatSession:
    id: str
    org_id: str
    user_id: str
    title: str = ""
    summary: str = ""
    message_count: int = 0
    summarized_upto: int = 0
    created_at: datetime | None = None
    updated_at: datetime | None = None


class ConversationStore:
    """CRUD for chat sessions and messages with explicit org_id filtering."""

    def __init__(self, db: Database):
        self._db = db

    async def create_session(self, org_id: str, user_id: str, title: str = "") -> ChatSession:
        session_id = f"sess_{uuid4().hex[:12]}"
        async with self._db.transaction() as cur:
            await cur.execute(
                """INSERT INTO chat_sessions (id, org_id, user_id, title)
                   VALUES (%s, %s, %s, %s)
                   RETURNING id, org_id, user_id, title, summary, message_count,
                             summarized_upto, created_at, updated_at""",
                (session_id, org_id, user_id, title),
            )
            row = await cur.fetchone()
        return self._row_to_session(row)

    async def get_session(self, org_id: str, session_id: str) -> ChatSession | None:
        async with self._db.transaction() as cur:
            await cur.execute(
                """SELECT id, org_id, user_id, title, summary, message_count,
                          summarized_upto, created_at, updated_at
                   FROM chat_sessions WHERE id = %s AND org_id = %s""",
                (session_id, org_id),
            )
            row = await cur.fetchone()
        return self._row_to_session(row) if row else None

    async def list_sessions(
        self, org_id: str, user_id: str, limit: int = 50, offset: int = 0
    ) -> list[ChatSession]:
        async with self._db.transaction() as cur:
            await cur.execute(
                """SELECT id, org_id, user_id, title, summary, message_count,
                          summarized_upto, created_at, updated_at
                   FROM chat_sessions
                   WHERE org_id = %s AND user_id = %s
                   ORDER BY updated_at DESC
                   LIMIT %s OFFSET %s""",
                (org_id, user_id, limit, offset),
            )
            rows = await cur.fetchall()
        return [self._row_to_session(r) for r in rows]

    async def append_message(
        self,
        org_id: str,
        session_id: str,
        role: str,
        content: str,
        citations: list[dict] | None = None,
        tool_calls: list[dict] | None = None,
        meta: dict | None = None,
    ) -> ChatMessage:
        msg_id = f"msg_{uuid4().hex[:12]}"
        async with self._db.transaction() as cur:
            await cur.execute(
                """INSERT INTO chat_messages (id, session_id, org_id, role, content, citations, tool_calls, meta)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                (
                    msg_id, session_id, org_id, role, content,
                    json.dumps(citations or []),
                    json.dumps(tool_calls or []),
                    json.dumps(meta or {}),
                ),
            )
            await cur.execute(
                """UPDATE chat_sessions
                   SET message_count = message_count + 1, updated_at = NOW()
                   WHERE id = %s AND org_id = %s""",
                (session_id, org_id),
            )
            if role == "user":
                await cur.execute(
                    """UPDATE chat_sessions SET title = %s
                       WHERE id = %s AND org_id = %s AND title = ''""",
                    (content[:60], session_id, org_id),
                )
        return ChatMessage(
            id=msg_id, session_id=session_id, role=role,
            content=content, citations=citations or [], tool_calls=tool_calls or [],
            meta=meta or {},
        )

    async def get_messages(
        self, org_id: str, session_id: str, limit: int = 20, offset: int = 0
    ) -> list[ChatMessage]:
        """Get recent messages for a session (newest last)."""
        async with self._db.transaction() as cur:
            await cur.execute(
                """SELECT id, session_id, role, content, citations, tool_calls, meta, created_at
                   FROM chat_messages
                   WHERE session_id = %s AND org_id = %s
                   ORDER BY created_at ASC
                   LIMIT %s OFFSET %s""",
                (session_id, org_id, limit, offset),
            )
            rows = await cur.fetchall()
        return [self._row_to_message(r) for r in rows]

    async def update_summary(
        self, org_id: str, session_id: str, summary: str, summarized_upto: int
    ) -> None:
        async with self._db.transaction() as cur:
            await cur.execute(
                """UPDATE chat_sessions
                   SET summary = %s, summarized_upto = %s
                   WHERE id = %s AND org_id = %s""",
                (summary, summarized_upto, session_id, org_id),
            )

    async def delete_session(self, org_id: str, session_id: str) -> None:
        async with self._db.transaction() as cur:
            await cur.execute(
                "DELETE FROM chat_sessions WHERE id = %s AND org_id = %s",
                (session_id, org_id),
            )

    @staticmethod
    def _row_to_session(row) -> ChatSession:
        return ChatSession(
            id=row[0], org_id=row[1], user_id=row[2], title=row[3],
            summary=row[4], message_count=row[5], summarized_upto=row[6],
            created_at=row[7], updated_at=row[8],
        )

    @staticmethod
    def _row_to_message(row) -> ChatMessage:
        citations = row[4] if isinstance(row[4], list) else json.loads(row[4] or "[]")
        tool_calls = row[5] if isinstance(row[5], list) else json.loads(row[5] or "[]")
        meta = row[6] if isinstance(row[6], dict) else json.loads(row[6] or "{}")
        return ChatMessage(
            id=row[0], session_id=row[1], role=row[2], content=row[3],
            citations=citations, tool_calls=tool_calls, meta=meta, created_at=row[7],
        )
