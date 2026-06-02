"""Append-only audit logging (auth events, approvals, admin actions).

Always emits a structured log line; when Postgres is configured it also writes an
``audit_log`` row inside the org transaction (RLS-scoped). Best-effort: an audit
write never breaks the request path.
"""

from __future__ import annotations

import json
from typing import Any


class AuditLogger:
    def __init__(self, logger, db=None) -> None:
        self.logger = logger
        self.db = db

    async def record(self, *, org_id: str, event: str, user_id: str = "", **payload: Any) -> None:
        self.logger.info(
            "audit.%s", event,
            extra={"event": "audit", "org_id": org_id, "user_id": user_id},
        )
        if self.db is not None and org_id:
            try:
                async with self.db.org_transaction(org_id) as conn:
                    await conn.execute(
                        "INSERT INTO audit_log (org_id, user_id, event, payload) VALUES (%s,%s,%s,%s)",
                        (org_id, user_id, event, json.dumps(payload, default=str)),
                    )
            except Exception:
                pass


def build_audit_logger(settings, logger) -> AuditLogger:
    db = None
    if settings.store_backend == "postgres" and settings.database_url:
        from app.core.db.postgres import get_database

        db = get_database(settings)
    return AuditLogger(logger, db)


__all__ = ["AuditLogger", "build_audit_logger"]
