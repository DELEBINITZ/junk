"""Append-only audit logging (auth events, approvals, admin actions).

AUDIT LOG vs ordinary logs: regular logs are for operators debugging; an audit log
is the SECURITY-of-record trail of *who did what, in which org* — the thing a
compliance review reads. It is append-only (you never edit/delete entries).

Two-tier write: it ALWAYS emits a structured log line (so the trail exists even on
the zero-infra path), and when Postgres is configured it ALSO writes an
``audit_log`` row INSIDE the org transaction — meaning the row is RLS-scoped to the
acting org, so audit data inherits the same tenant isolation as everything else.
Best-effort by design: a failed audit write is swallowed and never breaks the
request it's recording.
"""

from __future__ import annotations

import json
from typing import Any


class AuditLogger:
    """Writes audit entries. ``db`` is optional — present only when Postgres is the
    configured store; without it, the structured log line is the whole trail."""

    def __init__(self, logger, db=None) -> None:
        self.logger = logger
        self.db = db

    async def record(self, *, org_id: str, event: str, user_id: str = "", **payload: Any) -> None:
        """Record one auditable event. ``org_id`` and ``event`` are required (the
        WHO-tenant and the WHAT); ``payload`` is arbitrary event detail.

        Note the field discipline: only org/user/event go to the structured LOG
        (matching logging.py's whitelist), while the full ``payload`` is persisted
        only in the DB row — keeping potentially sensitive detail out of stdout
        logs but inside the tenant-scoped, access-controlled audit table."""
        self.logger.info(
            "audit.%s", event,
            extra={"event": "audit", "org_id": org_id, "user_id": user_id},
        )
        if self.db is not None and org_id:
            try:
                # org_transaction sets the RLS org context, so this INSERT lands in
                # the acting org's slice and can't be read by another tenant.
                async with self.db.org_transaction(org_id) as conn:
                    await conn.execute(
                        "INSERT INTO audit_log (org_id, user_id, event, payload) VALUES (%s,%s,%s,%s)",
                        (org_id, user_id, event, json.dumps(payload, default=str)),
                    )
            except Exception:
                pass   # best-effort: never let an audit write failure surface to the caller


def build_audit_logger(settings, logger) -> AuditLogger:
    """Config-gate factory (called by bootstrap): attach a real DB handle only when
    Postgres is configured; otherwise the audit logger is log-line-only."""
    db = None
    if settings.store_backend == "postgres" and settings.database_url:
        from app.core.db.postgres import get_database

        db = get_database(settings)
    return AuditLogger(logger, db)


__all__ = ["AuditLogger", "build_audit_logger"]
