"""Audit logging helpers."""

from __future__ import annotations

import logging

from app.db.repository import DataStore
from app.domain import AuditEvent, User, new_id
from app.guardrails.pii import redact_pii
from app.observability.logging import safe_extra


logger = logging.getLogger(__name__)


def log_event(
    store: DataStore,
    user: User | None,
    action: str,
    resource_type: str,
    resource_id: str | None,
    outcome: str,
    details: dict[str, object] | None = None,
) -> None:
    """Store an audit event after redacting string details for PII."""

    safe_details = {}
    for key, value in (details or {}).items():
        if isinstance(value, str):
            safe_details[key] = redact_pii(value)
        else:
            safe_details[key] = value

    event = AuditEvent(
        id=new_id(),
        organization_id=user.organization_id if user else None,
        user_id=user.id if user else None,
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        outcome=outcome,
        details=safe_details,
    )
    store.add_audit_event(event)
    logger.info(
        "audit.event.recorded",
        extra=safe_extra(
            audit_event_id=event.id,
            user_id=event.user_id,
            organization_id=event.organization_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            outcome=outcome,
        ),
    )
