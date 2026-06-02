"""Human-approval action gate (the safety boundary for side effects).

v1 ships the *machinery* with no auto-actions: every ``side_effecting`` tool is
held for human approval (an org-scoped approval inbox). Layer-A deterministic
auto-actions and per-module ``ActionHandler.execute`` slot in later (blueprint
§10) with no change to callers — the in-process MCP client already routes every
side-effecting call through ``auto_approves`` / ``request_approval`` here.
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from app.core.contracts import ActionResult, ToolContext


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class ApprovalRequest:
    id: str
    action_type: str
    tool_name: str
    module_id: str
    org_id: str
    user_id: str
    arguments: dict[str, Any]
    status: str = "pending"  # pending | approved | rejected | executed | failed
    created_at: str = field(default_factory=_now)
    decided_by: str = ""
    decided_at: str = ""
    result: dict | None = None

    def public(self) -> dict:
        return {
            "id": self.id, "action_type": self.action_type, "tool": self.tool_name,
            "module": self.module_id, "arguments": self.arguments, "status": self.status,
            "created_at": self.created_at, "decided_by": self.decided_by,
        }


class ActionGate:
    def __init__(self, settings=None, action_handlers: dict | None = None) -> None:
        self.settings = settings
        self._inbox: dict[str, ApprovalRequest] = {}
        self._handlers = action_handlers or {}  # action_type -> ActionHandler
        self._lock = threading.Lock()

    async def auto_approves(self, tool, arguments: Mapping[str, Any], ctx: ToolContext) -> bool:
        # v1: no deterministic auto-actions. Layer-A policy plugs in here later.
        return False

    async def request_approval(self, tool, arguments: Mapping[str, Any], ctx: ToolContext) -> str:
        req = ApprovalRequest(
            id=f"appr_{uuid.uuid4().hex[:20]}", action_type=getattr(tool, "name", "action"),
            tool_name=getattr(tool, "name", ""), module_id=getattr(tool, "module_id", ""),
            org_id=ctx.org_id, user_id=ctx.user_id, arguments=dict(arguments),
        )
        with self._lock:
            self._inbox[req.id] = req
        return req.id

    def list_pending(self, org_id: str) -> list[dict]:
        with self._lock:
            return [r.public() for r in self._inbox.values()
                    if r.org_id == org_id and r.status == "pending"]

    def get(self, org_id: str, approval_id: str) -> ApprovalRequest | None:
        r = self._inbox.get(approval_id)
        return r if r and r.org_id == org_id else None

    async def approve(self, org_id: str, approval_id: str, approver: str) -> ApprovalRequest | None:
        r = self.get(org_id, approval_id)
        if not r or r.status != "pending":
            return r
        r.status = "approved"
        r.decided_by = approver
        r.decided_at = _now()
        return r

    async def reject(self, org_id: str, approval_id: str, approver: str) -> ApprovalRequest | None:
        r = self.get(org_id, approval_id)
        if not r or r.status != "pending":
            return r
        r.status = "rejected"
        r.decided_by = approver
        r.decided_at = _now()
        return r

    async def execute(self, org_id: str, approval_id: str, ctx: ToolContext) -> ActionResult | None:
        r = self.get(org_id, approval_id)
        if not r or r.status != "approved":
            return None
        handler = self._handlers.get(r.action_type)
        if handler is None:
            r.status = "failed"
            r.result = {"detail": "no action handler registered"}
            return ActionResult(action_type=r.action_type, status="failed", detail="no handler")
        result = await handler.execute(r.arguments, ctx)
        r.status = "executed" if result.status == "executed" else "failed"
        r.result = result.model_dump()
        return result


def build_action_gate(settings, action_handlers: dict | None = None) -> ActionGate:
    return ActionGate(settings, action_handlers)


__all__ = ["ActionGate", "ApprovalRequest", "build_action_gate"]
