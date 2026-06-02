"""Human-approval action gate (the safety boundary for side effects).

================================ WHY A GATE EXISTS ========================
An LLM agent that can READ is low-risk; an agent that can ACT (block an IP,
delete a record, send a message) is high-risk — and crucially, an agent driven
by untrusted text could be tricked (prompt injection) into acting maliciously.
The action gate is the hard rule that defuses this: NO side-effecting tool ever
runs inline. Instead it is parked as an APPROVAL REQUEST in an org-scoped inbox,
and a HUMAN must approve it before it executes.

This is the structural reason injection can't cause damage: even if an attacker
fully hijacks the model's reasoning, the worst it can do is *queue* a request a
human still has to approve. Read access and write access are separated by a
person, not by the model's judgement.

v1 ships the *machinery* with no auto-actions: every ``side_effecting`` tool is
held for human approval (an org-scoped approval inbox). Layer-A deterministic
auto-actions and per-module ``ActionHandler.execute`` slot in later (blueprint
§10) with no change to callers — the in-process MCP client already routes every
side-effecting call through ``auto_approves`` / ``request_approval`` here.

Lifecycle of one action:  request_approval (pending) -> approve / reject (human)
-> execute (runs the module's ActionHandler) -> executed / failed.
===========================================================================
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
    return datetime.now(UTC).isoformat()         # UTC ISO-8601 timestamps for audit fields


@dataclass
class ApprovalRequest:
    """One pending side effect awaiting a human decision — a row in the approval
    inbox. It captures WHAT would run (tool + arguments), WHO requested it, and —
    critically — the ``org_id``, so the inbox is tenant-scoped: org A can never
    see or act on org B's pending actions. ``status`` walks the lifecycle from
    the module docstring."""
    id: str
    action_type: str
    tool_name: str
    module_id: str
    org_id: str                  # owning tenant — every inbox query is filtered by this
    user_id: str                 # who triggered the action
    arguments: dict[str, Any]    # the exact args the action would execute with
    status: str = "pending"  # pending | approved | rejected | executed | failed
    created_at: str = field(default_factory=_now)
    decided_by: str = ""         # the approver/rejecter (set when a human decides)
    decided_at: str = ""
    result: dict | None = None   # filled after execute()

    def public(self) -> dict:
        # The reviewer-facing view. (Internal bookkeeping like decided_at is
        # omitted; this is what the approval-inbox API returns.)
        return {
            "id": self.id, "action_type": self.action_type, "tool": self.tool_name,
            "module": self.module_id, "arguments": self.arguments, "status": self.status,
            "created_at": self.created_at, "decided_by": self.decided_by,
        }


class ActionGate:
    """The inbox + decision engine. The in-process MCP client calls this for every
    side-effecting tool: first ``auto_approves`` (always False in v1), and if not
    auto-approved, ``request_approval`` to park it. Humans then call approve/reject
    and finally execute. The ``org_id`` argument on every method is what enforces
    tenant scoping — you may only touch requests in your own org."""

    def __init__(self, settings=None, action_handlers: dict | None = None) -> None:
        self.settings = settings
        self._inbox: dict[str, ApprovalRequest] = {}   # id -> request (the approval inbox)
        self._handlers = action_handlers or {}  # action_type -> ActionHandler
        self._lock = threading.Lock()                  # guards the inbox across worker threads

    async def auto_approves(self, tool, arguments: Mapping[str, Any], ctx: ToolContext) -> bool:
        # v1: no deterministic auto-actions. Layer-A policy plugs in here later.
        # Returning False unconditionally means EVERYTHING side-effecting goes to a
        # human — the safest possible default while the auto-policy isn't proven.
        return False

    async def request_approval(self, tool, arguments: Mapping[str, Any], ctx: ToolContext) -> str:
        """Park a side-effecting call as a pending request and return its id. The
        action does NOT run here — this only records the intent for human review.
        Note ``org_id``/``user_id`` come from the trusted ``ctx`` (the verified
        identity), never from the tool arguments — the same tenant-isolation rule
        as everywhere else."""
        req = ApprovalRequest(
            id=f"appr_{uuid.uuid4().hex[:20]}", action_type=getattr(tool, "name", "action"),
            tool_name=getattr(tool, "name", ""), module_id=getattr(tool, "module_id", ""),
            org_id=ctx.org_id, user_id=ctx.user_id, arguments=dict(arguments),
        )
        with self._lock:
            self._inbox[req.id] = req
        return req.id

    def list_pending(self, org_id: str) -> list[dict]:
        # The inbox view for one tenant: ONLY this org's still-pending requests.
        # The ``r.org_id == org_id`` filter is the tenant boundary.
        with self._lock:
            return [r.public() for r in self._inbox.values()
                    if r.org_id == org_id and r.status == "pending"]

    def get(self, org_id: str, approval_id: str) -> ApprovalRequest | None:
        # Fetch by id BUT only if it belongs to the caller's org. Returning None
        # for a foreign org_id means an attacker can't even confirm a request from
        # another tenant exists, let alone approve it.
        r = self._inbox.get(approval_id)
        return r if r and r.org_id == org_id else None

    async def approve(self, org_id: str, approval_id: str, approver: str) -> ApprovalRequest | None:
        """Human APPROVE. Only flips a still-pending request (idempotent: a second
        call after it's decided is a no-op). Approval alone does NOT run the
        action — execute() is a separate, explicit step."""
        r = self.get(org_id, approval_id)            # org-scoped lookup
        if not r or r.status != "pending":
            return r
        r.status = "approved"
        r.decided_by = approver                      # record WHO approved (audit)
        r.decided_at = _now()
        return r

    async def reject(self, org_id: str, approval_id: str, approver: str) -> ApprovalRequest | None:
        """Human REJECT — the request is closed and can never execute."""
        r = self.get(org_id, approval_id)
        if not r or r.status != "pending":
            return r
        r.status = "rejected"
        r.decided_by = approver
        r.decided_at = _now()
        return r

    async def execute(self, org_id: str, approval_id: str, ctx: ToolContext) -> ActionResult | None:
        """Run an APPROVED action via its module's ActionHandler. The guard
        ``status != "approved"`` is the crux: an action can only run AFTER a human
        approved it — never straight from a pending (or rejected) state, so the
        agent can never skip the human step."""
        r = self.get(org_id, approval_id)
        if not r or r.status != "approved":
            return None                              # not ours, or not approved -> refuse to run
        handler = self._handlers.get(r.action_type)
        if handler is None:
            # Approved but nothing knows how to perform it (v1 ships no handlers).
            r.status = "failed"
            r.result = {"detail": "no action handler registered"}
            return ActionResult(action_type=r.action_type, status="failed", detail="no handler")
        # Hand off to the module's handler to actually perform the side effect, then
        # record the terminal status + result on the request for the audit trail.
        result = await handler.execute(r.arguments, ctx)
        r.status = "executed" if result.status == "executed" else "failed"
        r.result = result.model_dump()
        return result


def build_action_gate(settings, action_handlers: dict | None = None) -> ActionGate:
    """Factory used at boot to construct the single shared gate."""
    return ActionGate(settings, action_handlers)


__all__ = ["ActionGate", "ApprovalRequest", "build_action_gate"]
