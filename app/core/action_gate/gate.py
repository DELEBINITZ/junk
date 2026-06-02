"""The approval gate.

Every side-effecting action parks here until a human approves it; only then does
the core call the module's ActionHandler.execute(). v1 ships no action tools, so
nothing submits here yet — it exists so the autonomy model (plan §6.5) is
forward-compatible. org-scoped like everything else.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from threading import RLock
from typing import Any
from uuid import uuid4

from app.core.contracts import ToolContext
from app.domain import User


@dataclass
class PendingAction:
    id: str
    org_id: str
    user_id: str
    action_type: str
    args: dict[str, Any]
    preview: dict[str, Any]
    status: str = "pending"  # pending | approved | rejected
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class ApprovalGate:
    def __init__(self):
        self._lock = RLock()
        self._pending: dict[str, PendingAction] = {}

    def submit(self, ctx: ToolContext, action_type: str, args: dict, preview: dict) -> PendingAction:
        with self._lock:
            action = PendingAction(
                id=str(uuid4()), org_id=ctx.org_id, user_id=ctx.user.id,
                action_type=action_type, args=args, preview=preview,
            )
            self._pending[action.id] = action
            return action

    def list_pending(self, user: User) -> list[PendingAction]:
        with self._lock:
            return [
                a for a in self._pending.values()
                if a.org_id == user.organization_id and a.status == "pending"
            ]

    def decide(self, user: User, action_id: str, approve: bool) -> PendingAction | None:
        with self._lock:
            action = self._pending.get(action_id)
            if action is None or action.org_id != user.organization_id:
                return None
            action.status = "approved" if approve else "rejected"
            return action


_gate: ApprovalGate | None = None


def get_action_gate() -> ApprovalGate:
    global _gate
    if _gate is None:
        _gate = ApprovalGate()
    return _gate
