"""The trusted per-request identity.

A :class:`SecurityContext` is built once, from a *verified* token (local JWT or
OIDC), and threaded everywhere. Tools receive a derived ``ToolContext`` whose
``org_id`` comes from here — never from tool arguments — so a prompt can never
talk the agent into crossing tenants.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from app.core.contracts import role_satisfies


@dataclass(frozen=True)
class SecurityContext:
    org_id: str
    user_id: str
    roles: tuple[str, ...]
    email: str = ""
    token_id: str = ""  # jti, for revocation / audit
    claims: Mapping[str, Any] = field(default_factory=dict)

    def has_role(self, minimum: str) -> bool:
        return role_satisfies(self.roles, minimum)

    def require_org(self) -> str:
        if not self.org_id:
            raise ValueError("security context has no org_id")
        return self.org_id


__all__ = ["SecurityContext"]
