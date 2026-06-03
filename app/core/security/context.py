"""The trusted per-request identity.

================================ MENTAL MODEL =============================
SecurityContext = "who is making THIS request", proven cryptographically. It is
the one object the rest of the system is allowed to trust about identity.

Why it exists: in a multi-tenant system the single most dangerous question is
"which org does this data belong to?". If we ever answered that from something
the caller can lie about (a request body field, a tool argument, a query param),
one tenant could read another tenant's data — a catastrophic leak. So we draw a
hard line: identity is established ONCE, by VERIFYING the caller's token, and is
captured in this immutable object. Everything downstream reads org/roles from
HERE and nowhere else.

A :class:`SecurityContext` is built once, from a *verified* JWT (its signature +
expiry checked — see jwt.py / deps.py), and threaded everywhere. Tools receive a derived
``ToolContext`` whose ``org_id`` comes from here — never from tool arguments —
so a prompt can never talk the agent into crossing tenants. ("Verified token" =
its signature/expiry/issuer were checked; an attacker can't forge or alter it.)
===========================================================================
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from app.core.contracts import role_satisfies


# ``frozen=True`` makes the context IMMUTABLE: once built from the verified
# token, nothing — not a tool, not a node, not the LLM — can mutate the org or
# roles mid-request. Immutability is itself a security property here.
@dataclass(frozen=True)
class SecurityContext:
    org_id: str                  # THE tenant key. Set from the token; the basis of all isolation.
    user_id: str                 # the authenticated subject ("sub" claim).
    roles: tuple[str, ...]       # roles the token asserts (used for RBAC checks below).
    email: str = ""
    # The token's unique id ("jti"). We keep it so a logout/rotation can REVOKE
    # exactly this token (see tokens.py) and so audit logs can name the credential.
    token_id: str = ""  # jti, for revocation / audit
    # The raw verified claims, kept for anything not promoted to a typed field.
    claims: Mapping[str, Any] = field(default_factory=dict)

    def has_role(self, minimum: str) -> bool:
        # RBAC is ORDERED (viewer < analyst < admin) and ``minimum`` is the floor:
        # any role at or above it satisfies the check. The ordering logic lives in
        # one place — ``role_satisfies`` in contracts.py — so it can never drift.
        return role_satisfies(self.roles, minimum)

    def require_org(self) -> str:
        # Fail loudly if we somehow have an identity with no tenant. Better to
        # error than to run a query with a blank/ambiguous org scope.
        if not self.org_id:
            raise ValueError("security context has no org_id")
        return self.org_id


__all__ = ["SecurityContext"]
