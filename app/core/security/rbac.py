"""RBAC helpers shared by the API layer, the MCP tool runner, and tools.

================================ RBAC IN ONE LINE =========================
RBAC (Role-Based Access Control): what you can do is decided by your ROLE, not
your identity. Our roles are ORDERED — viewer < analyst < admin — and every
permission is expressed as a MINIMUM role. "Requires analyst" means analyst OR
admin; it never means "exactly analyst". That ordering lives in ONE place
(``role_satisfies`` in contracts.py); everything here just calls it, so the
rule can't fork into subtly different versions.

The minimum role for a tool comes from the module manifest (``manifest.rbac``)
or the tool's own ``rbac_role`` — both derived from declarative config, never
hardcoded in core. (Manifest can OVERRIDE a tool's default to tighten or relax
it per deployment.)
===========================================================================
"""

from __future__ import annotations

from collections.abc import Iterable

from app.core.contracts import Tool, role_satisfies
from app.core.errors import PermissionDenied
from app.core.security.context import SecurityContext


def ensure_role(sc: SecurityContext, minimum: str) -> None:
    """Assert the caller meets ``minimum`` or raise PermissionDenied. The
    raise-on-failure form, used where a denial should stop the request."""
    if not sc.has_role(minimum):
        raise PermissionDenied(
            f"role '{minimum}' required",
            details={"have": list(sc.roles), "need": minimum},   # surfaced to the client
        )


def can_call_tool(roles: Iterable[str], tool: Tool, manifest_rbac: dict[str, str] | None = None) -> bool:
    """The predicate form (returns bool, doesn't raise) the MCP boundary uses
    before running a tool. Resolve the tool's REQUIRED role — a manifest override
    wins over the tool's declared ``rbac_role`` — then test the caller's roles
    against it with the ordered comparison."""
    required = (manifest_rbac or {}).get(tool.name, tool.rbac_role)
    return role_satisfies(roles, required)


def required_role_for(tool: Tool, manifest_rbac: dict[str, str] | None = None) -> str:
    """Just resolve which role a tool needs (manifest override, else the tool's
    own default). Used for display/introspection — e.g. listing tools a user can
    or can't call — separate from the actual enforcement above."""
    return (manifest_rbac or {}).get(tool.name, tool.rbac_role)


__all__ = ["ensure_role", "can_call_tool", "required_role_for"]
