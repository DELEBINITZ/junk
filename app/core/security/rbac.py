"""RBAC helpers shared by the API layer, the MCP tool runner, and tools.

The minimum role for a tool comes from the module manifest (``manifest.rbac``)
or the tool's own ``rbac_role`` — both derived from declarative config, never
hardcoded in core.
"""

from __future__ import annotations

from collections.abc import Iterable

from app.core.contracts import Tool, role_satisfies
from app.core.errors import PermissionDenied
from app.core.security.context import SecurityContext


def ensure_role(sc: SecurityContext, minimum: str) -> None:
    if not sc.has_role(minimum):
        raise PermissionDenied(
            f"role '{minimum}' required",
            details={"have": list(sc.roles), "need": minimum},
        )


def can_call_tool(roles: Iterable[str], tool: Tool, manifest_rbac: dict[str, str] | None = None) -> bool:
    required = (manifest_rbac or {}).get(tool.name, tool.rbac_role)
    return role_satisfies(roles, required)


def required_role_for(tool: Tool, manifest_rbac: dict[str, str] | None = None) -> str:
    return (manifest_rbac or {}).get(tool.name, tool.rbac_role)


__all__ = ["ensure_role", "can_call_tool", "required_role_for"]
