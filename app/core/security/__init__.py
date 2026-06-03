"""Security: identity (API key + JWT), RBAC, and the MCP service token."""

from app.core.security.context import SecurityContext
from app.core.security.deps import build_security_context, require_role, require_user
from app.core.security.jwt import create_access_token, create_service_token, decode_token
from app.core.security.rbac import can_call_tool, ensure_role, required_role_for

__all__ = [
    "SecurityContext",
    "require_user",
    "require_role",
    "build_security_context",
    "create_access_token",
    "create_service_token",
    "decode_token",
    "ensure_role",
    "can_call_tool",
    "required_role_for",
]
