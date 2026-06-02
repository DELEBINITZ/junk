"""Security: identity, auth (local JWT + OIDC), RBAC, token revocation."""

from app.core.security.context import SecurityContext
from app.core.security.deps import build_security_context, require_role, require_user
from app.core.security.jwt import (
    create_access_token,
    create_refresh_token,
    decode_token,
)
from app.core.security.rbac import can_call_tool, ensure_role, required_role_for
from app.core.security.tokens import build_revocation_store, get_default_revocation_store
from app.core.security.users import User, build_default_user_store

__all__ = [
    "SecurityContext",
    "require_user",
    "require_role",
    "build_security_context",
    "create_access_token",
    "create_refresh_token",
    "decode_token",
    "ensure_role",
    "can_call_tool",
    "required_role_for",
    "build_revocation_store",
    "get_default_revocation_store",
    "User",
    "build_default_user_store",
]
