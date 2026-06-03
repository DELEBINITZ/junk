"""Auth endpoint: identity introspection (``/v1/auth/me``).

Auth itself is handled by the dependencies in security/deps.py — every request
carries a gateway API key + a JWT, and the JWT (minted upstream) supplies the
identity. There is no login/refresh/logout flow here; this router just exposes a
single read-only endpoint so a client can confirm "who am I and what roles do I
hold?" from its already-verified token.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.core.api.deps import require_user
from app.core.api.schemas import UserInfo
from app.core.security.context import SecurityContext

router = APIRouter(prefix="/v1/auth", tags=["auth"])


@router.get("/me", response_model=UserInfo)
async def me(sc: SecurityContext = Depends(require_user)) -> UserInfo:
    """Echo back the authenticated caller's identity. Pure projection of the
    already-verified SecurityContext — useful for a client to confirm its identity
    and roles without decoding the token itself."""
    return UserInfo(id=sc.user_id, email=sc.email, org_id=sc.org_id, roles=list(sc.roles))


__all__ = ["router"]
