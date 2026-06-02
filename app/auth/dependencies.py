"""FastAPI dependencies that turn bearer tokens into trusted user context.

The rest of the backend should depend on `User`, not raw JWT claims. That keeps
authorization decisions tied to current server-side state instead of whatever a
client sends in a token or request body.
"""

from __future__ import annotations

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.auth.jwt import decode_access_token
from app.config import settings
from app.core.security.oidc import user_from_claims, verify_oidc_token
from app.core.security.tokens import get_revocation_store
from app.db.repository import DataStore, get_store
from app.domain import Role, User


bearer_scheme = HTTPBearer(auto_error=False)


def require_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    store: DataStore = Depends(get_store),
) -> User:
    """Return the active authenticated user or fail the request.

    JWTs are useful for stateless authentication, but they should not become the
    only source of truth. We decode the token, load the user from the repository,
    and verify that organization and role still match. That makes role changes
    and account deactivation effective without waiting for every old token to
    naturally expire.
    """

    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token")

    # OIDC mode: the IdP is the source of identity; verify against its JWKS and
    # map claims to a User (no local user store lookup).
    if settings.auth_provider.lower() == "oidc":
        try:
            claims = verify_oidc_token(credentials.credentials)
        except Exception as exc:  # noqa: BLE001 - any verification failure is a 401
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token") from exc
        if not (claims.get("org") or claims.get("organization_id") or claims.get("tenant_id")):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token missing organization claim")
        return user_from_claims(claims)

    try:
        claims = decode_access_token(credentials.credentials)
    except jwt.PyJWTError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token") from exc
    if get_revocation_store().is_revoked(claims.get("jti")):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token revoked")

    user = store.get_user(claims["sub"])
    if user is None or not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Inactive or unknown user")
    if user.organization_id != claims["org"] or user.role != claims["role"]:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token/user mismatch")
    return user


def require_role(*roles: Role):
    """Build a route dependency that allows only the supplied roles."""

    def dependency(user: User = Depends(require_user)) -> User:
        if user.role not in roles:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient role")
        return user

    return dependency
