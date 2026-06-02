"""JWT helpers for encoding and validating authenticated user context."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

import jwt

from app.config import settings
from app.domain import User


ALGORITHM = "HS256"


def create_access_token(user: User) -> str:
    """Create a short-lived bearer token with identity, tenant, and role claims.
    Carries a `jti` so it can be revoked (logout) via the revocation store."""

    now = datetime.now(timezone.utc)
    payload: dict[str, Any] = {
        "sub": user.id,
        "email": user.email,
        "org": user.organization_id,
        "role": user.role,
        "typ": "access",
        "jti": str(uuid4()),
        "iss": settings.jwt_issuer,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=settings.access_token_minutes)).timestamp()),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=ALGORITHM)


def create_refresh_token(user: User) -> str:
    """Create a long-lived refresh token (typ=refresh) exchanged for new access
    tokens at /auth/refresh. Revocable by jti."""

    now = datetime.now(timezone.utc)
    payload: dict[str, Any] = {
        "sub": user.id,
        "org": user.organization_id,
        "role": user.role,
        "typ": "refresh",
        "jti": str(uuid4()),
        "iss": settings.jwt_issuer,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=settings.refresh_token_minutes)).timestamp()),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=ALGORITHM)


def decode_refresh_token(token: str) -> dict[str, Any]:
    claims = jwt.decode(
        token, settings.jwt_secret, algorithms=[ALGORITHM], issuer=settings.jwt_issuer,
        options={"require": ["sub", "exp", "iss"]},
    )
    if claims.get("typ") != "refresh":
        raise jwt.InvalidTokenError("not a refresh token")
    return claims


def read_jti_ignoring_expiry(token: str) -> str | None:
    """Read the jti without enforcing expiry (used at logout to revoke)."""

    try:
        claims = jwt.decode(
            token, settings.jwt_secret, algorithms=[ALGORITHM],
            options={"verify_exp": False, "verify_iss": False},
        )
        return claims.get("jti")
    except jwt.PyJWTError:
        return None


def decode_access_token(token: str) -> dict[str, Any]:
    """Decode a token and require the claims the authorization layer depends on."""

    return jwt.decode(
        token,
        settings.jwt_secret,
        algorithms=[ALGORITHM],
        issuer=settings.jwt_issuer,
        options={"require": ["sub", "org", "role", "iat", "exp", "iss"]},
    )
