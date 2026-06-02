"""Local JWT mint/verify (HS256). OIDC verification lives in ``oidc.py``."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any

import jwt

from app.config import Settings
from app.core.errors import AuthError


def _now() -> int:
    return int(time.time())


@dataclass(frozen=True)
class IssuedToken:
    token: str
    jti: str
    expires_at: int


def _encode(settings: Settings, claims: dict[str, Any]) -> str:
    return jwt.encode(claims, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def create_access_token(
    settings: Settings, *, sub: str, org_id: str, roles: tuple[str, ...], email: str = ""
) -> IssuedToken:
    jti = uuid.uuid4().hex
    exp = _now() + settings.access_token_ttl_seconds
    claims = {
        "sub": sub, "org_id": org_id, "roles": list(roles), "email": email,
        "type": "access", "jti": jti, "iat": _now(), "exp": exp,
    }
    return IssuedToken(_encode(settings, claims), jti, exp)


def create_refresh_token(settings: Settings, *, sub: str, org_id: str) -> IssuedToken:
    jti = uuid.uuid4().hex
    exp = _now() + settings.refresh_token_ttl_seconds
    claims = {
        "sub": sub, "org_id": org_id, "type": "refresh",
        "jti": jti, "iat": _now(), "exp": exp,
    }
    return IssuedToken(_encode(settings, claims), jti, exp)


def decode_token(settings: Settings, token: str, *, expected_type: str | None = None) -> dict[str, Any]:
    try:
        claims = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    except jwt.ExpiredSignatureError as exc:
        raise AuthError("token expired") from exc
    except jwt.PyJWTError as exc:
        raise AuthError(f"invalid token: {exc}") from exc
    if expected_type and claims.get("type") != expected_type:
        raise AuthError(f"expected {expected_type} token")
    if not claims.get("org_id") or not claims.get("sub"):
        raise AuthError("token missing org_id/sub")
    return claims


__all__ = ["IssuedToken", "create_access_token", "create_refresh_token", "decode_token"]
