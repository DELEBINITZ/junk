"""OIDC token verification (production auth, plan §8.1).

Verifies an IdP-issued JWT against the provider's JWKS (RS256) and maps standard
claims to our `User`. Active when AUTH_PROVIDER=oidc. `PyJWKClient` (PyJWT +
cryptography, in the `prod` extra) is imported lazily, so the default local
HS256 path needs no extra dependency.

Org and role come from configurable claims; do not build your own user store —
the IdP is the source of identity.
"""

from __future__ import annotations

from typing import Any

import jwt

from app.config import settings
from app.domain import User


_jwks_client = None


def _client():
    global _jwks_client
    if _jwks_client is None:
        from jwt import PyJWKClient  # lazy (needs cryptography)

        _jwks_client = PyJWKClient(settings.oidc_jwks_url)
    return _jwks_client


def verify_oidc_token(token: str) -> dict[str, Any]:
    signing_key = _client().get_signing_key_from_jwt(token)
    return jwt.decode(
        token,
        signing_key.key,
        algorithms=["RS256"],
        audience=settings.oidc_audience or None,
        issuer=settings.oidc_issuer or None,
        options={"verify_aud": bool(settings.oidc_audience)},
    )


def user_from_claims(claims: dict[str, Any]) -> User:
    org = claims.get("org") or claims.get("organization_id") or claims.get("tenant_id")
    role = claims.get("role")
    if not role:
        roles = claims.get("roles") or []
        role = roles[0] if roles else "viewer"
    return User(
        id=str(claims["sub"]),
        organization_id=str(org),
        email=claims.get("email", ""),
        name=claims.get("name", ""),
        role=role,
        password_hash="",
        is_active=True,
    )


def reset_jwks_client() -> None:
    global _jwks_client
    _jwks_client = None
