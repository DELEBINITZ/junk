"""Local JWT mint/verify (HS256). OIDC verification lives in ``oidc.py``.

================================ WHAT IS A JWT ============================
A JWT (JSON Web Token) is a small, SIGNED bundle of "claims" (facts) — here:
who you are (``sub``), your tenant (``org_id``), your ``roles``, when it expires
(``exp``), and a unique id (``jti``). The signature is computed with a secret
key, so the server can later VERIFY the token wasn't forged or tampered with: if
even one byte changed, the signature no longer matches and we reject it.

This file is the "local" identity provider — the zero-infrastructure default
used in dev / on-prem when no external IdP is wired. It does two jobs:
  * MINT tokens at login (``create_access_token`` / ``create_refresh_token``).
  * VERIFY a token on every request (``decode_token``).

Two token TYPES, a standard pattern:
  * ACCESS token  — short-lived, sent on every API call to prove identity.
  * REFRESH token — longer-lived, used ONLY to mint a fresh access token so the
    user doesn't have to log in again. Keeping access tokens short-lived limits
    the damage window if one leaks; the refresh token is presented far less often.

HS256 = a SYMMETRIC signature: the same ``jwt_secret`` both signs and verifies.
That is fine when one app owns identity. When an EXTERNAL IdP owns it, you switch
to OIDC (oidc.py), which uses ASYMMETRIC keys (the IdP signs with a private key,
we verify with its public key) so we never hold the signing secret.
===========================================================================
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any

import jwt

from app.config import Settings
from app.core.errors import AuthError


def _now() -> int:
    # Current UNIX time in WHOLE SECONDS — the unit JWT timestamps (iat/exp) use.
    return int(time.time())


@dataclass(frozen=True)
class IssuedToken:
    """One freshly minted token plus the metadata the auth layer needs without
    re-decoding it: the ``jti`` (so it can be tracked/revoked) and ``expires_at``
    (so a refresh-token store can set a matching TTL)."""
    token: str
    jti: str
    expires_at: int


def _encode(settings: Settings, claims: dict[str, Any]) -> str:
    # Sign the claims with the shared secret. This signature is exactly what makes
    # the token tamper-evident: change any claim and verification later fails.
    return jwt.encode(claims, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def create_access_token(
    settings: Settings, *, sub: str, org_id: str, roles: tuple[str, ...], email: str = ""
) -> IssuedToken:
    """Mint a short-lived ACCESS token carrying the caller's identity + roles.

    Note what goes INTO the token: ``org_id`` and ``roles`` are baked in here, at
    login, from the trusted user record — that is precisely why downstream code
    can trust them and must never re-read them from request input.
    """
    jti = uuid.uuid4().hex                          # unique token id => revocable + auditable
    exp = _now() + settings.access_token_ttl_seconds  # short TTL bounds the leak window
    claims = {
        "sub": sub, "org_id": org_id, "roles": list(roles), "email": email,
        # ``type`` lets verify() reject a refresh token used where an access token
        # is required (and vice-versa); ``iat`` is issued-at, ``exp`` is expiry.
        "type": "access", "jti": jti, "iat": _now(), "exp": exp,
    }
    return IssuedToken(_encode(settings, claims), jti, exp)


def create_refresh_token(settings: Settings, *, sub: str, org_id: str) -> IssuedToken:
    """Mint a longer-lived REFRESH token. Deliberately CARRIES NO ROLES: its only
    job is to obtain a new access token, at which point fresh roles are looked up
    again — so a role change isn't "frozen" into a long-lived credential."""
    jti = uuid.uuid4().hex
    exp = _now() + settings.refresh_token_ttl_seconds  # longer TTL than the access token
    claims = {
        "sub": sub, "org_id": org_id, "type": "refresh",
        "jti": jti, "iat": _now(), "exp": exp,
    }
    return IssuedToken(_encode(settings, claims), jti, exp)


def decode_token(settings: Settings, token: str, *, expected_type: str | None = None) -> dict[str, Any]:
    """VERIFY a token and return its claims, or raise AuthError. This is the gate
    every local-auth request passes through.

    ``jwt.decode`` does the cryptographic heavy lifting: it checks the signature
    (forgery/tamper) AND the ``exp`` expiry, raising if either fails. We then add
    two application-level checks on top:
      * the token is of the EXPECTED type (don't accept a refresh token as access);
      * it actually carries the tenant + subject we require (``org_id`` / ``sub``).
    Every failure is normalized to AuthError so callers handle auth uniformly.
    """
    try:
        claims = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    except jwt.ExpiredSignatureError as exc:        # signature was valid but the token aged out
        raise AuthError("token expired") from exc
    except jwt.PyJWTError as exc:                    # bad signature / malformed / etc.
        raise AuthError(f"invalid token: {exc}") from exc
    if expected_type and claims.get("type") != expected_type:
        raise AuthError(f"expected {expected_type} token")
    if not claims.get("org_id") or not claims.get("sub"):
        # No tenant/subject => we cannot establish a trustworthy identity. Refuse.
        raise AuthError("token missing org_id/sub")
    return claims


__all__ = ["IssuedToken", "create_access_token", "create_refresh_token", "decode_token"]
