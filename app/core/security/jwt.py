"""JWT verify/mint (HS256) + the org-scoped MCP service token.

================================ WHAT IS A JWT ============================
A JWT (JSON Web Token) is a small, SIGNED bundle of "claims" (facts) — here:
who you are (``sub``), your tenant (``org_id``), your ``roles``, when it expires
(``exp``), and a unique id (``jti``). The signature is computed with a secret
key, so the server can later VERIFY the token wasn't forged or tampered with: if
even one byte changed, the signature no longer matches and we reject it.

The auth model: a JWT carries the caller's identity on every request and this
service VERIFIES it (``decode_token``); the token itself is minted UPSTREAM (a
gateway / IdP). ``create_access_token`` is kept as a convenience for dev/tests to
mint a valid ACCESS token locally — the only token type used for auth.

HS256 = a SYMMETRIC signature: the same ``jwt_secret`` both signs and verifies.
This file ALSO mints the short-lived, org-scoped SERVICE token used for remote MCP
calls (``create_service_token``) — see core/mcp for how identity crosses that wire.
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


def _service_keys(settings: Settings) -> tuple[str, str, str]:
    """Resolve ``(sign_key, verify_key, algorithm)`` for SERVICE tokens.

    ASYMMETRIC when a keypair is configured: the CORE signs with the private key,
    and a remote MCP server verifies with the PUBLIC key only — so a promoted MCP
    server holds no signing key and CANNOT mint tokens for any org. This closes the
    "any promoted server can forge admin tokens" hole a shared symmetric secret
    leaves open.

    Falls back to the shared HS256 secret when NO service keypair is set — acceptable
    for the in-process/dev path (no token crosses a process boundary), but the prod
    guard REQUIRES the keypair once a remote MCP server is configured."""
    priv = getattr(settings, "jwt_service_private_key", "") or ""
    pub = getattr(settings, "jwt_service_public_key", "") or ""
    if priv or pub:
        return priv, pub, getattr(settings, "jwt_service_algorithm", "RS256")
    return settings.jwt_secret, settings.jwt_secret, settings.jwt_algorithm


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


def create_service_token(
    settings: Settings, *, sub: str, org_id: str, roles: tuple[str, ...],
    audience: str, ttl_seconds: int,
) -> str:
    """Mint a SHORT-LIVED, org-scoped SERVICE token for a remote MCP call.

    When a module is promoted to its own MCP server, the agent must prove WHO is
    calling and FOR WHICH ORG — but we never put the org in the tool arguments
    (an attacker could change those). Instead we mint this token from the trusted
    local identity and send it in the Authorization header; the remote server
    re-derives org/roles from it (same rule, across the wire). ``audience`` names
    the intended server (e.g. "easm-mcp") so a token can't be replayed elsewhere;
    the TTL is tiny because it's used immediately for one hop.

    SECURITY: this is a DISTINCT token ``type="service"`` (not "access"), signed with
    the SERVICE key (asymmetric private key when configured — see ``_service_keys``).
    Two consequences: (1) the main API verifies ``expected_type="access"`` so a
    service token is rejected there — it can't be replayed as a user token; (2) when
    a keypair is configured, an MCP server holding only the public key cannot mint
    tokens. Verified by ``decode_service_token`` on the server."""
    sign_key, _verify, alg = _service_keys(settings)
    exp = _now() + ttl_seconds
    claims = {
        "sub": sub, "org_id": org_id, "roles": list(roles), "email": "",
        "type": "service", "aud": audience, "jti": uuid.uuid4().hex,
        "iat": _now(), "exp": exp,
    }
    return jwt.encode(claims, sign_key, algorithm=alg)


def decode_token(
    settings: Settings, token: str, *,
    expected_type: str | None = None, expected_audience: str | None = None,
) -> dict[str, Any]:
    """VERIFY a token and return its claims, or raise AuthError. This is the gate
    every local-auth request passes through.

    ``jwt.decode`` does the cryptographic heavy lifting: it checks the signature
    (forgery/tamper) AND the ``exp`` expiry, raising if either fails. We then add
    application-level checks on top:
      * the token is of the EXPECTED type (don't accept a refresh token as access);
      * (for service tokens) the ``aud`` matches the server's EXPECTED AUDIENCE, so
        a token minted for one MCP server cannot be replayed against another — PyJWT
        does NOT enforce ``aud`` unless asked, so we verify it explicitly here;
      * it actually carries the tenant + subject we require (``org_id`` / ``sub``).
    Every failure is normalized to AuthError so callers handle auth uniformly, and
    the message stays generic so we never echo token internals back to a caller.
    """
    try:
        # verify_aud=False: PyJWT would otherwise REJECT any token that carries an
        # ``aud`` claim (every service token does) whenever no audience is passed —
        # so we disable its automatic audience check and enforce ``aud`` ourselves
        # below via ``expected_audience`` (None => skip, matching the main API path).
        claims = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm],
                            options={"verify_aud": False})
    except jwt.ExpiredSignatureError as exc:        # signature was valid but the token aged out
        raise AuthError("token expired") from exc
    except jwt.PyJWTError as exc:                    # bad signature / malformed / etc.
        # Keep the message generic (don't leak which check failed); the original
        # exception is still chained via ``from exc`` for server-side debugging.
        raise AuthError("invalid token") from exc
    if expected_type and claims.get("type") != expected_type:
        raise AuthError(f"expected {expected_type} token")
    if expected_audience is not None and claims.get("aud") != expected_audience:
        # Audience mismatch => this token was minted for a DIFFERENT server. Refuse,
        # so a valid service token for module A can't be replayed against module B.
        raise AuthError("token audience mismatch")
    if not claims.get("org_id") or not claims.get("sub"):
        # No tenant/subject => we cannot establish a trustworthy identity. Refuse.
        raise AuthError("token missing org_id/sub")
    return claims


def decode_service_token(
    settings: Settings, token: str, *, expected_audience: str | None = None
) -> dict[str, Any]:
    """VERIFY a SERVICE token on the remote MCP server side, or raise AuthError.

    Verifies with the SERVICE verify key (the PUBLIC key when an asymmetric keypair
    is configured — so this process cannot MINT tokens, only check them) and the
    service algorithm. Requires ``type="service"`` (a user/access token is rejected
    here) and, when given, the exact ``aud`` (so a token minted for another module's
    server can't be replayed). This is the cross-process counterpart to
    ``create_service_token`` and is what ``mcp/server.py`` calls."""
    _sign, verify_key, alg = _service_keys(settings)
    try:
        claims = jwt.decode(token, verify_key, algorithms=[alg], options={"verify_aud": False})
    except jwt.ExpiredSignatureError as exc:
        raise AuthError("token expired") from exc
    except jwt.PyJWTError as exc:
        raise AuthError("invalid token") from exc
    if claims.get("type") != "service":
        # A user/access token (or anything not minted as a service token) must not
        # authenticate a tool call on an MCP server.
        raise AuthError("expected service token")
    if expected_audience is not None and claims.get("aud") != expected_audience:
        raise AuthError("token audience mismatch")
    if not claims.get("org_id") or not claims.get("sub"):
        raise AuthError("token missing org_id/sub")
    return claims


__all__ = ["IssuedToken", "create_access_token", "create_service_token",
           "decode_token", "decode_service_token"]
