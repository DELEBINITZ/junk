"""FastAPI auth dependencies.

Token extraction supports three transports because browser ``EventSource`` (SSE)
cannot send an Authorization header:
  1. ``Authorization: Bearer <token>``     (normal API calls)
  2. ``?access_token=<token>``              (SSE / EventSource)
  3. ``access_token`` cookie               (browser sessions)
"""

from __future__ import annotations

from fastapi import Request

from app.config import Settings, get_settings
from app.core.errors import AuthError
from app.core.security.context import SecurityContext
from app.core.security.jwt import decode_token
from app.core.security.tokens import RevocationStore, get_default_revocation_store


def extract_bearer_token(request: Request) -> str | None:
    auth = request.headers.get("authorization") or request.headers.get("Authorization")
    if auth and auth.lower().startswith("bearer "):
        return auth[7:].strip()
    qp = request.query_params.get("access_token")
    if qp:
        return qp
    cookie = request.cookies.get("access_token")
    if cookie:
        return cookie
    return None


def _revocation_store(request: Request, settings: Settings) -> RevocationStore:
    store = getattr(request.app.state, "revocation_store", None)
    return store or get_default_revocation_store()


def build_security_context(request: Request) -> SecurityContext:
    settings: Settings = getattr(request.app.state, "settings", None) or get_settings()
    token = extract_bearer_token(request)
    if not token:
        raise AuthError("missing bearer token")

    if settings.auth_provider == "oidc":
        verifier = getattr(request.app.state, "oidc_verifier", None)
        if verifier is None:
            from app.core.security.oidc import get_oidc_verifier

            verifier = get_oidc_verifier(settings)
        return verifier.verify(token)

    # local JWT
    claims = decode_token(settings, token, expected_type="access")
    jti = str(claims.get("jti", ""))
    if jti and _revocation_store(request, settings).is_revoked(jti):
        raise AuthError("token revoked")
    return SecurityContext(
        org_id=str(claims["org_id"]),
        user_id=str(claims["sub"]),
        roles=tuple(str(r) for r in claims.get("roles", []) or ("viewer",)),
        email=str(claims.get("email", "")),
        token_id=jti,
        claims=claims,
    )


# FastAPI dependency callables ------------------------------------------------
async def require_user(request: Request) -> SecurityContext:
    return build_security_context(request)


def require_role(minimum: str):
    async def _dep(request: Request) -> SecurityContext:
        sc = build_security_context(request)
        if not sc.has_role(minimum):
            from app.core.errors import PermissionDenied

            raise PermissionDenied(
                f"role '{minimum}' required",
                details={"have": list(sc.roles), "need": minimum},
            )
        return sc

    return _dep


__all__ = [
    "extract_bearer_token",
    "build_security_context",
    "require_user",
    "require_role",
]
