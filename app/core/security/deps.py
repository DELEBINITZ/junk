"""FastAPI auth dependencies — where a raw HTTP request becomes a trusted identity.

================================ THE AUTH MODEL ===========================
ONE mechanism, two parts, on every protected request:

  1. API KEY (gateway) — an ``X-API-Key`` header (or ``?api_key=`` for SSE) must
     match a configured key. This authenticates that the request comes from a
     known caller/gateway at all. No key => 401, nothing else runs.
  2. JWT (identity)   — a Bearer token (or ``?access_token=`` / cookie) carries the
     caller's identity. We verify its signature + expiry and read org_id / user_id
     / roles STRAIGHT FROM ITS CLAIMS. Those claims are the trusted tenant + role
     identity that flows into every ToolContext and powers tenant isolation.

The JWT is minted upstream (a gateway / IdP / the dev mint helper in jwt.py); this
service only VERIFIES it. There is no login/password/OIDC/refresh flow here — that
was removed deliberately to keep auth simple and unambiguous.

The token may arrive three ways because a browser ``EventSource`` (SSE) cannot set
an Authorization header:
  1. ``Authorization: Bearer <token>``     (normal API calls)
  2. ``?access_token=<token>``              (SSE / EventSource)
  3. ``access_token`` cookie               (browser sessions)
===========================================================================
"""

from __future__ import annotations

from fastapi import Request

from app.config import Settings, get_settings
from app.core.errors import AuthError
from app.core.security.context import SecurityContext
from app.core.security.jwt import decode_token


def extract_bearer_token(request: Request) -> str | None:
    """Pull the raw JWT string off the request, trying the three transports in
    priority order. Returns ``None`` if none carried a token (caller then 401s).

    This only LOCATES the token; it does not trust it at all yet — verification
    happens later in ``build_security_context``."""
    auth = request.headers.get("authorization") or request.headers.get("Authorization")
    if auth and auth.lower().startswith("bearer "):
        return auth[7:].strip()                    # strip the "Bearer " prefix (7 chars)
    qp = request.query_params.get("access_token")  # SSE/EventSource can't set headers
    if qp:
        return qp
    cookie = request.cookies.get("access_token")   # browser session cookie
    if cookie:
        return cookie
    return None


def _api_key_from(request: Request) -> str:
    # The key may ride a header (normal calls) or the query string (SSE can't set
    # headers). Header lookup is case-insensitive in Starlette.
    return (request.headers.get("x-api-key") or request.query_params.get("api_key") or "")


def _verify_api_key(request: Request, settings: Settings) -> None:
    """Gateway check: the request must carry a configured API key, or it is refused
    before any identity work happens."""
    key = _api_key_from(request)
    if not settings.api_keys or key not in set(settings.api_keys):
        raise AuthError("invalid or missing API key")


def build_security_context(request: Request) -> SecurityContext:
    """Turn the incoming request into a verified SecurityContext, or raise.

    THE choke point where identity is established. After it returns, org/roles are
    trustworthy; before it, nothing about the caller is trusted. Two gates, in
    order: the API key authenticates the request, then the JWT supplies identity.
    """
    settings: Settings = getattr(request.app.state, "settings", None) or get_settings()

    # 1. Gateway API key — authenticate the request itself.
    _verify_api_key(request, settings)

    # 2. JWT — verify signature + expiry and require it be an ACCESS token (a refresh
    #    token, if one is ever presented, must not authenticate a request).
    token = extract_bearer_token(request)
    if not token:
        raise AuthError("missing bearer token")
    claims = decode_token(settings, token, expected_type="access")
    # Build the trusted identity STRAIGHT FROM THE VERIFIED CLAIMS — org_id, sub, and
    # roles come only from the token, never from request body/query/path/headers.
    # (Default to ``viewer`` if a token somehow carried no roles: least privilege.)
    return SecurityContext(
        org_id=str(claims["org_id"]),
        user_id=str(claims["sub"]),
        roles=tuple(str(r) for r in claims.get("roles", []) or ("viewer",)),
        email=str(claims.get("email", "")),
        token_id=str(claims.get("jti", "")),
        claims=claims,
    )


# FastAPI dependency callables ------------------------------------------------
# These are what routes actually depend on. ``require_user`` = "valid API key AND
# a valid JWT"; ``require_role(...)`` = that PLUS holding at least the given role.
async def require_user(request: Request) -> SecurityContext:
    """Dependency for any authenticated endpoint: succeeds with the SecurityContext,
    or raises (FastAPI turns the AuthError into a 401)."""
    return build_security_context(request)


def require_role(minimum: str):
    """Dependency FACTORY for role-gated endpoints. Call it with the MINIMUM role
    an endpoint needs (e.g. ``Depends(require_role("admin"))``); it returns the
    actual dependency. RBAC is ordered, so a higher role (admin) also satisfies a
    lower requirement (analyst) — ``has_role`` encodes that floor check.

    NOTE this is API-level coarse gating. Per-TOOL RBAC is enforced again deeper
    in, at the MCP boundary, so privilege is checked both at the door and at the
    point of use (defense in depth)."""
    async def _dep(request: Request) -> SecurityContext:
        sc = build_security_context(request)        # authenticate first
        if not sc.has_role(minimum):                # then authorize (role floor)
            from app.core.errors import PermissionDenied

            raise PermissionDenied(
                f"role '{minimum}' required",
                details={"have": list(sc.roles), "need": minimum},   # helpful 403 body
            )
        return sc

    return _dep


__all__ = [
    "extract_bearer_token",
    "build_security_context",
    "require_user",
    "require_role",
]
