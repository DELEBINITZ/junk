"""FastAPI auth dependencies — where a raw HTTP request becomes a trusted identity.

================================ WHERE THIS FITS ==========================
This is the BRIDGE between "an untrusted HTTP request arrived" and "we have a
verified SecurityContext". A FastAPI "dependency" is a function the framework
runs before your route handler; declaring ``sc: SecurityContext = Depends(...)``
on an endpoint means "authenticate first, then call me with the result". So
every protected route reuses the same vetted auth path instead of re-checking
tokens by hand.

The flow per request:
  extract the token (3 transports) -> verify it (OIDC or local JWT) -> for local
  JWT also reject it if it's on the revocation deny-list -> build SecurityContext.

Token extraction supports three transports because browser ``EventSource`` (SSE)
cannot send an Authorization header:
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
from app.core.security.tokens import RevocationStore, get_default_revocation_store


def extract_bearer_token(request: Request) -> str | None:
    """Pull the raw token string off the request, trying the three transports in
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


def _revocation_store(request: Request, settings: Settings) -> RevocationStore:
    # Use the app's configured store (wired at startup, e.g. Redis) if present,
    # else the in-memory default. Read off ``app.state`` so tests/replicas can
    # inject their own without touching this code.
    store = getattr(request.app.state, "revocation_store", None)
    return store or get_default_revocation_store()


# --- API-key auth (AUTH_PROVIDER=apikey) — TESTING / trusted-gateway only ------
def _api_key_from(request: Request) -> str:
    # The key may ride a header (normal calls) or the query string (SSE can't set
    # headers). Header lookup is case-insensitive in Starlette.
    return (request.headers.get("x-api-key") or request.query_params.get("api_key") or "")


def _verify_api_key(request: Request, settings: Settings) -> None:
    key = _api_key_from(request)
    if not settings.api_keys or key not in set(settings.api_keys):
        raise AuthError("invalid or missing API key")


def _roles_tuple(roles, request: Request, settings: Settings) -> tuple[str, ...]:
    if roles:
        return tuple(str(r) for r in roles)
    raw = request.headers.get("x-roles") or request.query_params.get("roles") or ""
    parsed = tuple(r.strip() for r in str(raw).split(",") if r.strip())
    return parsed or tuple(settings.apikey_default_roles) or ("viewer",)


def apikey_context(request: Request, settings: Settings, *,
                   org_id=None, user_id=None, roles=None) -> SecurityContext:
    """Authenticate with the API key, then build the identity from explicit params
    (chat body), falling back to X-Org-Id/X-User-Id/X-Roles headers or the query
    string. SECURITY: this TRUSTS the caller for org/user — whoever holds a key can
    set any org. Testing / trusted-backend only (the prod guard forbids apikey)."""
    _verify_api_key(request, settings)
    org = org_id or request.headers.get("x-org-id") or request.query_params.get("org_id")
    user = user_id or request.headers.get("x-user-id") or request.query_params.get("user_id")
    if not org or not user:
        raise AuthError("apikey mode requires org_id and user_id (in the chat body, "
                        "X-Org-Id/X-User-Id headers, or query string)")
    return SecurityContext(org_id=str(org), user_id=str(user),
                           roles=_roles_tuple(roles, request, settings), email="")


def build_security_context(request: Request) -> SecurityContext:
    """Turn the incoming request into a verified SecurityContext, or raise.

    This is THE choke point where identity is established. After this returns,
    org/roles are trustworthy; before it, nothing about the caller is trusted.
    It branches on the configured provider but both branches end at the same
    SecurityContext shape."""
    settings: Settings = getattr(request.app.state, "settings", None) or get_settings()

    if settings.auth_provider == "apikey":
        # API key authenticates; identity from headers/query (testing/gateway).
        return apikey_context(request, settings)

    token = extract_bearer_token(request)
    if not token:
        raise AuthError("missing bearer token")     # no credential at all -> 401

    if settings.auth_provider == "oidc":
        # Production path: an external IdP signed the token; verify it against the
        # IdP's JWKS public keys (see oidc.py). The verifier itself produces the
        # SecurityContext, so revocation isn't done here — the IdP owns lifecycle.
        verifier = getattr(request.app.state, "oidc_verifier", None)
        if verifier is None:
            from app.core.security.oidc import get_oidc_verifier

            verifier = get_oidc_verifier(settings)
        return verifier.verify(token)

    # local JWT (zero-infra default). Verify signature+expiry and require it be an
    # ACCESS token (a refresh token must not be usable to authenticate a request).
    claims = decode_token(settings, token, expected_type="access")
    jti = str(claims.get("jti", ""))
    # Revocation check: a signature-valid token can still be DENIED if it was
    # logged out / rotated. This is the stateful override on top of stateless JWT.
    if jti and _revocation_store(request, settings).is_revoked(jti):
        raise AuthError("token revoked")
    # Build the trusted identity STRAIGHT FROM THE VERIFIED CLAIMS — org_id, sub,
    # and roles come only from the token, never from request body/query/path.
    # (Default to ``viewer`` if a token somehow carried no roles: least privilege.)
    return SecurityContext(
        org_id=str(claims["org_id"]),
        user_id=str(claims["sub"]),
        roles=tuple(str(r) for r in claims.get("roles", []) or ("viewer",)),
        email=str(claims.get("email", "")),
        token_id=jti,
        claims=claims,
    )


# FastAPI dependency callables ------------------------------------------------
# These are what routes actually depend on. ``require_user`` = "must be logged
# in"; ``require_role(...)`` = "must be logged in AND hold at least this role".
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


def resolve_identity(request: Request, *, org_id=None, user_id=None, roles=None) -> SecurityContext:
    """Identity for the CHAT endpoints, where (in apikey mode) the user params are
    allowed to come from the request BODY. In apikey mode the explicit args win
    (then headers/query); in token/oidc mode the args are ignored and identity is
    taken from the verified token, exactly as everywhere else."""
    settings: Settings = getattr(request.app.state, "settings", None) or get_settings()
    if settings.auth_provider == "apikey":
        return apikey_context(request, settings, org_id=org_id, user_id=user_id, roles=roles)
    return build_security_context(request)


__all__ = [
    "extract_bearer_token",
    "build_security_context",
    "apikey_context",
    "resolve_identity",
    "require_user",
    "require_role",
]
