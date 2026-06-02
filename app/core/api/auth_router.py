"""Auth endpoints: local login + refresh (rotation) + logout (revocation).

This router issues and manages the JWTs the rest of the API trusts. The model is
the standard two-token scheme:
  * ACCESS token  — short-lived, sent on every request; carries the identity
    (sub/org/roles) that becomes the SecurityContext and drives RBAC + tenant
    isolation everywhere downstream.
  * REFRESH token — longer-lived, used only to mint a fresh access token. On
    refresh we ROTATE it (issue a new one and revoke the old) so a leaked refresh
    token has a short useful life. Logout REVOKES the current token.

A FastAPI ROUTER groups related endpoints under a shared prefix (here
``/v1/auth``); it's mounted onto the app at startup.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends, Request

from app.core.api.deps import get_services, require_user
from app.core.api.schemas import LoginRequest, RefreshRequest, TokenResponse, UserInfo
from app.core.errors import AuthError
from app.core.security.context import SecurityContext
from app.core.security.jwt import create_access_token, create_refresh_token, decode_token
from app.core.security.passwords import verify_password

router = APIRouter(prefix="/v1/auth", tags=["auth"])


@router.post("/login", response_model=TokenResponse)
async def login(body: LoginRequest, request: Request) -> TokenResponse:
    """Local email/password login -> a fresh access+refresh token pair.

    Only available when ``auth_provider=local`` (otherwise the deployment uses an
    external OIDC provider and this path is disabled). The credential check is
    deliberately VAGUE on failure — "invalid email or password" for missing user,
    disabled account, or wrong password alike — so an attacker can't probe which
    emails exist. The issued access token embeds org_id + roles, which is the
    seed of all later authorization."""
    services = get_services(request)
    settings = services.settings
    if settings.auth_provider != "local":
        raise AuthError("local login disabled; authenticate via the configured OIDC provider")
    user = services.user_store.get_by_email(body.email)
    # verify_password compares against a hash (never a plaintext password); the
    # combined check avoids leaking which specific condition failed.
    if not user or user.disabled or not verify_password(body.password, user.password_hash):
        raise AuthError("invalid email or password")
    access = create_access_token(settings, sub=user.user_id, org_id=user.org_id,
                                 roles=user.roles, email=user.email)
    refresh = create_refresh_token(settings, sub=user.user_id, org_id=user.org_id)
    await services.audit.record(org_id=user.org_id, user_id=user.user_id, event="login")
    return TokenResponse(
        access_token=access.token, refresh_token=refresh.token,
        expires_in=settings.access_token_ttl_seconds,
        user=UserInfo(id=user.user_id, email=user.email, org_id=user.org_id, roles=list(user.roles)),
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh(body: RefreshRequest, request: Request) -> TokenResponse:
    """Exchange a valid refresh token for a NEW access+refresh pair, rotating the
    refresh token in the process.

    Flow: verify the token is genuine AND of type "refresh" (decode_token rejects
    an access token used here); reject it if it's already been revoked (replay
    defense); then re-read the user so the new token reflects current roles. The
    ``jti`` is the token's unique id — we revoke the OLD one so a refresh token is
    single-use (rotation). Roles freshly come from the user store, never from the
    incoming token, so a role change takes effect on the next refresh."""
    services = get_services(request)
    settings = services.settings
    claims = decode_token(settings, body.refresh_token, expected_type="refresh")
    jti = str(claims.get("jti", ""))
    if jti and services.revocation_store.is_revoked(jti):
        raise AuthError("refresh token revoked")
    user = services.user_store.get_by_id(str(claims["sub"]))
    # Fall back to least-privilege ("viewer") if the user record vanished.
    roles = tuple(user.roles) if user else ("viewer",)
    email = user.email if user else ""
    if jti:  # rotate
        # Revoke the presented refresh token (until its original expiry) so it
        # can't be reused — this is what makes rotation single-use.
        services.revocation_store.revoke(jti, int(claims.get("exp", time.time())))
    access = create_access_token(settings, sub=str(claims["sub"]), org_id=str(claims["org_id"]),
                                 roles=roles, email=email)
    new_refresh = create_refresh_token(settings, sub=str(claims["sub"]), org_id=str(claims["org_id"]))
    return TokenResponse(
        access_token=access.token, refresh_token=new_refresh.token,
        expires_in=settings.access_token_ttl_seconds,
        user=UserInfo(id=str(claims["sub"]), email=email, org_id=str(claims["org_id"]), roles=list(roles)),
    )


@router.post("/logout")
async def logout(request: Request, sc: SecurityContext = Depends(require_user)) -> dict:
    """Revoke the caller's CURRENT access token so it can't be reused before its
    natural expiry. ``Depends(require_user)`` both authenticates the caller and
    hands us the verified SecurityContext (so we know which token id to revoke).
    We revoke until now + the access-token TTL — past that it would expire anyway."""
    services = get_services(request)
    if sc.token_id:
        services.revocation_store.revoke(
            sc.token_id, int(time.time()) + services.settings.access_token_ttl_seconds
        )
    await services.audit.record(org_id=sc.org_id, user_id=sc.user_id, event="logout")
    return {"status": "logged_out"}


@router.get("/me", response_model=UserInfo)
async def me(sc: SecurityContext = Depends(require_user)) -> UserInfo:
    """Echo back the authenticated caller's identity. Pure projection of the
    already-verified SecurityContext — useful for a client to confirm "who am I
    and what roles do I hold?" without decoding the token itself."""
    return UserInfo(id=sc.user_id, email=sc.email, org_id=sc.org_id, roles=list(sc.roles))


__all__ = ["router"]
