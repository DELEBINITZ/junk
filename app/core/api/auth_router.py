"""Auth endpoints: local login + refresh (rotation) + logout (revocation)."""

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
    services = get_services(request)
    settings = services.settings
    if settings.auth_provider != "local":
        raise AuthError("local login disabled; authenticate via the configured OIDC provider")
    user = services.user_store.get_by_email(body.email)
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
    services = get_services(request)
    settings = services.settings
    claims = decode_token(settings, body.refresh_token, expected_type="refresh")
    jti = str(claims.get("jti", ""))
    if jti and services.revocation_store.is_revoked(jti):
        raise AuthError("refresh token revoked")
    user = services.user_store.get_by_id(str(claims["sub"]))
    roles = tuple(user.roles) if user else ("viewer",)
    email = user.email if user else ""
    if jti:  # rotate
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
    services = get_services(request)
    if sc.token_id:
        services.revocation_store.revoke(
            sc.token_id, int(time.time()) + services.settings.access_token_ttl_seconds
        )
    await services.audit.record(org_id=sc.org_id, user_id=sc.user_id, event="logout")
    return {"status": "logged_out"}


@router.get("/me", response_model=UserInfo)
async def me(sc: SecurityContext = Depends(require_user)) -> UserInfo:
    return UserInfo(id=sc.user_id, email=sc.email, org_id=sc.org_id, roles=list(sc.roles))


__all__ = ["router"]
