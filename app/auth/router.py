"""Authentication endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

import jwt

from app.auth.jwt import (
    create_access_token,
    create_refresh_token,
    decode_refresh_token,
    read_jti_ignoring_expiry,
)
from app.auth.password import verify_password
from app.core.security.tokens import get_revocation_store
from app.db.repository import DataStore, get_store
from app.observability.logging import safe_extra


router = APIRouter(prefix="/auth", tags=["auth"])
logger = logging.getLogger(__name__)


class LoginRequest(BaseModel):
    email: str
    password: str


@router.post("/login")
def login(payload: LoginRequest, store: DataStore = Depends(get_store)):
    """Authenticate demo users and return a JWT used by all protected routes."""

    user = store.user_by_email(payload.email)
    if user is None or not verify_password(payload.password, user.password_hash):
        logger.warning(
            "auth.login.failed",
            extra=safe_extra(email=payload.email, reason="invalid_credentials"),
        )
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    logger.info(
        "auth.login.success",
        extra=safe_extra(user_id=user.id, organization_id=user.organization_id, role=user.role),
    )
    return {
        "access_token": create_access_token(user),
        "refresh_token": create_refresh_token(user),
        "token_type": "bearer",
        "user": {
            "id": user.id,
            "email": user.email,
            "organization_id": user.organization_id,
            "role": user.role,
        },
    }


class RefreshRequest(BaseModel):
    refresh_token: str


class LogoutRequest(BaseModel):
    token: str


@router.post("/refresh")
def refresh(payload: RefreshRequest, store: DataStore = Depends(get_store)):
    """Exchange a valid refresh token for a new access token."""

    try:
        claims = decode_refresh_token(payload.refresh_token)
    except jwt.PyJWTError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token") from exc
    if get_revocation_store().is_revoked(claims.get("jti")):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Refresh token revoked")
    user = store.get_user(claims["sub"])
    if user is None or not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Inactive or unknown user")
    return {"access_token": create_access_token(user), "token_type": "bearer"}


@router.post("/logout")
def logout(payload: LogoutRequest):
    """Revoke a token by jti (access or refresh). Idempotent."""

    jti = read_jti_ignoring_expiry(payload.token)
    get_revocation_store().revoke(jti)
    return {"revoked": bool(jti)}
