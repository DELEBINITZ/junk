"""Admin endpoints for tenant-local user and guardrail management."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict

from app.auth.dependencies import require_role, require_user
from app.auth.password import hash_password
from app.db.repository import DataStore, get_store
from app.domain import GuardrailConfig, Role, User
from app.rbac.permissions import can_configure_guardrails, can_manage_users


router = APIRouter(prefix="/admin", tags=["admin"])


class CreateUserRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    email: str
    name: str
    role: Role
    password: str
    organization_id: str | None = None


class PatchUserRequest(BaseModel):
    name: str | None = None
    role: Role | None = None
    is_active: bool | None = None
    organization_id: str | None = None


class UpdateGuardrailsRequest(BaseModel):
    hallucination_confidence_threshold: float = 0.7
    pii_redaction_enabled: bool = True
    blocked_keywords: list[str] = []
    require_citations: bool = True
    toxicity_threshold: float = 0.8


@router.get("/users")
def list_users(
    user: User = Depends(require_role("admin")),
    store: DataStore = Depends(get_store),
):
    """List users only inside the admin's organization."""

    return {
        "users": [
            _user_response(candidate)
            for candidate in store.users.values()
            if candidate.organization_id == user.organization_id
        ]
    }


@router.post("/users", status_code=status.HTTP_201_CREATED)
def create_user(
    payload: CreateUserRequest,
    user: User = Depends(require_role("admin")),
    store: DataStore = Depends(get_store),
):
    """Create a user in the admin's own organization."""

    if payload.organization_id is not None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="organization_id is derived from admin JWT and cannot be supplied",
        )
    if not can_manage_users(user, user.organization_id):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Cannot manage users")
    if payload.id in store.users or store.user_by_email(payload.email):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="User already exists")
    new_user = User(
        id=payload.id,
        organization_id=user.organization_id,
        email=payload.email,
        name=payload.name,
        role=payload.role,
        password_hash=hash_password(payload.password),
    )
    store.add_user(new_user)
    return _user_response(new_user)


@router.patch("/users/{user_id}")
def patch_user(
    user_id: str,
    payload: PatchUserRequest,
    user: User = Depends(require_role("admin")),
    store: DataStore = Depends(get_store),
):
    """Patch user attributes while preventing tenant reassignment."""

    if payload.organization_id is not None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="organization_id cannot be modified by request body",
        )
    target = store.users.get(user_id)
    if target is None or target.organization_id != user.organization_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    if payload.name is not None:
        target.name = payload.name
    if payload.role is not None:
        target.role = payload.role
    if payload.is_active is not None:
        target.is_active = payload.is_active
    return _user_response(target)


@router.get("/guardrails")
def get_guardrails(
    user: User = Depends(require_user),
    store: DataStore = Depends(get_store),
):
    """Return guardrail configuration for the caller's organization."""

    if not can_configure_guardrails(user, user.organization_id):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Cannot view guardrails")
    return store.guardrail_configs[user.organization_id]


@router.put("/guardrails")
def update_guardrails(
    payload: UpdateGuardrailsRequest,
    user: User = Depends(require_user),
    store: DataStore = Depends(get_store),
):
    """Replace guardrail configuration for the caller's organization."""

    if not can_configure_guardrails(user, user.organization_id):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Cannot configure guardrails")
    config = GuardrailConfig(
        organization_id=user.organization_id,
        hallucination_confidence_threshold=payload.hallucination_confidence_threshold,
        pii_redaction_enabled=payload.pii_redaction_enabled,
        blocked_keywords=payload.blocked_keywords,
        require_citations=payload.require_citations,
        toxicity_threshold=payload.toxicity_threshold,
    )
    store.guardrail_configs[user.organization_id] = config
    return config


def _user_response(user: User) -> dict[str, object]:
    return {
        "id": user.id,
        "organization_id": user.organization_id,
        "email": user.email,
        "name": user.name,
        "role": user.role,
        "is_active": user.is_active,
    }
