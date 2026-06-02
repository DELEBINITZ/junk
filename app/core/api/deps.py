"""API-layer dependencies: fetch wired services + re-export auth deps."""

from __future__ import annotations

from fastapi import Request

from app.core.security.deps import require_role, require_user  # re-exported for routers


def get_services(request: Request):
    services = getattr(request.app.state, "services", None)
    if services is None:
        from app.core.errors import AppError

        raise AppError("services not initialized")
    return services


__all__ = ["get_services", "require_user", "require_role"]
