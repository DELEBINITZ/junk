"""API-layer dependencies: fetch wired services + re-export auth deps.

FastAPI DEPENDENCIES are small callables a route declares via ``Depends(...)``;
FastAPI runs them before the handler and injects their result as an argument.
This module is the routers' one-stop shop for two of them:
  * ``get_services`` — reach the singleton service bundle built at startup.
  * ``require_user`` / ``require_role`` — auth guards, re-exported from the
    security package so routers can import everything they need from one place
    (the actual token verification lives in app.core.security.deps).
"""

from __future__ import annotations

from fastapi import Request

from app.core.security.deps import require_role, require_user  # re-exported for routers


def get_services(request: Request):
    """Return the application's wired service bundle (orchestrator, stores, metrics,
    registry, ...). It is constructed ONCE at startup and stashed on
    ``app.state.services``; handlers fetch it here rather than building anything
    per-request. If it's missing, the app didn't initialize properly — surface a
    clear AppError (mapped to a JSON error by the error handlers) instead of an
    opaque AttributeError. AppError is imported lazily to keep this module light."""
    services = getattr(request.app.state, "services", None)
    if services is None:
        from app.core.errors import AppError

        raise AppError("services not initialized")
    return services


__all__ = ["get_services", "require_user", "require_role"]
