"""Typed application errors with HTTP status mapping.

Internal code raises these; the API layer (``app/core/api/errors.py``) converts
them to JSON problem responses. Tools never raise — they return ``ToolError``.
"""

from __future__ import annotations

from typing import Any


class AppError(Exception):
    """Base for all expected application errors."""

    status_code: int = 500
    code: str = "internal_error"

    def __init__(self, message: str = "", *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message or self.code)
        self.message = message or self.code
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        return {"error": {"code": self.code, "message": self.message, "details": self.details}}


class ConfigError(AppError):
    status_code = 500
    code = "config_error"


class AuthError(AppError):
    status_code = 401
    code = "unauthorized"


class TokenRevokedError(AuthError):
    code = "token_revoked"


class PermissionDenied(AppError):
    status_code = 403
    code = "forbidden"


class NotFound(AppError):
    status_code = 404
    code = "not_found"


class GuardrailBlocked(AppError):
    status_code = 422
    code = "guardrail_blocked"


class RateLimited(AppError):
    status_code = 429
    code = "rate_limited"


class Overloaded(AppError):
    status_code = 503
    code = "overloaded"


class RegistryError(AppError):
    status_code = 500
    code = "registry_error"


class UpstreamError(AppError):
    status_code = 502
    code = "upstream_error"


__all__ = [
    "AppError",
    "ConfigError",
    "AuthError",
    "TokenRevokedError",
    "PermissionDenied",
    "NotFound",
    "GuardrailBlocked",
    "RateLimited",
    "Overloaded",
    "RegistryError",
    "UpstreamError",
]
