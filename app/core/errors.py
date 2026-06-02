"""Typed application errors with HTTP status mapping.

WHY THIS EXISTS: instead of scattering ``raise HTTPException(403, ...)`` through
the codebase, internal code raises a SEMANTIC error (``PermissionDenied``) and the
API layer (``app/core/api/errors.py``) translates it once into a uniform JSON
problem response. Each class carries its own ``status_code`` + machine-readable
``code``, so the HTTP contract lives in one place and the business logic stays
transport-agnostic (the same errors work for a CLI, a worker, or tests).

IMPORTANT CONTRAST: these are for the REQUEST path. Inside the agent, TOOLS never
raise — they return a ``ToolError`` value (the "errors-as-data" rule from
contracts.py) so one bad tool can't crash a turn. Raising vs returning is the line
between "this request failed" and "this tool call failed but the agent continues".
"""

from __future__ import annotations

from typing import Any


class AppError(Exception):
    """Base for all expected application errors. Subclasses just set two class
    attributes — ``status_code`` (the HTTP status) and ``code`` (a stable string
    clients can branch on) — and inherit the message/details plumbing. ``details``
    is an optional structured blob carried through to the JSON response."""

    status_code: int = 500
    code: str = "internal_error"

    def __init__(self, message: str = "", *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message or self.code)
        self.message = message or self.code
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        """The wire shape of an error response — one consistent envelope for every
        error type, so clients can always read ``error.code`` / ``error.message``."""
        return {"error": {"code": self.code, "message": self.message, "details": self.details}}


# Each subclass below is one (HTTP status, code) pair. The class IS the
# documentation of when to raise it; the API layer maps it to a response.
class ConfigError(AppError):
    status_code = 500
    code = "config_error"


class AuthError(AppError):
    status_code = 401   # not authenticated / bad token
    code = "unauthorized"


class TokenRevokedError(AuthError):
    # A revoked (logged-out / rotated) token — still a 401, but a distinct code so
    # the client can tell "log in again" from a generic auth failure.
    code = "token_revoked"


class PermissionDenied(AppError):
    status_code = 403   # authenticated but lacks the required role (RBAC denial)
    code = "forbidden"


class NotFound(AppError):
    status_code = 404
    code = "not_found"


class GuardrailBlocked(AppError):
    status_code = 422   # the safety spine refused this input/output
    code = "guardrail_blocked"


class RateLimited(AppError):
    status_code = 429   # per-caller throttle tripped
    code = "rate_limited"


class Overloaded(AppError):
    status_code = 503   # concurrency/queue limits hit — back-pressure, retry later
    code = "overloaded"


class RegistryError(AppError):
    # Raised at boot for a malformed/conflicting capability module (see registry.py).
    status_code = 500
    code = "registry_error"


class UpstreamError(AppError):
    status_code = 502   # a backend we depend on (LLM/vector store/...) failed
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
