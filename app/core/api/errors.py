"""Map exceptions to JSON problem responses.

STRUCTURED ERROR HANDLERS: instead of letting exceptions bubble up as default
HTML/500s, we register handlers that turn them into a consistent JSON envelope
(``{"error": {"code", "message"}}``) every client can rely on. Two handlers cover
the two cases: KNOWN domain errors (AppError and its subclasses — NotFound,
AuthError, ...) carry their own HTTP status + code; anything UNEXPECTED is caught
by a catch-all that returns a generic 500 and — importantly — never leaks internal
details to the client (full traceback goes to the logs only).
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.core.errors import AppError
from app.core.observability.logging import get_logger

_log = get_logger("asi.api")


async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    """Handle a KNOWN application error. AppError carries its intended HTTP status,
    a stable machine-readable ``code``, and a safe ``message``, so we just serialize
    it. 5xx means "our fault" -> log it; 4xx (client mistakes) are not logged as
    errors to avoid noise."""
    if exc.status_code >= 500:
        _log.error("app_error %s: %s", exc.code, exc.message)
    return JSONResponse(status_code=exc.status_code, content=exc.to_dict())


async def unhandled_handler(request: Request, exc: Exception) -> JSONResponse:
    """Last-resort catch-all for any exception we didn't model. Log the full
    stack trace for debugging, but return a deliberately VAGUE 500 to the caller —
    leaking internals (stack frames, messages) from an unexpected error is an
    information-disclosure risk."""
    _log.exception("unhandled error: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"error": {"code": "internal_error", "message": "internal server error"}},
    )


def install_error_handlers(app: FastAPI) -> None:
    """Register both handlers on the app (called from app construction). Order of
    specificity matters conceptually: the AppError handler catches our typed
    errors; the broad Exception handler is the safety net for everything else."""
    app.add_exception_handler(AppError, app_error_handler)
    app.add_exception_handler(Exception, unhandled_handler)


__all__ = ["install_error_handlers"]
