"""Map exceptions to JSON problem responses."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.core.errors import AppError
from app.core.observability.logging import get_logger

_log = get_logger("asi.api")


async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    if exc.status_code >= 500:
        _log.error("app_error %s: %s", exc.code, exc.message)
    return JSONResponse(status_code=exc.status_code, content=exc.to_dict())


async def unhandled_handler(request: Request, exc: Exception) -> JSONResponse:
    _log.exception("unhandled error: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"error": {"code": "internal_error", "message": "internal server error"}},
    )


def install_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(AppError, app_error_handler)
    app.add_exception_handler(Exception, unhandled_handler)


__all__ = ["install_error_handlers"]
