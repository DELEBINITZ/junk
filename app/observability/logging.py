"""Structured, PII-aware logging utilities.

The application has a separate audit log for business/security evidence. This
module is operational logging: request lifecycle, tool execution, guardrail
decisions, LLM fallback, and errors. Logs intentionally avoid raw contract text
and raw user prompts.
"""

from __future__ import annotations

import contextvars
import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

from app.guardrails.pii import redact_pii


_request_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "request_id",
    default=None,
)
_BASE_RECORD_KEYS = set(logging.makeLogRecord({}).__dict__)


def set_request_id(value: str):
    """Attach a request ID to the current async context."""

    return _request_id.set(value)


def reset_request_id(token) -> None:
    """Restore the previous request ID after a request completes."""

    _request_id.reset(token)


def get_request_id() -> str | None:
    """Return the request ID associated with the current request, if any."""

    return _request_id.get()


def configure_logging(level: str = "INFO", log_format: str = "json") -> None:
    """Configure application logging once at process startup.

    JSON logs are the default because they are easy to filter in Docker,
    CloudWatch, Datadog, ELK, or any other production log pipeline. Set
    `LOG_FORMAT=text` locally if a more human-readable stream is preferred.
    """

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(_level(level))

    handler = logging.StreamHandler(sys.stdout)
    formatter: logging.Formatter
    if log_format.lower() == "text":
        formatter = TextFormatter()
    else:
        formatter = JsonFormatter()
    handler.setFormatter(formatter)
    root.addHandler(handler)

    # Uvicorn access logs duplicate our request middleware logs. Keep server
    # lifecycle/error logs, but let this app own request-level observability.
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def safe_extra(**fields: Any) -> dict[str, Any]:
    """Return log `extra` fields with strings redacted for PII."""

    return {key: _sanitize(value) for key, value in fields.items()}


class JsonFormatter(logging.Formatter):
    """Emit compact structured JSON logs with request context."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": get_request_id(),
        }
        payload.update(_record_extras(record))
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, separators=(",", ":"))


class TextFormatter(logging.Formatter):
    """Readable local formatter that still keeps structured key/value fields."""

    def format(self, record: logging.LogRecord) -> str:
        pieces = [
            datetime.now(timezone.utc).isoformat(),
            record.levelname,
            record.name,
            f"request_id={get_request_id() or '-'}",
            record.getMessage(),
        ]
        extras = _record_extras(record)
        if extras:
            pieces.extend(f"{key}={json.dumps(value, default=str)}" for key, value in extras.items())
        if record.exc_info:
            pieces.append(self.formatException(record.exc_info))
        return " ".join(pieces)


def _record_extras(record: logging.LogRecord) -> dict[str, Any]:
    extras: dict[str, Any] = {}
    for key, value in record.__dict__.items():
        if key in _BASE_RECORD_KEYS or key.startswith("_"):
            continue
        extras[key] = _sanitize(value)
    return extras


def _sanitize(value: Any) -> Any:
    if isinstance(value, str):
        return redact_pii(value)
    if isinstance(value, dict):
        return {str(key): _sanitize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_sanitize(item) for item in value)
    return value


def _level(value: str) -> int:
    return getattr(logging, value.upper(), logging.INFO)
