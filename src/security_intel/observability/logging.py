"""Structured JSON logging with correlation IDs for production tracing."""

import logging
import json
import sys
from contextvars import ContextVar
from uuid import uuid4

from security_intel.config import Settings

_trace_id: ContextVar[str] = ContextVar("trace_id", default="")
_org_id: ContextVar[str] = ContextVar("org_id", default="")


def get_trace_id() -> str:
    return _trace_id.get()


def set_trace_context(trace_id: str = "", org_id: str = ""):
    if trace_id:
        _trace_id.set(trace_id)
    if org_id:
        _org_id.set(org_id)


def new_trace_id() -> str:
    tid = uuid4().hex[:16]
    _trace_id.set(tid)
    return tid


class JSONFormatter(logging.Formatter):
    """Structured JSON log formatter with trace context."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "ts": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "trace_id": _trace_id.get(),
            "org_id": _org_id.get(),
        }

        if record.exc_info and record.exc_info[0]:
            log_entry["error"] = self.formatException(record.exc_info)

        if hasattr(record, "extra_data"):
            log_entry.update(record.extra_data)

        return json.dumps(log_entry, default=str)


def setup_logging(settings: Settings) -> logging.Logger:
    """Configure structured logging for the application."""
    logger = logging.getLogger("security_intel")
    logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)

    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    return logger


def get_logger(name: str = "") -> logging.Logger:
    """Get a child logger with the given name."""
    base = "security_intel"
    return logging.getLogger(f"{base}.{name}" if name else base)
