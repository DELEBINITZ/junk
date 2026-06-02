"""Structured logging (JSON in prod, human-readable in dev).

WHY STRUCTURED LOGS: in production you don't grep prose, you query fields. Code
logs an event name plus structured context (``extra={"org_id": ..., "event":
...}``) and in JSON mode each line becomes a machine-parsable object a log system
can filter/aggregate on. In dev the same calls print readable text instead.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

# Module-level guard so logging is configured exactly once even if both the
# lifespan and bootstrap call configure_logging — re-running would duplicate
# handlers and double every line.
_CONFIGURED = False


class _JsonFormatter(logging.Formatter):
    """Renders each log record as a single JSON object. The key design choice is
    the FIELD WHITELIST below: rather than dumping every attribute on the record
    (noisy, and a risk of leaking arbitrary data), we copy only a known-safe set
    of operational fields. New structured fields must be added to this list to
    surface — an intentional allow-list, not a deny-list."""

    def format(self, record: logging.LogRecord) -> str:
        base: dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # The whitelist: only these ``extra=`` keys are promoted into the JSON.
        # They're the safe operational dimensions we slice logs by (tenant, trace,
        # which tool/module, latency, outcome) — deliberately no free-form payload.
        for k, v in record.__dict__.items():
            if k in ("org_id", "user_id", "trace_id", "tool", "cap_module", "cap_modules",
                     "event", "latency_ms", "ok", "code", "reason", "required", "approval_id"):
                base[k] = v
        if record.exc_info:
            base["exc"] = self.formatException(record.exc_info)   # attach traceback if logging an error
        return json.dumps(base, default=str)


def configure_logging(level: str = "INFO", json_logs: bool = False) -> None:
    """Install the root log handler ONCE. ``json_logs`` picks the formatter:
    structured JSON for prod, a terse human line for dev. Idempotent via the
    ``_CONFIGURED`` flag so repeated calls are harmless."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(sys.stdout)   # logs go to stdout (12-factor: let the platform collect them)
    if json_logs:
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.handlers.clear()        # drop any default handler so we don't log twice
    root.addHandler(handler)
    root.setLevel(level.upper())
    _CONFIGURED = True


def get_logger(name: str = "asi") -> logging.Logger:
    """Thin wrapper over ``logging.getLogger`` so callers go through one helper
    (and a consistent default name) rather than touching the stdlib directly."""
    return logging.getLogger(name)


__all__ = ["configure_logging", "get_logger"]
