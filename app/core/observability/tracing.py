"""Tracing (plan §12). No-op by default; Langfuse adapter when
OBSERVABILITY_PROVIDER=langfuse (lazy-imported, fails open to no-op).

Usage:
    with get_tracer().span("agent.query", org_id=..., trace_id=...):
        ...
Every span carries org_id so per-org cost/usage can be rolled up later.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Iterator

from app.config import settings


logger = logging.getLogger(__name__)


class NoOpTracer:
    @contextmanager
    def span(self, name: str, **attrs) -> Iterator[None]:
        yield None


class LangfuseTracer:  # pragma: no cover - exercised only when configured
    def __init__(self):
        from langfuse import Langfuse  # lazy

        self._client = Langfuse()

    @contextmanager
    def span(self, name: str, **attrs) -> Iterator[None]:
        trace = None
        try:
            trace = self._client.trace(name=name, metadata=attrs)
        except Exception as exc:
            logger.warning("tracing.langfuse_failed", extra={"error": str(exc)})
        try:
            yield None
        finally:
            if trace is not None:
                try:
                    trace.update(output="ok")
                except Exception:
                    pass


_tracer = None


def get_tracer():
    global _tracer
    if _tracer is None:
        provider = os.getenv("OBSERVABILITY_PROVIDER", settings.observability_provider).lower()
        if provider == "langfuse":
            try:
                _tracer = LangfuseTracer()
            except Exception as exc:  # pragma: no cover
                logger.warning("tracing.langfuse_unavailable", extra={"error": str(exc)})
                _tracer = NoOpTracer()
        else:
            _tracer = NoOpTracer()
    return _tracer


def reset_tracer() -> None:
    global _tracer
    _tracer = None
