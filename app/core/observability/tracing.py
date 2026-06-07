"""Tracing seam — NoOp by default, Langfuse when configured.

CONCEPT: a "trace" is a tree of timed "spans" — one span per unit of work (a chat
turn, a tool call, an LLM request) — that together show where time went and what
happened inside one request. This module is the SEAM: code always calls
``tracer.span(...)`` / ``tracer.event(...)`` the same way, and config decides
whether those calls do nothing (NoOp) or report to Langfuse.

WHY A NOOP DEFAULT: it keeps the hot path dependency-free and zero-cost — no
tracing backend, no SDK import, no network — until an operator opts in. This is
the same config-gating pattern as the rest of the platform. Both classes expose
an identical interface so the rest of the code never knows which one it has (the
"null object" pattern).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any


class NoOpSpan:
    """A span that records nothing. Its methods accept any args and discard them,
    so instrumentation calls are valid no-ops when tracing is off."""
    def set(self, **_: Any) -> None: ...
    def event(self, *_: Any, **__: Any) -> None: ...


class NoOpTracer:
    """The default tracer: every method is a no-op. Selected whenever tracing is
    not configured, so ``with tracer.span(...)`` is free and harmless."""
    provider = "none"

    @contextmanager
    def span(self, name: str, **attrs: Any) -> Iterator[NoOpSpan]:
        # Yields a throwaway span; the ``with`` block runs normally, nothing is recorded.
        yield NoOpSpan()

    def event(self, name: str, **attrs: Any) -> None: ...

    async def aclose(self) -> None: ...


class LangfuseTracer:
    """The real tracer: reports spans/events to Langfuse (an LLM-observability
    backend). Same interface as NoOpTracer, so swapping it in is invisible to
    callers. Every Langfuse call is wrapped in try/except — observability must
    NEVER break the request it's observing; a tracing failure is swallowed."""
    provider = "langfuse"

    def __init__(self, host: str, public_key: str, secret_key: str) -> None:
        from langfuse import Langfuse

        self._lf = Langfuse(host=host or None, public_key=public_key, secret_key=secret_key)

    @contextmanager
    def span(self, name: str, **attrs: Any) -> Iterator[Any]:
        span = None
        try:
            trace = self._lf.trace(name=name, metadata=attrs)
            span = trace.span(name=name, metadata=attrs)
            yield span
        except Exception:
            yield NoOpSpan()
        finally:
            try:
                if span and hasattr(span, "end"):
                    span.end()
            except Exception:
                pass

    def event(self, name: str, **attrs: Any) -> None:
        try:
            trace = self._lf.trace(name=name)
            trace.event(name=name, metadata=attrs)
        except Exception:
            pass

    async def aclose(self) -> None:
        try:
            self._lf.flush()
        except Exception:
            pass


def build_tracer(settings):
    """Config-gate factory (called by bootstrap): return the real Langfuse tracer
    only when it's both selected AND credentialed; otherwise the NoOp. This is the
    single decision point for tracing — everything else just uses what it's given."""
    if settings.tracing_provider == "langfuse" and settings.langfuse_public_key:
        return LangfuseTracer(settings.langfuse_host, settings.langfuse_public_key, settings.langfuse_secret_key)
    return NoOpTracer()


def build_langfuse_handler(settings):
    """Return a LangChain/LangGraph ``CallbackHandler`` that AUTO-traces the whole
    graph — every node, every LLM call, tokens, latency — to Langfuse, or ``None``
    when tracing isn't configured.

    WHY ALONGSIDE build_tracer: ``build_tracer`` gives manual spans (you wrap what
    you choose). This handler is the LangSmith-style auto-instrumenter: pass it in
    the LangGraph invoke ``config`` and the full per-node trace tree appears in the
    dashboard with zero manual spans. Same config gate as the tracer; same swallow-
    all-errors posture — observability must never break the request it observes.

    NOTE: this builds its OWN Langfuse client (separate from build_tracer's). The
    SDK flushes batched events on a background timer and at process exit, so a
    long-running server ships traces without an explicit flush; nothing is lost."""
    if settings.tracing_provider != "langfuse" or not settings.langfuse_public_key:
        return None
    try:
        from langfuse.callback import CallbackHandler  # langfuse v2 import path

        return CallbackHandler(
            host=settings.langfuse_host or None,
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
        )
    except Exception:
        # SDK missing / wrong version / bad creds -> no auto-tracing, but the app
        # runs exactly as before. Tracing is strictly best-effort.
        return None


__all__ = ["NoOpTracer", "LangfuseTracer", "build_tracer", "build_langfuse_handler"]
