"""Admission-control / backpressure middleware.

================================ MENTAL MODEL =============================
ADMISSION CONTROL: an LLM generation is expensive and slow, so a server can only
run so many at once. Rather than accept unlimited requests and let them all crawl
(or exhaust memory) under a burst — "browning out" — we put a fixed-size gate in
front of the chat path. This is BACKPRESSURE: when we're already full, we reject
NEW work fast and clearly (HTTP 503 + Retry-After) instead of degrading everyone.

Two limits, working together as BOUNDED CONCURRENCY:
  * ``max_global`` — how many requests may RUN concurrently (a semaphore).
  * ``queue_max``  — how many may WAIT for a slot before we start shedding load.
A request beyond both -> immediate graceful 503. Only ``/v1/chat`` paths are
gated; cheap endpoints (auth, health, sessions list) pass straight through.

MIDDLEWARE = code that wraps every request/response. Per-org fairness quotas
layer on top inside the orchestrator; THIS is the cheap global guard at the edge.
===========================================================================
"""

from __future__ import annotations

import asyncio

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse


class ConcurrencyMiddleware(BaseHTTPMiddleware):
    """Caps concurrent work on the protected paths and sheds excess as 503."""

    def __init__(self, app, *, max_global: int, queue_max: int, protected_prefixes=("/v1/chat",)) -> None:
        # The semaphore is the concurrency gate: it has ``max_global`` permits, so
        # at most that many requests hold one at the same time. ``max(1, ...)``
        # guards against a zero/negative config that would deadlock everything.
        super().__init__(app)
        self._sem = asyncio.Semaphore(max(1, max_global))
        self._queue_max = queue_max               # how many may queue before we reject
        self._waiting = 0                         # current in-flight + queued count (see below)
        self._protected = protected_prefixes      # only these path prefixes are gated

    async def dispatch(self, request: Request, call_next):
        # Fast path: anything outside the protected prefixes isn't rate-limited.
        if not any(request.url.path.startswith(p) for p in self._protected):
            return await call_next(request)
        # Already saturated (too many waiting) -> shed load immediately. Returning
        # 503 with Retry-After tells a well-behaved client to back off and retry,
        # which is far better than making it wait indefinitely.
        if self._waiting >= self._queue_max:
            return JSONResponse(
                status_code=503,
                content={"error": {"code": "overloaded", "message": "server is busy, retry shortly"}},
                headers={"Retry-After": "2"},
            )
        self._waiting += 1  # counts in-flight + queued requests on protected paths
        try:
            # Block here until a permit is free, then run the request holding it.
            # ``async with`` guarantees the permit is released even on error.
            async with self._sem:
                return await call_next(request)
        finally:
            # Always decrement, whether the request succeeded, failed, or was shed,
            # so the queue counter can never get stuck high and wedge the gate.
            self._waiting -= 1


__all__ = ["ConcurrencyMiddleware"]
