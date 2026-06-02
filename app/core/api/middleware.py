"""Admission-control / backpressure middleware.

Bounds concurrent generations and sheds load gracefully (503) instead of
browning out under burst — the first line of the multi-user concurrency story
(blueprint §12). Per-org fairness quotas layer on top via the orchestrator; this
is the global guard on the chat path.
"""

from __future__ import annotations

import asyncio

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse


class ConcurrencyMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, *, max_global: int, queue_max: int, protected_prefixes=("/v1/chat",)) -> None:
        super().__init__(app)
        self._sem = asyncio.Semaphore(max(1, max_global))
        self._queue_max = queue_max
        self._waiting = 0
        self._protected = protected_prefixes

    async def dispatch(self, request: Request, call_next):
        if not any(request.url.path.startswith(p) for p in self._protected):
            return await call_next(request)
        if self._waiting >= self._queue_max:
            return JSONResponse(
                status_code=503,
                content={"error": {"code": "overloaded", "message": "server is busy, retry shortly"}},
                headers={"Retry-After": "2"},
            )
        self._waiting += 1  # counts in-flight + queued requests on protected paths
        try:
            async with self._sem:
                return await call_next(request)
        finally:
            self._waiting -= 1


__all__ = ["ConcurrencyMiddleware"]
