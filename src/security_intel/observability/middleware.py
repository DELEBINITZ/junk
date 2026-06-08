"""FastAPI middleware for request tracing and logging."""

import time
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from security_intel.observability.logging import new_trace_id, set_trace_context, get_logger

logger = get_logger("http")


class TracingMiddleware(BaseHTTPMiddleware):
    """Inject trace_id into every request and log request lifecycle."""

    async def dispatch(self, request: Request, call_next) -> Response:
        trace_id = request.headers.get("X-Request-ID") or new_trace_id()
        set_trace_context(trace_id=trace_id)

        start = time.perf_counter()

        logger.info(
            "request_start",
            extra={"extra_data": {
                "method": request.method,
                "path": request.url.path,
                "client": request.client.host if request.client else "",
            }},
        )

        try:
            response = await call_next(request)
            duration_ms = (time.perf_counter() - start) * 1000

            logger.info(
                "request_end",
                extra={"extra_data": {
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                    "duration_ms": round(duration_ms, 1),
                }},
            )

            response.headers["X-Request-ID"] = trace_id
            return response

        except Exception as e:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.error(
                "request_error",
                extra={"extra_data": {
                    "method": request.method,
                    "path": request.url.path,
                    "error": str(e),
                    "duration_ms": round(duration_ms, 1),
                }},
                exc_info=True,
            )
            raise
