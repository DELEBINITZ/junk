"""Tracing seam — NoOp by default, Langfuse when configured.

Every turn/tool/LLM call can open a span; in prod Langfuse records the full
trace. The NoOp keeps the hot path dependency-free.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any


class NoOpSpan:
    def set(self, **_: Any) -> None: ...
    def event(self, *_: Any, **__: Any) -> None: ...


class NoOpTracer:
    provider = "none"

    @contextmanager
    def span(self, name: str, **attrs: Any) -> Iterator[NoOpSpan]:
        yield NoOpSpan()

    def event(self, name: str, **attrs: Any) -> None: ...

    async def aclose(self) -> None: ...


class LangfuseTracer:
    provider = "langfuse"

    def __init__(self, host: str, public_key: str, secret_key: str) -> None:
        from langfuse import Langfuse  # lazy

        self._lf = Langfuse(host=host or None, public_key=public_key, secret_key=secret_key)

    @contextmanager
    def span(self, name: str, **attrs: Any) -> Iterator[Any]:
        span = None
        try:
            span = self._lf.span(name=name, metadata=attrs)
            yield span
        finally:
            try:
                if span:
                    span.end()
            except Exception:
                pass

    def event(self, name: str, **attrs: Any) -> None:
        try:
            self._lf.event(name=name, metadata=attrs)
        except Exception:
            pass

    async def aclose(self) -> None:
        try:
            self._lf.flush()
        except Exception:
            pass


def build_tracer(settings):
    if settings.tracing_provider == "langfuse" and settings.langfuse_public_key:
        return LangfuseTracer(settings.langfuse_host, settings.langfuse_public_key, settings.langfuse_secret_key)
    return NoOpTracer()


__all__ = ["NoOpTracer", "LangfuseTracer", "build_tracer"]
