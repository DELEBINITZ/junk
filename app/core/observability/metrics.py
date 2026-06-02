"""Lightweight metrics: in-memory counters/histograms, Prometheus when available."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager


class Metrics:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._counters: dict[str, float] = {}
        self._hist: dict[str, list[float]] = {}
        self._lock = threading.Lock()
        self._prom = None
        if enabled:
            try:
                import prometheus_client as prom  # noqa: F401

                self._prom = prom
                self._prom_counters: dict[str, object] = {}
                self._prom_hist: dict[str, object] = {}
            except Exception:
                self._prom = None

    def inc(self, name: str, value: float = 1.0) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._counters[name] = self._counters.get(name, 0.0) + value
        if self._prom is not None:
            c = self._prom_counters.get(name)
            if c is None:
                c = self._prom.Counter(name, name)
                self._prom_counters[name] = c
            c.inc(value)

    def observe(self, name: str, value: float) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._hist.setdefault(name, []).append(value)
        if self._prom is not None:
            h = self._prom_hist.get(name)
            if h is None:
                h = self._prom.Histogram(name, name)
                self._prom_hist[name] = h
            h.observe(value)

    @contextmanager
    def timer(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.observe(name, (time.perf_counter() - start) * 1000.0)

    def snapshot(self) -> dict:
        with self._lock:
            hist = {
                k: {"count": len(v), "avg_ms": (sum(v) / len(v)) if v else 0.0}
                for k, v in self._hist.items()
            }
            return {"counters": dict(self._counters), "histograms": hist}

    def render_prometheus(self) -> bytes | None:
        if self._prom is None:
            return None
        return self._prom.generate_latest()


_metrics: Metrics | None = None


def get_metrics(settings=None) -> Metrics:
    global _metrics
    if _metrics is None:
        _metrics = Metrics(enabled=(settings.metrics_enabled if settings else True))
    return _metrics


__all__ = ["Metrics", "get_metrics"]
