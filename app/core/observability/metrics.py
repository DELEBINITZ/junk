"""Lightweight metrics: in-memory counters/histograms, Prometheus when available.

THE TWO METRIC SHAPES (the whole vocabulary):
  * COUNTER — a number that only goes up (requests served, tools called, errors).
    You read it as a rate over time.
  * HISTOGRAM / TIMER — a distribution of measured values (latencies). You read it
    as count + average (and percentiles in a real backend).

Design: always keep simple in-memory tallies (so ``/metrics`` works with zero
infra), and ALSO mirror into the ``prometheus_client`` registry IF that library
is installed — best-effort, never required. Same call sites either way.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager


class Metrics:
    """A tiny metrics registry. ``enabled=False`` turns every record into a no-op
    (cheap kill-switch via config). All mutation is guarded by a lock because
    metrics are written from many concurrent request tasks/threads."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._counters: dict[str, float] = {}        # name -> running total
        self._hist: dict[str, list[float]] = {}      # name -> observed values
        self._lock = threading.Lock()                # protects the dicts above
        self._prom = None
        # Optional Prometheus mirror: only wired if the client lib imports. We keep
        # parallel dicts of Prom metric objects so each name maps to one instrument.
        if enabled:
            try:
                import prometheus_client as prom  # noqa: F401

                self._prom = prom
                self._prom_counters: dict[str, object] = {}
                self._prom_hist: dict[str, object] = {}
            except Exception:
                self._prom = None   # library absent -> in-memory only, no error

    def inc(self, name: str, value: float = 1.0) -> None:
        """Increment a COUNTER. Lazily creates the Prometheus counter on first use
        (you can't pre-declare every metric name, so we register on demand)."""
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
        """Record one sample into a HISTOGRAM (e.g. a latency in ms). Same lazy-
        registration trick for the Prometheus histogram."""
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
        """Convenience TIMER: ``with metrics.timer("x"): ...`` records the wall-clock
        duration of the block (in ms) as a histogram sample. ``perf_counter`` is a
        monotonic clock, and the ``finally`` guarantees we record even if the block
        raises."""
        start = time.perf_counter()
        try:
            yield
        finally:
            self.observe(name, (time.perf_counter() - start) * 1000.0)

    def snapshot(self) -> dict:
        """A JSON-able view of all metrics: raw counters plus, for each histogram,
        its sample count and average. This backs the JSON form of ``/metrics``."""
        with self._lock:
            hist = {
                k: {"count": len(v), "avg_ms": (sum(v) / len(v)) if v else 0.0}
                for k, v in self._hist.items()
            }
            return {"counters": dict(self._counters), "histograms": hist}

    def render_prometheus(self) -> bytes | None:
        """The Prometheus exposition-format bytes, or None if the client lib isn't
        installed (in which case ``/metrics`` serves the JSON snapshot instead)."""
        if self._prom is None:
            return None
        return self._prom.generate_latest()


# Process-wide singleton — one metrics registry shared by all requests.
_metrics: Metrics | None = None


def get_metrics(settings=None) -> Metrics:
    """Accessor for the singleton, created on first call (honoring
    ``metrics_enabled`` from config). Returning the same instance everywhere is
    what lets counters accumulate across the whole process."""
    global _metrics
    if _metrics is None:
        _metrics = Metrics(enabled=(settings.metrics_enabled if settings else True))
    return _metrics


__all__ = ["Metrics", "get_metrics"]
