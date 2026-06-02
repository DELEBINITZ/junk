"""Minimal in-process counters (no external dependency).

Increment 1 keeps a tiny counter registry so the agent/MCP paths can record
volume and the /metrics endpoint can expose it. The production target is a
Prometheus client + Grafana dashboards (plan §12); the call sites (`metrics.incr`)
stay the same.
"""

from __future__ import annotations

from collections import defaultdict
from threading import Lock


class _Metrics:
    def __init__(self):
        self._lock = Lock()
        self._counters: dict[str, int] = defaultdict(int)

    def incr(self, name: str, amount: int = 1) -> None:
        with self._lock:
            self._counters[name] += amount

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counters)

    def render_prometheus(self) -> str:
        """Render counters in Prometheus exposition format (no external dep)."""

        lines: list[str] = []
        for name, value in sorted(self.snapshot().items()):
            metric = name.replace(".", "_").replace("-", "_")
            lines.append(f"# TYPE {metric} counter")
            lines.append(f"{metric} {value}")
        return "\n".join(lines) + "\n"

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()


metrics = _Metrics()
