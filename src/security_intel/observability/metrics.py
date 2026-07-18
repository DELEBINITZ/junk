"""SLO metrics + cost attribution — a small, dependency-free instrument registry.

Production needs to answer, per request: how long did it take, how many tokens did
it burn, what did it cost, which agents ran, for which tenant, and did the answer
trip a quality flag. This module provides:

- a minimal in-process metric registry (Counter + Histogram) that renders the
  Prometheus text exposition format, so a `/metrics` endpoint can be scraped;
- token→cost attribution (models here are SELF-HOSTED, so rates are config-driven
  estimates of compute cost, not vendor pricing — override via MODEL_PRICING);
- `record_request(...)`, the single call the request path makes to emit everything.

It is intentionally not a full Prometheus client (no extra dependency): correct
exposition for the handful of instruments we need, and a structured log line as a
second sink so the data survives even without a scraper.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from security_intel.observability.logging import get_logger

logger = get_logger("metrics")

# ---- cost attribution ------------------------------------------------------- #
# $ per 1,000,000 tokens (input, output). SELF-HOSTED estimate of compute cost —
# these are placeholders to override for your GPU economics, matched by substring
# against the model id so "Qwen/Qwen2.5-72B-Instruct" hits "Qwen2.5-72B".
DEFAULT_MODEL_PRICING: dict[str, tuple[float, float]] = {
    "Qwen2.5-7B": (0.05, 0.10),
    "Qwen2.5-72B": (0.40, 0.80),
    "DeepSeek-V3": (0.30, 0.90),
    "Qwen3-Embedding": (0.02, 0.0),
}


def cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
    pricing: dict[str, tuple[float, float]] | None = None,
) -> float:
    """Cost of one model call. Unknown models cost 0 and warn (never crash billing)."""
    table = pricing or DEFAULT_MODEL_PRICING
    rate = next((v for k, v in table.items() if k in (model or "")), None)
    if rate is None:
        logger.warning(f"no pricing for model '{model}' — cost counted as $0")
        return 0.0
    in_rate, out_rate = rate
    return (input_tokens / 1_000_000) * in_rate + (output_tokens / 1_000_000) * out_rate


def attribute_cost(
    usage_by_model: dict[str, dict],
    pricing: dict[str, tuple[float, float]] | None = None,
) -> dict:
    """Turn a UsageMetadataCallbackHandler's ``.usage_metadata`` into a cost report.

    Input shape: ``{model: {"input_tokens": N, "output_tokens": M, ...}}``.
    Returns ``{"total_usd", "input_tokens", "output_tokens", "by_model": {...}}``.
    """
    by_model, total, tin, tout = {}, 0.0, 0, 0
    for model, u in (usage_by_model or {}).items():
        i = int(u.get("input_tokens", 0) or 0)
        o = int(u.get("output_tokens", 0) or 0)
        c = cost_usd(model, i, o, pricing)
        by_model[model] = {"input_tokens": i, "output_tokens": o, "usd": round(c, 6)}
        total += c
        tin += i
        tout += o
    return {"total_usd": round(total, 6), "input_tokens": tin, "output_tokens": tout, "by_model": by_model}


# ---- metric registry -------------------------------------------------------- #
def _fmt_labels(labels: dict) -> str:
    if not labels:
        return ""
    inner = ",".join(f'{k}="{_esc(str(v))}"' for k, v in sorted(labels.items()))
    return "{" + inner + "}"


def _esc(v: str) -> str:
    return v.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")


def _key(labels: dict) -> tuple:
    return tuple(sorted(labels.items()))


@dataclass
class Counter:
    name: str
    help: str
    _vals: dict = field(default_factory=dict)

    def inc(self, amount: float = 1.0, **labels) -> None:
        self._vals[_key(labels)] = self._vals.get(_key(labels), 0.0) + amount

    def render(self) -> list[str]:
        out = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} counter"]
        for k, v in self._vals.items():
            out.append(f"{self.name}{_fmt_labels(dict(k))} {v}")
        return out


_DEFAULT_BUCKETS = (50, 100, 250, 500, 1000, 2500, 5000, 10000, 30000)


@dataclass
class Histogram:
    name: str
    help: str
    buckets: tuple = _DEFAULT_BUCKETS
    _obs: dict = field(default_factory=dict)  # label-key -> [counts-per-bucket, sum, count]

    def observe(self, value: float, **labels) -> None:
        k = _key(labels)
        rec = self._obs.get(k)
        if rec is None:
            rec = [[0] * (len(self.buckets) + 1), 0.0, 0]  # +1 for +Inf
            self._obs[k] = rec
        for i, b in enumerate(self.buckets):
            if value <= b:
                rec[0][i] += 1
        rec[0][-1] += 1  # +Inf always
        rec[1] += value
        rec[2] += 1

    def render(self) -> list[str]:
        out = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} histogram"]
        for k, (counts, s, cnt) in self._obs.items():
            base = dict(k)
            cumulative = 0
            for i, b in enumerate(self.buckets):
                cumulative = counts[i]
                out.append(f"{self.name}_bucket{_fmt_labels(base | {'le': str(b)})} {cumulative}")
            out.append(f"{self.name}_bucket{_fmt_labels(base | {'le': '+Inf'})} {counts[-1]}")
            out.append(f"{self.name}_sum{_fmt_labels(base)} {s}")
            out.append(f"{self.name}_count{_fmt_labels(base)} {cnt}")
        return out


class Registry:
    """Process-wide metric registry. Thread-safe for the simple inc/observe path."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.request_latency_ms = Histogram("request_latency_ms", "End-to-end request latency (ms).")
        self.request_tokens = Histogram(
            "request_tokens", "Total tokens per request.", buckets=(100, 500, 1000, 2000, 5000, 10000, 50000)
        )
        self.request_cost_usd = Counter("request_cost_usd_total", "Cumulative request cost in USD.")
        self.requests = Counter("requests_total", "Requests by outcome.")
        self.agent_invocations = Counter("agent_invocations_total", "Specialist invocations by agent.")
        self.routing_decisions = Counter("routing_decisions_total", "Router actions.")
        self.answer_flags = Counter("answer_flags_total", "Answer-quality flags by type.")

    def render_prometheus(self) -> str:
        with self._lock:
            lines: list[str] = []
            for inst in (
                self.requests, self.request_latency_ms, self.request_tokens,
                self.request_cost_usd, self.agent_invocations, self.routing_decisions,
                self.answer_flags,
            ):
                lines.extend(inst.render())
            return "\n".join(lines) + "\n"


REGISTRY = Registry()


@dataclass
class RequestMetrics:
    tenant: str = "unknown"
    latency_ms: float = 0.0
    agents_used: list[str] = field(default_factory=list)
    routing_action: str = ""
    cost: dict = field(default_factory=dict)          # attribute_cost() output
    answer_flags: list[str] = field(default_factory=list)
    outcome: str = "ok"                               # ok | error | empty

    def as_dict(self) -> dict:
        return {
            "tenant": self.tenant, "latency_ms": round(self.latency_ms, 1),
            "agents_used": self.agents_used, "routing_action": self.routing_action,
            "cost_usd": self.cost.get("total_usd", 0.0),
            "tokens": self.cost.get("input_tokens", 0) + self.cost.get("output_tokens", 0),
            "answer_flags": self.answer_flags, "outcome": self.outcome,
        }


def record_request(m: RequestMetrics, registry: Registry = REGISTRY) -> None:
    """Single sink the request path calls once per turn: updates the scrapeable
    registry AND emits one structured log line (survives without a scraper).

    Deliberately never raises — metrics must not break the response path."""
    try:
        with registry._lock:
            registry.requests.inc(outcome=m.outcome)
            registry.request_latency_ms.observe(m.latency_ms, outcome=m.outcome)
            tokens = m.cost.get("input_tokens", 0) + m.cost.get("output_tokens", 0)
            registry.request_tokens.observe(tokens)
            # cost attributed per tenant AND per model, so a heavy tenant/model is visible
            for model, mc in (m.cost.get("by_model") or {}).items():
                registry.request_cost_usd.inc(mc.get("usd", 0.0), tenant=m.tenant, model=model)
            if not m.cost.get("by_model"):
                registry.request_cost_usd.inc(m.cost.get("total_usd", 0.0), tenant=m.tenant, model="unknown")
            if m.routing_action:
                registry.routing_decisions.inc(action=m.routing_action)
            for a in m.agents_used:
                registry.agent_invocations.inc(agent=a)
            for f in m.answer_flags:
                registry.answer_flags.inc(type=f.split(":")[0])
        logger.info("request_metrics", extra={"metrics": m.as_dict()})
    except Exception as e:  # noqa: BLE001 — metrics must never break the response
        logger.warning(f"record_request failed (non-fatal): {e}")
