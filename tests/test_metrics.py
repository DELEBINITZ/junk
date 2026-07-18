"""Tests for the SLO metrics + cost-attribution module (pure, CI-safe)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from security_intel.observability.metrics import (  # noqa: E402
    cost_usd,
    attribute_cost,
    Registry,
    RequestMetrics,
    record_request,
)


def test_cost_usd_matches_by_substring():
    # 1M input tokens of Qwen2.5-72B at $0.40/1M
    assert cost_usd("Qwen/Qwen2.5-72B-Instruct", 1_000_000, 0) == 0.40
    assert cost_usd("Qwen/Qwen2.5-72B-Instruct", 0, 1_000_000) == 0.80


def test_cost_usd_unknown_model_is_zero():
    assert cost_usd("some-unlisted-model", 1_000_000, 1_000_000) == 0.0


def test_attribute_cost_aggregates_and_totals():
    usage = {
        "Qwen/Qwen2.5-7B-Instruct": {"input_tokens": 1_000_000, "output_tokens": 0},
        "deepseek-ai/DeepSeek-V3": {"input_tokens": 0, "output_tokens": 1_000_000},
    }
    rep = attribute_cost(usage)
    assert rep["input_tokens"] == 1_000_000
    assert rep["output_tokens"] == 1_000_000
    # 0.05 (7B in) + 0.90 (V3 out)
    assert rep["total_usd"] == 0.95
    assert set(rep["by_model"]) == set(usage)


def test_counter_render_prometheus_shape():
    reg = Registry()
    reg.agent_invocations.inc(agent="atlas")
    reg.agent_invocations.inc(agent="atlas")
    reg.agent_invocations.inc(agent="sentinel")
    text = reg.render_prometheus()
    assert "# TYPE agent_invocations_total counter" in text
    assert 'agent_invocations_total{agent="atlas"} 2.0' in text
    assert 'agent_invocations_total{agent="sentinel"} 1.0' in text


def test_histogram_render_has_buckets_sum_count():
    reg = Registry()
    reg.request_latency_ms.observe(120.0, outcome="ok")
    text = reg.render_prometheus()
    assert "# TYPE request_latency_ms histogram" in text
    assert "request_latency_ms_bucket" in text
    assert 'le="+Inf"' in text
    assert "request_latency_ms_sum" in text
    assert "request_latency_ms_count" in text


def test_record_request_updates_registry_and_is_safe():
    reg = Registry()
    m = RequestMetrics(
        tenant="orgA",
        latency_ms=812.0,
        agents_used=["aura", "sentinel"],
        routing_action="COMPLEX",
        cost=attribute_cost({"deepseek-ai/DeepSeek-V3": {"input_tokens": 1000, "output_tokens": 500}}),
        answer_flags=["unsupported_cves:CVE-2024-9999", "low_groundedness:0.20"],
        outcome="ok",
    )
    record_request(m, registry=reg)
    text = reg.render_prometheus()
    assert 'requests_total{outcome="ok"} 1.0' in text
    assert 'agent_invocations_total{agent="aura"} 1.0' in text
    assert 'routing_decisions_total{action="COMPLEX"} 1.0' in text
    # flags counted by TYPE (prefix before ':'), not the full value
    assert 'answer_flags_total{type="unsupported_cves"} 1.0' in text
    assert 'answer_flags_total{type="low_groundedness"} 1.0' in text
    assert "request_cost_usd_total" in text


def test_record_request_never_raises_on_bad_input():
    reg = Registry()
    bad = RequestMetrics(cost={"by_model": None})  # malformed
    record_request(bad, registry=reg)  # must not raise
