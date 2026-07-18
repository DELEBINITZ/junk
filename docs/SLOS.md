# SLOs, metrics & cost attribution

The system emits Prometheus metrics at **`GET /v1/metrics`** and a structured
`request_metrics` log line per turn. This doc defines the targets to alert on and
what each instrument measures. Source: `security_intel.observability.metrics`,
wired in `api/routes.py::chat`.

## SLO targets (customer-facing)

| SLO | Target | Metric |
|---|---|---|
| Availability (non-error turns) | ≥ 99.5% | `requests_total{outcome="error"}` / `requests_total` |
| Latency — single-agent (Atlas) | p95 ≤ 4 s | `request_latency_ms` histogram |
| Latency — multi-agent (complex) | p95 ≤ 12 s | `request_latency_ms` (critical path = max(agent) + synthesis) |
| Answer-quality flags | < 1% of agent-backed turns | `answer_flags_total` / agent-backed `requests_total` |
| Hallucinated CVE (hard) | **0 tolerated** | `answer_flags_total{type="unsupported_cves"}` |
| Cost per turn | budget per tenant tier | `request_cost_usd_total` (÷ turns) |

Latency numbers assume the Phase-2 changes (cheap routing, caching); revisit after
those land.

## Instruments

- `requests_total{outcome}` — turns by `ok` / `empty` / `error`.
- `request_latency_ms{outcome}` — end-to-end latency histogram.
- `request_tokens` — total tokens per turn histogram.
- `request_cost_usd_total{tenant,model}` — cumulative cost, attributed **per tenant
  AND per model** (so a heavy tenant or an expensive model is directly visible).
- `agent_invocations_total{agent}` — specialist usage (atlas/sentinel/aura).
- `routing_decisions_total{action}` — DIRECT / SIMPLE / COMPLEX / CLARIFY / OTHER.
- `answer_flags_total{type}` — online answer-quality sampling: `unsupported_cves`,
  `low_groundedness`, `low_citation_coverage` (scored by `eval_scoring`, LLM-free,
  only on agent-backed answers).

## Cost attribution

Models are **self-hosted** (Qwen 7B/72B, DeepSeek-V3), so `DEFAULT_MODEL_PRICING`
in `metrics.py` is a **placeholder estimate of compute cost per 1M tokens** — set it
to your real GPU economics (override the table / load from config). Token counts
come from a `UsageMetadataCallbackHandler` attached per request, so attribution is
automatic and per-model — no per-node plumbing.

## Recommended alerts

- `unsupported_cves` rate > 0 over 5m → **page** (hallucinated CVE reached a user).
- error-outcome ratio > 0.5% over 10m → page.
- latency p95 over target for 10m → ticket.
- `request_cost_usd_total` per-tenant slope over budget → ticket (cost runaway / abuse).

## Not yet wired (follow-ups)

- The `/v1/chat/stream` path emits metrics only if the same block is added there
  (currently only `/v1/chat` records). Mirror `_emit_chat_metrics` into the stream
  handler once its terminal state is available.
- Retrieval hit-rate and routing-accuracy-in-prod need labeled ground truth; drive
  them from `run_eval.py --score` sampling against mirrored traffic (Phase 1 harness).
- `/v1/metrics` should sit behind the same auth/network policy as the rest of `/v1`
  (or be moved to an internal-only port) before exposure.
