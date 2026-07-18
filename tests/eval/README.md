# Routing eval harness

Measures whether the orchestrator triggers the **right agent(s)** for a query —
the metric that separates "routing works" from "routing is production-grade".

## What it does

Runs each query in `golden_queries.json` through the **real orchestrator**
(the same classify → plan → dispatch path production uses) and checks which agents
actually ran against the expected set. DIRECT/REFUSE queries expect **no** agents.

- `expect_agents: []` → should stay agent-free (greeting / out-of-scope / refusal)
- `expect_agents: ["sentinel"]` → single-agent route
- `expect_agents: ["aura","sentinel"]` → cross-domain (planner) route

Cases whose expected agent isn't registered in the current deployment (e.g. `aura`
with no MCP server) are **SKIPPED**, so the eval adapts to what's actually running.

## Run

```bash
# Needs live fast+standard LLM and Qdrant. No Postgres required.
uv run python tests/eval/run_eval.py
uv run python tests/eval/run_eval.py --json    # machine-readable, for CI
uv run python tests/eval/run_eval.py --score   # ALSO score answer quality (below)
```

Exit code is non-zero if any non-skipped case fails or errors — wire it into CI as
a routing regression gate.

## Answer-quality gate (`--score`, real-LLM only)

`--score` additionally scores each synthesized answer against what the agents
actually retrieved, using the LLM-free scorers in
`security_intel.observability.eval_scoring`:

- **groundedness** — fraction of the answer's content supported by the findings;
- **citation coverage** — how well the answer points back to its sources;
- **unsupported CVEs** — CVE ids asserted in the answer that appear in **no**
  retrieved source (highest-signal hallucination check for a security product).

With `--score`, the exit code is also non-zero if any answer carries a hard flag
(unsupported CVE / low groundedness). Same scorers can be sampled online in prod
(see Phase 4).

## Related gates

- **`tests/test_routing.py`** — routing PLUMBING regression, runs in CI with **no
  services** (scripted LLM + canned tools).
- **`tests/test_tenant_isolation.py`** — proves the Qdrant access filter never lets
  one org see another's private data (structural + semantic, no services; plus a
  live-Qdrant integration test that auto-skips without `QDRANT_URL`).
- **`tests/test_eval_scoring.py`** — unit tests for the answer-quality scorers.

## Extending

Add cases to `golden_queries.json`. Keep them intent-diverse: greetings, refusals,
each single agent, follow-ups, and genuine cross-domain queries. Aim to cover every
new agent and every ambiguous boundary you find misrouting in production.
