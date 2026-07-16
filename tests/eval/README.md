# Routing eval harness

Measures whether the orchestrator triggers the **right agent(s)** for a query —
the metric that separates "routing works" from "routing is production-grade".

## What it does

Runs each query in `golden_queries.json` through the **real orchestrator**
(the same classify → plan → dispatch path production uses) and checks which agents
actually ran against the expected set. DIRECT/REFUSE queries expect **no** agents.

- `expect_agents: []` → should stay agent-free (greeting / out-of-scope / refusal)
- `expect_agents: ["reports"]` → single-agent route
- `expect_agents: ["easm","reports"]` → cross-domain (planner) route

Cases whose expected agent isn't registered in the current deployment (e.g. `easm`
with no MCP server) are **SKIPPED**, so the eval adapts to what's actually running.

## Run

```bash
# Needs live fast+standard LLM and Qdrant. No Postgres required.
uv run python tests/eval/run_eval.py
uv run python tests/eval/run_eval.py --json   # machine-readable, for CI
```

Exit code is non-zero if any non-skipped case fails or errors — wire it into CI as
a routing regression gate.

## Extending

Add cases to `golden_queries.json`. Keep them intent-diverse: greetings, refusals,
each single agent, follow-ups, and genuine cross-domain queries. Aim to cover every
new agent and every ambiguous boundary you find misrouting in production.
