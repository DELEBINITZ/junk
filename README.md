# Agentic Security Intelligence

Production-grade **agentic RAG** over org-scoped security intelligence. A
supervisor/planner agent **dynamically routes** each question to **capability
modules** — **reports** today; **EASM**, **Brand Protection**, **ACI**, and
MCP-backed modules as drop-ins — backed by a shared retrieval pipeline, typed MCP
tools, multi-tenant isolation, guardrails, token streaming, and chat memory.

- **Dynamic routing + deep reasoning**: modules are chosen by *meaning* (LLM router,
  or embedding similarity over each module's description + tools — no keyword lists).
  The default orchestrator plans → dispatches → reflects (a bounded re-query loop).
- **Real-infra stack**: a real LLM (SGLang/OpenAI-compatible), a real embedder
  (TEI/OpenAI), Qdrant, and Postgres+RLS are required (the offline stubs were removed).
- **Core + modules**: adding a feature is a small, **no-core-edit** module; promoting
  one to its own MCP server is config-only (`MCP_URLS`).

> Design + code map: **[`ARCHITECTURE.md`](ARCHITECTURE.md)**. Deploy/ops:
> **[`infra/README.md`](infra/README.md)**. Add a feature:
> **[`app/capabilities/_template/README.md`](app/capabilities/_template/README.md)**.

## Quickstart

The platform needs real infra. The fastest local setup uses OpenAI for the
LLM + embeddings (just an API key) and local Qdrant + Postgres.

```bash
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -e ".[dev,qdrant,postgres]"      # add ,observability,langgraph,redis as needed
# bring up Qdrant + Postgres (see infra/), then:
cp .env.example .env                            # set OPENAI_API_KEY, DATABASE_URL, QDRANT_URL, API_KEYS, JWT_SECRET
#   LLM_PROVIDER=openai  EMBEDDING_PROVIDER=openai   (easiest, no GPU)
asi-migrate                                     # create the Postgres schema (RLS)
uvicorn app.main:app --reload                   # http://localhost:8000
```

Report documents are written to Qdrant by an **external ingestion cron** (operated
separately); the agent only *retrieves* from `reports_kb` to answer. Point the agent
at a Qdrant that the cron has populated.

## Auth — API key + JWT

Every request carries **both**:
- `X-API-Key: <key>` — gateway key (must be in `API_KEYS`).
- `Authorization: Bearer <jwt>` — identity; org_id / user_id / roles come from the
  verified JWT claims. The JWT is minted upstream; this service only verifies it.

Mint a dev JWT locally:
```python
from app.config import get_settings
from app.core.security.jwt import create_access_token
print(create_access_token(get_settings(), sub="u-alice", org_id="org_acme", roles=("admin",)).token)
```
```bash
curl -H "X-API-Key: dev-api-key-change-me" -H "Authorization: Bearer $JWT" \
     -d '{"message":"What critical CVE is on our Confluence server?"}' localhost:8000/v1/chat
```

## API

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/v1/auth/me` | echo the verified identity |
| POST | `/v1/chat` | one grounded, cited answer |
| GET | `/v1/chat/stream` | **SSE** token stream (EventSource: `?access_token=` + `?api_key=`) |
| GET/POST | `/v1/sessions` | list / create chats |
| GET/PATCH/DELETE | `/v1/sessions/{id}` | history / rename / delete |
| GET | `/v1/sessions/search?q=` | cross-session recall |
| GET | `/v1/capabilities` | modules + tools visible to the user |
| POST | `/v1/route/preview` | inspect routing for a question |
| POST | `/v1/ingest` | generic document ingest (analyst+) — reports use the external cron |
| GET/POST | `/v1/approvals[/{id}/approve\|reject]` | action approval inbox |
| GET | `/healthz` · `/readyz` · `/metrics` | ops (`?fmt=prometheus`) |

## Configuration

Config-gated via env (see `.env.example`). Key knobs:

| Concern | Setting | Options |
|--------|---------|---------|
| LLM | `LLM_PROVIDER` | `sglang` (Qwen 7B/72B + DeepSeek lanes) · `openai` |
| Embeddings | `EMBEDDING_PROVIDER` | `tei` (Qwen3-Embedding) · `openai` |
| Retrieval | `RETRIEVAL_BACKEND` | `qdrant` |
| Chat store | `STORE_BACKEND` | `postgres` (RLS) — run `asi-migrate` |
| Routing | `ROUTER_MODE` / `ROUTING_STRATEGY` | `llm` / `semantic` |
| Orchestration | `ORCHESTRATOR_MODE` | `planner` (plan→dispatch→reflect) · `heuristic` |
| MCP servers | `MCP_URLS` | `{module_id: url}` — promote any module to remote, no code edit |
| Agent engine | `AGENT_ENGINE` | `internal` · `langgraph` (durable checkpointing) |
| Tracing | `TRACING_PROVIDER` | `none` · `langfuse` (recommended on — see the reasoning loop) |

## Testing

```bash
python -m pytest        # infra-free tests (auth/jwt, config guard, guardrail providers)
```
The agent/RAG/API tests were removed when the offline stubs were (they required the
deterministic engine). Re-add an integration layer (testcontainers for Qdrant/Postgres,
a stub MCP server, the LLM stubbed at the HTTP boundary) before relying on CI.

## Add a capability module (no core edit)

```bash
cp -r app/capabilities/_template app/capabilities/<feature>
# rename *.template → *.py, write tools + prompt, fill the manifest (a crisp
# `description` + well-described tools IS the routing signal — no keywords),
# add cap_<feature>_enabled to app/config.py, flip the flag.
```
Routing is automatic (the planner/router picks it by meaning). To back it with an
MCP server, add `"<feature>": "<url>"` to `MCP_URLS` — RBAC + the action gate still
apply locally. See `app/capabilities/_template/README.md`.

## Project layout

```
app/
  core/            # the frozen platform (see ARCHITECTURE.md §3)
    agent/ llm/ rag/ memory/ guardrails/ security/ mcp/ db/ ingestion/
    observability/ streaming/ action_gate/ api/  contracts.py registry.py bootstrap.py
  capabilities/    # drop-in features: reports, easm, brand, aci, testkit, _template
  config.py  main.py  eval/
tests/   frontend/   infra/
```

## Security posture (this is a security product)

Multi-tenant isolation at every layer (API key + verified JWT → Postgres RLS →
vector-store org filter → per-org KG/gate), injection-resistant prompting, secrets
redaction, **all side-effecting actions human-gated**, append-only audit. Remote MCP
calls carry a short-lived, audience-bound, org-scoped service token. Asset
domains/IPs are treated as subjects, not PII. Details in `ARCHITECTURE.md` §6.
