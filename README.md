# Agentic Security Intelligence

Production-grade **agentic RAG** over org-scoped security intelligence. A
supervisor agent (LangGraph) routes each question to **capability modules** —
**reports** today; **EASM**, **Brand Protection**, **ACI** as drop-in modules —
backed by a shared retrieval pipeline, typed MCP tools, multi-tenant isolation,
guardrails, token streaming, and ChatGPT-style chat memory.

- **Runs with zero infra** out of the box: deterministic LLM + embeddings,
  in-memory stores, built-in graph engine. No GPU, keys, or network.
- **Flip config + add creds** for the real self-hosted stack: SGLang (Qwen),
  TEI embeddings/reranker, Qdrant, Postgres+RLS, Redis, Langfuse, real LangGraph.
- **Core + modules**: adding a feature is a ~1-day module, **no core edit**.

> Design + code map: **[`ARCHITECTURE.md`](ARCHITECTURE.md)**. Deploy/ops:
> **[`infra/README.md`](infra/README.md)**. Add a feature:
> **[`app/capabilities/_template/README.md`](app/capabilities/_template/README.md)**.

## Quickstart

```bash
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -e ".[dev]"          # add ,langgraph,qdrant,postgres,redis,observability for real backends
uvicorn app.main:app --reload       # http://localhost:8000  (open / for the chat UI)
```

The app boots on the deterministic path and **seeds a two-org demo corpus**.
Open `http://localhost:8000/` and sign in.

### Demo users (password: `password`)
| Email | Org | Role |
|-------|-----|------|
| alice@acme.test | org_acme | admin |
| bob@acme.test | org_acme | analyst |
| carol@acme.test | org_acme | viewer |
| dave@globex.test | org_globex | admin |

Try: *"What critical CVE is exposed on our Confluence server?"*, *"What are our
top risks this quarter?"*, *"who is the threat actor behind it?"* (follow-up).
Sign in as `dave@globex.test` and confirm you **never** see Acme's data.

## API

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/v1/auth/login` · `/refresh` · `/logout` · GET `/me` | auth (local JWT; OIDC via config) |
| POST | `/v1/chat` | one grounded, cited answer |
| GET | `/v1/chat/stream` | **SSE** token stream (EventSource: `?access_token=`) |
| GET/POST | `/v1/sessions` | list / create chats |
| GET/PATCH/DELETE | `/v1/sessions/{id}` | history / rename / delete |
| GET | `/v1/sessions/search?q=` | cross-session recall |
| GET | `/v1/capabilities` | modules + tools visible to the user |
| POST | `/v1/route/preview` | inspect routing for a question |
| POST | `/v1/ingest` | add documents to the corpus (analyst+) |
| GET/POST | `/v1/approvals[/{id}/approve|reject]` | action approval inbox |
| GET | `/healthz` · `/readyz` · `/metrics` | ops (`?fmt=prometheus`) |

## Configuration

Everything is config-gated with safe defaults (see `.env.example`). Switch on a
real backend by setting its provider + creds:

| Concern | Default | Production |
|--------|---------|-----------|
| LLM | `LLM_PROVIDER=deterministic` | `sglang` (Qwen 7B/72B + DeepSeek lanes) |
| Embeddings | `deterministic` | `tei` (Qwen3-Embedding) |
| Retrieval | `RETRIEVAL_BACKEND=memory` | `qdrant` (+ `RERANK_ENABLED=true`) |
| Chat store | `STORE_BACKEND=memory` | `postgres` (RLS) — run `asi-migrate` |
| Agent engine | `internal` | `langgraph` (durable checkpointing) |
| Auth | `local` (JWT) | `oidc` (JWKS) |
| Tracing | `none` | `langfuse` |

## Testing & eval

```bash
python -m pytest                    # 32 tests: contracts, isolation, agent, guardrails, RAG, API
python -m app.eval.runner           # golden-set eval + CI gate (route/refusal/block)
```
The eval gate (`EVAL_MIN_ROUTE`/`_REFUSAL`/`_BLOCK`) fails CI on regressions.
Live HTTP regression: `eval/promptfoo.yaml`.

## Add a capability module (~1 day, no core edit)

```bash
cp -r app/capabilities/_template app/capabilities/<feature>
# rename *.template → *.py, write tools + prompt, fill the manifest,
# add cap_<feature>_enabled to app/config.py, add evals/golden.jsonl, flip the flag.
```
See `app/capabilities/_template/README.md`. The supervisor routes to it, RBAC +
the gate apply, and it appears in `/v1/capabilities` — all from your manifest.

## Project layout

```
app/
  core/            # the frozen platform (see ARCHITECTURE.md §3)
    agent/ llm/ rag/ memory/ guardrails/ security/ mcp/ db/ ingestion/
    observability/ streaming/ action_gate/ api/  contracts.py registry.py bootstrap.py
  capabilities/    # drop-in features: reports (full), easm (mock), brand/aci (stubs), _template
  config.py  main.py  eval/
tests/   frontend/   infra/   eval/
```

## Security posture (this is a security product)

Multi-tenant isolation at every layer (JWT/OIDC → RLS → vector-store org filter →
per-org KG/gate), injection-resistant prompting, secrets redaction, **all
side-effecting actions human-gated**, append-only audit. Asset domains/IPs are
treated as subjects, not PII. Details in `ARCHITECTURE.md` §6.
