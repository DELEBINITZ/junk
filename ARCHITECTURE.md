# Architecture — Agentic Security Intelligence Platform

A production-grade **agentic RAG** platform over org-scoped security intelligence.
Today it answers questions about **security reports**; it is built as a **frozen
core + drop-in capability modules** so EASM, Brand Protection, and ACI (and
anything after) bolt on as ~1-day modules with **no core change**.

This document is the design + a map into the code. Built greenfield in FastAPI;
runs with **zero infra** by default and flips to a self-hosted GPU stack by config.

---

## 1. The one idea: core + modules

```
┌──────────────── CORE  (app/core — built once, rarely touched) ───────────────┐
│ supervisor/router · graph engine (built-in + LangGraph) · retrieval pipeline │
│ guardrails · memory (sessions + KG seam) · MCP tool boundary · action gate   │
│ streaming (SSE) · security (authN/Z, org isolation) · observability · registry│
└───────────────────────────────┬──────────────────────────────────────────────┘
                                 │ discovers + wires by manifest at boot
┌────────────────────────────────┴──────────────────────────────────────────────┐
│      CAPABILITY MODULES  (app/capabilities/* — drop-in features)               │
│   reports │ easm │ brand │ aci │ <next> ...   each = manifest + tools (+ prompt,│
│                                                retriever, evals, ontology, …)   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

The core **never names a feature**. The list of features is *computed* from
registered manifests (`app/core/registry.py`). With only `reports` registered,
the supervisor behaves like a single agent; register EASM/Brand/ACI and it becomes
the multi-pillar platform — same core code.

> **Litmus test:** a new engineer ships a working, evaluated, permission-gated
> feature by adding one module dir + a manifest — opening **no file under
> `app/core/`**. See `app/capabilities/_template/README.md`.

---

## 2. Request flow (one chat turn)

`app/core/agent/orchestrator.py` owns the turn; reasoning is a graph of nodes
(`app/core/agent/nodes.py`) run by an engine (`engines.py`).

```
client ──HTTPS/SSE──► /v1/chat[/stream]  (app/core/api/chat_router.py)
   │ auth: verify token → SecurityContext (org_id, user, roles)   [security/]
   ▼
Orchestrator.run_turn / stream_turn
   load history + rolling summary (memory/) → persist user msg
   ▼  graph:
   input_guardrail ─(blocked?)─► END
        │ PII/secret redact · injection screen · topic safety   (guardrails/)
   route                         supervisor picks module(s) from manifest hints
        │                        (heuristic or LLM)             (agent/supervisor.py)
   dispatch                      ONE specialist per routed module, IN PARALLEL —
        │                        each scoped to its OWN tools (retrievers + tools
        │                        via MCP, RBAC+gate); findings merged → one
        │                        org-scoped, numbered context block (agent/specialist.py)
   answer (synthesize)           LLM joins findings, cites [n]; streams tokens (llm/)
   output_guardrail              groundedness + citation check; PII-leak redact
   ▼
persist assistant msg · update summary · KG observation · emit `done`
```

Tenant identity flows from the **verified token only** (never tool args), so a
prompt cannot make the agent cross orgs.

---

## 3. Component map (where things live)

| Concern | Path | Notes |
|--------|------|------|
| Frozen contracts | `app/core/contracts.py` | `Tool`, `CapabilityManifest`, `ToolContext`, `Retriever`, `ActionHandler`, … (SemVer'd) |
| Config | `app/config.py` | every backend config-gated; deterministic defaults |
| Registry | `app/core/registry.py` | manifest discovery, per-org capability views |
| Supervisor / router | `app/core/agent/supervisor.py` | manifest-driven; heuristic + LLM modes |
| Specialists | `app/core/agent/specialist.py` | one per module, run **in parallel**, each **scoped to its own tools** (tool schemas never co-locate → scales to 100s of tools); pluggable per module via `manifest.specialist` |
| Graph engines | `app/core/agent/graph.py`, `engines.py` | built-in (zero-dep) + real **LangGraph** + checkpointer |
| Nodes | `app/core/agent/nodes.py` | guardrail→route→gather→answer→guardrail |
| Orchestrator | `app/core/agent/orchestrator.py` | sessions, persistence, SSE streaming |
| LLM lanes | `app/core/llm/` | deterministic · SGLang · OpenAI-compat; fast/standard/deep |
| Retrieval | `app/core/rag/` | embeddings, **org-filtered** vector store (mem/Qdrant), reranker, time filters, citations |
| Memory | `app/core/memory/` | sessions+messages (mem/Postgres-RLS), rolling summary, KG seam |
| Guardrails | `app/core/guardrails/` | input/output spine; model seams (Prompt/Llama Guard) |
| MCP boundary | `app/core/mcp/` | in-process now; remote client + standalone server seams |
| Action gate | `app/core/action_gate/` | human-approval inbox for side-effecting tools |
| Security | `app/core/security/` | local JWT + OIDC, RBAC, revocation, org context |
| DB / RLS | `app/core/db/` | async pool, `org_transaction` (RLS), migrations |
| Ingestion | `app/core/ingestion/` | `/ingest` path + event-bus seam + parsers |
| Observability | `app/core/observability/` | logging, Langfuse seam, metrics, audit |
| API | `app/core/api/` | routers, schemas, error handlers, concurrency middleware |
| App factory | `app/main.py` | lifespan, boot/seed, system endpoints |
| Composition root | `app/core/bootstrap.py` | wires everything from config |
| Modules | `app/capabilities/` | `reports` (full), `easm` (mock), `brand`/`aci` (stubs), `_template` |

---

## 4. The capability-module contract (chassis)

A module is a directory with a `MANIFEST` (`app/core/contracts.py:CapabilityManifest`):

```python
MANIFEST = CapabilityManifest(
    id="easm", version="1.0.0", display_name="...",
    enabled_flag="cap_easm_enabled",          # deployment composition
    tools=TOOLS,                              # typed, MCP-exposable, RBAC'd
    retrievers=(CollectionRetriever(...),),   # optional corpus binding (RAG)
    routing_hints=(RoutingHint(intents=..., examples=...),),  # supervisor routes from these
    default_autonomy=Autonomy.SUGGEST,
    rbac={"trigger_rescan": "analyst"},       # min role per tool
)
```

At boot the registry **discovers** manifests, validates contracts, merges
ontology, and computes per-org views. The supervisor routes from `routing_hints`;
RBAC + the action gate come from the manifest. **Adding a feature edits no core
file.** Two corpus shapes are demonstrated: `reports` (RAG corpus + retriever) and
`easm` (tool-backed structured data — its "MCP surface").

### Contract tests (the safety net)
`tests/test_contracts.py` runs against **every** registered module: tool-schema
validity, errors-as-data, tenant context, **gate enforcement** for side-effecting
tools, and golden-question routing. Green contracts + green eval = mergeable.

---

## 5. Orchestrator + MCP strategy (answers "what's the MCP architecture?")

- **Now:** tools run **in-process** behind `InProcessMCPClient` (one deployable).
  The registry the client sees is computed from manifests; **RBAC and the action
  gate are enforced at this boundary** for every call, whatever agent makes it.
- **Later (promotion):** when a module needs its own deploy/scale, team ownership,
  or a hard security boundary around adversary-controlled data, package its tools
  as a **standalone MCP server** (`app/core/mcp/server.py:make_mcp_app` → run
  `easm-mcp`/`brand-mcp`/`aci-mcp`) and point a `RemoteMCPClient` at it. **The tool
  contracts, schemas, and org-scoping don't change** — it's a transport swap.
- The trusted `org_id` is propagated as a short-lived **service token**, never as
  a tool argument; the remote server re-derives org from it.

This gives the orchestrator-over-MCP pattern now, without paying for a distributed
system before a module needs it.

---

## 6. Security & multi-tenant isolation (org_id at every layer)

| Layer | Enforcement |
|------|-------------|
| AuthN | local JWT (`security/jwt.py`) or OIDC JWKS (`security/oidc.py`); refresh + revocation |
| AuthZ | ordered RBAC (`viewer<analyst<admin`), per-tool min role from manifest |
| Vector store | **mandatory `org_id` filter** on every search (`rag/vector_store.py`, `qdrant_backend.py`) |
| Chat store | Postgres **RLS** via `org_transaction` (`db/postgres.py`, `0001_init.sql`) |
| Tools | `org_id` from `ToolContext` (token-derived), never args |
| Action gate | approvals are org-scoped (`action_gate/gate.py`) |
| KG | per-`org:user` namespace (`memory/kg.py`) |

**Injection defense** (the agent reads adversary-controlled text): retrieved
content is labeled untrusted in the prompt and treated as data; user input is
screened; **all side-effecting actions are gated**, so injection can't fire one.
Secrets are redacted before the LLM/logs; asset domains/IPs are *not* PII (this is
a security product). Isolation has dedicated tests (`tests/test_isolation.py`) and
is a launch gate.

---

## 7. Memory (ChatGPT/Claude-style)

`app/core/memory/` — durable sessions + full message history per (org, user),
auto titles, **rolling summaries** to bound context, and **cross-session recall**
(`/v1/sessions/search`, Postgres FTS in prod). Long-term entity memory is a seam
(`kg.py`, NoOp default → Zep/Graphiti when `KG_PROVIDER=zep`) — the connective
tissue for future cross-pillar reasoning.

---

## 8. Scalability & reliability

- **Three inference lanes** (`llm/lanes.py`): fast (routing/summaries), standard
  (answers), deep (hard analysis, opt-in + quota-capped) — most traffic runs cheap.
- **Admission control / backpressure** (`api/middleware.py`): bounded concurrency
  → graceful 503, never a brownout.
- **Stateless chat service** → horizontal scale; **resumable runs** via the
  LangGraph checkpointer (`engines.py`) — the basis for the durable approval gate.
- **Retrieval** scales via Qdrant sharding/quantization; **ingestion** is off the
  chat path (external cron + `/ingest` + event-bus/Temporal seam).
- Targets & the go-live gate: blueprint §12/§16 and `infra/README.md`.

---

## 9. Roadmap (this build → full platform)

| Stage | State |
|------|-------|
| **v1 — production report chat** | ✅ built here: chassis + reports module, guardrails, streaming, memory, isolation, eval |
| EASM (read) | ✅ example module (mock data) — proves multi-module + MCP routing |
| **Parallel specialist sub-agents** | ✅ active: one specialist/module, parallel fan-out, tool-isolated, synthesize step (`agent/specialist.py`) — the 100s-of-tools scaling design |
| Brand + ACI (read) | ✅ drop-in stubs (flip the flag) — wire real backends/MCP servers |
| Scheduled briefings + KG entity-join | ⛶ seams: shared KG (`memory/kg.py`), proactive cron |
| Action layer + approval inbox | ⛶ gate machinery + `ActionHandler` interface shipped; handlers later |
| Graduated autonomy (Layer A) | ⛶ `auto_approves` hook present; promote per-action after proven precision |

Source design: `agentic-rag-system/MASTER_BLUEPRINT.md` (files 00–16).
