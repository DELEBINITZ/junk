# Architecture Review — User Guide Agent & Multi-Agent Routing Hardening

| | |
|---|---|
| **Status** | Proposed for review |
| **Author** | Platform / Agent team |
| **Reviewers** | CTO, Engineering |
| **Scope** | `security_intel` orchestrator + new `services/userguide-ingest` |
| **Change set** | commit `6dab840` on `remove-deterministic-stubs` |
| **Decision requested** | Approve for staging validation; sign off on 1 open design decision |

---

## 1. Executive summary

We added an **agentic capability to answer product how-to / dashboard-walkthrough
questions** from the FortiRecon user guide, and used the opportunity to **harden the
multi-agent routing architecture** for production.

Two workstreams:

- **A — User Guide RAG:** a standalone, independently-deployable ingestion service
  that parses the product documentation, embeds it, and loads it into a dedicated
  Qdrant collection; plus a new specialist agent that retrieves from it.
- **B — Orchestrator hardening:** structured routing, per-agent execution modes, a
  reflection/recovery loop, a routing eval harness, and one latent-bug fix.

The foundation (LangGraph **supervisor / orchestrator-workers** pattern) was already
sound; this work removes specific production gaps and adds the missing evaluation
layer. **Everything is verified locally except LLM-in-loop behavior**, which by
definition requires the staging model. We are asking the ARB to approve promotion to
staging for that final validation gate, and to sign off on one execution-mode
tradeoff.

---

## 2. Context & problem

1. **Capability gap:** the assistant could search threat reports but could not answer
   "how do I use the dashboard / where do I find X" — product-usage questions.
2. **Production-readiness question:** is the query→agent routing robust enough to
   ship, and does it extend to future multi-agent, multi-MCP collaboration?

## 3. Scope

**In scope:** doc ingestion + retrieval, the user-guide agent, routing/dispatch
hardening, execution-model efficiency, recovery on failure, routing evaluation.

**Out of scope:** the answer-LLM itself, EASM/BP agents (future), UI, auth changes.

---

## 4. What was delivered

### A. User Guide RAG

- **`services/userguide-ingest/`** — standalone service (own `pyproject.toml` +
  `Dockerfile`, **no imports from the app**). Contract with the platform is env-only
  (collection name, TEI endpoint, embedding dim). Deployable as a one-shot Job/Cron.
- **Ingestion pipeline:** HTML (local dir or crawl) → structure-aware parse → chunk →
  embed (same TEI model) → upsert to a **dedicated** Qdrant collection `user_guide_kb`.
- **Chunking strategy** (the hard part — iterated against the real Fortinet HTML):
  - sections split by `h1–h4`;
  - **tables rendered row-wise** as atomic `"Name — description"` per-topic chunks
    (Fortinet lays module lists out as 2-column tables) → precise single-topic recall;
  - prose packed to a word budget **with overlap** — no mid-sentence cuts;
  - **adjacent-duplicate dedup** (pages render each block twice for responsive layout);
  - **per-page navigation breadcrumb** extracted from the embedded TOC tree
    (`ul.toc`), stored as `breadcrumb`/`section_path` and prepended to the embedded
    text — hierarchical queries match, and the agent can give real nav directions
    (e.g. *"Attack Surface Management > EASM > EASM Dashboard"*).
- **`userguide` agent** — `search_user_guide` + `get_user_guide_page` tools;
  registered only when the collection is non-empty (fail-closed).

### B. Orchestrator hardening

- **Structured router** — routing decision returned via the model's function-calling
  (`RouterDecision` schema) instead of parsing free-text JSON; **text-parse fallback**
  if the backend can't do structured output. Guarantees a schema-valid decision.
- **Shared agent catalog** — router and planner now read the **same full**
  descriptions + capabilities (previously the router saw descriptions truncated to 60
  chars). Routing is no longer decided on lossy agent info.
- **Coverage-driven REFUSE** — refuse only when *no registered agent's domain covers*
  the request, so adding an agent expands scope without prompt edits.
- **Per-agent execution modes** — `tool_call` (single deterministic search, **no
  ReAct loop**) vs `react` (full tool-reasoning loop). Pure-retrieval agents skip an
  entire LLM round-trip. Current: `reports`, `userguide` = `tool_call`; `easm` =
  `react`.
- **Reflection loop** — if no agent returns productive content, the turn escalates to
  the planner **once** (capped) to recover a mis-routed query.
- **Routing observability** — decision confidence logged per query.

### C. Bug fixes

| Bug | Impact | Fix |
|---|---|---|
| `_is_error_finding` markers didn't match `"Agent '<id>' failed…"` (id interpolated) | Per-agent error/timeout text could leak into the user-facing synthesized answer | Broadened markers (`' failed`, `' timed out`) |
| Chunk double-counting — extractor matched both `<td>` and its child `<p>` | Every table cell embedded twice → duplicate vectors + duplicate hits | DFS walker emits each block exactly once; tables handled row-wise |
| Dirty titles — double-encoded `&nbsp;` + `\| User Guide` suffix | Bad titles + duplicated breadcrumb leaf | Title normalization + suffix strip |
| Router agent context truncated to 60 chars | Lossy routing for the ~80% SIMPLE path | Shared full catalog |

---

## 5. Architecture

Supervisor / orchestrator-workers on LangGraph. A central orchestrator owns control
flow deterministically — agents never call each other; the supervisor coordinates.
(Full detail: `docs/ARCHITECTURE.md`.)

```
security_gate ─┐                              ┌─ replan ─┐   (reflection, capped)
               ├─ route ─┬─ chitchat ─────────┤          │
classify ──────┘         ├─ plan ─ validate ──┤          ▼
                         └─ dispatch ──────────┴─ synthesize ─ output_guard ─ persist
```

- **Two-tier routing:** a cheap FAST-lane classifier handles ~80% of queries and
  generates the agent task inline (**planner bypassed**); only genuinely
  multi-domain queries reach the planner, which builds a DAG.
- **Parallel dispatch:** independent agents run concurrently (topological batches);
  dependent agents receive upstream findings.
- **Registry-driven:** adding an agent = one `AgentSpec` registration; router and
  planner auto-discover it. No routing-prompt or graph changes.

---

## 6. Key design decisions (rationale & alternatives)

| Decision | Chosen | Alternative | Why |
|---|---|---|---|
| Doc corpus location | **Separate collection** `user_guide_kb` | Reuse `reports_kb` | Docs are how-to pages, not threat intel; mixing pollutes both retrievals |
| Ingestion coupling | **Standalone service** | In-app cron | Independent deploy/scale/versioning; env-only contract; no app dependency |
| Chunk granularity | **Table row = atomic chunk**; prose packed | Fixed word windows | Single-topic queries need tight vectors; word windows split topics mid-sentence |
| Hierarchy | **Breadcrumb from TOC tree** into metadata + embedding | Page title only | Enables nav directions + hierarchical query match |
| Retrieval-agent execution | **`tool_call` (no ReAct)** | Always ReAct | Cuts an LLM round-trip + failure modes for "run one search and return" |
| Router output | **Structured + text fallback** | Free-text JSON parse | 7B models emit malformed JSON; schema-guarantee with graceful degradation |
| Failure recovery | **Reflection re-plan (capped 1)** | One-shot | Recovers mis-routed SIMPLE queries; bounded latency |

---

## 7. Testing & verification (honest status)

| Area | Status | How |
|---|---|---|
| Ingestion → Qdrant → retrieval | ✅ Verified | End-to-end on 6 pages (29 chunks) **at dim 384** via a local embed shim |
| Chunking quality | ✅ Verified | Inspected real Fortinet HTML; dedup + table + breadcrumb confirmed; retrieval returns correct page/heading |
| Orchestrator graph build | ✅ Verified | Compiles; reflection edges present |
| Routing plumbing | ✅ Verified | `run_eval.py --fake-llm` 13/13; `tests/test_routing.py` 5/5 (no external services) |
| Compile + lint | ✅ Verified | ruff/py_compile on all changes |
| **Routing decision quality** | ❌ Not verified | Needs live LLM (no vLLM locally) |
| **Structured-output happy path on Qwen** | ❌ Not verified | Fallback covers failure; happy path unproven |
| **Planner ReAct / synthesis / reflection replan end-to-end** | ❌ Not verified | Needs live LLM |
| **Retrieval at production dim (2560)** | ❌ Not verified | Local used 384-dim shim; must re-ingest in staging |

The unverified items are structurally sound and guarded; they are the **staging
validation gate** (`docs/STAGING_CHECKLIST.md`), not open code risks.

---

## 8. Risks & mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| Embedding-dim mismatch (local 384 vs prod 2560) → silent empty retrieval | High | **Blocker:** re-ingest in staging at `EMBEDDING_DIM=2560`; verify collection size |
| Structured output unsupported on the serving backend | Medium | Automatic text-parse fallback; log-monitored |
| `reports=tool_call` drops by-ID/filter query paths | Medium | **Open decision (below)**; one-line flip to `react` |
| Cross-domain queries mis-routed to a single agent | Medium | Reflection loop + eval golden cases; see roadmap for one-sided-reflection |
| Per-query LLM cost/latency (4–6 calls on complex path) | Medium | FAST-lane router, SIMPLE bypass, `tool_call` mode, parallel dispatch; measure under load |
| Flat routing prompt scales to ~15–20 agents | Low (3 today) | Hierarchical/embedding agent selection when needed |

---

## 9. Rollout plan

1. Deploy `services/userguide-ingest`; **ingest the full guide at dim 2560** before
   app start (agent gates on a non-empty collection).
2. Deploy the app; confirm `userguide` registers.
3. Run `tests/eval/run_eval.py` (real LLM) → routing accuracy number.
4. Full-turn smoke + reflection check (`docs/STAGING_CHECKLIST.md`).
5. Run `pytest tests/`.
6. Promote only after the gate passes. CI already runs the routing plumbing gate
   (`.github/workflows/tests.yml`) on every PR.

---

## 10. Open decisions for the ARB

1. **`reports` execution mode.** `tool_call` is cheaper but narrows the reports agent
   to semantic search at runtime — the by-ID (`summarize report <id>`) and filter
   (`TLP:RED`) tool paths are not exercised. **Recommendation:** keep `tool_call` if
   those query types are rare; else flip to `react` (one line). *Requires a call.*
2. **Investment in precise multi-agent correlation** (see §11) — approve now or defer.

---

## 11. Extensibility & roadmap

**Adding agents (e.g. EASM with DB + MCP, Brand Protection with a different MCP):**
supported today with **no architecture change** — register an `AgentSpec` with its own
tool set (local + DB + MCP tools, mixed), `mode="react"`, RBAC (`min_role`), and
approval-gated side-effecting tools. The router/planner auto-discover it.

**Two agents collaborating on an overlapping query (e.g. BP ∩ EASM):** supported by the
COMPLEX → planner → DAG → parallel/sequential dispatch → synthesis path (verified
plumbing). To make the **join precise** (not just "both ran"), three incremental
additions — all extensions to the existing graph, not redesigns:

1. **Correlation step/tool** — deterministic intersection of two agents' result sets
   instead of the synthesis LLM inferring the overlap from concatenated text.
2. **One-sided reflection** — re-plan when a query looks cross-domain but only one
   domain answered (today reflection triggers only on *empty* results). Can reuse the
   already-logged routing `confidence`.
3. **Cross-domain eval coverage** — expand the golden set; gate on it.

**Scale:** flat single-prompt routing is fine to ~15–20 agents; beyond that, move to
hierarchical or embedding-based agent selection.

---

## 12. Appendix — change surface

- **New service:** `services/userguide-ingest/` (script, Dockerfile, pyproject, env, README)
- **New agent:** `src/security_intel/agents/userguide/`, `tools/userguide_search.py`,
  `prompts/userguide.py`
- **Orchestrator/registry:** structured router, catalog, execution modes, reflection,
  bug fix
- **Config:** `USER_GUIDE_COLLECTION`
- **Tests/eval:** `tests/eval/` (`run_eval.py`, `_fake_llm.py`, `golden_queries.json`),
  `tests/test_routing.py`, `.github/workflows/tests.yml`
- **Docs:** `docs/ARCHITECTURE.md`, `docs/STAGING_CHECKLIST.md`, this ARB
