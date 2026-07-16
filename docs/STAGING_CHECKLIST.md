# Staging validation checklist — User Guide agent + orchestrator hardening

Everything below was implemented and verified locally EXCEPT the LLM-in-loop
behavior (no vLLM/Qwen locally) and retrieval at the real embedding dim (local
tests used a 384-dim shim; production is 2560). Run this gate in staging before
promoting. Do the blockers in order; then the validation steps in any order.

## 0. Config (once)

- `USER_GUIDE_COLLECTION=user_guide_kb` present in the app env (added to
  `.env.staging` / `.env.production`).
- Ingestion service env points at the SAME Qdrant + TEI as the app, at
  `EMBEDDING_DIM=2560` (see `services/userguide-ingest/.env.example`).

## 1. BLOCKER — ingest the user guide at the real embedding dim

The local `user_guide_kb` is dim 384 (throwaway). Re-ingest with the real TEI:

```bash
# from the staging host (or a job with network to Qdrant + TEI)
docker build -t fortirecon-userguide-ingest services/userguide-ingest
docker run --rm \
  -e QDRANT_URL=http://<STAGING_HOST>:6333 -e QDRANT_API_KEY=... \
  -e TEI_EMBED_URL=http://<STAGING_HOST>:9001 -e EMBEDDING_DIM=2560 \
  -e USER_GUIDE_COLLECTION=user_guide_kb \
  -v /path/to/full/user-guide/html:/data \
  fortirecon-userguide-ingest --html-dir /data
```

- Download the **full** guide (you have 6 pages; the child pages — "Viewing
  discovered assets summary" etc. — are not in yet). `--html-dir` is the reliable
  path (the docs TOC is JS-rendered; `--crawl` misses siblings).
- Verify: `curl -s $QDRANT/collections/user_guide_kb | jq '.result.points_count,.result.config.params.vectors.size'`
  → non-zero points, **size 2560**.

## 2. BLOCKER — ingest before app start

The User Guide agent registers only when `user_guide_kb` is non-empty. Ingest first,
then (re)start the app. Confirm startup log: `Registered agent: userguide (…, mode=tool_call)`.

## 3. Run the routing eval (real LLM) — the accuracy number

```bash
uv run python tests/eval/run_eval.py           # human table
uv run python tests/eval/run_eval.py --json     # for a dashboard/CI
```

Needs live fast+standard LLM + Qdrant (no Postgres). Fake-mode proved plumbing
(13/13); this gives real decision accuracy. Investigate any FAIL; extend
`tests/eval/golden_queries.json` with real misroutes.

## 4. Confirm the structured router works on Qwen/vLLM

Grep app logs while running the eval:

- If you see `Structured router unavailable …` or `Structured router failed …`, the
  model backend isn't honoring function-calling and it silently fell back to the
  text parser (functional, but less reliable). Either enable tool/guided-JSON on the
  vLLM server or accept the fallback.

## 5. Full-turn smoke (manual, via the chat API)

| Query | Expect |
|---|---|
| "hi" | DIRECT, no agent |
| "walk me through the EASM dashboard" | userguide answer with nav path (`Attack Surface Management > EASM > EASM Dashboard`) |
| "what do we know about CVE-2024-3400?" | reports answer |
| "compare our exposed assets against recent threats" | planner → easm + reports (needs EASM MCP configured) |
| "write me a python script" | REFUSE |

Also confirm: streaming works, PII redaction runs (Presidio warmed), and no internal
fields (scores/ids) leak into answers.

## 6. Reflection loop (needs real LLM)

Ask something a registered agent genuinely can't answer → confirm one re-plan
(`No productive agent results — escalating to planner`) then a graceful answer, and
that it does NOT loop (capped at 1).

## 7. Run the test suites

```bash
uv run pytest tests/test_routing.py -v          # routing plumbing (no services) — also in CI
uv run pytest tests/ -v                          # full suite (needs Presidio + spaCy model)
```

## Open decision

- `reports` is `mode="tool_call"` → only `search_reports` runs; the by-ID
  (`summarize report <id>`) and filter (`TLP:RED`) paths are NOT exercised. If those
  queries matter, set `mode="react"` on the reports `AgentSpec` in
  `main._register_agents` (one line) and re-test.

## Not blockers, but track

- Per-query LLM cost/latency (pipeline can be 4–6 calls) — measure under load.
- Only a subset of the guide ingested — ingest the full tree.
- Flat single-prompt routing scales to ~15–20 agents; revisit if you add many more.
