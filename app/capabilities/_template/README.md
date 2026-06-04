# Capability module template

Add a product feature in ~1 day **without opening any file under `app/core/`**.

## Recipe

1. **Copy** this directory:
   ```bash
   cp -r app/capabilities/_template app/capabilities/<feature>
   cd app/capabilities/<feature>
   mv manifest.py.template manifest.py
   mv tools.py.template tools.py
   ```
2. **Write the tools** (`tools.py`) — typed args, a docstring the LLM reads,
   structured `ToolResult`/`ToolError`. Mark side-effecting tools
   (`side_effecting=True`) — they auto-route through the human-approval gate.
3. **Write the prompt** at `prompts/v1.md` (persona + few-shots).
4. **Fill the manifest** (`manifest.py`): id, a crisp `description`, RBAC, autonomy.
   Routing is by MEANING (the description + tool descriptions ARE the routing signal —
   no keyword lists). If the feature has a corpus, bind a `CollectionRetriever` (the
   corpus is filled by the external ingestion cron; the module only reads). All optional.
5. **Add a flag** `cap_<feature>_enabled: bool = False` to `app/config.py`.
6. **Add evals** at `evals/golden.jsonl` (golden questions + expected routing).
   CI runs them.
7. **Enable + ship**: set `CAP_<FEATURE>_ENABLED=true`, canary, promote.

The supervisor will route to it, RBAC + the action gate will apply, and it will
appear in `/v1/capabilities` — all derived from your manifest. No core edit.

## What a module may contain

| File | Required? | Purpose |
|------|-----------|---------|
| `manifest.py` | yes | declarative wiring (`MANIFEST`) |
| `tools.py` | yes (or a retriever) | the typed tools |
| `prompts/v1.md` | recommended | specialist persona |
| `evals/golden.jsonl` | yes (CI gate) | golden questions |
| `retrievers` (in manifest) | if it has a corpus | RAG binding (read; the corpus is cron-fed) |
| `tools.py` (remote) | to back with MCP | add `MCP_URLS[<id>]` — no code edit |

## Promote to a standalone MCP server (later)

When the feature needs its own deploy/scale or a hard security boundary, package
its tools as a standalone MCP server (`app/core/mcp/server.py:make_mcp_app`) and
point a `RemoteMCPClient` at it. The tool contracts don't change.
