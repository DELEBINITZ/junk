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
4. **Fill the manifest** (`manifest.py`): id, routing hints, RBAC, autonomy.
   If the feature has a corpus, bind a `CollectionRetriever`; if it has its own
   data feed, add an `IngestionConnector`; if it adds entities, declare an
   `OntologyContribution`. All optional.
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
| `retrievers` (in manifest) | if it has a corpus | RAG binding |
| `ingestion.py` | if it has a data feed | event → corpus/KG |
| `ontology` (in manifest) | if it adds entities | KG slice |
| `seed.py` (`seed_demo`) | dev only | demo data |

## Promote to a standalone MCP server (later)

When the feature needs its own deploy/scale or a hard security boundary, package
its tools as a standalone MCP server (`app/core/mcp/server.py:make_mcp_app`) and
point a `RemoteMCPClient` at it. The tool contracts don't change.
