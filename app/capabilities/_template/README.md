# Capability module template

Target: ship a working, evaluated, permission-gated feature in ~1 day, touching
no file under `app/core/`.

## Steps
1. `cp -r app/capabilities/_template app/capabilities/<feature>`
2. Rename `manifest.py.template` -> `manifest.py`, `tools.py.template` -> `tools.py`.
   Replace every `<feature>` placeholder.
3. Write 3-6 typed tools (rich docstrings, structured returns, errors-as-data via
   `ToolException`; mark `side_effecting=True` for actions).
4. Optional: add `retrievers.py` (bind a corpus to the shared pipeline),
   `ingestion.py` (a connector), `ontology.py` (KG node/edge types).
5. Add `prompts/v1.md` and `evals/golden.jsonl`.
6. Fill the manifest (id, tiers, routing_hints, rbac, autonomy, min_core_version).
7. The registry discovers the module at boot. Ship behind a flag → canary → promote.

## Contract tests a module must pass (plan §5.5)
schema validity · errors-as-data · ORG ISOLATION · gate enforcement ·
ontology merge · routing · prompt-injection resistance.

## Promoting to a standalone MCP server (plan §6.4)
When the module needs independent ownership/scale or a hard boundary around
adversary-controlled data, package its tools as an MCP server and point the
tool handlers at a remote MCP client. The Tool contract is unchanged.
