"""EASM tools (MOCK data backend) — the module's typed MCP surface.

Each function below is a :class:`Tool` (contracts.py): a typed function the agent
can call, declared with ``@tool``. Together they ARE the EASM module's data path
(it has no retriever). They prove the multi-module + MCP pattern end to end WITHOUT
a real EASM database: every tool serves org-scoped MOCK data exactly as a real
``easm-mcp`` server would — same inputs, same ToolResult/ToolError shapes — except
a real server would query the asset DB instead of the ``_MOCK`` dict.

Three tools are READ (viewer); the fourth, ``trigger_rescan``, is SIDE-EFFECTING
and ANALYST-only, so it is gated by the human action gate. Read it last — it is
the whole reason this module exists as a teaching example.

To go live: replace the ``_MOCK`` lookups with calls to your asset store (and,
optionally, promote this module to a standalone MCP server — see core/mcp/server.py).
The tools' signatures and return contracts would not change.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from app.core.contracts import Autonomy, Citation, ToolContext, ToolResult, tool

# Org-scoped MOCK inventory: a stand-in for the EASM asset database, keyed by tenant.
# CRITICAL SECURITY POINT: the data is indexed by ``org_id`` and the tools look it up
# using ``ctx.org_id`` (the trusted, token-derived tenant) — never an org passed in
# args. That is what makes cross-tenant access impossible here, and is exactly how a
# real backend would scope its queries. In production this dict becomes a DB call.
_MOCK: dict[str, dict] = {
    "org_acme": {
        "assets": [
            {"asset_id": "A-1", "host": "admin.acme.test", "type": "host", "tech": "Atlassian Confluence", "exposed": True},
            {"asset_id": "A-2", "host": "shop.acme.test", "type": "host", "tech": "nginx", "exposed": True},
            {"asset_id": "A-3", "host": "vpn.acme.test", "type": "vpn", "tech": "OpenVPN", "exposed": True},
            {"asset_id": "A-4", "host": "api.acme.test", "type": "host", "tech": "Kong", "exposed": True},
        ],
        "exposures": [
            {"exposure_id": "E-1", "asset": "admin.acme.test", "issue": "CVE-2023-22515 critical auth bypass on Confluence", "severity": "critical"},
            {"exposure_id": "E-2", "asset": "shop.acme.test", "issue": "expired TLS certificate", "severity": "medium"},
        ],
        "changes": [
            {"asset": "api.acme.test", "change": "new subdomain discovered, port 443 open", "at": "2026-05-30"},
        ],
    },
    "org_globex": {
        "assets": [
            {"asset_id": "G-1", "host": "vpn.globex.test", "type": "vpn", "tech": "RDP", "exposed": True},
            {"asset_id": "G-2", "host": "www.globex.test", "type": "host", "tech": "IIS", "exposed": True},
        ],
        "exposures": [
            {"exposure_id": "GE-1", "asset": "vpn.globex.test", "issue": "RDP (3389) exposed to the internet — brute-force/ransomware risk", "severity": "high"},
        ],
        "changes": [],
    },
}


# The single tenant-scoping helper every read tool funnels through. It reads the org
# STRICTLY from ``ctx.org_id`` and returns that tenant's slice (or an empty slice for
# an unknown org). Because args never carry an org, no tool can reach another tenant.
def _org(ctx: ToolContext) -> dict:
    return _MOCK.get(ctx.org_id, {"assets": [], "exposures": [], "changes": []})


# Per-tool argument schemas. pydantic gives validation + a JSON schema for the LLM.
# Note all read-tool args are optional (defaults) and one carries free text, so the
# heuristic planner can auto-invoke them on a broad question (see specialist.py).
class QueryAssetsArgs(BaseModel):
    query: str = Field(default="", description="Free-text filter over host/tech/type. Empty = all assets.")
    limit: int = Field(default=10, ge=1, le=100)


class ExposuresArgs(BaseModel):
    severity: str | None = Field(default=None, description="Filter: critical|high|medium|low.")


class ChangesArgs(BaseModel):
    since: str | None = Field(default=None, description="ISO date lower bound; omit for all recent.")


# Args for the gated action. ``asset`` is REQUIRED (no default): the heuristic planner
# cannot infer it from a plain question, so the rescan is never auto-fired — it only
# runs when something explicitly supplies the target and the human gate approves.
class RescanArgs(BaseModel):
    asset: str = Field(description="Host or asset_id to rescan.")


# READ tool, viewer-level, not side-effecting. ``description`` is what the LLM reads
# to know when to call it; ``args_schema`` validates the input it passes.
@tool(name="query_assets",
      description="List the organization's external assets (domains, hosts, VPNs) matching a query.",
      args_schema=QueryAssetsArgs, rbac_role="viewer")
async def query_assets(args: QueryAssetsArgs, ctx: ToolContext):
    # Start from THIS tenant's assets only (org comes from ctx, never args).
    all_assets = _org(ctx)["assets"]
    tokens = {t for t in re.findall(r"[a-z0-9.]+", args.query.lower()) if len(t) > 2}
    def matches(a: dict) -> bool:
        hay = f"{a['host']} {a['tech']} {a['type']}".lower()
        return any(t in hay for t in tokens)
    # Specific term -> filter; broad natural-language question -> return all.
    narrowed = [a for a in all_assets if matches(a)] if tokens else []
    assets = (narrowed or all_assets)[: args.limit]
    # Build one Citation per asset so each fact in the answer can be grounded back to a
    # specific asset. The Citation snippet is the human-readable evidence string.
    cites = [Citation(doc_id=a["asset_id"], source="easm", title=a["host"],
                      snippet=f"Exposed external asset {a['host']} running {a['tech']} "
                              f"(type {a['type']}), internet-exposed={a['exposed']}.") for a in assets]
    # SUCCESS: structured ``data`` (for the agent to reason over) + citations (evidence).
    return ToolResult(data={"assets": assets, "count": len(assets)}, citations=cites)


# READ tool, viewer-level. Optional ``severity`` filter; org scoped via ctx.
@tool(name="get_exposures",
      description="Get current external exposures/findings for the organization, optionally by severity.",
      args_schema=ExposuresArgs, rbac_role="viewer")
async def get_exposures(args: ExposuresArgs, ctx: ToolContext):
    exps = _org(ctx)["exposures"]
    if args.severity:
        exps = [e for e in exps if e["severity"] == args.severity.lower()]
    cites = [Citation(doc_id=e["exposure_id"], source="easm", title=e["asset"],
                      snippet=f"Exposure on asset {e['asset']}: {e['issue']}. Severity {e['severity']}.")
             for e in exps]
    return ToolResult(data={"exposures": exps, "count": len(exps)}, citations=cites)


# READ tool, viewer-level. Optional ``since`` lower bound; org scoped via ctx.
@tool(name="get_asset_changes",
      description="Get recent changes to the external attack surface (new subdomains, newly opened ports).",
      args_schema=ChangesArgs, rbac_role="viewer")
async def get_asset_changes(args: ChangesArgs, ctx: ToolContext):
    changes = _org(ctx)["changes"]
    if args.since:
        changes = [c for c in changes if c.get("at", "") >= args.since]
    cites = [Citation(doc_id=f"chg-{i}", source="easm", title=c["asset"],
                      snippet=f"Attack-surface change on {c['asset']}: {c['change']} ({c['at']}).")
             for i, c in enumerate(changes)]
    return ToolResult(data={"changes": changes, "count": len(changes)}, citations=cites)


# THE GATED TOOL. ``side_effecting=True`` is the key flag: it means this tool does NOT
# run inline when the agent "calls" it. The MCP boundary intercepts it and routes it to
# the human action gate (rule #2 in contracts.py) — a reviewer must approve before it
# ever executes. ``rbac_role="analyst"`` additionally bars viewers, and the specialist
# never auto-invokes it at all (its ``_read_tools`` filters out side-effecting tools).
# ``autonomy=Autonomy.SUGGEST`` records that the most this tool does on its own is DRAFT
# the action for approval. This is how the platform lets an agent propose an action
# (a rescan) while keeping a human firmly in the loop.
@tool(name="trigger_rescan",
      description="Request an on-demand external rescan of an asset. Side-effecting: requires approval.",
      args_schema=RescanArgs, side_effecting=True, rbac_role="analyst", autonomy=Autonomy.SUGGEST)
async def trigger_rescan(args: RescanArgs, ctx: ToolContext):
    # By the time this body runs, the gate has ALREADY approved it — so the handler can
    # assume authorization and just do the work. Here the mock simply acknowledges; a
    # real backend would enqueue the scan job (scoped to ctx.org_id) and return its id.
    return ToolResult(data={"asset": args.asset, "status": "rescan_queued"})


# The module's full tool surface, imported by the manifest as ``tools=TOOLS``.
TOOLS = (query_assets, get_exposures, get_asset_changes, trigger_rescan)

__all__ = ["TOOLS", "query_assets", "get_exposures", "get_asset_changes", "trigger_rescan"]
