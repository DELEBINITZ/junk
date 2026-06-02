"""EASM tools (mock data backend).

This module proves the multi-module + MCP pattern end to end WITHOUT a real EASM
database: tools serve org-scoped mock data exactly as a real ``easm-mcp`` server
would (which would query the asset DB instead). Includes a gated, side-effecting
``trigger_rescan`` to exercise the human-approval action gate via a real module.

To go live: replace ``_MOCK`` lookups with calls to your asset store (and,
optionally, promote this module to a standalone MCP server — see core/mcp/server.py).
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from app.core.contracts import Autonomy, Citation, ToolContext, ToolResult, tool

# Org-scoped mock inventory (stand-in for the EASM asset database).
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


def _org(ctx: ToolContext) -> dict:
    return _MOCK.get(ctx.org_id, {"assets": [], "exposures": [], "changes": []})


class QueryAssetsArgs(BaseModel):
    query: str = Field(default="", description="Free-text filter over host/tech/type. Empty = all assets.")
    limit: int = Field(default=10, ge=1, le=100)


class ExposuresArgs(BaseModel):
    severity: str | None = Field(default=None, description="Filter: critical|high|medium|low.")


class ChangesArgs(BaseModel):
    since: str | None = Field(default=None, description="ISO date lower bound; omit for all recent.")


class RescanArgs(BaseModel):
    asset: str = Field(description="Host or asset_id to rescan.")


@tool(name="query_assets",
      description="List the organization's external assets (domains, hosts, VPNs) matching a query.",
      args_schema=QueryAssetsArgs, rbac_role="viewer")
async def query_assets(args: QueryAssetsArgs, ctx: ToolContext):
    all_assets = _org(ctx)["assets"]
    tokens = {t for t in re.findall(r"[a-z0-9.]+", args.query.lower()) if len(t) > 2}
    def matches(a: dict) -> bool:
        hay = f"{a['host']} {a['tech']} {a['type']}".lower()
        return any(t in hay for t in tokens)
    # Specific term -> filter; broad natural-language question -> return all.
    narrowed = [a for a in all_assets if matches(a)] if tokens else []
    assets = (narrowed or all_assets)[: args.limit]
    cites = [Citation(doc_id=a["asset_id"], source="easm", title=a["host"],
                      snippet=f"Exposed external asset {a['host']} running {a['tech']} "
                              f"(type {a['type']}), internet-exposed={a['exposed']}.") for a in assets]
    return ToolResult(data={"assets": assets, "count": len(assets)}, citations=cites)


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


@tool(name="trigger_rescan",
      description="Request an on-demand external rescan of an asset. Side-effecting: requires approval.",
      args_schema=RescanArgs, side_effecting=True, rbac_role="analyst", autonomy=Autonomy.SUGGEST)
async def trigger_rescan(args: RescanArgs, ctx: ToolContext):
    # Only runs post-approval (the gate guarantees this). Mock acknowledges.
    return ToolResult(data={"asset": args.asset, "status": "rescan_queued"})


TOOLS = (query_assets, get_exposures, get_asset_changes, trigger_rescan)

__all__ = ["TOOLS", "query_assets", "get_exposures", "get_asset_changes", "trigger_rescan"]
