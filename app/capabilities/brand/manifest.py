"""Brand Protection / Digital Risk Protection — drop-in module stub.

Another single-file capability (tool + manifest in one place), structurally a twin
of the ACI stub. It detects impersonation: lookalike domains, phishing sites, fake
apps, and would drive takedowns.

DISABLED BY DEFAULT (``cap_brand_enabled=False`` via the flag + enabled_default
below): set the flag and the registry loads it as a working, routable module with
NO core change — the "feature in a day" claim made concrete. Tool-backed with MOCK
data; swap ``_MOCK`` for your brand-protection backend (or a ``brand-mcp`` server).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.core.contracts import (
    Autonomy,
    CapabilityManifest,
    Citation,
    ToolContext,
    ToolResult,
    tool,
)

# Org-scoped MOCK alerts, keyed by tenant. Note org_globex is intentionally EMPTY:
# the tool must correctly return "no alerts" for a tenant with none — and crucially
# must NOT leak org_acme's BP-1 to a globex user. Lookup is by ctx.org_id only.
_MOCK = {
    "org_acme": [
        {"id": "BP-1", "domain": "acme-support.test", "kind": "phishing lookalike", "status": "active"},
    ],
    "org_globex": [],
}


# Single fully-optional arg, so the heuristic specialist can auto-invoke this tool.
class BrandArgs(BaseModel):
    query: str = Field(default="", description="Optional filter over domain/kind.")


# The module's only tool: READ, viewer-level, not side-effecting.
@tool(name="get_brand_alerts",
      description="List brand-abuse alerts: lookalike domains, phishing sites, fake apps impersonating the org.",
      args_schema=BrandArgs, rbac_role="viewer")
async def get_brand_alerts(args: BrandArgs, ctx: ToolContext):
    # Tenant-scoped read: this org's alerts only, from ctx.org_id (never from args).
    rows = _MOCK.get(ctx.org_id, [])
    cites = [Citation(doc_id=r["id"], source="brand", title=r["domain"],
                      snippet=f"Brand threat: lookalike/phishing domain {r['domain']} impersonating the "
                              f"organization ({r['kind']}), status {r['status']}.") for r in rows]
    # Success-as-data: structured ``data`` + citations. An empty org just yields count 0.
    return ToolResult(data={"alerts": rows, "count": len(rows)}, citations=cites)


# The cartridge (see reports/manifest.py for the full per-field walkthrough).
MANIFEST = CapabilityManifest(
    id="brand",
    version="0.1.0",
    display_name="Brand Protection",
    description="Detect impersonation, lookalike domains, phishing, and fake apps; drive takedowns.",
    license_tiers=("platform", "brand"),
    # DEPLOYMENT: flag-gated and off by default — load it by setting cap_brand_enabled.
    enabled_flag="cap_brand_enabled",
    enabled_default=False,
    # AGENT/MCP surface: the single read tool; no retriever (tool-backed module).
    tools=(get_brand_alerts,),
    # SUPERVISOR routing is DYNAMIC: brand-abuse / impersonation questions reach this
    # module by MEANING, scored against the ``description`` + the tool descriptions —
    # no curated keywords.
    # SUGGEST: the pillar is destined to drive gated takedown actions, even though the
    # only tool today is read-only.
    default_autonomy=Autonomy.SUGGEST,
    # RBAC: the read tool is viewer-level; enforced by the MCP boundary per call.
    rbac={"get_brand_alerts": "viewer"},
    owners=("team-brand",),
)

__all__ = ["MANIFEST", "get_brand_alerts"]
