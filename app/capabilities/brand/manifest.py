"""Brand Protection / Digital Risk Protection — drop-in stub.

Disabled by default (``cap_brand_enabled=False``). Set the flag to load it; it
becomes a working module with no core change — the 1-day-feature claim in
practice. Replace the mock tool with calls to your brand-protection backend (or
a standalone ``brand-mcp`` server).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.core.contracts import (
    Autonomy,
    CapabilityManifest,
    Citation,
    RoutingHint,
    ToolContext,
    ToolResult,
    tool,
)

_MOCK = {
    "org_acme": [
        {"id": "BP-1", "domain": "acme-support.test", "kind": "phishing lookalike", "status": "active"},
    ],
    "org_globex": [],
}


class BrandArgs(BaseModel):
    query: str = Field(default="", description="Optional filter over domain/kind.")


@tool(name="get_brand_alerts",
      description="List brand-abuse alerts: lookalike domains, phishing sites, fake apps impersonating the org.",
      args_schema=BrandArgs, rbac_role="viewer")
async def get_brand_alerts(args: BrandArgs, ctx: ToolContext):
    rows = _MOCK.get(ctx.org_id, [])
    cites = [Citation(doc_id=r["id"], source="brand", title=r["domain"],
                      snippet=f"Brand threat: lookalike/phishing domain {r['domain']} impersonating the "
                              f"organization ({r['kind']}), status {r['status']}.") for r in rows]
    return ToolResult(data={"alerts": rows, "count": len(rows)}, citations=cites)


MANIFEST = CapabilityManifest(
    id="brand",
    version="0.1.0",
    display_name="Brand Protection",
    description="Detect impersonation, lookalike domains, phishing, and fake apps; drive takedowns.",
    license_tiers=("platform", "brand"),
    enabled_flag="cap_brand_enabled",
    enabled_default=False,
    tools=(get_brand_alerts,),
    routing_hints=(
        RoutingHint(
            intents=("brand", "lookalike domain", "phishing", "impersonation", "fake app", "takedown"),
            examples=("are there phishing sites targeting our brand?",
                      "any lookalike domains impersonating us?"),
        ),
    ),
    default_autonomy=Autonomy.SUGGEST,
    rbac={"get_brand_alerts": "viewer"},
    owners=("team-brand",),
)

__all__ = ["MANIFEST", "get_brand_alerts"]
