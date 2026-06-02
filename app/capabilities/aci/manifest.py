"""Adversary-Centric Intelligence (ACI) — drop-in stub.

Disabled by default (``cap_aci_enabled=False``). Profiles who attacks the org,
why, and how (threat actors, campaigns, TTPs, credential leaks, dark-web
mentions). Replace the mock with your threat-intel backend / ``aci-mcp`` server.
"""

from __future__ import annotations

import re

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
        {"id": "TA-1", "actor": "FIN-Acme", "motive": "financial", "ttps": ["T1566", "T1190"],
         "note": "weaponizes CVE-2023-22515; phishes Acme staff for credentials"},
    ],
    "org_globex": [
        {"id": "TA-9", "actor": "Lazarus-G", "motive": "financial/espionage", "ttps": ["T1566"],
         "note": "BEC and invoice fraud against Globex finance"},
    ],
}


class ActorArgs(BaseModel):
    query: str = Field(default="", description="Optional filter over actor name / TTP / motive.")


@tool(name="get_threat_actors",
      description="Profile threat actors targeting the org: motives, TTPs (MITRE ATT&CK), and notes.",
      args_schema=ActorArgs, rbac_role="viewer")
async def get_threat_actors(args: ActorArgs, ctx: ToolContext):
    all_rows = _MOCK.get(ctx.org_id, [])
    tokens = {t for t in re.findall(r"[a-z0-9]+", args.query.lower()) if len(t) > 2}
    def matches(r: dict) -> bool:
        hay = f"{r['actor']} {r['note']} {' '.join(r['ttps'])}".lower()
        return any(t in hay for t in tokens)
    narrowed = [r for r in all_rows if matches(r)] if tokens else []
    rows = narrowed or all_rows  # broad NL question -> all actors
    cites = [Citation(doc_id=r["id"], source="aci", title=r["actor"],
                      snippet=f"Threat actor {r['actor']} ({r['motive']} motive) is targeting the "
                              f"organization. TTPs (MITRE ATT&CK): {', '.join(r['ttps'])}. {r['note']}")
             for r in rows]
    return ToolResult(data={"actors": rows, "count": len(rows)}, citations=cites)


MANIFEST = CapabilityManifest(
    id="aci",
    version="0.1.0",
    display_name="Adversary-Centric Intelligence",
    description="Profile the threat actors, campaigns, and TTPs targeting the organization.",
    license_tiers=("platform", "aci"),
    enabled_flag="cap_aci_enabled",
    enabled_default=False,
    tools=(get_threat_actors,),
    routing_hints=(
        RoutingHint(
            intents=("threat actor", "adversary", "who is attacking", "campaign", "TTP",
                     "MITRE", "dark web", "credential leak", "attribution"),
            examples=("who is targeting us?", "which threat actors weaponize our exposed CVE?"),
        ),
    ),
    default_autonomy=Autonomy.SUGGEST,
    rbac={"get_threat_actors": "viewer"},
    owners=("team-aci",),
)

__all__ = ["MANIFEST", "get_threat_actors"]
