"""Adversary-Centric Intelligence (ACI) — drop-in module stub.

A deliberately MINIMAL module kept in ONE file (tool + manifest together) to show
the smallest shape a capability can take. It profiles who attacks the org, why, and
how (threat actors, campaigns, TTPs, credential leaks, dark-web mentions).

DISABLED BY DEFAULT: ``enabled_flag="cap_aci_enabled"`` plus ``enabled_default=False``
mean the registry skips this module unless the flag is turned on. Flipping that one
config value makes it a live, routable module — no core edit, no other change. This
is the concrete demonstration of "a new feature is a manifest + a tool".

Like easm/brand it is tool-backed with MOCK data; replace ``_MOCK`` with your
threat-intel backend (or a standalone ``aci-mcp`` server) to go live.
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

# Org-scoped MOCK threat-intel, keyed by tenant. Looked up by ``ctx.org_id`` (the
# trusted token-derived tenant), never by an arg — so one org can't read another's.
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


# Single, fully-optional arg (free-text ``query``) — so the heuristic specialist can
# auto-invoke this tool on any broad question routed to ACI.
class ActorArgs(BaseModel):
    query: str = Field(default="", description="Optional filter over actor name / TTP / motive.")


# The module's only tool: a READ, viewer-level function (not side-effecting). The
# ``@tool`` decorator wraps this async handler into the Tool the manifest advertises.
@tool(name="get_threat_actors",
      description="Profile threat actors targeting the org: motives, TTPs (MITRE ATT&CK), and notes.",
      args_schema=ActorArgs, rbac_role="viewer")
async def get_threat_actors(args: ActorArgs, ctx: ToolContext):
    # Tenant scoping: read THIS org's rows from ctx.org_id only (never from args).
    all_rows = _MOCK.get(ctx.org_id, [])
    tokens = {t for t in re.findall(r"[a-z0-9]+", args.query.lower()) if len(t) > 2}
    def matches(r: dict) -> bool:
        hay = f"{r['actor']} {r['note']} {' '.join(r['ttps'])}".lower()
        return any(t in hay for t in tokens)
    narrowed = [r for r in all_rows if matches(r)] if tokens else []
    rows = narrowed or all_rows  # broad NL question -> all actors
    # One Citation per actor = grounded evidence the answer can cite as [n].
    cites = [Citation(doc_id=r["id"], source="aci", title=r["actor"],
                      snippet=f"Threat actor {r['actor']} ({r['motive']} motive) is targeting the "
                              f"organization. TTPs (MITRE ATT&CK): {', '.join(r['ttps'])}. {r['note']}")
             for r in rows]
    # Errors-as-data success shape: structured ``data`` + citations, never a raise.
    return ToolResult(data={"actors": rows, "count": len(rows)}, citations=cites)


# The cartridge. Field meanings are documented field-by-field in reports/manifest.py;
# the two ACI-specific points are the disabled-by-default switch and the single tool.
MANIFEST = CapabilityManifest(
    id="aci",
    version="0.1.0",
    display_name="Adversary-Centric Intelligence",
    description="Profile the threat actors, campaigns, and TTPs targeting the organization.",
    license_tiers=("platform", "aci"),
    # DEPLOYMENT: gated behind this flag AND off by default (next line) — the registry
    # won't load the module until someone sets cap_aci_enabled. That is the entire
    # difference between this stub and a shipping feature.
    enabled_flag="cap_aci_enabled",
    enabled_default=False,
    # AGENT/MCP surface: just the one read tool. No retriever (tool-backed module).
    tools=(get_threat_actors,),
    # SUPERVISOR routing vocabulary for adversary/threat-intel questions.
    routing_hints=(
        RoutingHint(
            intents=("threat actor", "adversary", "who is attacking", "campaign", "TTP",
                     "MITRE", "dark web", "credential leak", "attribution"),
            examples=("who is targeting us?", "which threat actors weaponize our exposed CVE?"),
        ),
    ),
    # SUGGEST autonomy is the module default even though today's only tool is read-only —
    # it signals where this pillar is headed (it would later add gated actions).
    default_autonomy=Autonomy.SUGGEST,
    # RBAC: the read tool is viewer-level; enforced at the MCP boundary before each call.
    rbac={"get_threat_actors": "viewer"},
    owners=("team-aci",),
)

__all__ = ["MANIFEST", "get_threat_actors"]
