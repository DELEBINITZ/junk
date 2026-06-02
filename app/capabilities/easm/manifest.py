"""EASM (External Attack Surface Management) capability manifest.

The second module — proof that adding a feature is just a manifest + tools, no
core edit. Data comes from typed tools (its 'MCP surface'), not a RAG corpus, so
it shows the tool-backed module shape alongside the corpus-backed reports module.
"""

from __future__ import annotations

from app.capabilities.easm.tools import TOOLS
from app.core.contracts import Autonomy, CapabilityManifest, RoutingHint

MANIFEST = CapabilityManifest(
    id="easm",
    version="1.0.0",
    display_name="External Attack Surface Management",
    description=(
        "Query the organization's external attack surface — exposed assets, "
        "exposures/findings, and surface changes — and request rescans (gated)."
    ),
    license_tiers=("platform", "easm"),
    enabled_flag="cap_easm_enabled",
    tools=TOOLS,
    system_prompt="prompts/v1.md",
    routing_hints=(
        RoutingHint(
            intents=(
                "attack surface", "exposed asset", "exposure", "open port", "asset inventory",
                "external scan", "what's exposed", "subdomain", "shadow IT", "internet-facing",
                "rescan", "surface change",
            ),
            examples=(
                "what assets do we have exposed to the internet?",
                "show me our current exposures",
                "what changed on our attack surface this week?",
                "rescan admin.acme.test",
            ),
        ),
    ),
    default_autonomy=Autonomy.SUGGEST,
    rbac={
        "query_assets": "viewer", "get_exposures": "viewer", "get_asset_changes": "viewer",
        "trigger_rescan": "analyst",
    },
    owners=("team-easm",),
)

__all__ = ["MANIFEST"]
