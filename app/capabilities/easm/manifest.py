"""EASM (External Attack Surface Management) capability MODULE — manifest.

THE SECOND MODULE, and the contrast to reports. Where reports is corpus-backed
(its evidence is RAG over documents), EASM is TOOL-BACKED: it ships NO retriever,
and all of its data flows through typed tools (its "MCP surface", in tools.py).
That makes it the reference for "a module that wraps a backend/API" rather than a
document store. It also ships a GATED, side-effecting tool (``trigger_rescan``),
so it is the module that exercises the human-approval action gate end to end.

Like every module it plugs in with NO core edit: the registry discovers
``MANIFEST``, and the ``cap_easm_enabled`` flag below is the entire on/off switch.
Read this file alongside reports/manifest.py — same fields, different choices
(no ``retrievers``, autonomy SUGGEST not READ, one analyst-level tool).
"""

from __future__ import annotations

from app.capabilities.easm.tools import TOOLS
from app.core.contracts import Autonomy, CapabilityManifest, RoutingHint

MANIFEST = CapabilityManifest(
    # Stable id + display metadata (see reports/manifest.py for the field-by-field
    # walkthrough; only the EASM-specific choices are called out below).
    id="easm",
    version="1.0.0",
    display_name="External Attack Surface Management",
    description=(
        "Query the organization's external attack surface — exposed assets, "
        "exposures/findings, and surface changes — and request rescans (gated)."
    ),
    license_tiers=("platform", "easm"),
    # DEPLOYMENT switch: load this module only when this Settings flag is on.
    enabled_flag="cap_easm_enabled",
    # AGENT/MCP surface. NOTE there is no ``retrievers=`` field here at all — EASM is
    # tool-backed, so its specialist gathers evidence purely by calling these tools.
    tools=TOOLS,
    system_prompt="prompts/v1.md",
    # SUPERVISOR routing signals — attack-surface vocabulary. The supervisor scores a
    # question against these to decide EASM should answer; nothing is hardcoded in core.
    routing_hints=(
        RoutingHint(
            intents=(
                "attack surface", "exposed asset", "exposure", "open port", "asset inventory",
                "asset count", "how many assets", "live assets", "external scan", "what's exposed",
                "subdomain", "shadow IT", "internet-facing", "rescan", "surface change",
            ),
            examples=(
                "what assets do we have exposed to the internet?",
                "how many assets are live?",
                "show me our current exposures",
                "what changed on our attack surface this week?",
                "rescan admin.acme.test",
            ),
        ),
    ),
    # AUTONOMY SUGGEST (not READ): this module can DRAFT an action for a human to
    # approve — because of the side-effecting ``trigger_rescan`` tool. The drafting
    # still goes through the gate; SUGGEST never means "act on its own".
    default_autonomy=Autonomy.SUGGEST,
    # RBAC: the three read tools are viewer-level; ``trigger_rescan`` requires ANALYST.
    # The MCP boundary enforces this minimum role BEFORE invoking each tool, so the
    # agent can never escalate privilege. (This mirrors each tool's own ``rbac_role``.)
    rbac={
        "query_assets": "viewer", "get_exposures": "viewer", "get_asset_changes": "viewer",
        "get_live_asset_count": "viewer", "trigger_rescan": "analyst",
    },
    owners=("team-easm",),
)

__all__ = ["MANIFEST"]
