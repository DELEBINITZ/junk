"""EASM manifest. enabled_by_default=False — ships dark; activate with
CAP_EASM_ENABLED=true. Discovered and routable with no core change once enabled."""

from __future__ import annotations

from app.capabilities.easm.tools import EASM_TOOLS
from app.core.contracts import Autonomy, CapabilityManifest, RoutingHint


MANIFEST = CapabilityManifest(
    id="easm",
    version="0.1.0",
    display_name="External Attack Surface Management",
    tools=EASM_TOOLS,
    routing_hints=[
        RoutingHint(
            intents=[
                "asset", "assets", "exposed", "exposure", "attack surface",
                "subdomain", "domain", "ip", "port", "certificate", "cve",
                "vulnerability", "vulnerabilities", "misconfiguration", "shadow it",
            ],
            examples=[
                "what assets are exposed on the internet?",
                "which critical vulnerabilities should we patch first?",
                "what changed in our attack surface this week?",
                "are we affected by CVE-2026-12345?",
            ],
        )
    ],
    license_tiers=["easm", "platform"],
    default_autonomy=Autonomy.READ,
    rbac={tool.name: "analyst" for tool in EASM_TOOLS},
    owners=["team-easm"],
    min_core_version="1.0.0",
    enabled_by_default=False,
)
