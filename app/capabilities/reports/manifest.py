"""Reports capability manifest — the first drop-in module.

The platform discovers this file at boot and derives routing, the tool registry,
RBAC, and autonomy from it. Adding EASM/BP/ACI later means adding sibling
directories each with their own manifest — no core change. See plan §5.
"""

from __future__ import annotations

from app.capabilities.reports.ingestion import ReportConnector
from app.capabilities.reports.retrievers import reports_retriever
from app.capabilities.reports.tools import REPORT_TOOLS
from app.core.contracts import Autonomy, CapabilityManifest, GenericSpecialist, RoutingHint


MANIFEST = CapabilityManifest(
    id="reports",
    version="1.0.0",
    display_name="Security & Contract Reports",
    tools=REPORT_TOOLS,
    routing_hints=[
        RoutingHint(
            intents=[
                "report", "contract", "agreement", "clause", "termination",
                "liability", "renewal", "expiry", "expiring", "expire", "risk",
                "obligation", "notice", "compare", "metadata", "value",
            ],
            examples=[
                "what does the termination clause say?",
                "which contracts expire next quarter?",
                "compare the liability clauses across our vendors",
                "summarize the obligations in this agreement",
                "what is the contract value and notice period?",
            ],
        )
    ],
    license_tiers=["reports", "platform"],
    specialist=GenericSpecialist(
        id="reports",
        tools=REPORT_TOOLS,
        system_prompt="app/capabilities/reports/prompts/v1.md",
    ),
    system_prompt="app/capabilities/reports/prompts/v1.md",
    retrievers=[reports_retriever],
    ingestion=[ReportConnector()],
    default_autonomy=Autonomy.READ,
    rbac={tool.name: "analyst" for tool in REPORT_TOOLS},
    owners=["team-reports"],
    min_core_version="1.0.0",
)
