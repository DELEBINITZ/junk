"""Reports capability manifest — the first (and reference) module.

Everything the platform needs to wire report-chat is declared here. With only
this module registered, the supervisor behaves like today's single agent; adding
EASM/Brand/ACI later changes nothing in core.
"""

from __future__ import annotations

from app.capabilities.reports.tools import TOOLS
from app.core.contracts import Autonomy, CapabilityManifest, RoutingHint
from app.core.rag.pipeline import CollectionRetriever

REPORTS_COLLECTION = "reports_kb"

_retriever = CollectionRetriever(
    id="reports_kb_retriever", collection=REPORTS_COLLECTION, source="reports"
)

MANIFEST = CapabilityManifest(
    id="reports",
    version="1.0.0",
    display_name="Security Reports",
    description=(
        "Q&A over analyst- and AI-generated security reports — EASM scan results, "
        "brand-protection findings, threat-intel write-ups, and executive briefings."
    ),
    license_tiers=("platform", "reports"),
    enabled_flag="cap_reports_enabled",
    tools=TOOLS,
    retrievers=(_retriever,),
    system_prompt="prompts/v1.md",
    routing_hints=(
        RoutingHint(
            intents=(
                "report", "finding", "summary", "analyst", "briefing", "scan result",
                "executive summary", "remediation", "vulnerability", "what does the report say",
                "top risks", "credential leak", "phishing", "threat intel",
            ),
            examples=(
                "summarize the latest report",
                "what did the EASM scan find on our confluence server?",
                "what are our top risks this quarter?",
                "were any of our credentials leaked?",
            ),
        ),
    ),
    default_autonomy=Autonomy.READ,
    rbac={"get_report_metadata": "viewer", "find_expiring_items": "viewer"},
    owners=("team-reports",),
)

__all__ = ["MANIFEST", "REPORTS_COLLECTION"]
