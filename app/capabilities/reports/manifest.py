"""Reports capability manifest — the first (and reference) capability MODULE.

WHAT A MODULE IS (read this first): a capability module is a self-contained
"feature cartridge". This file is its MANIFEST — the single declarative object
(:class:`CapabilityManifest`, defined in app/core/contracts.py) that tells the
platform everything it needs to wire the feature in. The console (app/core) reads
this manifest at boot and DERIVES all of its behaviour from it; you never edit
core to add a module. See the "games console + cartridges" mental model at the
top of contracts.py.

WHAT THIS MODULE COVERS: question-answering over the security-report corpus.
Unlike the tool-backed modules (easm/aci/brand), reports is a CORPUS-backed
module — its evidence comes from a RAG document collection (see the retriever
below), not from typed tools. The two structured tools it ships are just targeted
shortcuts over that same corpus.

HOW IT PLUGS IN: the registry discovers ``MANIFEST`` and wires it automatically.
Enabling/disabling is purely a config flag (``cap_reports_enabled`` below) — no
code change. With ONLY this module registered the supervisor routes everything
here and the system behaves like a single agent; registering EASM/Brand/ACI later
adds routing targets without touching core or this file.
"""

from __future__ import annotations

from app.capabilities.reports.tools import TOOLS
from app.core.contracts import Autonomy, CapabilityManifest
from app.core.rag.pipeline import CollectionRetriever

# The Qdrant collection holding this module's report documents (per-tenant isolation
# is layered on top via org_id at query time — see the retriever and the tools).
#
# DATA LIFECYCLE — READ ONLY in this app. The platform does NOT write reports here.
# An EXTERNAL CRON (owned/operated separately) embeds real reports into this Qdrant
# collection out-of-band. This module's job is purely to RETRIEVE: at query time the
# agent searches ``reports_kb`` (via the retriever + the search_reports tool) and, if
# it finds documents relevant to the user's question, answers from them with
# citations. So there is no ingest/seed code here — only the read path below.
REPORTS_COLLECTION = "reports_kb"

# The module's RAG binding (a :class:`Retriever` from contracts.py). It adapts the
# shared retrieval pipeline to THIS module's collection, tagging every Chunk it
# returns with ``source="reports"`` so provenance is preserved. The specialist
# (specialist.py) calls ``retrieve`` on this during ``_retrieve`` to gather corpus
# evidence; tenant scoping is applied from the trusted ToolContext, never the query.
# This is a pure READ against Qdrant — the documents were written by the external cron.
_retriever = CollectionRetriever(
    id="reports_kb_retriever", collection=REPORTS_COLLECTION, source="reports"
)

# The cartridge itself. Each field below maps to a core subsystem that the
# platform DERIVES behaviour from — nothing here is "called" by you; the registry
# reads it once at boot and wires the rest.
MANIFEST = CapabilityManifest(
    # ``id`` is the module's stable handle: the registry keys the module by it, the
    # supervisor names it in routing decisions, and every Chunk/tool is stamped with
    # it. ``version``/``display_name``/``description`` are metadata (the description
    # also gives a human + the supervisor a one-line sense of scope).
    id="reports",
    version="1.0.0",
    display_name="Security Reports",
    description=(
        "Q&A over the organization's analyst- and AI-generated security REPORTS, "
        "scans and findings — what a report or scan FOUND or SAID. Covers: EASM scan "
        "results and exposed internet-facing assets/open ports; critical CVEs and "
        "vulnerabilities (e.g. on a Confluence server); leaked employee credentials "
        "and dark-web exposure; phishing and lookalike-domain findings; threat-intel "
        "write-ups; top risks for the quarter; remediation guidance; and executive "
        "briefings."
    ),
    # Licensing tiers this module belongs to (commercial packaging metadata).
    license_tiers=("platform", "reports"),
    # DEPLOYMENT: the name of a Settings attribute. The registry only loads this
    # module if that flag is truthy. This is the whole on/off switch — flip the
    # config, no code change. ("" would mean "always enabled".)
    enabled_flag="cap_reports_enabled",
    # AGENT / MCP surface: the tuple of Tools (from tools.py). The MCP boundary
    # exposes exactly these to the agent, and the per-module specialist may only
    # call THESE — tool isolation, the core scaling property (see specialist.py).
    tools=TOOLS,
    # RAG: the module's retrievers. The specialist runs each one to pull corpus
    # evidence. reports is corpus-backed, so it binds one; pure tool modules bind none.
    retrievers=(_retriever,),
    # The module's own system prompt, given as a path RELATIVE to this module dir
    # (resolves to app/capabilities/reports/prompts/v1.md). When this module is
    # routed, answer_node prepends that text to the base persona for domain flavour.
    system_prompt="prompts/v1.md",
    # SUPERVISOR ROUTING is DYNAMIC: the supervisor/planner route to this module by
    # the MEANING of the question, scored against this manifest's ``description`` +
    # the tools' names/descriptions (semantic similarity, or an LLM router). Keep the
    # ``description`` above crisp and the tools well-described — that text IS the
    # routing signal. No keywords are maintained here.
    # AUTONOMY: this module only READS (answers/recommends), it never acts. Compare
    # with easm, which is SUGGEST because it ships a gated side-effecting tool.
    default_autonomy=Autonomy.READ,
    # RBAC: per-tool minimum-role overrides, consumed by the MCP boundary before
    # each call. Both reports tools are viewer-level (read-only Q&A). A tool's own
    # ``rbac_role`` is the default; this map can tighten it per deployment.
    rbac={"get_report_metadata": "viewer", "find_expiring_items": "viewer",
          "search_reports": "viewer"},
    # Ownership metadata (who maintains the module) — informational only.
    owners=("team-reports",),
)

# Re-export the manifest (what the registry imports) and the collection name (used
# by seed.py and the tools). Strictly the module's public surface.
__all__ = ["MANIFEST", "REPORTS_COLLECTION"]
