"""Demo corpus for the reports module (DEV/TEST ONLY — mock data).

The reports module is corpus-backed, so it needs documents in its vector store to
have anything to retrieve. In PRODUCTION an external ingestion path (a cron / event
bus, off the chat hot-path — see SourceEvent/IngestionConnector in contracts.py)
embeds real reports into the ``reports_kb`` collection. This file is the stand-in:
it indexes a small fixed set of synthetic reports so the platform answers out of the
box and so the eval/test suite has deterministic ground truth.

WHY TWO ORGS: the rows below are seeded under two DIFFERENT tenant keys
(``org_acme`` and ``org_globex``). Because retrieval is org-scoped from the trusted
context, an acme user can never retrieve globex's R-9001 — seeding both lets tests
PROVE that tenant isolation actually holds end to end.
"""

from __future__ import annotations

from app.capabilities.reports.manifest import REPORTS_COLLECTION
from app.core.contracts import CoreDeps
from app.core.rag.pipeline import IndexItem

# Synthetic reports for tenant "org_acme". Each row is (doc_id, title, date, body).
# These deliberately mirror the other modules' mock data (the same Confluence CVE,
# lookalike domain, threat actor, leaked credentials) so a cross-module question can
# be demonstrated. This is MOCK content — replaced by real ingested reports in prod.
_ACME = [
    ("R-1001", "EASM scan: critical exposure", "2026-05-28",
     "EASM scan found the Confluence server admin.acme.test exposed to the internet and "
     "vulnerable to CVE-2023-22515, a critical authentication-bypass flaw. Severity: critical. "
     "Recommended remediation: restrict access and patch immediately."),
    ("R-1002", "Certificate audit", "2026-05-10",
     "The TLS certificate on shop.acme.test is expired and must be renewed within 30 days. "
     "Severity: medium. An expired certificate breaks customer trust and may block checkout."),
    ("R-1003", "Brand protection finding", "2026-05-20",
     "A lookalike domain acme-support.test is hosting a phishing page impersonating Acme "
     "customer support to harvest credentials. Severity: high. Recommended action: submit a "
     "takedown request and add the domain to the blocklist."),
    ("R-1004", "Adversary intelligence", "2026-05-22",
     "Threat actor tracked as FIN-Acme is targeting Acme employees with credential phishing and "
     "is known to weaponize CVE-2023-22515 against exposed Confluence instances. TTPs map to "
     "MITRE ATT&CK T1566 (Phishing) and T1190 (Exploit public-facing application)."),
    ("R-1005", "Executive briefing Q2", "2026-04-02",
     "Top organizational risks this quarter: (1) the exposed Confluence server on a critical CVE, "
     "(2) phishing lookalike domains targeting customers, and (3) leaked VPN credentials on the "
     "dark web. Overall risk posture: elevated."),
    ("R-1006", "Credential leak notice", "2026-05-25",
     "14 sets of acme.test employee credentials were found for sale on a dark-web forum, including "
     "credentials that front the VPN admin panel. Severity: high. Recommended response: force "
     "password resets and enable MFA on affected accounts."),
    ("R-0900", "Legacy certificate report", "2024-02-15",
     "In 2024 the certificate on legacy.acme.test expired and the host was decommissioned. This "
     "is a historical record retained for audit."),
]

# Synthetic reports for the SECOND tenant "org_globex". A distinct, smaller set —
# the point is purely that it is a different org's data, never visible to org_acme.
_GLOBEX = [
    ("R-9001", "EASM scan: exposed RDP", "2026-05-18",
     "Globex exposes Remote Desktop (RDP) on vpn.globex.test to the public internet, creating a "
     "brute-force and ransomware risk. Severity: high. Recommended remediation: place behind VPN."),
    ("R-9002", "Adversary intelligence", "2026-05-21",
     "A threat actor tracked as Lazarus-G is targeting Globex's finance department with invoice "
     "fraud and business-email-compromise lures."),
]


# Turn the plain (doc_id, title, date, text) rows into IndexItems the pipeline can
# embed. Each report becomes one chunk here (id ``<doc_id>::0``) with its title folded
# into the text; a real ingester would chunk long documents into many pieces.
def _items(rows) -> list[IndexItem]:
    out = []
    for doc_id, title, date, text in rows:
        out.append(IndexItem(
            id=f"{doc_id}::0", text=f"{title}. {text}", source="reports",
            doc_id=doc_id, title=title, published_at=f"{date}T00:00:00Z",
            metadata={"severity": "see-text"},
        ))
    return out


# Index each org's items UNDER THAT ORG'S KEY. The second positional arg is the
# tenant the chunks are stamped with — this is what scopes them, so acme and globex
# end up isolated in the same collection. Called once at dev/test startup.
async def seed_demo(deps: CoreDeps) -> None:
    await deps.rag.index(REPORTS_COLLECTION, "org_acme", _items(_ACME))
    await deps.rag.index(REPORTS_COLLECTION, "org_globex", _items(_GLOBEX))


__all__ = ["seed_demo"]
