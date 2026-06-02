"""Demo corpus for the reports module (dev only).

In production the external cron embeds reports into ``reports_kb``; this seed
just makes the platform answerable out of the box and powers eval/tests. Two
orgs are seeded so tenant isolation is demonstrable.
"""

from __future__ import annotations

from app.capabilities.reports.manifest import REPORTS_COLLECTION
from app.core.contracts import CoreDeps
from app.core.rag.pipeline import IndexItem

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

_GLOBEX = [
    ("R-9001", "EASM scan: exposed RDP", "2026-05-18",
     "Globex exposes Remote Desktop (RDP) on vpn.globex.test to the public internet, creating a "
     "brute-force and ransomware risk. Severity: high. Recommended remediation: place behind VPN."),
    ("R-9002", "Adversary intelligence", "2026-05-21",
     "A threat actor tracked as Lazarus-G is targeting Globex's finance department with invoice "
     "fraud and business-email-compromise lures."),
]


def _items(rows) -> list[IndexItem]:
    out = []
    for doc_id, title, date, text in rows:
        out.append(IndexItem(
            id=f"{doc_id}::0", text=f"{title}. {text}", source="reports",
            doc_id=doc_id, title=title, published_at=f"{date}T00:00:00Z",
            metadata={"severity": "see-text"},
        ))
    return out


async def seed_demo(deps: CoreDeps) -> None:
    await deps.rag.index(REPORTS_COLLECTION, "org_acme", _items(_ACME))
    await deps.rag.index(REPORTS_COLLECTION, "org_globex", _items(_GLOBEX))


__all__ = ["seed_demo"]
