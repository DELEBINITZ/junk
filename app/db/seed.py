"""Deterministic demo seed data.

The seed deliberately covers positive and negative security cases: two tenants,
multiple roles, explicit document shares, a PII-heavy NDA, and a Frank-owned
document used by RBAC denial tests.
"""

from __future__ import annotations

from pathlib import Path

from app.auth.password import hash_password
from app.db.repository import DataStore
from app.documents.parser import parse_contract_file, parse_contract_text
from app.domain import DocumentShare, Organization, User


DEMO_PASSWORD = "password123"


USER_FIXTURES = [
    ("alice", "techcorp", "alice@techcorp.com", "Alice Chen", "admin"),
    ("bob", "techcorp", "bob@techcorp.com", "Bob Williams", "analyst"),
    ("charlie", "techcorp", "charlie@techcorp.com", "Charlie Davis", "viewer"),
    ("diana", "medicare", "diana@medicareplus.com", "Diana Martinez", "admin"),
    ("eve", "medicare", "eve@medicareplus.com", "Eve Thompson", "analyst"),
    ("frank", "techcorp", "frank@techcorp.com", "Frank Negative Test", "analyst"),
]

OWNER_BY_CONTRACT_ID = {
    "TC-1001": "alice",
    "TC-1055": "alice",
    "TC-1089": "alice",
    "TC-1042": "bob",
    "MC-2001": "diana",
    "MC-2015": "diana",
    "MC-2033": "eve",
}


def seed_demo_data(store: DataStore, corpus_dir: Path) -> None:
    """Reset the repository and load organizations, users, contracts, and shares."""

    store.reset()
    store.add_organization(Organization(id="techcorp", name="TechCorp Inc."))
    store.add_organization(Organization(id="medicare", name="MediCare Plus"))

    for user_id, org_id, email, name, role in USER_FIXTURES:
        store.add_user(
            User(
                id=user_id,
                organization_id=org_id,
                email=email,
                name=name,
                role=role,  # type: ignore[arg-type]
                password_hash=hash_password(DEMO_PASSWORD),
            )
        )

    for path in sorted(corpus_dir.glob("*.txt")):
        parsed = parse_contract_file(path)
        contract_id = str(parsed.metadata.get("contract_id"))
        owner = OWNER_BY_CONTRACT_ID.get(contract_id, "alice")
        tags = tags_for_contract(contract_id)
        document = store.add_parsed_contract(parsed, uploaded_by=owner, tags=tags)
        if contract_id == "TC-1001":
            # Bob is an analyst, not an admin. This explicit share lets him run
            # the Q2 renewal demo without giving him blanket TechCorp access.
            store.add_share(DocumentShare(document_id=document.id, user_id="bob", access_level="query"))
        if contract_id == "TC-1089":
            # Charlie is a viewer; this gives him a read-only positive case
            # while AI query and upload remain forbidden.
            store.add_share(
                DocumentShare(document_id=document.id, user_id="charlie", access_level="read")
            )

    frank_contract = parse_contract_text(FRANK_NEGATIVE_RBAC_CONTRACT, "FR-3001_Test-Document.txt")
    store.add_parsed_contract(
        frank_contract,
        uploaded_by="frank",
        tags=["negative-rbac-test"],
        contract_id_override="FR-3001",
        organization_id_override="techcorp",
    )


def tags_for_contract(contract_id: str) -> list[str]:
    if contract_id.startswith("MC-"):
        return ["healthcare"]
    if contract_id in {"TC-1055"}:
        return ["software"]
    if contract_id in {"TC-1089"}:
        return ["lease"]
    return ["technology"]


FRANK_NEGATIVE_RBAC_CONTRACT = """
Contract ID: FR-3001
Contract Title: Frank Negative RBAC Test Agreement

This Agreement ("Agreement") is entered into as of January 1, 2024
("Effective Date") by and between:

PARTY A: TechCorp Inc.
PARTY B: Example Vendor LLC

1. SCOPE OF SERVICES/PRODUCTS
Example vendor support services.

2. TERM AND DURATION
2.1 Effective Date: This Agreement shall commence on January 1, 2024
    and shall continue until December 31, 2024 ("Initial Term").
2.2 Renewal: No automatic renewal applies.

3. PAYMENT TERMS
3.1 Total Contract Value: $1,000 USD

7. LIABILITY AND INDEMNIFICATION
7.1 Limitation of Liability: Liability is capped at fees paid.

8. TERMINATION
8.1 Termination Notice Period: 30 days

9. FORCE MAJEURE
Standard force majeure terms.

10. DISPUTE RESOLUTION
10.1 Governing Law: This Agreement shall be governed by the laws of the State of California.
"""
