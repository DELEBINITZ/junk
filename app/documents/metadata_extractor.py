"""Deterministic metadata extraction for the seeded contract corpus.

Exact contract facts are safer to parse than to ask an LLM to infer. The agent
uses this module for high-confidence answers about dates, notice periods,
values, renewal flags, and action deadlines.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from typing import Any


MONTH_DATE_FORMATS = ["%B %d, %Y", "%b %d, %Y"]
NUMBER_WORDS = {
    "thirty": 30,
    "sixty": 60,
    "ninety": 90,
    "one hundred eighty": 180,
    "hundred eighty": 180,
}


def parse_contract_date(value: str) -> date | None:
    """Parse known date formats, returning None for perpetual terms."""

    value = value.strip().strip(".")
    if value.lower().startswith("perpetual"):
        return None
    for fmt in MONTH_DATE_FORMATS:
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    return None


def normalize_money_to_number(value: str) -> float | None:
    """Normalize contract value text into a sortable numeric amount."""

    lowered = value.lower()
    if "no monetary" in lowered or "not applicable" in lowered:
        return 0.0
    match = re.search(r"\$([\d,.]+)\s*(m|million)?", value, re.IGNORECASE)
    if not match:
        return None
    amount = float(match.group(1).replace(",", ""))
    if match.group(2):
        amount *= 1_000_000
    return amount


def _extract_number_or_word_days(text: str) -> int | None:
    """Extract notice periods written as either digits or common words."""

    numeric = re.search(r"(\d+)\s+days?", text, re.IGNORECASE)
    if numeric:
        return int(numeric.group(1))
    lowered = text.lower()
    for word, value in NUMBER_WORDS.items():
        if word in lowered:
            return value
    return None


def extract_metadata(text: str, filename: str = "") -> dict[str, Any]:
    """Extract structured metadata used by tools, tests, and agent answers."""

    contract_id = _first_match(r"Contract ID:\s*([A-Z]{2}-\d{4})", text)
    title = _first_match(r"Contract Title:\s*(.+)", text)
    party_a = _first_match(r"PARTY A:\s*(.+)", text)
    party_b = _first_match(r"PARTY B:\s*(.+)", text)

    effective_raw = _first_match(r"entered into as of\s+([A-Za-z]+ \d{1,2}, \d{4})", text)
    if not effective_raw:
        effective_raw = _first_match(r"commence on\s+([A-Za-z]+ \d{1,2}, \d{4})", text)
    effective_date = parse_contract_date(effective_raw) if effective_raw else None

    expiration_raw = _first_match(r"continue until\s+(.+?)\s+\(\"Initial Term\"\)", text)
    expiration_date = parse_contract_date(expiration_raw) if expiration_raw else None
    is_perpetual = bool(expiration_raw and expiration_raw.lower().startswith("perpetual"))

    value_raw = _first_match(r"3\.1 Total Contract Value:\s*(.+)", text)
    governing_law = _first_match(r"10\.1 Governing Law:\s*(.+?)(?:\.|\n)", text)
    notice_text = _first_match(r"8\.1 Termination Notice Period:\s*(.+)", text) or ""
    notice_period_days = _extract_number_or_word_days(notice_text)
    renewal_text = _first_match(r"2\.2 Renewal:\s*(.+?)(?:\n\n|\n3\.|\Z)", text, flags=re.DOTALL) or ""
    auto_renewal = "automatically renew" in renewal_text.lower()

    action_required_by = None
    if expiration_date and notice_period_days is not None:
        action_required_by = expiration_date - timedelta(days=notice_period_days)

    organization_id = infer_organization_id(contract_id, party_a)

    return {
        "contract_id": contract_id,
        "title": title or title_from_filename(filename),
        "parties": [party for party in [party_a, party_b] if party],
        "effective_date": effective_date.isoformat() if effective_date else None,
        "expiration_date": expiration_date.isoformat() if expiration_date else None,
        "term_text": expiration_raw,
        "is_perpetual": is_perpetual,
        "contract_value": value_raw,
        "contract_value_numeric": normalize_money_to_number(value_raw or ""),
        "currency": "USD" if value_raw and "$" in value_raw else None,
        "governing_law": governing_law,
        "auto_renewal": auto_renewal,
        "notice_period_days": notice_period_days,
        "action_required_by": action_required_by.isoformat() if action_required_by else None,
        "organization_id": organization_id,
    }


def infer_organization_id(contract_id: str | None, party_a: str | None = None) -> str:
    """Infer tenant from the contract ID prefix or Party A name."""

    if contract_id and contract_id.startswith("TC-"):
        return "techcorp"
    if contract_id and contract_id.startswith("MC-"):
        return "medicare"
    if party_a and "medicare" in party_a.lower():
        return "medicare"
    if party_a and "techcorp" in party_a.lower():
        return "techcorp"
    return "sandbox"


def title_from_filename(filename: str) -> str:
    """Fallback title when the contract header does not contain one."""

    stem = filename.rsplit("/", 1)[-1].removesuffix(".txt")
    parts = stem.split("_", 1)
    return parts[-1].replace("-", " ") if parts else stem


def _first_match(pattern: str, text: str, flags: int = 0) -> str | None:
    """Return a whitespace-normalized first capture group for a regex pattern."""

    match = re.search(pattern, text, flags)
    if not match:
        return None
    return " ".join(match.group(1).strip().split())
