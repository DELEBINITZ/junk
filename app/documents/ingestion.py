"""Small ingestion facade used by routers and seed data."""

from __future__ import annotations

from pathlib import Path

from app.db.repository import DataStore
from app.documents.parser import parse_contract_file, parse_contract_text


def ingest_contract_file(store: DataStore, path: Path, uploaded_by: str, tags: list[str] | None = None):
    """Parse and persist a contract file through the repository boundary."""

    parsed = parse_contract_file(path)
    return store.add_parsed_contract(parsed, uploaded_by=uploaded_by, tags=tags or [])


def ingest_contract_text(
    store: DataStore,
    raw_text: str,
    filename: str,
    uploaded_by: str,
    tags: list[str] | None = None,
):
    """Parse and persist uploaded raw contract text."""

    parsed = parse_contract_text(raw_text, filename)
    return store.add_parsed_contract(parsed, uploaded_by=uploaded_by, tags=tags or [])
