"""Batch ingestion for the reports corpus.

Wraps the existing parser + section chunker + repository commit behind the
connector interface. The production target runs this on a Temporal worker and
upserts to Qdrant + KG; the interface (`handle`) does not change. See plan §7.1.
"""

from __future__ import annotations

from pathlib import Path

from app.core.ingestion.connectors import IngestStats
from app.db.repository import DataStore
from app.documents.parser import parse_contract_file


class ReportConnector:
    id = "reports.batch"
    source = "file"

    def handle(self, path: str | Path, store: DataStore, uploaded_by: str, tags: list[str] | None = None) -> IngestStats:
        parsed = parse_contract_file(Path(path))
        document = store.add_parsed_contract(parsed, uploaded_by=uploaded_by, tags=tags or [])
        chunk_count = len(store.chunks_for_documents([document.id]))
        return IngestStats(documents=1, chunks=chunk_count)
