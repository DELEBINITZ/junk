"""Ingestion connector contract: normalize a source event and land it in the
vector store (+ KG). A module owns one connector per source. See plan §7.1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass
class IngestStats:
    documents: int = 0
    chunks: int = 0


@runtime_checkable
class Connector(Protocol):
    id: str
    source: str

    def handle(self, event: Any, store: Any) -> IngestStats: ...
