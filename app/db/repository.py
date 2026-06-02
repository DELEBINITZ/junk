"""Repository abstraction for the PoC runtime.

The default implementation is in-memory so tests and demos are deterministic.
The rest of the application talks through this narrow repository shape, which
keeps RBAC, MCP, RAG, and agent logic independent from the storage backend.
"""

from __future__ import annotations

import os
from pathlib import Path
from threading import RLock
from typing import Iterable

from app.config import settings
from app.documents.chunker import section_chunks
from app.documents.parser import ParsedContract
from app.domain import (
    AuditEvent,
    Chunk,
    Document,
    DocumentShare,
    GuardrailConfig,
    Organization,
    QueryRecord,
    Report,
    Section,
    User,
    new_id,
)
from app.guardrails.pii import redact_pii
from app.rag.embeddings import create_embedding_provider


class DataStore:
    """In-memory repository used by the PoC and tests.

    The code keeps repository calls explicit so it is easy to replace this class
    with a PostgreSQL/pgvector adapter without changing RBAC, MCP, or agent code.
    """

    def __init__(self):
        self.lock = RLock()
        self.embedder = create_embedding_provider()
        self.reset()

    def reset(self) -> None:
        with self.lock:
            self.organizations: dict[str, Organization] = {}
            self.users: dict[str, User] = {}
            self.documents: dict[str, Document] = {}
            self.shares: dict[tuple[str, str], DocumentShare] = {}
            self.sections: dict[str, Section] = {}
            self.chunks: dict[str, Chunk] = {}
            self.guardrail_configs: dict[str, GuardrailConfig] = {}
            self.reports: dict[str, Report] = {}
            self.audit_events: list[AuditEvent] = []
            self.queries: dict[str, QueryRecord] = {}

    def add_organization(self, organization: Organization) -> None:
        with self.lock:
            self.organizations[organization.id] = organization
            self.guardrail_configs.setdefault(
                organization.id, GuardrailConfig(organization_id=organization.id)
            )

    def add_user(self, user: User) -> None:
        with self.lock:
            self.users[user.id] = user

    def add_share(self, share: DocumentShare) -> None:
        with self.lock:
            self.shares[(share.document_id, share.user_id)] = share

    def add_audit_event(self, event: AuditEvent) -> None:
        with self.lock:
            self.audit_events.append(event)

    def add_query_record(self, record: QueryRecord) -> None:
        with self.lock:
            self.queries[record.id] = record

    def add_report(self, report: Report) -> None:
        with self.lock:
            self.reports[report.id] = report

    def user_by_email(self, email: str) -> User | None:
        normalized = email.lower()
        return next((user for user in self.users.values() if user.email.lower() == normalized), None)

    def document_by_contract_id(self, contract_id: str, organization_id: str | None = None) -> Document | None:
        return next(
            (
                document
                for document in self.documents.values()
                if document.contract_id == contract_id
                and document.deleted_at is None
                and (organization_id is None or document.organization_id == organization_id)
            ),
            None,
        )

    def sections_for_document(self, document_id: str) -> list[Section]:
        return sorted(
            [section for section in self.sections.values() if section.document_id == document_id],
            key=lambda section: int(section.section_number),
        )

    def section_by_number(
        self, document_id: str, section_number: str, organization_id: str | None = None
    ) -> Section | None:
        return next(
            (
                section
                for section in self.sections.values()
                if section.document_id == document_id
                and section.section_number == section_number
                and (organization_id is None or section.organization_id == organization_id)
            ),
            None,
        )

    def chunks_for_documents(
        self, document_ids: Iterable[str], organization_id: str | None = None
    ) -> list[Chunk]:
        allowed = set(document_ids)
        return [
            chunk
            for chunk in self.chunks.values()
            if chunk.document_id in allowed
            and (organization_id is None or chunk.organization_id == organization_id)
        ]

    def get_user(self, user_id: str) -> User | None:
        return self.users.get(user_id)

    def documents_for_org(self, organization_id: str) -> list[Document]:
        return [d for d in self.documents.values() if d.organization_id == organization_id]

    def share_for(self, document_id: str, user_id: str) -> DocumentShare | None:
        return self.shares.get((document_id, user_id))

    def is_empty(self) -> bool:
        return not self.users

    def count_documents(self) -> int:
        return len(self.documents)

    def count_organizations(self) -> int:
        return len(self.organizations)

    def add_parsed_contract(
        self,
        parsed: ParsedContract,
        uploaded_by: str,
        tags: list[str] | None = None,
        contract_id_override: str | None = None,
        organization_id_override: str | None = None,
    ) -> Document:
        """Persist a parsed contract with sections, redacted chunks, and embeddings.

        This method is the ingestion commit point. It creates the document row,
        preserves legal sections for citation, redacts chunk text before vector
        indexing, and stores embeddings with enough metadata to trace a result
        back to a contract section.
        """

        metadata = dict(parsed.metadata)
        contract_id = contract_id_override or str(metadata.get("contract_id") or new_id())
        organization_id = organization_id_override or str(metadata.get("organization_id") or "sandbox")
        title = str(metadata.get("title") or contract_id)
        redacted_text = redact_pii(parsed.raw_text)

        with self.lock:
            document = Document(
                id=new_id(),
                organization_id=organization_id,
                contract_id=contract_id,
                title=title,
                filename=parsed.filename,
                uploaded_by=uploaded_by,
                tags=tags or [],
                raw_text=parsed.raw_text,
                redacted_text=redacted_text,
                metadata=metadata,
            )
            self.documents[document.id] = document

            section_lookup: dict[str, Section] = {}
            for parsed_section in parsed.sections:
                # Sections are first-class records because citations reference
                # legal sections, not arbitrary embedding chunks.
                section = Section(
                    id=new_id(),
                    document_id=document.id,
                    organization_id=organization_id,
                    section_number=parsed_section.section_number,
                    section_title=parsed_section.section_title,
                    text=parsed_section.text,
                    line_start=parsed_section.line_start,
                    line_end=parsed_section.line_end,
                )
                self.sections[section.id] = section
                section_lookup[section.section_number] = section

            for chunk_data in section_chunks(parsed.sections):
                section = section_lookup[str(chunk_data["section_number"])]
                chunk_text = redact_pii(str(chunk_data["text"]))
                # Store redacted chunk text to reduce the chance that retrieval
                # surfaces PII. Final responses are redacted again as a second
                # line of defense.
                chunk = Chunk(
                    id=new_id(),
                    document_id=document.id,
                    organization_id=organization_id,
                    section_id=section.id,
                    chunk_index=int(chunk_data["chunk_index"]),
                    text=chunk_text,
                    embedding=self.embedder.embed(chunk_text),
                    metadata={
                        "contract_id": contract_id,
                        "section_number": section.section_number,
                        "section_title": section.section_title,
                        "line_start": section.line_start,
                        "line_end": section.line_end,
                    },
                )
                self.chunks[chunk.id] = chunk

            return document

    def contract_paths(self, corpus_dir: Path | None = None) -> list[Path]:
        root = corpus_dir or settings.corpus_dir
        return sorted(path for path in root.glob("*.txt") if path.is_file())


_store = None


def get_store():
    """Return the configured datastore. Default in-memory (no driver). With
    STORE_BACKEND=postgres, returns the RLS-backed PostgresDataStore (requires
    the `prod` extra + a reachable DATABASE_URL). See plan §8.2, §15."""

    global _store
    if _store is None:
        backend = os.getenv("STORE_BACKEND", settings.store_backend).lower()
        if backend == "postgres":
            from app.core.db.pg_datastore import PostgresDataStore

            _store = PostgresDataStore()
        else:
            _store = DataStore()
    return _store


def reset_store() -> None:
    global _store
    _store = None
