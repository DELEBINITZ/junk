"""PostgreSQL-backed datastore for the primary corpus (organizations, users,
documents, sections, chunks, shares).

Org-scoped reads run inside `org_transaction`, so row-level security enforces
tenant isolation at the database (defense in depth behind the Python RBAC).
Two operations legitimately cross orgs and use `privileged_transaction` (run as a
BYPASSRLS role): authentication user-lookup (no org context exists at login) and
admin seeding. Selected with STORE_BACKEND=postgres; requires the `prod` extra
(psycopg) + a migrated database. Verify against a live DB. See plan §8.2, §15.

Implements the same surface the application depends on (see
app/core/db/repository_protocol.py and the in-memory DataStore).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from app.config import settings
from app.core.db.postgres import PostgresDatabase, get_database
from app.documents.chunker import section_chunks
from app.documents.parser import ParsedContract
from app.domain import (
    AuditEvent, Chunk, Document, DocumentShare, Organization, QueryRecord,
    Report, Section, User, new_id,
)
from app.guardrails.pii import redact_pii
from app.rag.embeddings import create_embedding_provider


_CORE_TABLES = ["chunks", "sections", "document_shares", "documents", "users", "guardrail_configs", "organizations"]


def _vec_literal(vector: list[float]) -> str:
    return "[" + ",".join(repr(float(x)) for x in vector) + "]"


def _parse_vec(value) -> list[float]:
    if isinstance(value, list):
        return [float(x) for x in value]
    if isinstance(value, str) and value.strip().startswith("["):
        return [float(x) for x in value.strip()[1:-1].split(",") if x.strip()]
    return []


def _jsonb(value) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, (str, bytes)):
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            return {}
    return {}


class PostgresDataStore:
    def __init__(self, database: PostgresDatabase | None = None):
        self._db = database or get_database()
        self.embedder = create_embedding_provider()

    # ---- document reconstruction -------------------------------------------------
    def _row_to_document(self, row) -> Document:
        return Document(
            id=str(row[0]), organization_id=row[1], contract_id=row[2], title=row[3],
            filename=row[4], uploaded_by=row[5], tags=list(row[6] or []),
            raw_text=row[7] or "", redacted_text=row[8] or "", metadata=_jsonb(row[9]),
            deleted_at=row[10], created_at=row[11],
        )

    _DOC_COLS = ("id, organization_id, contract_id, title, filename, uploaded_by, tags, "
                 "raw_text, redacted_text, metadata, deleted_at, created_at")

    # ---- seeding / admin writes (privileged, cross-org) --------------------------
    def reset(self) -> None:
        with self._db.privileged_transaction() as conn:
            conn.execute("TRUNCATE " + ", ".join(_CORE_TABLES) + " CASCADE")

    def add_organization(self, organization: Organization) -> None:
        with self._db.privileged_transaction() as conn:
            conn.execute(
                "INSERT INTO organizations (id, name) VALUES (%s, %s) ON CONFLICT (id) DO NOTHING",
                (organization.id, organization.name),
            )
            conn.execute(
                "INSERT INTO guardrail_configs (organization_id) VALUES (%s) ON CONFLICT DO NOTHING",
                (organization.id,),
            )

    def add_user(self, user: User) -> None:
        with self._db.privileged_transaction() as conn:
            conn.execute(
                "INSERT INTO users (id, organization_id, email, name, role, password_hash, is_active) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s) ON CONFLICT (id) DO NOTHING",
                (user.id, user.organization_id, user.email, user.name, user.role,
                 user.password_hash, user.is_active),
            )

    def add_share(self, share: DocumentShare) -> None:
        with self._db.privileged_transaction() as conn:
            conn.execute(
                "INSERT INTO document_shares (document_id, user_id, access_level) VALUES (%s, %s, %s) "
                "ON CONFLICT (document_id, user_id) DO UPDATE SET access_level = EXCLUDED.access_level",
                (share.document_id, share.user_id, share.access_level),
            )

    def add_parsed_contract(
        self, parsed: ParsedContract, uploaded_by: str, tags: list[str] | None = None,
        contract_id_override: str | None = None, organization_id_override: str | None = None,
    ) -> Document:
        metadata = dict(parsed.metadata)
        contract_id = contract_id_override or str(metadata.get("contract_id") or new_id())
        organization_id = organization_id_override or str(metadata.get("organization_id") or "sandbox")
        title = str(metadata.get("title") or contract_id)
        redacted_text = redact_pii(parsed.raw_text)
        document = Document(
            id=new_id(), organization_id=organization_id, contract_id=contract_id, title=title,
            filename=parsed.filename, uploaded_by=uploaded_by, tags=tags or [],
            raw_text=parsed.raw_text, redacted_text=redacted_text, metadata=metadata,
        )
        with self._db.privileged_transaction() as conn:
            conn.execute(
                "INSERT INTO documents (id, organization_id, contract_id, title, filename, uploaded_by, "
                "tags, raw_text, redacted_text, metadata) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (document.id, organization_id, contract_id, title, parsed.filename, uploaded_by,
                 list(tags or []), parsed.raw_text, redacted_text, json.dumps(metadata)),
            )
            section_lookup: dict[str, Section] = {}
            for parsed_section in parsed.sections:
                section = Section(
                    id=new_id(), document_id=document.id, organization_id=organization_id,
                    section_number=parsed_section.section_number, section_title=parsed_section.section_title,
                    text=parsed_section.text, line_start=parsed_section.line_start,
                    line_end=parsed_section.line_end,
                )
                conn.execute(
                    "INSERT INTO sections (id, document_id, organization_id, section_number, section_title, "
                    "line_start, line_end, text) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
                    (section.id, document.id, organization_id, section.section_number,
                     section.section_title, section.line_start, section.line_end, section.text),
                )
                section_lookup[section.section_number] = section
            for chunk_data in section_chunks(parsed.sections):
                section = section_lookup[str(chunk_data["section_number"])]
                chunk_text = redact_pii(str(chunk_data["text"]))
                embedding = self.embedder.embed(chunk_text)
                conn.execute(
                    "INSERT INTO chunks (id, document_id, organization_id, section_id, chunk_index, text, "
                    "embedding, metadata) VALUES (%s,%s,%s,%s,%s,%s,%s::vector,%s)",
                    (new_id(), document.id, organization_id, section.id, int(chunk_data["chunk_index"]),
                     chunk_text, _vec_literal(embedding),
                     json.dumps({"contract_id": contract_id, "section_number": section.section_number,
                                 "section_title": section.section_title,
                                 "line_start": section.line_start, "line_end": section.line_end})),
                )
        return document

    def add_audit_event(self, event: AuditEvent) -> None:
        with self._db.privileged_transaction() as conn:
            conn.execute(
                "INSERT INTO audit_log (id, organization_id, user_id, action, resource_type, "
                "resource_id, outcome, details) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
                (event.id, event.organization_id, event.user_id, event.action, event.resource_type,
                 event.resource_id, event.outcome, json.dumps(event.details)),
            )

    def add_query_record(self, record: QueryRecord) -> None:
        with self._db.privileged_transaction() as conn:
            conn.execute(
                "INSERT INTO reports (id, organization_id, created_by, title, query, result) "
                "VALUES (%s,%s,%s,%s,%s,%s)",
                (record.id, record.organization_id, record.user_id, "query", record.query,
                 json.dumps(record.result)),
            )

    def add_report(self, report: Report) -> None:
        with self._db.privileged_transaction() as conn:
            conn.execute(
                "INSERT INTO reports (id, organization_id, created_by, title, query, result) "
                "VALUES (%s,%s,%s,%s,%s,%s)",
                (report.id, report.organization_id, report.created_by, report.title, report.query,
                 json.dumps(report.result)),
            )

    # ---- auth bootstrap (privileged, pre-org-context) ----------------------------
    def get_user(self, user_id: str) -> User | None:
        with self._db.privileged_transaction() as conn:
            row = conn.execute(
                "SELECT id, organization_id, email, name, role, password_hash, is_active "
                "FROM users WHERE id = %s", (user_id,),
            ).fetchone()
        return self._row_to_user(row) if row else None

    def user_by_email(self, email: str) -> User | None:
        with self._db.privileged_transaction() as conn:
            row = conn.execute(
                "SELECT id, organization_id, email, name, role, password_hash, is_active "
                "FROM users WHERE lower(email) = lower(%s)", (email,),
            ).fetchone()
        return self._row_to_user(row) if row else None

    def _row_to_user(self, row) -> User:
        return User(id=row[0], organization_id=row[1], email=row[2], name=row[3],
                    role=row[4], password_hash=row[5], is_active=row[6])

    def share_for(self, document_id: str, user_id: str) -> DocumentShare | None:
        with self._db.privileged_transaction() as conn:
            row = conn.execute(
                "SELECT document_id, user_id, access_level FROM document_shares "
                "WHERE document_id = %s AND user_id = %s", (document_id, user_id),
            ).fetchone()
        return DocumentShare(document_id=str(row[0]), user_id=row[1], access_level=row[2]) if row else None

    # ---- org-scoped reads (RLS-enforced) -----------------------------------------
    def documents_for_org(self, organization_id: str) -> list[Document]:
        with self._db.org_transaction(organization_id) as conn:
            rows = conn.execute(
                f"SELECT {self._DOC_COLS} FROM documents WHERE deleted_at IS NULL"
            ).fetchall()
        return [self._row_to_document(r) for r in rows]

    def document_by_contract_id(self, contract_id: str, organization_id: str | None = None) -> Document | None:
        if organization_id is None:
            with self._db.privileged_transaction() as conn:
                row = conn.execute(
                    f"SELECT {self._DOC_COLS} FROM documents WHERE contract_id = %s AND deleted_at IS NULL",
                    (contract_id,),
                ).fetchone()
        else:
            with self._db.org_transaction(organization_id) as conn:
                row = conn.execute(
                    f"SELECT {self._DOC_COLS} FROM documents WHERE contract_id = %s AND deleted_at IS NULL",
                    (contract_id,),
                ).fetchone()
        return self._row_to_document(row) if row else None

    def sections_for_document(self, document_id: str, organization_id: str | None = None) -> list[Section]:
        ctx = self._db.org_transaction(organization_id) if organization_id else self._db.privileged_transaction()
        with ctx as conn:
            rows = conn.execute(
                "SELECT id, document_id, organization_id, section_number, section_title, line_start, line_end, text "
                "FROM sections WHERE document_id = %s ORDER BY (section_number)::text", (document_id,),
            ).fetchall()
        return [self._row_to_section(r) for r in rows]

    def section_by_number(self, document_id: str, section_number: str, organization_id: str | None = None) -> Section | None:
        ctx = self._db.org_transaction(organization_id) if organization_id else self._db.privileged_transaction()
        with ctx as conn:
            row = conn.execute(
                "SELECT id, document_id, organization_id, section_number, section_title, line_start, line_end, text "
                "FROM sections WHERE document_id = %s AND section_number = %s", (document_id, section_number),
            ).fetchone()
        return self._row_to_section(row) if row else None

    def _row_to_section(self, row) -> Section:
        return Section(id=str(row[0]), document_id=str(row[1]), organization_id=row[2],
                       section_number=row[3], section_title=row[4], line_start=row[5],
                       line_end=row[6], text=row[7])

    def chunks_for_documents(self, document_ids: Iterable[str], organization_id: str | None = None) -> list[Chunk]:
        ids = list(document_ids)
        if not ids:
            return []
        ctx = self._db.org_transaction(organization_id) if organization_id else self._db.privileged_transaction()
        with ctx as conn:
            rows = conn.execute(
                "SELECT id, document_id, organization_id, section_id, chunk_index, text, embedding, metadata "
                "FROM chunks WHERE document_id = ANY(%s)", (ids,),
            ).fetchall()
        return [
            Chunk(id=str(r[0]), document_id=str(r[1]), organization_id=r[2], section_id=str(r[3]),
                  chunk_index=r[4], text=r[5], embedding=_parse_vec(r[6]), metadata=_jsonb(r[7]))
            for r in rows
        ]

    # ---- counts / lifecycle ------------------------------------------------------
    def is_empty(self) -> bool:
        with self._db.privileged_transaction() as conn:
            return conn.execute("SELECT count(*) FROM users").fetchone()[0] == 0

    def count_documents(self) -> int:
        with self._db.privileged_transaction() as conn:
            return conn.execute("SELECT count(*) FROM documents WHERE deleted_at IS NULL").fetchone()[0]

    def count_organizations(self) -> int:
        with self._db.privileged_transaction() as conn:
            return conn.execute("SELECT count(*) FROM organizations").fetchone()[0]

    def contract_paths(self, corpus_dir: Path | None = None) -> list[Path]:
        root = corpus_dir or settings.corpus_dir
        return sorted(path for path in Path(root).glob("*.txt") if path.is_file())
