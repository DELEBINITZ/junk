#!/usr/bin/env python3
"""Reports ingestion cron — embeds security reports and upserts them into the SAME
Qdrant collection the platform RETRIEVES from (``reports_kb``).

WHY THIS REWRITE EXISTS (the bugs it fixes vs. the old Ollama version):
  1. COLLECTION NAME — must equal the app's ``REPORTS_COLLECTION`` ("reports_kb"),
     not "aci_reports", or the app queries an empty collection.
  2. EMBEDDER + DIM — must be the SAME embedder the app queries with: TEI
     Qwen3-Embedding-4B at dim 2560 (settings.embedding_dim, the 24GB quant profile).
     The old Ollama ``qwen3-embedding:4b`` was the right dim by accident but a
     different runtime/space; index via the SAME TEI server the app queries, or
     vectors won't match and Qdrant rejects dim mismatches.
  3. PAYLOAD KEYS — must match what the retriever reads (qdrant_backend.py): the
     chunk text goes in ``text`` (not ``chunk_content``), the id in ``doc_id`` (not
     ``_id``), the date in ``published_at`` as RFC3339 (not ``information_date``).
  4. VISIBILITY — this corpus is SHARED INTEL. Each point carries ``customer_tags``
     (an allow-list of org_ids). EMPTY/absent = PUBLIC to every org. The app's
     ``visibility="shared"`` filter reads this exact top-level key.

DOCUMENTS ARE EMBEDDED RAW. Qwen3-Embedding wants its instruction prefix on the
QUERY side only (the app applies it via EMBEDDING_QUERY_INSTRUCTION); passages here
must NOT carry it, or query/passage vectors stop matching.

Idempotent: a chunk's point id is a deterministic uuid5 of ``{doc_id}_chunk_{idx}``,
so re-running overwrites rather than duplicating. Set REINDEX_DELETE_FIRST=1 to also
purge stale chunks of a doc (when a re-indexed report has FEWER chunks than before).

Run:  uv run python scripts/index_reports_cron.py
Env:  QDRANT_URL, QDRANT_API_KEY, TEI_EMBED_URL, EMBEDDING_DIM, COLLECTION_NAME,
      REPORTS_SOURCE, REINDEX_DELETE_FIRST, plus your source-API vars (see fetch_reports).
"""

from __future__ import annotations

import os
import sys
import uuid

import requests
from qdrant_client import QdrantClient, models

# ---------------------------------------------------------------------------
# 1. Configuration — defaults mirror app/config.py so index == query by default.
# ---------------------------------------------------------------------------
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "") or None
TEI_EMBED_URL = os.getenv("TEI_EMBED_URL", "http://localhost:8080").rstrip("/")
EMBEDDING_DIM = int(
    os.getenv("EMBEDDING_DIM", "2560")
)  # MUST equal settings.embedding_dim (Qwen3-Embedding-4B)
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "reports_kb")  # MUST equal REPORTS_COLLECTION
SOURCE = os.getenv("REPORTS_SOURCE", "aci_reports")  # provenance tag, NOT the collection name
DELETE_FIRST = os.getenv("REINDEX_DELETE_FIRST", "0") == "1"

CHUNK_MAX_WORDS = int(os.getenv("CHUNK_MAX_WORDS", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
EMBED_TIMEOUT = float(os.getenv("EMBED_TIMEOUT", "60"))

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


# ---------------------------------------------------------------------------
# 2. Chunking — split long reports into retrieval-sized passages.
# ---------------------------------------------------------------------------
def chunk_text_by_words(
    text: str, max_words: int = CHUNK_MAX_WORDS, overlap: int = CHUNK_OVERLAP
) -> list[str]:
    words = (text or "").split()
    if not words:
        return []
    step = max(1, max_words - overlap)  # guard: never a zero/negative stride
    return [" ".join(words[i : i + max_words]) for i in range(0, len(words), step)]


# ---------------------------------------------------------------------------
# 3. Embedding — TEI, the SAME server the app queries with. Documents embedded RAW
#    (no Qwen3 query instruction). Batched: one HTTP call per report.
# ---------------------------------------------------------------------------
def embed_documents(chunks: list[str]) -> list[list[float]]:
    """POST a batch of passages to TEI ``/embed`` and return their vectors.
    Raises on failure so the caller can skip the whole report rather than upsert
    a partially-embedded doc."""
    r = requests.post(f"{TEI_EMBED_URL}/embed", json={"inputs": chunks}, timeout=EMBED_TIMEOUT)
    r.raise_for_status()
    vectors = r.json()
    if len(vectors) != len(chunks):
        raise RuntimeError(f"TEI returned {len(vectors)} vectors for {len(chunks)} chunks")
    for v in vectors:
        if len(v) != EMBEDDING_DIM:
            raise RuntimeError(
                f"TEI vector dim {len(v)} != EMBEDDING_DIM {EMBEDDING_DIM}. "
                f"The collection and the app's settings.embedding_dim must all agree."
            )
    return vectors


# ---------------------------------------------------------------------------
# 4. Collection — create ONCE, sized + indexed to match the app's ensure_collection.
#    Fails LOUD on a dimension mismatch instead of silently upserting into a
#    wrong-dim collection (which would corrupt retrieval).
# ---------------------------------------------------------------------------
def ensure_collection() -> None:
    if client.collection_exists(COLLECTION_NAME):
        info = client.get_collection(COLLECTION_NAME)
        existing_dim = info.config.params.vectors.size
        if existing_dim != EMBEDDING_DIM:
            sys.exit(
                f"FATAL: collection '{COLLECTION_NAME}' exists at dim {existing_dim}, "
                f"but EMBEDDING_DIM={EMBEDDING_DIM}. Qdrant can't resize — DROP it first:\n"
                f'  python -c "from qdrant_client import QdrantClient; '
                f"QdrantClient('{QDRANT_URL}').delete_collection('{COLLECTION_NAME}')\""
            )
    else:
        print(f"Creating collection '{COLLECTION_NAME}' (dim={EMBEDDING_DIM}, COSINE)")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=EMBEDDING_DIM, distance=models.Distance.COSINE),
        )
    # Payload indexes — EXACTLY the fields the retriever filters on (qdrant_backend.py).
    # Creating an index that already exists raises; swallow it (idempotent).
    indexes = [
        ("source", models.PayloadSchemaType.KEYWORD),
        ("doc_id", models.PayloadSchemaType.KEYWORD),
        ("published_at", models.PayloadSchemaType.DATETIME),  # recency range filtering
        ("customer_tags", models.PayloadSchemaType.KEYWORD),  # shared-visibility allow-list (array)
        ("public", models.PayloadSchemaType.BOOL),  # explicit "visible to every org" flag
    ]
    for field, schema in indexes:
        try:
            client.create_payload_index(COLLECTION_NAME, field_name=field, field_schema=schema)
        except Exception:  # noqa: BLE001 — already exists
            pass


# ---------------------------------------------------------------------------
# 5. Payload mapping — report JSON -> the retriever's contract.
# ---------------------------------------------------------------------------
def _published_at(report: dict) -> str | None:
    """``information_date`` ("2026-01-28") -> RFC3339 ("2026-01-28T00:00:00Z") so the
    app's DatetimeRange recency filter parses it. None when absent."""
    d = report.get("information_date")
    if not d:
        return None
    return f"{d}T00:00:00Z" if len(d) == 10 else d


def _title(report: dict) -> str:
    insight = report.get("insight") or {}
    return (
        report.get("title")
        or insight.get("title")
        or " ".join((insight.get("content_text") or "").split()[:12])  # first ~12 words
        or "Security advisory"
    )


def _light_metadata(report: dict, idx: int, total: int) -> dict:
    """Small, FILTERABLE extras (reachable via SearchFilters.extra -> metadata.<key>).
    Deliberately does NOT copy the full content_text — that bloats every chunk."""
    return {
        "information_date": report.get("information_date"),
        "geographies": [
            g.get("country") for g in report.get("geographies", []) if g.get("country")
        ],
        "industries": [i.get("name") for i in report.get("industries", []) if i.get("name")],
        "adversaries": [a.get("name") for a in report.get("adversaries", []) if a.get("name")],
        "threat_types": report.get(
            "threat_types", []
        ),  # present on some reports; harmless when absent
        "tlp": report.get("tlp"),
        "report_type": report.get("report_type"),
        "chunk_index": idx,
        "total_chunks": total,
    }


def _is_public(report: dict) -> bool:
    """A shared-corpus document is PUBLIC (visible to every tenant) only when the
    feed says so EXPLICITLY: a truthy ``public`` flag, or a TLP:CLEAR/WHITE marking
    (the Traffic Light Protocol value for 'freely shareable'). Everything else is
    restricted — visible only to the orgs named in ``customer_tags``. This is the
    fail-CLOSED rule: absence of a signal means PRIVATE, never public."""
    if bool(report.get("public")):
        return True
    return str(report.get("tlp") or "").strip().upper() in {
        "CLEAR",
        "WHITE",
        "TLP:CLEAR",
        "TLP:WHITE",
    }


def build_points(report: dict) -> list[models.PointStruct]:
    doc_id = str(report["_id"])
    full_text = (report.get("insight") or {}).get("content_text", "")
    chunks = chunk_text_by_words(full_text)
    if not chunks:
        return []

    # SHARED-corpus visibility. ``customer_tags`` allow-lists specific orgs; absent/[]
    # means PUBLIC to every org under the app's default policy (SHARED_UNTAGGED_PUBLIC).
    # ``public`` is an EXPLICIT "visible to everyone" marker (set from the feed's
    # TLP:CLEAR/WHITE or a public flag) — it makes a doc public even when the app runs
    # in the strict SHARED_UNTAGGED_PUBLIC=false mode, and documents intent regardless.
    customer_tags = report.get("customer_tags", []) or []
    public = _is_public(report)

    vectors = embed_documents(chunks)
    title, published_at = _title(report), _published_at(report)

    points = []
    for idx, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True)):
        rid = f"{doc_id}_chunk_{idx}"
        points.append(
            models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_DNS, rid)),  # deterministic => re-run overwrites
                vector=vector,
                payload={
                    # ---- keys the retriever READS (qdrant_backend.py search) ----
                    "rid": rid,
                    "text": chunk,  # the passage the LLM grounds on
                    "doc_id": doc_id,
                    "source": SOURCE,
                    "title": title,
                    "section": f"chunk {idx}",
                    "published_at": published_at,
                    # ---- shared-visibility (top-level; matched by _filter, DEFAULT-DENY) ----
                    "customer_tags": customer_tags,  # orgs explicitly allow-listed for this doc
                    "public": public,  # True => visible to EVERY org (explicit only)
                    # ---- arbitrary extras, filterable as metadata.<key> ----
                    "metadata": _light_metadata(report, idx, len(chunks)),
                    # NOTE: no org_id — shared corpus has no single owner; the shared
                    # filter never reads it.
                },
            )
        )
    return points


def index_report(report: dict) -> int:
    doc_id = str(report["_id"])
    if DELETE_FIRST:
        # Purge any prior chunks of this doc so a shrunk report leaves no orphans.
        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))
                    ]
                )
            ),
        )
    try:
        points = build_points(report)
    except Exception as e:  # noqa: BLE001 — skip the whole report on embed failure
        print(f"  ! skip report {doc_id}: {e}")
        return 0
    if points:
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"  upserted {len(points)} chunks for {doc_id}")
    return len(points)


# ---------------------------------------------------------------------------
# 6. Source API — WIRE THIS TO YOUR REAL FEED. Kept as the same shape as your
#    original (sliding-window backfill -> paged GET -> items). Fill in url/headers/
#    payload/sliding_window from your environment; everything below is source-agnostic.
# ---------------------------------------------------------------------------
def sliding_window(backfilling: bool = True):
    """Yield the timestamps (pages) to pull. Replace with your real window logic."""
    raise NotImplementedError("wire sliding_window() to your source's pagination")


def fetch_reports():
    """Generator over report dicts from your source API. Replace the request block
    with your real endpoint; it must yield dicts shaped like your sample
    (``_id``, ``insight.content_text``, ``information_date``, optional ``customer_tags``)."""
    url = os.environ["REPORTS_API_URL"]  # e.g. "https://.../reports?ts={timestamp}"
    headers = {"Authorization": f"Bearer {os.environ.get('REPORTS_API_TOKEN', '')}"}
    payload: dict = {}
    for t in sliding_window(backfilling=True):
        resp = requests.get(
            url.format(timestamp=t), headers=headers, json=payload, timeout=30
        ).json()
        if resp.get("total", 0) == 0:
            continue
        yield from resp.get("items", [])


# ---------------------------------------------------------------------------
# 7. Entry point.
# ---------------------------------------------------------------------------
def main() -> None:
    print(
        f"Indexing -> {QDRANT_URL} / '{COLLECTION_NAME}' via TEI {TEI_EMBED_URL} (dim {EMBEDDING_DIM})"
    )
    ensure_collection()
    total_reports = total_chunks = 0
    for report in fetch_reports():
        total_reports += 1
        total_chunks += index_report(report)
    print(f"Done. {total_reports} reports -> {total_chunks} chunks in '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    main()
