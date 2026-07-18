#!/usr/bin/env python3
"""FortiRecon user-guide ingestion service — parses the product documentation
(HTML), embeds it with the SAME TEI model the app queries with, and upserts it
into a DEDICATED Qdrant collection (``user_guide_kb``) that the Atlas agent
retrieves from.

WHY A SEPARATE COLLECTION (not reports_kb):
  Docs are how-to / navigation pages, not threat intel. Mixing them into the
  report corpus would let a "what ransomware do we see?" query surface a
  "How to read the Dashboard" page (and vice-versa). Separate collection =
  clean retrieval for both agents. Same embedder + dim, so no infra changes.

SCHEMA CONTRACT (must match tools/qdrant_search.py access filter + userguide_search):
  text            the passage the LLM grounds on (chunk body)
  doc_id          stable per-PAGE id (re-running overwrites, never duplicates)
  source          provenance tag (default "fortirecon_user_guide")
  title           page title (user-facing)
  heading         nearest section heading for the chunk (user-facing)
  url             canonical page URL when known (user-facing citation)
  section         "chunk <idx>"
  published_at    None (docs are not time-bound; key kept for schema parity)
  customer_tags   [] — product docs are shared
  public          True — visible to EVERY org
  is_deleted      False — the retriever's access filter REQUIRES this key
  metadata        {chunk_index, total_chunks, heading, url, product, guide_version}

EMBEDDING: passages are embedded with a small "<title> / <heading>" context prefix
(asymmetric augmentation — improves doc recall). Queries are embedded raw by the
app. Both hit the SAME TEI server, so dims and space agree.

Idempotent + self-syncing (safe to re-run any time):
  - point id = uuid5("<doc_id>_chunk_<idx>") -> a chunk is overwritten in place, never
    duplicated.
  - doc_id = the file's path RELATIVE to the ingest root (offline export) or the URL
    slug (scraped) -> unique per page (same-named files in different folders don't
    collide) and stable across runs.
  - per re-run each page is UPSERTED then PRUNED -> a shrunk/edited page leaves no stale
    chunks behind.
  - after a full --html-dir run, sync_removed() deletes pages that vanished from the
    source (deleted/renamed) so the collection always mirrors the current docs.
So: new pages created, changed pages overwritten, removed pages deleted, zero dupes.
REINDEX_DELETE_FIRST=1 forces a hard per-page purge; PRUNE_REMOVED=0 disables the
removed-page sync (use when ingesting only a PARTIAL subset of the guide).

This is a STANDALONE, independently-deployable service (its own pyproject.toml +
Dockerfile under services/userguide-ingest/). It imports NOTHING from the main app;
the only contract is the Qdrant collection + payload schema + embedder, coordinated
via env vars. See services/userguide-ingest/README.md.

Run (local HTML you already downloaded — the primary path):
    python index_user_guide.py --html-dir ./fortirecon_userguide_html

Run (scrape/crawl from the live docs, best-effort):
    python index_user_guide.py \
        --url https://docs.fortinet.com/document/fortirecon/26.2.a/user-guide/897693/introduction \
        --crawl --max-pages 400

Env: QDRANT_URL, QDRANT_API_KEY, TEI_EMBED_URL, EMBEDDING_DIM, USER_GUIDE_COLLECTION,
     GUIDE_SOURCE, GUIDE_VERSION, GUIDE_PRODUCT, REINDEX_DELETE_FIRST, PRUNE_REMOVED,
     CHUNK_MAX_WORDS, CHUNK_OVERLAP, CRAWL_DELAY, EMBED_TIMEOUT
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
import uuid
from pathlib import Path
from urllib.parse import urldefrag, urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
from qdrant_client import QdrantClient, models

# ---------------------------------------------------------------------------
# 1. Configuration — defaults mirror the app so index == query by default.
# ---------------------------------------------------------------------------
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "") or None
TEI_EMBED_URL = os.getenv("TEI_EMBED_URL", os.getenv("EMBEDDING_BASE_URL", "http://localhost:8080")).rstrip("/")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "2560"))  # MUST equal settings.embedding_dim
COLLECTION_NAME = os.getenv("USER_GUIDE_COLLECTION", "user_guide_kb")  # MUST equal settings.user_guide_collection
SOURCE = os.getenv("GUIDE_SOURCE", "fortirecon_user_guide")
GUIDE_VERSION = os.getenv("GUIDE_VERSION", "26.2.a")
GUIDE_PRODUCT = os.getenv("GUIDE_PRODUCT", "FortiRecon")
# Canonical doc URL is reconstructed from each page's own <meta data-pageid> (present
# in every MadCap offline export) + the title slug — so offline pages get an exact,
# citable docs.fortinet.com link with ZERO manual mapping. The numeric pageid is
# authoritative on the live site (the slug is cosmetic; a wrong slug still resolves).
DOCS_URL_TEMPLATE = os.getenv(
    "DOCS_URL_TEMPLATE",
    "https://docs.fortinet.com/document/{product}/{version}/user-guide/{pageid}/{slug}",
)
DELETE_FIRST = os.getenv("REINDEX_DELETE_FIRST", "0") == "1"
# Full-sync: after a COMPLETE --html-dir ingest, delete pages that vanished from the
# source (removed or renamed) so the collection always matches the current docs. On by
# default for --html-dir (a full snapshot); set PRUNE_REMOVED=0 to ingest a partial set
# without deleting the rest.
PRUNE_REMOVED = os.getenv("PRUNE_REMOVED", "1") != "0"

CHUNK_MAX_WORDS = int(os.getenv("CHUNK_MAX_WORDS", "350"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
EMBED_TIMEOUT = float(os.getenv("EMBED_TIMEOUT", "60"))
EMBED_BATCH = int(os.getenv("EMBED_BATCH", "32"))
CRAWL_DELAY = float(os.getenv("CRAWL_DELAY", "0.5"))
FETCH_TIMEOUT = float(os.getenv("FETCH_TIMEOUT", "30"))

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
_http = httpx.Client(
    timeout=FETCH_TIMEOUT,
    follow_redirects=True,
    headers={"User-Agent": "fortirecon-userguide-indexer/1.0"},
)

# Containers we strip before extracting text — chrome, not content.
_STRIP_TAGS = ("script", "style", "noscript", "nav", "header", "footer", "aside", "form", "svg")
# Priority-ordered candidate selectors for the real article body across doc themes.
# MadCap Flare offline exports (the Fortinet user-guide HTML download) put the article
# in #mc-main-content / .body-container — listed explicitly so extraction isn't
# accidental and survives theme tweaks.
_CONTENT_SELECTORS = (
    "main",
    "article",
    "[role=main]",
    "#mc-main-content",
    "div.mc-main-content",
    "div.body-container",
    "div.content-container",
    "div#content",
    "div.content",
    "div.body",
    "div.article",
    "div.documentation",
)


# ---------------------------------------------------------------------------
# 2. Chunking — line/paragraph-aware packing.
#
# A section is a list of atomic text blocks (paragraphs, list items, table rows).
# We PACK whole blocks up to a word budget rather than blindly slicing a flat word
# stream, so a chunk never cuts a sentence — or a table row's "Module — description"
# — in half. Only a single block that is itself larger than the budget is
# hard-split (rare: a very long paragraph).
# ---------------------------------------------------------------------------
def _word_windows(text: str, max_words: int, overlap: int) -> list[str]:
    """Hard word-window split for a single oversized block."""
    words = (text or "").split()
    if not words:
        return []
    step = max(1, max_words - overlap)
    return [" ".join(words[i : i + max_words]) for i in range(0, len(words), step)]


def pack_lines(
    blocks: list[tuple[str, bool]], max_words: int = CHUNK_MAX_WORDS, overlap: int = CHUNK_OVERLAP
) -> list[str]:
    """Pack blocks into chunks up to ``max_words``.

    Each block is ``(text, atomic)``. An ATOMIC block (a table row rendered as
    "Name — description") is a self-contained topic and becomes its OWN chunk — so
    a single-topic query ("Red Teaming module") matches a tight per-topic vector
    instead of one diluted vector holding five modules. Non-atomic prose blocks are
    packed together up to the budget, with a small trailing overlap for continuity.
    """
    chunks: list[str] = []
    cur: list[str] = []
    cur_words = 0

    def flush_prose():
        nonlocal cur, cur_words
        if cur:
            chunks.append("\n".join(cur))
            cur, cur_words = [], 0

    for text, atomic in blocks:
        w = len(text.split())
        if atomic:
            flush_prose()
            if w > max_words:
                chunks.extend(_word_windows(text, max_words, overlap))
            else:
                chunks.append(text)
            continue
        if w > max_words:
            flush_prose()
            chunks.extend(_word_windows(text, max_words, overlap))
            continue
        if cur and cur_words + w > max_words:
            chunks.append("\n".join(cur))
            # Overlap: carry trailing prose blocks totaling ~overlap words forward.
            carry: list[str] = []
            c = 0
            for pp in reversed(cur):
                carry.insert(0, pp)
                c += len(pp.split())
                if c >= overlap:
                    break
            cur = list(carry)
            cur_words = sum(len(x.split()) for x in cur)
        cur.append(text)
        cur_words += w
    flush_prose()
    return chunks


# ---------------------------------------------------------------------------
# 3. HTML parsing — title, canonical URL, and (heading, text) sections.
# ---------------------------------------------------------------------------
def _clean(text: str) -> str:
    return re.sub(r"[ \t ]+", " ", re.sub(r"\n{3,}", "\n\n", text or "")).strip()


def _pick_content(soup: BeautifulSoup):
    for sel in _CONTENT_SELECTORS:
        node = soup.select_one(sel)
        if node and len(node.get_text(strip=True)) > 200:
            return node
    return soup.body or soup


def _fin_title(text: str) -> str:
    """Normalize a title: decode a double-encoded ``&nbsp;``, collapse whitespace,
    and drop the ' | User Guide' site suffix."""
    t = _clean((text or "").replace("&nbsp;", " "))
    return re.split(r"\s+\|\s+", t)[0].strip() or t


def _extract_title(soup: BeautifulSoup, content) -> str:
    og = soup.find("meta", attrs={"property": "og:title"})
    if og and og.get("content"):
        return _fin_title(og["content"])
    h1 = content.find(["h1"]) or soup.find("h1")
    if h1 and h1.get_text(strip=True):
        return _fin_title(h1.get_text())
    if soup.title and soup.title.string:
        return _fin_title(soup.title.string)
    return "FortiRecon user guide"


def _extract_canonical_url(soup: BeautifulSoup) -> str:
    link = soup.find("link", attrs={"rel": lambda v: v and "canonical" in v})
    if link and link.get("href"):
        return link["href"].strip()
    og = soup.find("meta", attrs={"property": "og:url"})
    if og and og.get("content"):
        return og["content"].strip()
    return ""


def _slugify(text: str) -> str:
    """Title -> URL slug ('Adding watermarks' -> 'adding-watermarks')."""
    return re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")


def _extract_pageid(soup: BeautifulSoup) -> str:
    """Numeric page id from the MadCap export's <meta data-pageid="630160" ...>.

    This id is exactly the number in the live URL (.../user-guide/630160/...), so it
    lets an OFFLINE page (which ships no canonical <link>) still be cited with its real
    docs URL — no hand-maintained filename->URL table.
    """
    m = soup.find("meta", attrs={"data-pageid": True})
    return (m.get("data-pageid") or "").strip() if m else ""


def _canonical_url_from_pageid(pageid: str, title: str) -> str:
    """Build the live docs URL from the page id + title slug (empty if no pageid)."""
    if not pageid:
        return ""
    return DOCS_URL_TEMPLATE.format(
        product=re.sub(r"\s+", "", GUIDE_PRODUCT).lower(),
        version=GUIDE_VERSION,
        pageid=pageid,
        slug=_slugify(title) or "page",
    )


# Leaf block tags whose text is emitted as ONE atomic block. Deliberately does NOT
# include container tags (td/tr/div): descending into a container AND matching its
# child <p> double-counts the same text — the source of the duplicate chunks seen
# on the Fortinet pages (their module lists are 2-column tables of <td><p>…).
_LEAF_BLOCKS = ("p", "li", "pre", "blockquote", "figcaption", "dt", "dd")
# MadCap Flare (the Fortinet offline export) uses h5/h6 for procedure lead-ins
# ("To create a digital watermark:", "To edit…"). Excluding them silently DROPPED
# that text (h5/h6 are neither a recursed wrapper nor a leaf block). Include them so
# the lead-in survives AND becomes the chunk's heading (feeds the embed prefix).
_HEADINGS = ("h1", "h2", "h3", "h4", "h5", "h6")


def _table_rows(table) -> list[str]:
    """Render a table as one line per row: 'cell1 — cell2 — …'.

    Fortinet doc pages lay out feature/module lists as 2-column tables
    (name | description). Emitting a row as 'Name — description' keeps each
    topic self-contained in a single chunkable block, instead of scattering the
    name away from its text. Header rows collapse to the joined header text.
    """
    lines: list[str] = []
    for tr in table.find_all("tr"):
        cells = []
        for cell in tr.find_all(["th", "td"], recursive=False) or tr.find_all(["th", "td"]):
            t = _clean(cell.get_text(" "))
            if t:
                cells.append(t)
        # Drop exact-duplicate cells within a row (e.g. name repeated as an xref link),
        # preserving order.
        cells = list(dict.fromkeys(cells))
        line = " — ".join(cells)
        if line:
            lines.append(line)
    return lines


def _dedup_adjacent(blocks: list[tuple[str, bool]]) -> list[tuple[str, bool]]:
    """Collapse consecutive blocks with identical text (responsive DOM variants
    render the same paragraph twice on these pages)."""
    out: list[tuple[str, bool]] = []
    for blk in blocks:
        if not out or out[-1][0] != blk[0]:
            out.append(blk)
    return out


def _walk(node, sections: list[tuple[str, list[tuple[str, bool]]]]) -> None:
    """Depth-first walk that emits each block of text exactly once, opens a new
    section at every heading, and renders tables row-wise (see _table_rows).

    Blocks are ``(text, atomic)``: table rows are atomic (each a standalone topic,
    chunked on its own); prose paragraphs/list items are non-atomic (packed).
    """
    for child in node.children:
        name = getattr(child, "name", None)
        if name is None:  # bare string between blocks — skip
            continue
        if name in _HEADINGS:
            heading = _clean(child.get_text(" "))
            if heading:
                sections.append((heading, []))
        elif name == "table":
            for row in _table_rows(child):  # do NOT recurse into cells
                sections[-1][1].append((row, True))
        elif name in _LEAF_BLOCKS:
            t = _clean(child.get_text(" "))
            if t:
                sections[-1][1].append((t, False))  # atomic unit — do NOT recurse
        else:
            _walk(child, sections)  # wrapper (div/section/ul/…): recurse


def _sections(content, page_title: str) -> list[tuple[str, list[tuple[str, bool]]]]:
    """Split content into (heading, [(text, atomic)]) sections by h1-h4 boundaries.

    Each block is emitted exactly once. Content before the first heading is
    attached to the page title.
    """
    sections: list[tuple[str, list[tuple[str, bool]]]] = [(page_title, [])]
    _walk(content, sections)

    out: list[tuple[str, list[tuple[str, bool]]]] = []
    for heading, blocks in sections:
        blocks = _dedup_adjacent([b for b in blocks if b[0]])
        if blocks:
            out.append((heading, blocks))
    # Fallback: unusual layout produced no blocks — take the raw text as one block.
    if not out:
        raw = _clean(content.get_text("\n"))
        if raw:
            out = [(page_title, [(raw, False)])]
    return out


def _url_id_slug(url: str) -> tuple[str, str]:
    """(id, slug) from a user-guide URL: .../user-guide/531751/easm-dashboard."""
    path = urlparse(url).path if url else ""
    m = re.search(r"/user-guide/([^/]+)/([^/]+)/?$", path)
    if m:
        return m.group(1), m.group(2)
    tail = path.strip("/").rsplit("/", 1)[-1]
    return "", tail


def _breadcrumb_from_toc(soup: BeautifulSoup, url: str, page_title: str) -> list[str]:
    """Ancestry path for this page from the embedded ``ul.toc`` nav tree.

    Every doc page ships the full navigation tree (nested <ul class="toc">). We
    locate THIS page's own anchor (by URL id/slug) and walk up its <li> ancestors,
    yielding e.g. ["Attack Surface Management", "EASM", "EASM Dashboard"]. This is
    the authoritative section hierarchy — far richer than the generic
    "Home > User Guide" breadcrumb bar. Falls back to [page_title] when the page
    isn't in the tree (e.g. an unlinked/crawled page).

    MUST be called BEFORE chrome is stripped (the toc may live inside <nav>).
    """
    toc = soup.select_one("ul.toc")
    pid, slug = _url_id_slug(url)
    if not toc or (not pid and not slug):
        return [page_title]

    target = None
    for a in toc.find_all("a", href=True):
        href = a["href"].rstrip("/")
        if (pid and f"/{pid}/" in href + "/") or (slug and href.endswith("/" + slug)):
            target = a
            break
    if not target:
        return [page_title]

    anc: list[str] = []
    li = target.find_parent("li")
    while li is not None:
        a = li.find("a")
        label = _clean(a.get_text(" ").replace("&nbsp;", " ")) if a else ""
        if label:
            anc.append(label)
        li = li.find_parent("li")
    return list(reversed(anc)) if anc else [page_title]


def parse_html(html: str, fallback_url: str = "") -> dict | None:
    soup = BeautifulSoup(html, "html.parser")
    # Resolve URL + hierarchy from the FULL document BEFORE stripping chrome — the
    # nav tree (ul.toc) is chrome we otherwise remove.
    url = _extract_canonical_url(soup) or fallback_url
    pageid = _extract_pageid(soup)  # read before chrome strip (meta lives in <head>)
    pre_title = _extract_title(soup, soup)
    section_path = _breadcrumb_from_toc(soup, url, pre_title)

    for tag in soup.find_all(_STRIP_TAGS):
        tag.decompose()
    content = _pick_content(soup)
    title = _extract_title(soup, content)
    sections = _sections(content, title)
    if not sections:
        return None
    # Ensure the page's own title is the leaf of the path.
    if not section_path or section_path[-1] != title:
        if title not in section_path:
            section_path = [*section_path, title] if section_path else [title]
    return {
        "title": title,
        "url": url,
        "pageid": pageid,
        "sections": sections,
        "section_path": section_path,
    }


# ---------------------------------------------------------------------------
# 4. Embedding — TEI, the SAME server the app queries with. Batched.
# ---------------------------------------------------------------------------
def embed_documents(texts: list[str]) -> list[list[float]]:
    vectors: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i : i + EMBED_BATCH]
        r = _http.post(f"{TEI_EMBED_URL}/embed", json={"inputs": batch}, timeout=EMBED_TIMEOUT)
        r.raise_for_status()
        vecs = r.json()
        if len(vecs) != len(batch):
            raise RuntimeError(f"TEI returned {len(vecs)} vectors for {len(batch)} inputs")
        for v in vecs:
            if len(v) != EMBEDDING_DIM:
                raise RuntimeError(
                    f"TEI vector dim {len(v)} != EMBEDDING_DIM {EMBEDDING_DIM}. "
                    f"Collection and app settings.embedding_dim must agree."
                )
        vectors.extend(vecs)
    return vectors


# ---------------------------------------------------------------------------
# 5. Collection — create once, sized + indexed to match the retriever's filter.
# ---------------------------------------------------------------------------
def ensure_collection() -> None:
    if client.collection_exists(COLLECTION_NAME):
        info = client.get_collection(COLLECTION_NAME)
        existing_dim = info.config.params.vectors.size
        if existing_dim != EMBEDDING_DIM:
            sys.exit(
                f"FATAL: collection '{COLLECTION_NAME}' exists at dim {existing_dim}, "
                f"but EMBEDDING_DIM={EMBEDDING_DIM}. Qdrant can't resize — drop it first."
            )
    else:
        print(f"Creating collection '{COLLECTION_NAME}' (dim={EMBEDDING_DIM}, COSINE)")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=EMBEDDING_DIM, distance=models.Distance.COSINE),
        )
    indexes = [
        ("source", models.PayloadSchemaType.KEYWORD),
        ("doc_id", models.PayloadSchemaType.KEYWORD),
        ("url", models.PayloadSchemaType.KEYWORD),
        ("section_path", models.PayloadSchemaType.KEYWORD),  # nav ancestry (array) — filterable
        ("is_deleted", models.PayloadSchemaType.BOOL),  # retriever access filter REQUIRES this
        ("customer_tags", models.PayloadSchemaType.KEYWORD),
        ("public", models.PayloadSchemaType.BOOL),
    ]
    for field, schema in indexes:
        try:
            client.create_payload_index(COLLECTION_NAME, field_name=field, field_schema=schema)
        except Exception:  # noqa: BLE001 — already exists
            pass


# ---------------------------------------------------------------------------
# 6. Page id + point building.
# ---------------------------------------------------------------------------
def _doc_id(url: str, fallback: str) -> str:
    """Stable, UNIQUE per-page id (re-runs overwrite the same page, never duplicate).

    Prefer the user-guide path tail from the URL (scraped pages have a canonical URL,
    e.g. .../user-guide/897693/introduction -> '897693-introduction'). Offline HTML
    exports have NO canonical URL, so ``fallback`` MUST be the file's path RELATIVE to
    the ingest root (e.g. '0800_ASM/0000_EASM'), not just the bare filename — otherwise
    same-named pages in different module folders ('0000_Dashboard' under EASM vs IASM)
    collide onto ONE doc_id and overwrite each other. The full relative path keeps
    each page unique AND stable across re-runs (files don't move)."""
    if url:
        path = urlparse(url).path
        m = re.search(r"/user-guide/(.+?)/?$", path)
        tail = m.group(1) if m else path.strip("/").rsplit("/", 2)[-1]
        slug = re.sub(r"[^a-zA-Z0-9]+", "-", tail).strip("-").lower()
        if slug:
            return slug
    return re.sub(r"[^a-zA-Z0-9]+", "-", fallback).strip("-").lower() or "page"


def _breadcrumb_from_path(rel_id: str, title: str) -> list[str]:
    """Nav hierarchy from the file's folder path when the HTML carries none.

    Offline exports have no ToC/URL, so the module structure lives only in the
    directory layout ('0800_ASM/0000_EASM.htm'). Turn the folders into a breadcrumb
    ('ASM > EASM'), stripping the leading sort-order prefixes ('0800_', '0000_') and
    underscores, so navigation answers and hierarchical retrieval still work.
    """
    parts = [p for p in rel_id.replace("\\", "/").split("/") if p]
    labels: list[str] = []
    for folder in parts[:-1]:  # folders only; the leaf file is represented by `title`
        lbl = re.sub(r"^[\d]+[_\-\s]*", "", folder).replace("_", " ").strip()
        if lbl:
            labels.append(lbl)
    labels.append(title)
    return labels or [title]


def build_points(page: dict, fallback_id: str) -> list[models.PointStruct]:
    title = page["title"]
    url = page.get("url", "")
    doc_id = _doc_id(url, fallback_id)  # identity: unchanged (empty url -> folder id)
    # User-facing citation link. Offline pages have no canonical <link>, so synthesize
    # the live docs URL from the page's own <meta data-pageid> + title slug. This is
    # citation-only — it deliberately does NOT feed doc_id, so re-ingest stays an
    # in-place update with no id churn.
    citation_url = url or _canonical_url_from_pageid(page.get("pageid", ""), title)
    section_path = page.get("section_path") or [title]
    # No real hierarchy from the page (offline export: no ToC/URL) -> rebuild it from
    # the file's folder path so breadcrumbs and "where do I go" answers still work.
    # Gated on `not url`: only offline files have a path fallback_id; a crawled page's
    # fallback_id is its URL (which must NOT be split into a breadcrumb).
    if len(section_path) <= 1 and not url and ("/" in fallback_id or "\\" in fallback_id):
        section_path = _breadcrumb_from_path(fallback_id, title)
    breadcrumb = " > ".join(section_path)

    # Flatten sections into (heading, chunk) pairs — pack whole blocks per section
    # so a chunk never splits a sentence or a table row's "Name — description".
    chunk_meta: list[tuple[str, str]] = []
    for heading, blocks in page["sections"]:
        for piece in pack_lines(blocks):
            chunk_meta.append((heading, piece))
    if not chunk_meta:
        return []

    # Embed with the full section BREADCRUMB + in-page heading as an asymmetric
    # prefix. This makes hierarchical queries ("view discovered assets in the EASM
    # dashboard") match the right chunk, and encodes where the content lives so the
    # agent can give navigation directions. The in-page heading is only added when
    # it differs from the breadcrumb leaf (avoids "…Dashboard\nEASM Dashboard").
    def _prefix(heading: str) -> str:
        return breadcrumb if heading == section_path[-1] else f"{breadcrumb} > {heading}"

    embed_inputs = [f"{_prefix(heading)}\n{body}" for heading, body in chunk_meta]
    vectors = embed_documents(embed_inputs)

    total = len(chunk_meta)
    points = []
    for idx, ((heading, body), vector) in enumerate(zip(chunk_meta, vectors, strict=True)):
        rid = f"{doc_id}_chunk_{idx}"
        points.append(
            models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_DNS, rid)),
                vector=vector,
                payload={
                    "rid": rid,
                    "text": body,
                    "doc_id": doc_id,
                    "source": SOURCE,
                    "title": title,
                    "heading": heading,
                    "url": citation_url,
                    "section": f"chunk {idx}",
                    # section hierarchy from the nav tree — user-facing nav path +
                    # filterable ancestry (e.g. show me EASM pages).
                    "breadcrumb": breadcrumb,
                    "section_path": section_path,
                    "published_at": None,  # docs are not time-bound
                    # shared-visibility contract read by the retriever's access filter
                    "customer_tags": [],
                    "public": True,
                    "is_deleted": False,
                    "metadata": {
                        "chunk_index": idx,
                        "total_chunks": total,
                        "heading": heading,
                        "url": citation_url,
                        "breadcrumb": breadcrumb,
                        "section_path": section_path,
                        "product": GUIDE_PRODUCT,
                        "guide_version": GUIDE_VERSION,
                    },
                },
            )
        )
    return points


def _existing_point_ids(doc_id: str) -> set[str]:
    """Every point id currently stored for this page (paginated scroll, ids only)."""
    ids: set[str] = set()
    offset = None
    flt = models.Filter(
        must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]
    )
    while True:
        pts, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=flt,
            with_payload=False,
            with_vectors=False,
            limit=256,
            offset=offset,
        )
        ids.update(str(p.id) for p in pts)
        if offset is None:
            break
    return ids


def index_page(page: dict, fallback_id: str) -> int:
    doc_id = _doc_id(page.get("url", ""), fallback_id)

    # Hard-purge override: wipe the page entirely before rebuilding (use for a schema
    # change or a full clean reindex). Normal updates don't need it — the prune below
    # keeps re-runs idempotent on their own.
    if DELETE_FIRST:
        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]
                )
            ),
        )

    try:
        points = build_points(page, fallback_id)
    except Exception as e:  # noqa: BLE001 — skip the page on embed failure
        print(f"  ! skip page {doc_id}: {e}")
        return 0
    if not points:
        return 0

    # Idempotent update, no duplicates, no stale orphans:
    # 1) UPSERT the current chunks — point ids are uuid5("{doc_id}_chunk_{idx}"), so an
    #    existing chunk is overwritten in place (never a second copy), a genuinely new
    #    chunk/page is created.
    # 2) PRUNE any chunk left over from a PREVIOUS, LONGER version of this page (idx >=
    #    the new chunk count). Without this, a shrunk page keeps stale chunks that would
    #    surface outdated answers. Upsert BEFORE prune -> no moment where the page is empty;
    #    diff on ids -> only true orphans are deleted.
    new_ids = {str(p.id) for p in points}
    client.upsert(collection_name=COLLECTION_NAME, points=points)

    if not DELETE_FIRST:
        stale = _existing_point_ids(doc_id) - new_ids
        if stale:
            client.delete(
                collection_name=COLLECTION_NAME,
                points_selector=models.PointIdsList(points=list(stale)),
            )
            print(f"  pruned {len(stale)} stale chunk(s) from a prior version of [{doc_id}]")

    print(f"  upserted {len(points)} chunks for '{page['title']}' [{doc_id}]")
    return len(points)


def sync_removed(seen_doc_ids: set[str]) -> int:
    """Delete pages that vanished from the source since the last run (removed/renamed).

    Turns a full --html-dir ingest into a true SYNC: after ingesting the whole export,
    any page still in Qdrant whose doc_id we did NOT just ingest is stale (the source
    file was deleted, or renamed/moved -> new doc_id) and is removed, so restructured
    docs never leave orphaned pages that surface as outdated answers. Scoped to THIS
    guide's ``source``.

    Guarded: never prunes on an empty run (a failed ingest must not wipe the corpus),
    and ``seen`` tracks every ATTEMPTED page (even one whose embed failed this run), so
    a transient failure never deletes a still-present page.
    """
    if not seen_doc_ids:
        print("Sync: skipped — 0 pages ingested this run (refusing to prune).")
        return 0

    source_match = models.FieldCondition(key="source", match=models.MatchValue(value=SOURCE))
    existing: set[str] = set()
    offset = None
    while True:
        pts, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=models.Filter(must=[source_match]),
            with_payload=["doc_id"],
            with_vectors=False,
            limit=512,
            offset=offset,
        )
        for p in pts:
            d = (p.payload or {}).get("doc_id")
            if d:
                existing.add(d)
        if offset is None:
            break

    stale = existing - seen_doc_ids
    if not stale:
        print(f"Sync: collection matches source ({len(seen_doc_ids)} pages) — nothing to remove.")
        return 0

    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    source_match,
                    models.FieldCondition(key="doc_id", match=models.MatchAny(any=list(stale))),
                ]
            )
        ),
    )
    sample = sorted(stale)[:8]
    print(f"Sync: removed {len(stale)} page(s) no longer in the source: {sample}"
          f"{' …' if len(stale) > 8 else ''}")
    return len(stale)


# ---------------------------------------------------------------------------
# 7. Sources — local HTML directory (primary) or best-effort URL crawl.
# ---------------------------------------------------------------------------
def iter_local(html_dir: str):
    root = Path(html_dir)
    files = sorted([*root.rglob("*.html"), *root.rglob("*.htm")])
    if not files:
        sys.exit(f"No .html files under '{html_dir}'.")
    print(f"Found {len(files)} HTML file(s) under {html_dir}")
    for f in files:
        try:
            html = f.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:  # noqa: BLE001
            print(f"  ! read failed {f}: {e}")
            continue
        page = parse_html(html)
        if page:
            # Pass the path RELATIVE to the ingest root (not just f.stem) as the id
            # fallback — this is what makes doc_id unique for same-named pages in
            # different folders, and feeds the folder-derived breadcrumb.
            yield page, str(f.relative_to(root).with_suffix(""))


def _same_guide(url: str, root_url: str) -> bool:
    a, b = urlparse(url), urlparse(root_url)
    if a.netloc != b.netloc:
        return False
    # Stay within the same document's user-guide tree.
    prefix = b.path.split("/user-guide/")[0] + "/user-guide/"
    return a.path.startswith(prefix)


def iter_crawl(root_url: str, max_pages: int):
    seen: set[str] = set()
    queue: list[str] = [urldefrag(root_url)[0]]
    yielded = 0
    while queue and yielded < max_pages:
        url = queue.pop(0)
        if url in seen:
            continue
        seen.add(url)
        try:
            resp = _http.get(url)
            resp.raise_for_status()
        except Exception as e:  # noqa: BLE001
            print(f"  ! fetch failed {url}: {e}")
            continue

        soup = BeautifulSoup(resp.text, "html.parser")
        for a in soup.find_all("a", href=True):
            nxt = urldefrag(urljoin(url, a["href"]))[0]
            if nxt not in seen and _same_guide(nxt, root_url):
                queue.append(nxt)

        page = parse_html(resp.text, fallback_url=url)
        if page:
            if not page.get("url"):
                page["url"] = url
            yielded += 1
            yield page, url
        time.sleep(CRAWL_DELAY)

    if yielded <= 1:
        print(
            "WARNING: crawl yielded <=1 page. docs.fortinet.com renders its table of "
            "contents with JavaScript, so a static crawl often can't discover sibling "
            "pages. Prefer downloading the pages and using --html-dir."
        )


# ---------------------------------------------------------------------------
# 8. Entry point.
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest the FortiRecon user guide into Qdrant.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--html-dir", help="Directory of downloaded .html pages (recursive).")
    src.add_argument("--url", help="Root user-guide URL to crawl from.")
    ap.add_argument("--crawl", action="store_true", help="With --url: follow same-guide links (BFS).")
    ap.add_argument("--max-pages", type=int, default=500, help="Crawl page cap (default 500).")
    args = ap.parse_args()

    print(
        f"Indexing FortiRecon user guide -> {QDRANT_URL} / '{COLLECTION_NAME}' "
        f"via TEI {TEI_EMBED_URL} (dim {EMBEDDING_DIM})"
    )
    ensure_collection()

    if args.html_dir:
        source = iter_local(args.html_dir)
    elif args.crawl:
        source = iter_crawl(args.url, args.max_pages)
    else:
        resp = _http.get(args.url)
        resp.raise_for_status()
        page = parse_html(resp.text, fallback_url=args.url)
        source = [(page, args.url)] if page else []

    pages = chunks = 0
    seen: set[str] = set()
    for page, fallback_id in source:
        pages += 1
        # Track every ATTEMPTED page (before embedding) so a transient embed failure
        # doesn't make sync_removed treat a still-present page as deleted.
        seen.add(_doc_id(page.get("url", ""), fallback_id))
        chunks += index_page(page, fallback_id)
    print(f"Done. {pages} page(s) -> {chunks} chunks in '{COLLECTION_NAME}'.")

    # Full-sync: on a complete --html-dir snapshot, remove pages that no longer exist
    # in the source (deleted/renamed) so the collection stays a mirror of the docs.
    if args.html_dir and PRUNE_REMOVED:
        sync_removed(seen)
    elif not args.html_dir and PRUNE_REMOVED:
        print("Sync: skipped for crawl/url mode (partial snapshot — set --html-dir for full sync).")


if __name__ == "__main__":
    main()
