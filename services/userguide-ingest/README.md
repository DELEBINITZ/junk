# FortiRecon User Guide Ingestion Service

Standalone, independently-deployable service that parses the FortiRecon product
user guide (HTML), embeds it, and upserts it into a **dedicated Qdrant collection**
(`user_guide_kb`) that the platform's **User Guide agent** retrieves from.

It is intentionally decoupled from the main `security_intel` app: no shared code,
its own `pyproject.toml` and `Dockerfile`. The only contract with the platform is
via environment variables:

| Contract | Env var | Must match app setting |
|---|---|---|
| Collection name | `USER_GUIDE_COLLECTION` | `settings.user_guide_collection` |
| Embedding server | `TEI_EMBED_URL` | `settings.embedding_base_url` |
| Embedding dim | `EMBEDDING_DIM` | `settings.embedding_dim` (2560) |

Get these wrong and retrieval silently returns nothing (wrong collection) or Qdrant
rejects the upsert (wrong dim). Keep them aligned.

## What it produces

One Qdrant point per chunk, schema matching the retriever's access filter:

- `text`, `title`, `heading`, `url`, `doc_id`, `section`
- `is_deleted=false`, `public=true`, `customer_tags=[]` (docs are shared/public)
- `metadata`: `{chunk_index, total_chunks, heading, url, product, guide_version}`

Point ids are deterministic (`uuid5("<doc_id>_chunk_<idx>")`), so re-runs overwrite
instead of duplicating. Set `REINDEX_DELETE_FIRST=1` to also purge a page's stale
chunks (when a re-indexed page has fewer chunks than before).

## Run locally

```bash
cd services/userguide-ingest
cp .env.example .env            # edit QDRANT_URL / TEI_EMBED_URL

# Option A — HTML you downloaded (recommended; docs.fortinet.com TOC is JS-rendered)
uv run --with beautifulsoup4 --with qdrant-client --with httpx \
  python index_user_guide.py --html-dir /path/to/downloaded/html

# Option B — best-effort crawl from the live docs
uv run --with beautifulsoup4 --with qdrant-client --with httpx \
  python index_user_guide.py \
  --url https://docs.fortinet.com/document/fortirecon/26.2.a/user-guide/897693/introduction \
  --crawl --max-pages 400
```

> Downloading the pages and using `--html-dir` is the reliable path: the Fortinet
> docs site renders its table of contents with JavaScript, so a static crawl often
> can't discover sibling pages.

## Deploy (one-shot job)

```bash
docker build -t fortirecon-userguide-ingest services/userguide-ingest

# Crawl
docker run --rm --env-file services/userguide-ingest/.env \
  fortirecon-userguide-ingest \
  --url https://docs.fortinet.com/document/fortirecon/26.2.a/user-guide/897693/introduction --crawl

# Local HTML mounted into the container
docker run --rm --env-file services/userguide-ingest/.env \
  -v /path/to/html:/data fortirecon-userguide-ingest --html-dir /data
```

Run it as a Kubernetes `Job` / CronJob or an ECS scheduled task — it exits when done.
After a successful run, the platform auto-registers the User Guide agent on next
startup (it checks the collection is non-empty before exposing the agent).
