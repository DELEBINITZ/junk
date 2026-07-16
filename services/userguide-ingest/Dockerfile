# Standalone, independently-deployable ingestion service.
# Build:  docker build -t fortirecon-userguide-ingest services/userguide-ingest
# Run (crawl):
#   docker run --rm --env-file services/userguide-ingest/.env fortirecon-userguide-ingest \
#     --url https://docs.fortinet.com/document/fortirecon/26.2.a/user-guide/897693/introduction --crawl
# Run (local HTML, mounted):
#   docker run --rm --env-file services/userguide-ingest/.env \
#     -v /path/to/html:/data fortirecon-userguide-ingest --html-dir /data
FROM python:3.12-slim

WORKDIR /app

# Install deps first for layer caching.
COPY pyproject.toml ./
RUN pip install --no-cache-dir "httpx>=0.27" "beautifulsoup4>=4.12" "qdrant-client>=1.12"

COPY index_user_guide.py ./

# It's a one-shot job: pass source args (--html-dir / --url [--crawl]) to `docker run`.
ENTRYPOINT ["python", "index_user_guide.py"]
