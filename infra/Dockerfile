# Production image for the agentic platform (the chat service).
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1

WORKDIR /srv

# System deps for psycopg + builds.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libpq5 curl && rm -rf /var/lib/apt/lists/*

# Install dependencies first (better layer caching).
COPY pyproject.toml README.md ./
COPY app ./app
RUN pip install --upgrade pip && pip install ".[prod]"

# Non-root.
RUN useradd -m -u 10001 appuser && chown -R appuser /srv
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=4s --retries=5 \
    CMD curl -fsS http://localhost:8000/healthz || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
