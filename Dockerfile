FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_TRUSTED_HOST="pypi.org files.pythonhosted.org pypi.python.org"

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY app ./app
COPY Assignment_org ./Assignment_org

RUN python -m pip install --no-cache-dir \
    "fastapi>=0.110" \
    "uvicorn>=0.30" \
    "pydantic>=2" \
    "PyJWT>=2.8" \
    "httpx>=0.27"

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --start-period=10s --retries=12 \
  CMD curl -fsS http://127.0.0.1:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
