"""Environment-driven settings for the backend."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """Small immutable settings object.

    Defaults keep the PoC runnable locally. Production deployments should supply
    a real JWT secret and can switch embedding/LLM providers through env vars.
    """

    app_name: str = os.getenv("APP_NAME", "Enterprise Contract Intelligence PoC")
    jwt_secret: str = os.getenv("JWT_SECRET", "demo-local-secret-change-before-prod")
    jwt_issuer: str = os.getenv("JWT_ISSUER", "contract-intelligence-poc")
    access_token_minutes: int = int(os.getenv("ACCESS_TOKEN_MINUTES", "120"))
    corpus_dir: Path = Path(os.getenv("CONTRACT_CORPUS_DIR", "Assignment_org"))
    embedding_dimensions: int = int(os.getenv("EMBEDDING_DIMENSIONS", "384"))
    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "hash")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    llm_provider: str = os.getenv("LLM_PROVIDER", "deterministic")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")
    vllm_base_url: str = os.getenv("VLLM_BASE_URL", "http://localhost:8001/v1")
    vllm_model: str = os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_format: str = os.getenv("LOG_FORMAT", "json")
    # Durable storage. Default "memory" keeps local dev/tests dependency-free;
    # "postgres" routes chat persistence through the RLS-backed Postgres layer.
    store_backend: str = os.getenv("STORE_BACKEND", "memory")
    database_url: str = os.getenv(
        "DATABASE_URL",
        "postgresql://contract_user:contract_pass@localhost:5432/contract_intelligence",
    )
    rls_setting_name: str = os.getenv("RLS_SETTING_NAME", "app.organization_id")
    # Inference lanes (self-hosted SGLang, OpenAI-compatible). Used when
    # LLM_PROVIDER=sglang; the deterministic default serves every lane locally.
    sglang_base_url: str = os.getenv("SGLANG_BASE_URL", "http://localhost:30000/v1")
    sglang_model_standard: str = os.getenv("SGLANG_MODEL_STANDARD", "Qwen/Qwen2.5-72B-Instruct")
    sglang_model_fast: str = os.getenv("SGLANG_MODEL_FAST", "Qwen/Qwen2.5-7B-Instruct")
    sglang_model_deep: str = os.getenv("SGLANG_MODEL_DEEP", "deepseek-ai/DeepSeek-V3.1")
    # Retrieval backend + vector store + embeddings service.
    retrieval_backend: str = os.getenv("RETRIEVAL_BACKEND", "memory")
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "reports_kb")
    tei_base_url: str = os.getenv("TEI_BASE_URL", "http://localhost:8080")
    tei_model: str = os.getenv("TEI_MODEL", "Qwen/Qwen3-Embedding-8B")
    # Guardrails: topic/safety classifier provider ("off" | "llama_guard").
    topic_safety_provider: str = os.getenv("TOPIC_SAFETY_PROVIDER", "off")
    # Router: "keyword" (default, deterministic) | "llm" (fast-lane LLM planner).
    router_mode: str = os.getenv("ROUTER_MODE", "keyword")
    # Observability tracer: "none" (default) | "langfuse".
    observability_provider: str = os.getenv("OBSERVABILITY_PROVIDER", "none")
    # Reranker (cross-encoder over candidates, biggest retrieval-quality win).
    rerank_enabled: bool = os.getenv("RERANK_ENABLED", "false").lower() == "true"
    tei_rerank_base_url: str = os.getenv("TEI_RERANK_BASE_URL", "http://localhost:8081")
    tei_rerank_model: str = os.getenv("TEI_RERANK_MODEL", "Qwen/Qwen3-Reranker-4B")
    # EASM product backend (empty -> tools return errors-as-data).
    easm_api_url: str = os.getenv("EASM_API_URL", "")
    easm_api_token: str = os.getenv("EASM_API_TOKEN", "")
    # Auth: "local" (HS256 demo) | "oidc" (verify via IdP JWKS).
    auth_provider: str = os.getenv("AUTH_PROVIDER", "local")
    oidc_issuer: str = os.getenv("OIDC_ISSUER", "")
    oidc_jwks_url: str = os.getenv("OIDC_JWKS_URL", "")
    oidc_audience: str = os.getenv("OIDC_AUDIENCE", "")
    refresh_token_minutes: int = int(os.getenv("REFRESH_TOKEN_MINUTES", "43200"))  # 30 days
    # Guardrail model endpoints (empty/heuristic -> built-in heuristics).
    injection_provider: str = os.getenv("INJECTION_PROVIDER", "heuristic")  # heuristic | prompt_guard
    prompt_guard_url: str = os.getenv("PROMPT_GUARD_URL", "")
    llama_guard_url: str = os.getenv("LLAMA_GUARD_URL", "")
    # Agent engine: "builtin" (our graph runner) | "langgraph" (real lib + checkpointer).
    agent_engine: str = os.getenv("AGENT_ENGINE", "builtin")
    # Knowledge graph (long-term entity memory): "none" | "zep".
    kg_provider: str = os.getenv("KG_PROVIDER", "none")
    zep_api_url: str = os.getenv("ZEP_API_URL", "")
    zep_api_key: str = os.getenv("ZEP_API_KEY", "")
    # Temporal (durable ingestion workflows).
    temporal_host: str = os.getenv("TEMPORAL_HOST", "localhost:7233")
    temporal_task_queue: str = os.getenv("TEMPORAL_TASK_QUEUE", "ingestion")
    # CORS: comma-separated allowed origins for the browser UI; empty disables.
    cors_origins: str = os.getenv("CORS_ORIGINS", "")


settings = Settings()
