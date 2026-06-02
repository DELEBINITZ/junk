"""Central configuration via pydantic-settings.

Every external dependency is *config-gated* and defaults to a zero-infra local
path so the system boots and serves with no GPU, keys, or network:

  * ``llm_provider=deterministic``      -> no LLM server needed
  * ``embedding_provider=deterministic``-> hash embeddings, no TEI
  * ``retrieval_backend=memory``        -> in-process vector store, no Qdrant
  * ``store_backend=memory``            -> in-process chat/session store, no PG
  * ``agent_engine=internal``           -> built-in graph engine, no langgraph dep

Flip the corresponding env var + provide creds to use the real production
backends (SGLang, TEI, Qdrant, Postgres+RLS, Redis, Langfuse, LangGraph).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="",
        extra="ignore",
        case_sensitive=False,
    )

    # ---- App ---------------------------------------------------------------
    app_name: str = "agentic-security-intelligence"
    environment: Literal["dev", "staging", "prod"] = "dev"
    debug: bool = True
    log_level: str = "INFO"
    log_json: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    # ---- Auth --------------------------------------------------------------
    auth_provider: Literal["local", "oidc"] = "local"
    jwt_secret: str = "dev-insecure-change-me-please-32byte-minimum-secret"  # noqa: S105 - dev only
    jwt_algorithm: str = "HS256"
    access_token_ttl_seconds: int = 3600
    refresh_token_ttl_seconds: int = 1209600  # 14 days
    # OIDC (auth_provider=oidc)
    oidc_issuer: str = ""
    oidc_audience: str = ""
    oidc_jwks_url: str = ""
    oidc_org_claim: str = "org_id"
    oidc_roles_claim: str = "roles"

    # ---- Stores ------------------------------------------------------------
    store_backend: Literal["memory", "postgres"] = "memory"
    database_url: str = ""  # postgresql://user:pass@host:5432/db
    rls_setting_name: str = "app.organization_id"
    db_pool_min: int = 1
    db_pool_max: int = 10

    cache_backend: Literal["memory", "redis"] = "memory"
    redis_url: str = ""
    semantic_cache_enabled: bool = False
    semantic_cache_threshold: float = 0.95

    # ---- LLM ---------------------------------------------------------------
    llm_provider: Literal["deterministic", "sglang", "openai"] = "deterministic"
    # SGLang (OpenAI-compatible) — three lanes
    sglang_base_url: str = "http://localhost:30000/v1"
    sglang_api_key: str = "EMPTY"
    model_fast: str = "Qwen/Qwen2.5-7B-Instruct"
    model_standard: str = "Qwen/Qwen2.5-72B-Instruct"
    model_deep: str = "deepseek-ai/DeepSeek-V3.1"
    # OpenAI-compatible dev provider (off by default; never for sensitive prod data)
    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1024
    llm_timeout_seconds: float = 60.0

    # ---- Embeddings --------------------------------------------------------
    embedding_provider: Literal["deterministic", "tei", "openai"] = "deterministic"
    tei_embed_url: str = "http://localhost:8080"
    embedding_dim: int = 1024
    embedding_model: str = "Qwen/Qwen3-Embedding-8B"

    # ---- Retrieval ---------------------------------------------------------
    retrieval_backend: Literal["memory", "qdrant"] = "memory"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    retrieval_top_k: int = 20
    rerank_enabled: bool = False
    rerank_provider: Literal["none", "tei"] = "none"
    tei_rerank_url: str = "http://localhost:8081"
    rerank_top_k: int = 6
    recency_half_life_days: float = 180.0

    # ---- Guardrails --------------------------------------------------------
    guardrails_enabled: bool = True
    pii_redaction: bool = True
    injection_detection: bool = True
    topic_safety: bool = True
    groundedness_check: bool = True
    prompt_guard_url: str = ""   # optional model endpoint (heuristic backstop always on)
    llama_guard_url: str = ""

    # ---- Memory / knowledge graph -----------------------------------------
    kg_provider: Literal["none", "zep"] = "none"
    zep_api_url: str = ""
    zep_api_key: str = ""

    # ---- Agent -------------------------------------------------------------
    agent_engine: Literal["internal", "langgraph"] = "internal"
    max_tool_iterations: int = 4
    history_window_messages: int = 12
    summary_trigger_messages: int = 20
    router_mode: Literal["heuristic", "llm"] = "heuristic"

    # ---- Observability -----------------------------------------------------
    tracing_provider: Literal["none", "langfuse"] = "none"
    langfuse_host: str = "http://localhost:3000"
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    metrics_enabled: bool = True

    # ---- Capabilities (deployment composition flags) -----------------------
    cap_reports_enabled: bool = True
    cap_easm_enabled: bool = True
    cap_brand_enabled: bool = False
    cap_aci_enabled: bool = False
    # Seed each enabled module's demo corpus at boot (dev convenience; off in prod).
    seed_demo_data: bool = True

    # ---- Concurrency / fairness -------------------------------------------
    max_concurrent_generations: int = 32
    per_org_concurrency: int = 8
    request_timeout_seconds: float = 120.0
    queue_max_size: int = 256

    # ---- Validators --------------------------------------------------------
    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_csv(cls, v: object) -> object:
        if isinstance(v, str):
            s = v.strip()
            if not s:
                return ["*"]
            if s.startswith("["):
                return v  # JSON list — let pydantic parse
            return [item.strip() for item in s.split(",") if item.strip()]
        return v

    @property
    def is_prod(self) -> bool:
        return self.environment == "prod"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reload_settings() -> Settings:
    """Clear the cache (tests / hot config changes)."""
    get_settings.cache_clear()
    return get_settings()


__all__ = ["Settings", "get_settings", "reload_settings"]
