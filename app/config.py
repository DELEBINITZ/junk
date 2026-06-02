"""Central configuration via pydantic-settings — the single source of truth that
the whole app is assembled from.

MENTAL MODEL: this one Settings object is the "control panel" for the platform.
bootstrap.py reads it and decides which concrete backend to wire for every seam
(LLM, embeddings, vector store, chat store, agent engine, tracing). That pattern
is "config-gating": the *same* code path serves either a deterministic, self-
hosted default or a real production backend, chosen purely by a value here. No
code change is needed to switch — only an env var.

WHY DETERMINISTIC DEFAULTS MATTER (read this twice): every external dependency
defaults to a zero-infra local path, so the system boots and serves with no GPU,
no API keys, and no network. Two payoffs:
  * onboarding / tests / CI run anywhere, instantly and reproducibly; and
  * SECURITY — on the defaults, no contract or security-intelligence text is ever
    sent to an external LLM/embedding API. Sensitive data only leaves this process
    if an operator EXPLICITLY flips a provider to a hosted service and supplies
    creds. Self-hosted-by-default is a deliberate data-governance posture.

  * ``llm_provider=deterministic``      -> no LLM server needed
  * ``embedding_provider=deterministic``-> hash embeddings, no TEI
  * ``retrieval_backend=memory``        -> in-process vector store, no Qdrant
  * ``store_backend=memory``            -> in-process chat/session store, no PG
  * ``agent_engine=internal``           -> built-in graph engine, no langgraph dep

Flip the corresponding env var + provide creds to use the real production
backends (SGLang, TEI, Qdrant, Postgres+RLS, Redis, Langfuse, LangGraph).

HOW pydantic-settings WORKS: each field below is read from the environment (or a
``.env`` file) by its UPPER_CASE name; if absent, the default here is used and is
fully type-validated. That is why every value has a safe default — the app is
always runnable, and overriding is just setting an env var.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ``model_config`` tells pydantic-settings HOW to load values: read a ``.env``
    # file, take env vars with no prefix, ignore unknown vars (so an env shared
    # with other tools won't crash boot), and match names case-insensitively.
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="",
        extra="ignore",
        case_sensitive=False,
    )

    # ---- App ---------------------------------------------------------------
    # Process identity and HTTP-server basics. ``log_json`` flips logging from
    # human-readable (dev) to structured JSON (prod) — see observability/logging.py.
    app_name: str = "agentic-security-intelligence"
    environment: Literal["dev", "staging", "prod"] = "dev"
    debug: bool = True
    log_level: str = "INFO"
    log_json: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    # ---- Auth --------------------------------------------------------------
    # Identity. ``local`` mints/verifies its own JWTs (great for dev); ``oidc``
    # defers to a real identity provider and reads org/roles from token claims.
    # The org claim is critical: it is the verified tenant key that flows into
    # every request's ToolContext and powers tenant isolation. The dev jwt_secret
    # is intentionally insecure and MUST be overridden outside dev.
    auth_provider: Literal["local", "oidc"] = "local"
    jwt_secret: str = "dev-insecure-change-me-please-32byte-minimum-secret"  # noqa: S105 - dev only
    jwt_algorithm: str = "HS256"
    access_token_ttl_seconds: int = 3600
    refresh_token_ttl_seconds: int = 1209600  # 14 days
    # OIDC (auth_provider=oidc): where to validate tokens and which claims carry
    # the tenant id and the caller's roles (the inputs to multi-tenant RBAC).
    oidc_issuer: str = ""
    oidc_audience: str = ""
    oidc_jwks_url: str = ""
    oidc_org_claim: str = "org_id"
    oidc_roles_claim: str = "roles"

    # ---- Stores ------------------------------------------------------------
    # Where conversations/sessions persist. ``memory`` is in-process (zero infra);
    # ``postgres`` uses real Postgres with ROW-LEVEL SECURITY. ``rls_setting_name``
    # is the Postgres session variable set to the org id inside each transaction,
    # so the database itself enforces "an org can only read its own rows" — a
    # second, defense-in-depth layer of tenant isolation under the app checks.
    store_backend: Literal["memory", "postgres"] = "memory"
    database_url: str = ""  # postgresql://user:pass@host:5432/db
    rls_setting_name: str = "app.organization_id"
    db_pool_min: int = 1
    db_pool_max: int = 10

    # Optional cache + semantic (embedding-similarity) response cache. Off/in-
    # memory by default; Redis is the shared backend for multi-replica deploys.
    cache_backend: Literal["memory", "redis"] = "memory"
    redis_url: str = ""
    semantic_cache_enabled: bool = False
    semantic_cache_threshold: float = 0.95

    # ---- LLM ---------------------------------------------------------------
    # The model backend. ``deterministic`` is a stub that returns fixed, grounded
    # output with NO network call — the security-relevant default (no prompt or
    # retrieved text leaves the box). ``sglang`` is the self-hosted production
    # path; ``openai`` is a hosted dev convenience and, by design, never for
    # sensitive prod data. Three model "lanes" (fast/standard/deep) let cheap
    # steps (routing, tool planning) use a small model and the answer use a bigger
    # one — see llm/lanes.py.
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
    llm_temperature: float = 0.1   # low temperature => more deterministic, factual answers
    llm_max_tokens: int = 1024
    llm_timeout_seconds: float = 60.0

    # ---- Embeddings --------------------------------------------------------
    # Vectorizer for RAG. ``deterministic`` hashes text into a vector locally (no
    # model, no network — keeps the default offline and reproducible); ``tei`` is
    # the self-hosted embedding server. ``embedding_dim`` MUST match whatever the
    # vector store was created with.
    embedding_provider: Literal["deterministic", "tei", "openai"] = "deterministic"
    tei_embed_url: str = "http://localhost:8080"
    embedding_dim: int = 1024
    embedding_model: str = "Qwen/Qwen3-Embedding-8B"

    # ---- Retrieval ---------------------------------------------------------
    # The vector store + RAG knobs. ``memory`` is an in-process store (zero infra);
    # ``qdrant`` is the production vector DB. ``top_k`` = how many candidates to
    # fetch; reranking (a cross-encoder, optional) reorders them for precision and
    # keeps the best ``rerank_top_k``. ``recency_half_life_days`` lets fresher docs
    # outrank stale ones — important for security intel that ages fast.
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
    # The safety spine, applied to BOTH the incoming question and the outgoing
    # answer. Each toggle gates one check: redact PII, detect prompt injection,
    # screen unsafe topics, and verify the answer is grounded in retrieved context
    # (anti-hallucination). The optional *_url endpoints are ML backstops; a
    # heuristic implementation is always on so guardrails never silently disable.
    guardrails_enabled: bool = True
    pii_redaction: bool = True
    injection_detection: bool = True
    topic_safety: bool = True
    groundedness_check: bool = True
    prompt_guard_url: str = ""   # optional model endpoint (heuristic backstop always on)
    llama_guard_url: str = ""

    # ---- Memory / knowledge graph -----------------------------------------
    # Long-term/cross-session memory backend. ``none`` => a NoOp graph (default);
    # ``zep`` wires a real temporal knowledge-graph service.
    kg_provider: Literal["none", "zep"] = "none"
    zep_api_url: str = ""
    zep_api_key: str = ""

    # ---- Agent -------------------------------------------------------------
    # How the agent runs a turn. ``agent_engine`` chooses the built-in graph engine
    # (zero deps) vs real LangGraph (adds durable checkpointing) — same node set
    # either way (see agent/graph.py). ``router_mode`` chooses the supervisor's
    # routing strategy: ``heuristic`` (no LLM) vs ``llm`` tool-calling. The window/
    # trigger settings bound how much chat history is kept verbatim before it gets
    # rolled into a running summary.
    agent_engine: Literal["internal", "langgraph"] = "internal"
    max_tool_iterations: int = 4
    history_window_messages: int = 12
    summary_trigger_messages: int = 20
    router_mode: Literal["heuristic", "llm"] = "heuristic"

    # ---- Observability -----------------------------------------------------
    # Tracing + metrics. ``tracing_provider=none`` uses a NoOp tracer (default, hot
    # path stays dependency-free); ``langfuse`` records full request traces. These
    # map directly to observability/tracing.py and metrics.py.
    tracing_provider: Literal["none", "langfuse"] = "none"
    langfuse_host: str = "http://localhost:3000"
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    metrics_enabled: bool = True

    # ---- Capabilities (deployment composition flags) -----------------------
    # WHICH capability modules are live in this deployment. The registry discovers
    # every module on disk but only ENABLES the ones whose flag here is true (a
    # manifest names its flag via ``enabled_flag``). This is how the same codebase
    # ships different product bundles per customer without any code change.
    cap_reports_enabled: bool = True
    cap_easm_enabled: bool = True
    cap_brand_enabled: bool = False
    cap_aci_enabled: bool = False
    # Seed each enabled module's demo corpus at boot (dev convenience; off in prod).
    seed_demo_data: bool = True

    # ---- Concurrency / fairness -------------------------------------------
    # Back-pressure knobs. A global cap plus a per-org cap give fairness (one noisy
    # tenant can't starve others); past the limits requests queue, then 503 if the
    # queue is full. Enforced by the ConcurrencyMiddleware (see main.py).
    max_concurrent_generations: int = 32
    per_org_concurrency: int = 8
    request_timeout_seconds: float = 120.0
    queue_max_size: int = 256

    # ---- Validators --------------------------------------------------------
    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_csv(cls, v: object) -> object:
        """Accept CORS origins as either a comma-separated string (the natural way
        to set a list in an env var, e.g. ``a.com,b.com``) or a JSON array, and
        normalize both to a Python list. ``mode="before"`` runs this BEFORE
        pydantic's own type coercion, so we get the raw env string to split."""
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
    """The app-wide accessor for config. ``lru_cache`` makes it a singleton: the
    env is read and validated ONCE, then the same Settings object is reused
    everywhere (bootstrap, API deps, tools), so config is consistent within a run.
    """
    return Settings()


def reload_settings() -> Settings:
    """Clear the cached singleton and rebuild it. Needed by tests and the eval
    runner, which mutate environment variables and then want a fresh Settings that
    reflects them (otherwise the lru_cache would keep handing back the old one)."""
    get_settings.cache_clear()
    return get_settings()


__all__ = ["Settings", "get_settings", "reload_settings"]
