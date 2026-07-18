from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # App
    environment: str = "dev"
    debug: bool = True
    log_level: str = "INFO"
    host: str = "0.0.0.0"
    port: int = 8000

    # Assistant identity (personality is otherwise DERIVED from the enabled agents).
    # Leave blank to auto-derive a name/tagline from whichever agents are active —
    # so an atlas-only deployment presents as a product guide, a sentinel+aura
    # deployment as a security analyst, with no code change.
    # The MASTER/orchestrator's user-facing identity. This is NOT a specialist agent —
    # the specialists (Atlas = product guide, Sentinel = reports, Aura = EASM) are
    # internal, and the master answers in one consistent voice. Blank = auto-derive.
    assistant_name: str = "FortiRecon Assistant"
    # Capability-agnostic so it stays right as specialists are added. Blank = auto-derive.
    assistant_tagline: str = "a knowledgeable assistant for the FortiRecon platform"

    # Capability gating — which agents are active. Comma-separated allowlist of agent
    # ids (e.g. "atlas" or "sentinel,atlas"). BLANK = every agent that is
    # otherwise available (corpus ingested / MCP reachable). The whole system
    # personality reshapes to match the enabled set.
    #
    # SHIP CONFIG (current): ship the Atlas agent ONLY (product guidance capability).
    # To bring the security agents online later, widen this (e.g. "sentinel,atlas,aura")
    # or set ENABLED_AGENTS="" in the environment to enable everything available.
    enabled_agents: str = "atlas"

    # Auth
    api_keys: str = "dev-api-key-change-me"
    jwt_secret: str = "dev-insecure-change-me-please-32byte-minimum-secret"
    jwt_algorithm: str = "HS256"

    # LLM (ChatOpenAI → vLLM/SGLang)
    llm_base_url: str = "http://localhost:30000/v1"
    llm_api_key: str = "EMPTY"
    llm_model: str = "Qwen/Qwen2.5-72B-Instruct"
    llm_fast_model: str = "Qwen/Qwen2.5-7B-Instruct"
    llm_deep_model: str = "deepseek-ai/DeepSeek-V3"
    llm_temperature: float = 0.1
    # Output budget. Kept generous so multi-step how-to walkthroughs (which now draw on
    # full user-guide pages) are not truncated mid-answer by the synthesizer/agent.
    llm_max_tokens: int = 4096

    # Chat context
    history_window_messages: int = 20

    # MCP Servers (JSON: {"agent_id": {"url": "...", "transport": "sse", "api_key": "..."}})
    mcp_servers: str = "{}"

    # Embeddings
    embedding_base_url: str = "http://localhost:8080"
    embedding_model: str = "Qwen/Qwen3-Embedding-4B"
    embedding_dim: int = 2560

    # Reranker (TEI /rerank endpoint)
    reranker_base_url: str = "http://localhost:9092"
    reranker_enabled: bool = True
    reranker_top_n: int = 0  # 0 = use top_k from search, >0 = override final count
    reranker_overfetch_multiplier: int = 3  # fetch N*top_k from Qdrant, rerank, take top_k
    reranker_score_threshold: float = 0.0  # drop reranked passages below this relevance (0 = keep all)

    # Query enrichment
    query_enrichment_enabled: bool = True  # adaptive multi-query, HyDE, step-back

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "reports_kb"
    # Atlas capability corpus: product user-guide / documentation (separate collection so
    # doc how-to pages never pollute threat-report retrieval). Capability-named (not
    # "atlas_kb") so Atlas can gain further capabilities without renaming this corpus.
    # Ingested by services/userguide-ingest/index_user_guide.py.
    user_guide_collection: str = "user_guide_kb"

    # Postgres (checkpointing + sessions)
    database_url: str = "postgresql://asi:asi@localhost:5432/asi"

    # EASM MCP Server
    easm_mcp_url: str = ""
    easm_mcp_api_key: str = ""
    easm_mcp_transport: str = "streamable_http"  # streamable_http | sse

    # Security
    guardrails_enabled: bool = True
    pii_redaction: bool = True
    human_approval_required: bool = True
    security_llm_check: bool = True
    security_timeout: float = 8.0
    security_fail_open: bool = True

    # Observability - Langfuse
    langfuse_host: str = ""
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""

    # Observability - LangSmith (alternative)
    langsmith_api_key: str = ""
    langsmith_project: str = "security-intel"

    # CORS
    cors_origins: str = "*"

    @property
    def api_key_list(self) -> list[str]:
        return [k.strip() for k in self.api_keys.split(",") if k.strip()]

    @property
    def enabled_agents_list(self) -> list[str]:
        """Allowlist of active agent ids; empty list = no explicit filter (all available)."""
        return [a.strip() for a in self.enabled_agents.split(",") if a.strip()]

    @property
    def mcp_servers_config(self) -> dict:
        import json
        try:
            return json.loads(self.mcp_servers)
        except (json.JSONDecodeError, TypeError):
            return {}
