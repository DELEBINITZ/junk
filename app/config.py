"""Central configuration via pydantic-settings — the single source of truth that
the whole app is assembled from.

MENTAL MODEL: this one Settings object is the "control panel" for the platform.
bootstrap.py reads it and decides which concrete backend to wire for every seam
(LLM, embeddings, vector store, chat store, agent engine, tracing). That pattern
is "config-gating": the *same* code path serves whichever real backend a value
here selects. No code change is needed to switch — only an env var.

REAL INFRA IS REQUIRED: the offline/deterministic stubs have been removed. The
platform needs a real LLM (SGLang/OpenAI-compatible), a real embedder (TEI/OpenAI),
Qdrant, and Postgres to boot and serve. Two payoffs of the data-governance posture:
  * onboarding / tests / CI run anywhere, instantly and reproducibly; and
  * SECURITY — on the defaults, no contract or security-intelligence text is ever
    sent to an external LLM/embedding API. Sensitive data only leaves this process
    if an operator EXPLICITLY flips a provider to a hosted service and supplies
    creds. Self-hosted-by-default is a deliberate data-governance posture.

  * ``llm_provider=sglang|openai``      -> a real LLM server is required
  * ``embedding_provider=tei|openai``   -> a real embedder is required
  * ``retrieval_backend=qdrant``        -> Qdrant is required
  * ``store_backend=postgres``          -> Postgres (+RLS) is required
  * ``agent_engine=internal``           -> built-in graph engine, no langgraph dep

Provide each backend's URL + creds via env. There is no zero-infra fallback.

HOW pydantic-settings WORKS: each field below is read from the environment (or a
``.env`` file) by its UPPER_CASE name; if absent, the default here is used and is
fully type-validated. That is why every value has a safe default — the app is
always runnable, and overriding is just setting an env var.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

# The insecure development JWT secret. Referenced as the dev default AND by the
# production guard, which refuses to boot prod while this is still in use.
_DEV_JWT_SECRET = "dev-insecure-change-me-please-32byte-minimum-secret"  # noqa: S105


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
    cors_origins: Annotated[list[str], NoDecode] = Field(default_factory=lambda: ["*"])

    # ---- Auth --------------------------------------------------------------
    # ONE auth model (see security/deps.py): every request must carry (1) a valid
    # API KEY (``X-API-Key`` header, or ``?api_key=`` for SSE) that gates access at
    # the gateway, AND (2) a JWT (Bearer / ``?access_token=`` / cookie) whose claims
    # carry the verified identity — ``org_id`` (the tenant key that powers isolation),
    # ``sub`` (user id), and ``roles``. The JWT is minted upstream; this service only
    # VERIFIES it. There is no login/password/OIDC/refresh flow.
    #
    # ``api_keys``: the set of accepted gateway keys. ``jwt_secret``: the HS256 key
    # used to verify (and, via the dev mint helper, sign) JWTs — the dev default is
    # intentionally insecure and the prod guard rejects it.
    api_keys: Annotated[list[str], NoDecode] = Field(default_factory=lambda: ["dev-api-key-change-me"])
    jwt_secret: str = _DEV_JWT_SECRET  # CHANGE in prod — the prod guard enforces this
    jwt_algorithm: str = "HS256"
    access_token_ttl_seconds: int = 3600

    # ---- Stores ------------------------------------------------------------
    # Where conversations/sessions persist. ``memory`` is in-process (zero infra);
    # ``postgres`` uses real Postgres with ROW-LEVEL SECURITY. ``rls_setting_name``
    # is the Postgres session variable set to the org id inside each transaction,
    # so the database itself enforces "an org can only read its own rows" — a
    # second, defense-in-depth layer of tenant isolation under the app checks.
    store_backend: Literal["postgres"] = "postgres"
    database_url: str = ""  # postgresql://user:pass@host:5432/db (REQUIRED — no in-memory fallback)
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
    # The model backend (all OpenAI-compatible HTTP). A real LLM server is REQUIRED —
    # there is no offline stub. Three model "lanes" (fast/standard/deep) let cheap
    # steps (routing, tool planning) use a small model and the answer use a bigger
    # one — see llm/lanes.py.
    #   sglang -> self-hosted, serves THREE models (one per lane).
    #   vllm   -> self-hosted (e.g. one 32B for staging); the three lanes collapse to
    #             the single served ``vllm_model``.
    #   openai -> any hosted/OpenAI-compatible endpoint; lanes collapse to one model.
    llm_provider: Literal["sglang", "vllm", "openai"] = "vllm"  # in-house: one 72B on vLLM
    # SGLang (OpenAI-compatible) — three lanes, three models
    sglang_base_url: str = "http://localhost:30000/v1"
    sglang_api_key: str = "EMPTY"
    model_fast: str = "Qwen/Qwen2.5-7B-Instruct"
    model_standard: str = "Qwen/Qwen2.5-72B-Instruct"
    model_deep: str = "deepseek-ai/DeepSeek-V3.1"
    # vLLM (OpenAI-compatible) — one served model, all three lanes map to it.
    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_api_key: str = "EMPTY"
    vllm_model: str = "Qwen/Qwen2.5-72B-Instruct"  # in-house vLLM: one 72B serves all lanes
    # OpenAI-compatible hosted provider (never for sensitive prod data)
    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.1   # low temperature => more deterministic, factual answers
    llm_max_tokens: int = 1024
    llm_timeout_seconds: float = 60.0
    # OVERFLOW FUSE: cap the answer prompt to this many CHARACTERS before sending, so a
    # large retrieved-context block can't exceed the model window and crash the turn
    # (UpstreamError). Char-based, not exact token accounting — the lowest-ranked context
    # is trimmed first. Keep comfortably below the window in chars (~3-4 chars/token);
    # 80k suits a 32k-token model. LOWER this if you serve a smaller context window.
    max_prompt_chars: int = 80000

    # ---- Embeddings --------------------------------------------------------
    # Vectorizer for RAG. ``tei`` is the self-hosted embedding server; ``openai`` is
    # an OpenAI-compatible embedding endpoint. A real embedder is REQUIRED — there is
    # no offline stub. ``embedding_dim`` MUST match whatever the vector store was
    # created with.
    embedding_provider: Literal["tei", "openai"] = "tei"
    tei_embed_url: str = "http://localhost:8080"
    # QUANT PROFILE (fits a single 24GB GPU): Qwen3-Embedding-4B, native dim 2560.
    # ~1 MTEB point below the 8B — negligible. ``embedding_dim`` MUST equal what TEI
    # returns AND the Qdrant collection dim AND the ingest cron's EMBEDDING_DIM.
    embedding_dim: int = 2560
    embedding_model: str = "Qwen/Qwen3-Embedding-4B"
    # Qwen3-Embedding (and similar INSTRUCT embedders) expect an instruction prefix on
    # the QUERY side ONLY — documents/chunks are embedded raw (the ingest cron must NOT
    # add it). Off by default ("") so behavior is unchanged; set it ONLY if your TEI
    # server doesn't already apply the instruction, e.g.:
    #   "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "
    # Adding it when the server already does = double instruction = WORSE recall, so verify first.
    embedding_query_instruction: str = ""

    # ---- Retrieval ---------------------------------------------------------
    # The vector store + RAG knobs. ``qdrant`` is the production vector DB and the
    # only backend — there is no in-process fallback. ``top_k`` = how many candidates
    # to fetch; reranking (a cross-encoder, optional) reorders them for precision and
    # keeps the best ``rerank_top_k``. ``recency_half_life_days`` lets fresher docs
    # outrank stale ones — important for security intel that ages fast.
    retrieval_backend: Literal["qdrant"] = "qdrant"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    retrieval_top_k: int = 20
    rerank_enabled: bool = False
    rerank_provider: Literal["none", "tei"] = "none"
    tei_rerank_url: str = "http://localhost:8081"
    rerank_top_k: int = 6
    recency_half_life_days: float = 180.0

    # ---- Guardrails --------------------------------------------------------
    # The safety spine, applied to BOTH the incoming question and the outgoing answer.
    # NO dedicated guard-model deployments: injection + content safety run on the
    # MAIN deployed LLM (LLMJudgeGuard — the single 72B in prod / 32B in staging
    # doubles as a hardened security judge; one extra small completion per turn).
    # PII is Microsoft Presidio (the ``pii`` extra; fails fast if absent).
    # The only hand-coded pieces are the ones no library covers inline: secret
    # redaction, indirect-injection neutralization of retrieved content, output
    # exfiltration defang, and groundedness verification (see guardrails/detectors.py).
    # Nothing extra to require in PROD — the LLM is already mandatory.
    guardrails_enabled: bool = True
    pii_redaction: bool = True
    injection_detection: bool = True
    topic_safety: bool = True
    groundedness_check: bool = True
    # INDIRECT prompt-injection defense (custom, no library): neutralize injection
    # instructions found in RETRIEVED documents + tool outputs (adversary-controlled)
    # before they enter the answer prompt, and frame that context as untrusted data.
    # Critical for an agentic RAG system that reads attacker-influenced content.
    indirect_injection_defense: bool = True
    # Defang data-exfiltration vectors (auto-loading markdown images, script links)
    # in the generated answer (custom — LLM-output-specific).
    output_exfiltration_guard: bool = True
    # ---- PII (Microsoft Presidio) ----
    # NER + context + checksums — catches free-text PERSON names, which no regex
    # can. Security-tuned entity set: IP_ADDRESS/URL/DOMAIN are intentionally
    # OMITTED — they are the product's subject matter, not PII.
    pii_score_threshold: float = 0.5
    pii_entities: Annotated[list[str], NoDecode] = Field(default_factory=lambda: [
        "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "US_SSN", "CREDIT_CARD",
        "IBAN_CODE", "US_BANK_NUMBER", "CRYPTO", "MEDICAL_LICENSE", "US_PASSPORT",
    ])
    # Fail policy for the model classifiers (injection/safety) on a model error:
    # True = fail CLOSED (block on classifier error — the production-safe default for a
    # security product); False = fail OPEN (availability over strictness).
    guardrails_fail_closed: bool = True

    # ---- Agent -------------------------------------------------------------
    # How the agent runs a turn. ``agent_engine`` chooses the built-in graph engine
    # (zero deps) vs real LangGraph (adds durable checkpointing) — same node set
    # either way (see agent/graph.py). ``router_mode`` chooses the supervisor's
    # routing strategy: ``heuristic`` (no LLM) vs ``llm`` tool-calling. The window/
    # trigger settings bound how much chat history is kept verbatim before it gets
    # rolled into a running summary.
    agent_engine: Literal["internal", "langgraph"] = "internal"
    max_tool_iterations: int = 4
    # Small-talk / scope triage: greet + steer on "hi" / "what can you do" / off-topic
    # WITHOUT routing, retrieval, or tool calls. Off => every message runs the full agent.
    smalltalk_handling: bool = True
    history_window_messages: int = 12     # how many prior messages run_turn loads
    answer_history_turns: int = 6         # how many of those the answer prompt includes verbatim
    summary_trigger_messages: int = 20
    # CROSS-SESSION RECALL: pull a few relevant snippets from the user's OTHER (past)
    # conversations into context, so follow-ups that reference earlier chats have
    # continuity ("like we discussed last week"). Org+user scoped by the store — never
    # crosses tenant/user. Injected as background context only, NOT citable evidence.
    cross_session_recall: bool = True
    cross_session_recall_k: int = 3       # max past-conversation snippets to inject
    # The agentic path is ON by default: the planner brain + LLM tool-calling. Both
    # DEGRADE GRACEFULLY to deterministic behaviour when no real LLM is wired (the
    # provider gate inside the supervisor/planner/specialist makes them no-ops on
    # the deterministic stub), so the zero-infra path still works. Point the LLM at
    # your Qwen/SGLang and the full agentic loop (plan -> tools -> reflect) lights up.
    router_mode: Literal["heuristic", "llm"] = "llm"
    # How the supervisor picks which module(s)/app(s) handle a query when NOT using
    # the LLM router. Routing is ALWAYS dynamic (by meaning), never by curated
    # keywords — the brittle keyword/``routing_hints`` path has been removed because
    # it silently mis-routed any query whose wording differed from the hand-written
    # phrases. "semantic" (default) = EMBEDDING similarity between the query and each
    # module's natural-language profile (display_name + description + tool
    # names/descriptions). The default embedder is deterministic + offline, so this
    # works with no model; a real embedder sharpens recall. "llm" defers entirely to
    # the LLM router (router_mode=llm + a real model).
    routing_strategy: Literal["semantic", "llm"] = "semantic"
    # Orchestration strategy. ``heuristic`` = the v1 supervisor->specialists graph
    # (route -> parallel dispatch -> answer). ``planner`` = the LLM-brain graph
    # (plan -> dispatch-with-dependencies -> synthesize -> bounded replan): it
    # decomposes a query into steps across modules and supports cross-module data
    # dependencies (a step can consume an earlier step's findings). The planner
    # falls back to a deterministic supervisor-style plan when no real LLM is wired,
    # so the zero-infra path still works and tests stay green.
    orchestrator_mode: Literal["heuristic", "planner"] = "planner"
    max_plan_steps: int = 6      # hard cap on steps the planner may emit (bounds fan-out + cost)
    # DEEP REASONING depth: how many times the reflect gate may revise the plan after
    # finding a gap (LLM mode only; the deterministic path always finishes in one
    # pass). 3 lets the agent iterate a few times — gather, notice what's missing,
    # fetch the missing data, and re-check — before answering. Bounds cost/latency
    # (each round is a full plan+answer+critic pass); raise for harder questions, set
    # 0 to disable reflection. On a single 72B every round is ~several model calls.
    max_replans: int = 3
    planner_max_fanout: int = 2  # max modules a single (deterministic) plan spreads across

    # ---- Remote MCP servers + tool-context budget --------------------------
    # Per-module MCP server URLs. Empty => that module runs IN-PROCESS (local
    # tools). Set one (e.g. a FastMCP server for EASM) and that module's tool
    # EXECUTION is routed there over MCP, while its manifest stays local as the
    # contract (so RBAC, the action gate, and planner cards are unchanged). Brand
    # and ACI follow the same pattern when their servers exist.
    easm_mcp_url: str = ""
    brand_mcp_url: str = ""
    aci_mcp_url: str = ""
    # Utility module backed by the external `mcp-test-kits` MCP server. Empty =>
    # the testkit module runs its LOCAL stub tools; set to e.g.
    # http://localhost:3000/mcp to route execution to the real server.
    testkit_mcp_url: str = ""
    # GENERIC MCP wiring (the easy-integration path). A {module_id: url} map that
    # promotes ANY capability module to its own MCP server with ZERO code edits — no
    # new config field, no bootstrap change. Set it as JSON in the env, e.g.
    #   MCP_URLS={"easm":"http://easm-mcp:8000/mcp","newmod":"http://newmod:8000/mcp"}
    # bootstrap iterates the registered modules and wires a FastMCPRemote for any
    # whose id appears here (this map wins) OR that has a legacy ``<id>_mcp_url`` field
    # above (kept for back-compat). Adding a brand-new MCP-backed module = write its
    # manifest + add one entry here. Routing/RBAC/the action gate are unchanged.
    mcp_urls: dict[str, str] = Field(default_factory=dict)
    # TTL of the short-lived, org-scoped service token minted to authenticate a
    # remote MCP call (identity travels in this token, never in tool args).
    mcp_service_token_ttl_seconds: int = 120
    # CONTEXT-FLOOD GUARD: the max number of tools a specialist advertises to the
    # tool-calling LLM at once. A remote MCP server may expose hundreds of tools;
    # the specialist shortlists the most relevant ones (tool-RAG) up to this budget
    # so a single agent context is never flooded with tool schemas.
    max_tools_advertised: int = 8
    # DYNAMIC TOOL DISCOVERY (pure-dynamic MCP). When a module is backed by an
    # external MCP server, the agent discovers its tools at runtime via tools/list
    # instead of declaring them locally. The discovered list is cached per
    # (module, org) for this many seconds (servers filter by the caller's org).
    mcp_tool_cache_ttl_seconds: int = 300
    # Coarse LOCAL role floor applied to a dynamically-discovered tool (the remote
    # server still enforces its own fine-grained RBAC from the service token; this
    # is just a local floor since there's no local manifest entry to read).
    mcp_dynamic_tool_role: str = "viewer"
    # SAFETY POLICY for discovered tools the server did NOT annotate with a
    # read-only/destructive hint: True => trust as read-callable; False => exclude
    # (require a local manifest stub so the human action gate applies). A tool the
    # server explicitly marks destructive/non-read-only is NEVER dynamically
    # callable regardless of this flag — side effects must be declared + gated.
    mcp_dynamic_trust_unannotated: bool = True

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
    cap_testkit_enabled: bool = True
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

    @field_validator("api_keys", "pii_entities", mode="before")
    @classmethod
    def _csv_list(cls, v: object) -> object:
        """Same CSV-or-JSON convenience for the other list-valued settings, so an
        env var like ``API_KEYS=key1,key2`` works without JSON."""
        if isinstance(v, str):
            s = v.strip()
            if not s:
                return []
            if s.startswith("["):
                return v
            return [item.strip() for item in s.split(",") if item.strip()]
        return v

    @model_validator(mode="after")
    def _enforce_prod_no_stub(self) -> Settings:
        """PRODUCTION GUARD — fail-closed if ENVIRONMENT=prod is running on an
        insecure or incomplete setting. The deterministic/in-memory stubs no longer
        exist (a real LLM/embedder/Qdrant/Postgres is always required), so this now
        checks the remaining footguns: a missing DB URL, dev secrets, and demo
        toggles. The process refuses to boot and prints exactly what to fix."""
        if self.environment != "prod":
            return self
        bad: list[str] = []
        # Postgres is the only store backend — a DB URL is mandatory to run at all.
        if not self.database_url:
            bad.append("DATABASE_URL is empty — Postgres is required (no in-memory store)")
        if self.seed_demo_data:
            bad.append("SEED_DEMO_DATA=true — must be false in prod (no demo corpora)")
        if self.debug:
            bad.append("DEBUG=true — must be false in prod")
        # JWT verifies caller identity — the secret must be strong in prod.
        if self.jwt_secret == _DEV_JWT_SECRET:
            bad.append("JWT_SECRET is the insecure dev default — set a strong secret")
        elif len(self.jwt_secret) < 32:
            bad.append("JWT_SECRET shorter than 32 bytes")
        # API keys gate the gateway — the dev default must not ship to prod.
        if not self.api_keys or "dev-api-key-change-me" in set(self.api_keys):
            bad.append("API_KEYS is empty or still the dev default — set real gateway key(s)")
        # Guard models need nothing here: injection/content-safety run on the main
        # LLM (LLMJudgeGuard), and a real LLM endpoint is already mandatory.
        # Tool-backed modules MUST point at a real MCP server in prod, else they
        # would serve their built-in mock data. (Corpus modules like reports are
        # covered by RETRIEVAL_BACKEND=qdrant above.)
        for enabled, url, var in (
            (self.cap_easm_enabled, self.easm_mcp_url, "EASM_MCP_URL"),
            (self.cap_brand_enabled, self.brand_mcp_url, "BRAND_MCP_URL"),
            (self.cap_aci_enabled, self.aci_mcp_url, "ACI_MCP_URL"),
        ):
            if enabled and not url:
                bad.append(f"{var} not set while its module is enabled — would serve mock data")
        if bad:
            raise ValueError(
                "ENVIRONMENT=prod but insecure/incomplete configuration detected. Fix:\n  - "
                + "\n  - ".join(bad)
            )
        return self

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
