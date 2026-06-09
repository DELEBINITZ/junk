"""FastAPI application — production-grade LangGraph multi-agent system.

Startup sequence:
1. Settings + structured logging
2. Database (Postgres with RLS)
3. LLM LaneRouter (FAST/STANDARD/DEEP)
4. Agent Registry (auto-discover from config)
5. Build all agents (LangGraph create_react_agent)
6. Build orchestrator (LangGraph StateGraph)
7. Start serving

Adding a new MCP agent requires ONLY:
- Set MCP_SERVERS env: {"new_agent": {"url": "...", "transport": "sse"}}
- Register in _register_agents() with description
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from security_intel.config import Settings
from security_intel.llm.provider import LaneRouter
from security_intel.db.postgres import Database
from security_intel.db.migrations import run_migrations
from security_intel.memory.checkpointer import get_checkpointer
from security_intel.memory.conversations import ConversationStore
from security_intel.memory.summarizer import RollingSummarizer
from security_intel.agents.registry import AgentRegistry, AgentSpec
from security_intel.agents.orchestrator import build_orchestrator
from security_intel.tools.mcp_loader import load_mcp_tools_for_agent
from security_intel.agents.reports.tools import get_reports_tools
from security_intel.agents.easm.tools import get_easm_tools
from security_intel.prompts.reports import REPORTS_SYSTEM_PROMPT
from security_intel.prompts.easm import EASM_SYSTEM_PROMPT
from security_intel.tools.query_enrichment import QueryEnricher
from security_intel.observability.logging import setup_logging, get_logger
from security_intel.observability.middleware import TracingMiddleware
from security_intel.observability.tracing import get_langfuse_handler
from security_intel.api.routes import router


async def _register_agents(registry: AgentRegistry, settings: Settings, lane_router=None) -> None:
    """Register all specialist agents with the registry.

    To add a new agent:
    1. Define its tools (local or MCP)
    2. Register with AgentSpec here
    3. Done — planner auto-discovers it, orchestrator auto-routes to it.
    """
    logger = get_logger("registry")

    # Query enricher for RAG optimization (uses fast LLM for expansion)
    enricher = None
    if settings.query_enrichment_enabled and lane_router:
        enricher = QueryEnricher(lane_router.fast)

    # Reports Agent
    reports_tools = get_reports_tools(settings, enricher=enricher)
    registry.register(
        AgentSpec(
            id="reports",
            display_name="Security Reports Agent",
            description="Searches security reports corpus (threat intel, ai generated reports and more). ",
            capabilities=[
                "Semantic search over security reports",
                "Filter by threat type, TLP level, report type",
                "Get report metadata and details",
            ],
            system_prompt=REPORTS_SYSTEM_PROMPT,
            tools=reports_tools,
        )
    )

    # EASM Agent — only registered when a real MCP server provides tools.
    # No stubs: without tools the agent is omitted rather than serving fake data.
    easm_tools = await get_easm_tools(settings)
    if not easm_tools:
        logger.warning("EASM agent not registered — no MCP tools available.")
    else:
        registry.register(
            AgentSpec(
                id="easm",
                display_name="EASM Agent",
                description="Queries external attack surface (assets, exposures, changes, rescans). "
                "Use for exposed infrastructure, asset inventory, misconfigurations, surface changes.",
                capabilities=[
                    "Query external-facing assets (domains, IPs, services)",
                    "Get current exposures and security findings",
                    "Track attack surface changes over time",
                    "Trigger asset rescans (requires approval)",
                ],
                system_prompt=EASM_SYSTEM_PROMPT,
                tools=easm_tools,
                side_effecting_tools={"trigger_rescan"},
            )
        )

    # --- ADD NEW AGENTS HERE ---
    # Example: Brand Protection Agent
    # brand_tools = await load_mcp_tools_for_agent("brand", settings)
    # if brand_tools:
    #     registry.register(AgentSpec(
    #         id="brand",
    #         display_name="Brand Protection Agent",
    #         description="Monitors brand abuse, phishing sites, typosquatting...",
    #         capabilities=["Detect brand impersonation", "Track phishing domains", ...],
    #         system_prompt="...",
    #         tools=brand_tools,
    #     ))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle — initialize all services, cleanup on shutdown."""
    settings = Settings()
    app.state.settings = settings

    # Structured logging
    setup_logging(settings)
    logger = get_logger("startup")
    logger.info("Starting Security Intelligence Platform v2.0.0")

    # Database
    db = None
    conversations = None
    summarizer = None
    try:
        db = await Database.connect(settings)
        await run_migrations(db)
        conversations = ConversationStore(db)
        app.state.db = db
        app.state.conversations = conversations
        logger.info("Postgres connected (explicit org_id filtering)")
    except Exception as e:
        logger.error(f"Postgres unavailable: {e}. Chat persistence disabled.")

    # LLM Lane Router
    lane_router = LaneRouter(settings)
    app.state.lane_router = lane_router
    logger.info(
        f"LLM lanes: fast={settings.llm_fast_model}, standard={settings.llm_model}, deep={settings.llm_deep_model}"
    )

    # Summarizer
    if conversations:
        summarizer = RollingSummarizer(conversations, lane_router.fast)

    # LangGraph Checkpointer
    checkpointer = None
    checkpoint_pool = None
    try:
        checkpointer, checkpoint_pool = await get_checkpointer(settings)
        logger.info("LangGraph checkpointer ready (Postgres)")
    except Exception as e:
        logger.warning(f"Checkpointer unavailable: {e}")

    # Langfuse tracing
    langfuse_handler = get_langfuse_handler(settings)
    app.state.langfuse_handler = langfuse_handler

    # Agent Registry
    registry = AgentRegistry()
    await _register_agents(registry, settings, lane_router=lane_router)

    # Build LangGraph agents
    registry.build_agents(lane_router.standard)
    app.state.registry = registry

    # Query enricher for orchestrator-level task enrichment
    orchestrator_enricher = None
    if settings.query_enrichment_enabled:
        orchestrator_enricher = QueryEnricher(lane_router.fast)

    # Build orchestrator (main LangGraph StateGraph)
    orchestrator = build_orchestrator(
        lane_router=lane_router,
        registry=registry,
        conversations=conversations,
        summarizer=summarizer,
        checkpointer=checkpointer,
        query_enricher=orchestrator_enricher,
    )
    app.state.orchestrator = orchestrator
    logger.info(f"Orchestrator ready. Agents: {registry.agent_ids}")

    # Warm up Presidio engines (avoids ~2s cold start on first request)
    if settings.guardrails_enabled:
        try:
            from security_intel.security.guardrails import _get_analyzer, _get_anonymizer
            _get_analyzer()
            _get_anonymizer()
            logger.info("Presidio engines warmed up")
        except Exception as e:
            logger.warning(f"Presidio warmup failed: {e}")

    yield

    # Cleanup
    if checkpoint_pool:
        await checkpoint_pool.close()
    if db:
        await db.close()
        logger.info("Postgres pool closed")


def create_app() -> FastAPI:
    """Application factory."""
    settings = Settings()

    app = FastAPI(
        title="Security Intelligence Platform",
        version="2.0.0",
        description="Multi-agent security intelligence powered by LangGraph",
        lifespan=lifespan,
    )

    # Middleware
    app.add_middleware(TracingMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins.split(","),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)
    return app


app = create_app()
