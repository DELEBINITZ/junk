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

from security_intel.agents.easm.tools import get_easm_tools
from security_intel.agents.orchestrator import build_orchestrator
from security_intel.agents.registry import AgentRegistry, AgentSpec
from security_intel.agents.reports.tools import get_reports_tools
from security_intel.agents.userguide.tools import get_user_guide_tools
from security_intel.api.routes import router
from security_intel.config import Settings
from security_intel.db.migrations import run_migrations
from security_intel.db.postgres import Database
from security_intel.llm.provider import LaneRouter
from security_intel.memory.checkpointer import get_checkpointer
from security_intel.memory.conversations import ConversationStore
from security_intel.memory.summarizer import RollingSummarizer
from security_intel.observability.logging import get_logger, setup_logging
from security_intel.observability.middleware import TracingMiddleware
from security_intel.observability.tracing import get_langfuse_handler
from security_intel.prompts.easm import EASM_SYSTEM_PROMPT
from security_intel.prompts.reports import REPORTS_SYSTEM_PROMPT
from security_intel.prompts.userguide import USER_GUIDE_SYSTEM_PROMPT
from security_intel.tools.mcp_loader import load_mcp_tools_for_agent, mcp_agent_ids
from security_intel.tools.query_enrichment import QueryEnricher


async def _user_guide_corpus_ready(settings: Settings) -> bool:
    """True when the user-guide collection exists and holds at least one point.

    Gates registration of the User Guide agent so it is never exposed against an
    empty/absent corpus (which would let the planner route to a dead end). Any
    connection/error is treated as "not ready" — fail closed, log, skip the agent.
    """
    from security_intel.tools.qdrant_search import _get_qdrant_client

    try:
        qdrant = _get_qdrant_client(settings)
        collection = settings.user_guide_collection
        if not await qdrant.collection_exists(collection):
            return False
        info = await qdrant.get_collection(collection)
        return (info.points_count or 0) > 0
    except Exception as e:
        get_logger("registry").warning(f"User-guide corpus check failed: {e}")
        return False


async def _register_agents(registry: AgentRegistry, settings: Settings, lane_router=None) -> None:
    """Register all specialist agents with the registry.

    To add a new agent:
    1. Define its tools (local or MCP)
    2. Register with AgentSpec here
    3. Done — planner auto-discovers it, orchestrator auto-routes to it.
    """
    logger = get_logger("registry")

    # Capability gating — the supported way to run "only some agents" WITHOUT editing
    # code. ENABLED_AGENTS is an allowlist of agent ids; blank means "every agent that
    # is otherwise available". The orchestrator then DERIVES its whole persona from
    # whatever ends up registered (see agents/identity.py), so a userguide-only
    # deployment presents as a product guide, not a security tool.
    allowlist = set(settings.enabled_agents_list)
    if allowlist:
        logger.info(f"ENABLED_AGENTS allowlist active: {sorted(allowlist)}")

    def _enabled(agent_id: str) -> bool:
        return not allowlist or agent_id in allowlist

    def _enricher(domain: str):
        """Per-agent query enricher, labeled with the agent's corpus domain so RAG
        expansions stay on-topic (a docs query is not expanded like a threat query)."""
        if settings.query_enrichment_enabled and lane_router:
            return QueryEnricher(lane_router.fast, domain=domain)
        return None

    # Reports Agent
    if _enabled("reports"):
        reports_tools = get_reports_tools(settings, enricher=_enricher("security reports corpus"))
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
                # tool_call: single-pass semantic search, no ReAct loop (cheaper/faster).
                # TRADEOFF: this narrows the reports agent to search_reports at runtime —
                # the by-ID (get_report_content) and filter (search_reports_by_filter)
                # paths are NOT exercised in tool_call mode. If "summarize report <id>" or
                # "show all TLP:RED reports" style queries matter, set mode="react" here.
                mode="tool_call",
                primary_tool="search_reports",
            )
        )
    else:
        logger.info("Reports agent disabled by ENABLED_AGENTS allowlist.")

    # User Guide Agent — answers product how-to / dashboard-walkthrough / navigation
    # questions from the FortiRecon documentation corpus (Qdrant: user_guide_collection).
    # Registered only when that collection has been ingested (services/userguide-ingest),
    # so we never route users to an empty docs corpus.
    if not _enabled("userguide"):
        logger.info("User Guide agent disabled by ENABLED_AGENTS allowlist.")
    elif await _user_guide_corpus_ready(settings):
        user_guide_tools = get_user_guide_tools(
            settings, enricher=_enricher("product user guide / documentation")
        )
        registry.register(
            AgentSpec(
                # id stays "userguide" (internal capability key, tied to the docs corpus).
                # display_name is a CAPABILITY of Atlas, phrased as such — Atlas is the
                # agent; helping with the product/user guide is one thing it can do.
                id="userguide",
                display_name="FortiRecon Product Guide",
                description=(
                    "Helps you use the FortiRecon platform: step-by-step how-to, navigation, "
                    "dashboards, features, and configuration — answered from the product "
                    "documentation. Use for 'how do I…', 'where do I find…', 'walk me "
                    "through…', feature explanations, and configuration steps (NOT threat-report content)."
                ),
                capabilities=[
                    "Explain FortiRecon dashboards, menus, and features",
                    "Give step-by-step how-to and configuration instructions",
                    "Walk through navigation and where to find things in the product",
                ],
                system_prompt=USER_GUIDE_SYSTEM_PROMPT,
                tools=user_guide_tools,
                # react: the agent REASONS over retrieval — it can run search_user_guide,
                # then follow up with get_user_guide_page(<doc_id>) to pull a full page for
                # a complete walkthrough (its system prompt directs exactly this). tool_call
                # mode would fire a single search and stop, truncating multi-step how-tos.
                mode="react",
            )
        )
    else:
        logger.warning(
            "User Guide agent not registered — collection "
            f"'{settings.user_guide_collection}' is empty or unavailable. "
            "Run the services/userguide-ingest service to ingest the FortiRecon user guide."
        )

    # EASM Agent — only registered when a real MCP server provides tools.
    # No stubs: without tools the agent is omitted rather than serving fake data.
    if not _enabled("easm"):
        logger.info("EASM agent disabled by ENABLED_AGENTS allowlist.")
    else:
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

    # --- Config-driven MCP agents (add a new agent with NO code changes) ---
    # Any server declared in MCP_SERVERS (or the EASM shortcut) that isn't already
    # registered above is auto-registered here as a react agent: its tools are
    # discovered from the server, its identity/capabilities come from the config (or are
    # derived from the tool names), and it gets a generated system prompt. The router,
    # planner, and derived persona pick it up automatically — no orchestrator edits.
    #
    #   MCP_SERVERS='{"brand": {"url": "https://brand-mcp/…", "transport": "streamable_http",
    #                           "api_key": "…", "display_name": "Brand Protection Agent",
    #                           "description": "Monitors brand abuse, phishing, typosquatting.",
    #                           "capabilities": ["Detect impersonation", "Track phishing domains"]}}'
    for agent_id in mcp_agent_ids(settings):
        if agent_id in registry.specs:
            continue  # already registered explicitly above (e.g. easm)
        if not _enabled(agent_id):
            logger.info(f"MCP agent '{agent_id}' disabled by ENABLED_AGENTS allowlist.")
            continue
        server_cfg = settings.mcp_servers_config.get(agent_id, {})
        tools = await load_mcp_tools_for_agent(agent_id, settings)
        if not tools:
            logger.warning(
                f"MCP agent '{agent_id}' not registered — its server returned no tools "
                "(unreachable or misconfigured)."
            )
            continue
        display_name = server_cfg.get("display_name") or f"{agent_id.replace('_', ' ').title()} Agent"
        description = server_cfg.get("description") or (
            f"Tools from the {agent_id} service: " + ", ".join(t.name for t in tools) + "."
        )
        capabilities = server_cfg.get("capabilities") or [t.name for t in tools]
        registry.register(
            AgentSpec(
                id=agent_id,
                display_name=display_name,
                description=description,
                capabilities=capabilities,
                tools=tools,
                # react + no primary_tool: MCP tool names are only known at runtime, so
                # hand all server tools to the ReAct loop rather than pinning one.
                mode="react",
                side_effecting_tools=set(server_cfg.get("side_effecting_tools", [])),
            )
        )
        logger.info(f"Auto-registered MCP agent '{agent_id}' ({len(tools)} tools)")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle — initialize all services, cleanup on shutdown."""
    settings = Settings()
    app.state.settings = settings

    # Structured logging
    setup_logging(settings)
    logger = get_logger("startup")
    logger.info("Starting agentic assistant platform v2.0.0 (identity derived from enabled agents)")

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

    # Derive the assistant identity from whatever agents actually built. This is what
    # makes the persona/scope/boundaries follow the enabled agent set instead of a
    # hardcoded security persona.
    from security_intel.agents.identity import build_system_profile
    profile = build_system_profile(registry, settings)
    app.state.profile = profile
    logger.info(f"System identity: '{profile.name}' — scope: {profile.domains}")

    # Query enricher for orchestrator-level task enrichment (neutral domain — it spans
    # whatever agents are active).
    orchestrator_enricher = None
    if settings.query_enrichment_enabled:
        orchestrator_enricher = QueryEnricher(lane_router.fast, domain="knowledge base")

    # Build orchestrator (main LangGraph StateGraph)
    orchestrator = build_orchestrator(
        lane_router=lane_router,
        registry=registry,
        conversations=conversations,
        summarizer=summarizer,
        checkpointer=checkpointer,
        query_enricher=orchestrator_enricher,
        profile=profile,
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

    # Title honors the operator's assistant name when set; otherwise a neutral default.
    # (The chat PERSONA is derived per-request from the enabled agents; this is only the
    # HTTP/OpenAPI label, which exists before the registry/profile do.)
    app_title = settings.assistant_name or "Agentic Assistant Platform"

    app = FastAPI(
        title=app_title,
        version="2.0.0",
        description="Multi-agent assistant powered by LangGraph; capabilities depend on the enabled agents.",
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
