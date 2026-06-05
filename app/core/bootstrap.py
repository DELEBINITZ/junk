"""Composition root — wires the whole platform from config.

MENTAL MODEL: this is the "main assembly line". A composition root is the ONE
place that constructs every concrete service and hands the wiring out; everything
else just RECEIVES what it needs (this is dependency injection — components never
build their own dependencies, which is what makes the app config-driven and
testable). Read top to bottom and you see the entire object graph being built:

    config (Settings)  ->  build_services()  ->  AppServices bundle
                                |
        logging/metrics/tracer  |  registry (discovers capability modules)
        llm / rag / store       |  mcp boundary  +  supervisor  +  orchestrator

WHY IT STAYS FEATURE-AGNOSTIC: nothing here names a specific capability (reports,
easm, aci, ...). Modules are DISCOVERED by the registry from disk and ENABLED by
config flags, so adding a feature never touches this file — the core never learns
a feature's name. Every backend chosen below is config-gated (see config.py), so
on the defaults this builds the fully deterministic, self-hosted, offline stack.

Used by the FastAPI lifespan (app/main.py) and by tests/eval, so the exact same
wiring backs the server, the test suite, and the eval gate — no drift.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

from app.config import Settings
from app.core.action_gate import ActionGate, build_action_gate
from app.core.agent import Orchestrator, Supervisor, build_checkpointer
from app.core.contracts import CoreDeps
from app.core.guardrails import build_input_guardrails, build_output_guardrails
from app.core.llm import build_llm
from app.core.mcp import InProcessMCPClient
from app.core.memory import RollingSummarizer, build_conversation_store
from app.core.observability import build_tracer, configure_logging, get_logger, get_metrics
from app.core.observability.audit import build_audit_logger
from app.core.rag import build_rag
from app.core.registry import CapabilityRegistry


@dataclass
class AppServices:
    """The fully-assembled application — every long-lived service in one bundle.

    Built once at startup and stashed on ``app.state`` so request handlers can
    reach any service without re-constructing it. Think of this as the output of
    the composition root: the live object graph the running app uses.
    """

    settings: Settings
    registry: CapabilityRegistry        # discovered capability modules + per-org views
    deps: CoreDeps                      # the shared service bundle handed to tools/specialists
    mcp: InProcessMCPClient             # the tool boundary: enforces RBAC + the action gate
    orchestrator: Orchestrator          # runs a chat turn end to end (the agent graph)
    supervisor: Supervisor              # routes a question to the right capability module(s)
    conversations: Any
    summarizer: RollingSummarizer
    llm: Any
    rag: Any
    action_gate: ActionGate
    input_guard: Any
    output_guard: Any
    tracer: Any
    metrics: Any
    logger: Any
    audit: Any = None
    checkpointer: Any = None

    async def aclose(self) -> None:
        """Graceful shutdown: close every service that owns external resources
        (HTTP clients, DB pools, tracer flush). Best-effort and order-tolerant —
        we swallow per-service errors so one stubborn handle can't block the
        rest of shutdown. Called from the FastAPI lifespan's ``finally``."""
        for c in (self.llm, self.rag, self.conversations, self.mcp, self.tracer):
            close = getattr(c, "aclose", None)   # duck-typed: only close what can be closed
            if close:
                try:
                    await close()
                except Exception:
                    pass


def _service_token_minters(settings: Settings, audience: str):
    """Return the (for_ctx, for_sc) token minters for one remote audience. Defined
    at module level (not inside the loop below) so each call binds its OWN
    ``audience`` — the closures turn the trusted local identity into a short-lived,
    org-scoped service token for that specific server."""
    from app.core.security.jwt import create_service_token

    def for_ctx(ctx):
        return create_service_token(
            settings, sub=ctx.user_id, org_id=ctx.org_id, roles=ctx.roles,
            audience=audience, ttl_seconds=settings.mcp_service_token_ttl_seconds)

    def for_sc(sc):
        return create_service_token(
            settings, sub=sc.user_id, org_id=sc.org_id, roles=sc.roles,
            audience=audience, ttl_seconds=settings.mcp_service_token_ttl_seconds)

    return for_ctx, for_sc


def _mcp_url_for(settings: Settings, module_id: str) -> str:
    """Resolve the MCP server URL for one module, or "" if it runs in-process.

    Two sources, generic map first: the ``mcp_urls`` {id: url} map (the zero-code
    integration path — add an entry, done) wins; otherwise a legacy per-module
    ``<id>_mcp_url`` Settings field (easm/brand/aci/testkit, kept for back-compat).
    Convention over configuration: there is no hardcoded list of module ids here, so
    a brand-new MCP-backed module is wired purely from config."""
    return (settings.mcp_urls.get(module_id) or getattr(settings, f"{module_id}_mcp_url", "") or "").strip()


def _build_remote_executors(settings: Settings, registry: CapabilityRegistry) -> dict[str, Any]:
    """Build the per-module REMOTE tool executors from config. For each REGISTERED
    module that has an MCP URL (see ``_mcp_url_for``) we create a FastMCPRemote whose
    calls carry a short-lived, org-scoped service token (identity rides in the token,
    never in tool args). Empty by default => every module runs in-process, so this
    adds no dependency and no behaviour change unless a URL is set. This is the ONE
    place a module is "promoted" to its own server — and it now iterates the registry
    (like seed_demo) instead of a hardcoded id list, so promoting ANY module to MCP is
    config-only: write its manifest, add its url to ``MCP_URLS``, done. No core edit."""
    from app.core.mcp.fastmcp_client import FastMCPRemote

    executors: dict[str, Any] = {}
    for module in registry.modules():
        url = _mcp_url_for(settings, module.id)
        if not url:
            continue
        for_ctx, for_sc = _service_token_minters(settings, f"{module.id}-mcp")
        executors[module.id] = FastMCPRemote(url, token_for_ctx=for_ctx, token_for_sc=for_sc)
    return executors


def build_services(settings: Settings) -> AppServices:
    """THE composition root. Construct every service from ``settings`` and wire
    them into the running app. Order matters and reads like a dependency chain:

      1. observability first (logging/metrics/tracer) so everything after can log;
      2. the registry DISCOVERS capability modules (reports/easm/...) from disk;
      3. the config-gated backends (llm, rag, store) — each is the
         deterministic default or the real backend purely per config.py;
      4. bundle those into ``CoreDeps`` (the bag injected into every tool call);
      5. build the boundaries/agent — the MCP tool boundary (RBAC + action gate),
         the supervisor (routing), and the orchestrator (runs the whole turn);
      6. return everything in the AppServices bundle.

    Each ``build_*`` factory hides one config-gate decision, so this function
    never branches on provider names — it just composes the pieces it's handed.
    """
    # 1. Observability is wired FIRST so every later step can emit logs/metrics
    #    and traces. ``configure_logging`` flips JSON vs human output per config.
    configure_logging(settings.log_level, settings.log_json)
    logger = get_logger("asi")
    metrics = get_metrics(settings)
    tracer = build_tracer(settings)   # NoOp by default; Langfuse if configured

    # 2. Discover capability modules from app/capabilities/*. The registry reads
    #    each manifest, validates its contract, and marks it enabled per its flag.
    registry = CapabilityRegistry().discover(settings)
    # 3. Config-gated core backends. Every factory returns the deterministic/self-
    #    hosted default unless config.py points it at a real backend.
    llm = build_llm(settings)
    rag = build_rag(settings)
    conversations = build_conversation_store(settings)
    summarizer = RollingSummarizer(llm)               # compresses old turns into a running summary
    action_gate = build_action_gate(settings)         # holds side-effecting actions for human approval

    # 4. CoreDeps is the shared service bundle. It is what gets handed to every
    #    tool/specialist (via ToolContext.deps), so a tool can reach core services
    #    without importing anything global — the dependency-injection seam.
    deps = CoreDeps(
        settings=settings, llm=llm, rag=rag, registry=registry, conversations=conversations,
        action_gate=action_gate, tracer=tracer, logger=logger,
    )
    # 5. The MCP client is the SINGLE chokepoint every tool call passes through; it
    #    enforces RBAC and routes side-effecting tools to the action gate. The
    #    supervisor does routing. The checkpointer (durable run state) only exists
    #    on the LangGraph engine — None otherwise.
    mcp = InProcessMCPClient(registry, action_gate=action_gate, logger=logger,
                             remote_executors=_build_remote_executors(settings, registry), settings=settings)
    supervisor = Supervisor(registry, llm, settings, embedder=getattr(rag, "embedder", None))
    checkpointer = build_checkpointer(settings) if settings.agent_engine == "langgraph" else None

    # The orchestrator ties it together: guardrails on input/output, the supervisor
    # for routing, conversations+summarizer for memory, and the chosen engine. This
    # is the object a request actually calls to run one chat turn.
    orchestrator = Orchestrator(
        settings=settings, registry=registry, deps=deps, mcp=mcp,
        # ``llm`` is handed to the input guard: the main model doubles as the
        # security judge (LLMJudgeGuard — no dedicated guard-model deployments).
        input_guard=build_input_guardrails(settings, llm=llm), output_guard=build_output_guardrails(settings),
        supervisor=supervisor, conversations=conversations, summarizer=summarizer,
        checkpointer=checkpointer,
    )

    # One structured "ready" line naming exactly which modules went live — the
    # fastest way to confirm a deployment's capability bundle at a glance.
    logger.info(
        "platform.ready",
        extra={"event": "ready", "cap_modules": ",".join(m.id for m in registry.modules() if m.enabled)},
    )
    # 6. Hand back the whole object graph. Note the audit logger is built inline
    #    here because nothing upstream needed it during wiring.
    return AppServices(
        settings=settings, registry=registry, deps=deps, mcp=mcp, orchestrator=orchestrator,
        supervisor=supervisor, conversations=conversations, summarizer=summarizer, llm=llm,
        rag=rag, action_gate=action_gate,
        input_guard=orchestrator.input_guard, output_guard=orchestrator.output_guard,
        tracer=tracer, metrics=metrics, logger=logger,
        audit=build_audit_logger(settings, logger),
        checkpointer=checkpointer,
    )


async def seed_demo(services: AppServices) -> None:
    """Convention-based dev seeding: call each enabled module's optional
    ``seed.seed_demo(deps)``. Keeps the core decoupled from feature modules.

    "Convention over configuration": the core never imports a feature by name. It
    just looks for ``app.capabilities.<id>.seed`` for each enabled module and, if
    that module happens to ship a ``seed_demo`` function, runs it. A module without
    one is simply skipped (the ModuleNotFoundError below). This is the same
    discovery pattern as the registry — features opt in, the core stays generic.
    Gated by ``seed_demo_data`` so it never populates demo corpora in prod.
    """
    if not services.settings.seed_demo_data:
        return
    for mod in services.registry.modules(include_disabled=False):
        try:
            seed_mod = importlib.import_module(f"app.capabilities.{mod.id}.seed")
        except ModuleNotFoundError:
            continue   # this module simply has no seeder — fine, move on
        fn = getattr(seed_mod, "seed_demo", None)
        if fn:
            try:
                await fn(services.deps)
                services.logger.info("seed.demo", extra={"event": "seed", "cap_module": mod.id})
            except Exception as exc:  # noqa: BLE001 - one module's seed failing must not abort boot
                services.logger.warning("seed.failed %s: %s", mod.id, exc)


__all__ = ["AppServices", "build_services", "seed_demo"]
