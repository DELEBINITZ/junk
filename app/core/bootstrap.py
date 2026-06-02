"""Composition root — wires the whole platform from config.

Used by the FastAPI lifespan and by tests/eval. Discovers capability modules,
builds core services, and assembles the orchestrator. The ONLY place that knows
how everything fits together; nothing here names a specific feature (modules are
discovered), keeping the core feature-agnostic.
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
from app.core.ingestion import IngestionService
from app.core.llm import build_llm
from app.core.mcp import InProcessMCPClient
from app.core.memory import RollingSummarizer, build_conversation_store, build_kg
from app.core.observability import build_tracer, configure_logging, get_logger, get_metrics
from app.core.observability.audit import build_audit_logger
from app.core.rag import build_rag
from app.core.registry import CapabilityRegistry
from app.core.security.tokens import build_revocation_store
from app.core.security.users import build_default_user_store


@dataclass
class AppServices:
    settings: Settings
    registry: CapabilityRegistry
    deps: CoreDeps
    mcp: InProcessMCPClient
    orchestrator: Orchestrator
    supervisor: Supervisor
    conversations: Any
    summarizer: RollingSummarizer
    llm: Any
    rag: Any
    kg: Any
    action_gate: ActionGate
    input_guard: Any
    output_guard: Any
    tracer: Any
    metrics: Any
    logger: Any
    revocation_store: Any
    user_store: Any
    ingestion: Any = None
    audit: Any = None
    checkpointer: Any = None

    async def aclose(self) -> None:
        for c in (self.llm, self.rag, self.conversations, self.kg, self.mcp, self.tracer):
            close = getattr(c, "aclose", None)
            if close:
                try:
                    await close()
                except Exception:
                    pass


def build_services(settings: Settings) -> AppServices:
    configure_logging(settings.log_level, settings.log_json)
    logger = get_logger("asi")
    metrics = get_metrics(settings)
    tracer = build_tracer(settings)

    registry = CapabilityRegistry().discover(settings)
    llm = build_llm(settings)
    rag = build_rag(settings)
    kg = build_kg(settings)
    conversations = build_conversation_store(settings)
    summarizer = RollingSummarizer(llm)
    action_gate = build_action_gate(settings)

    deps = CoreDeps(
        settings=settings, llm=llm, rag=rag, registry=registry, conversations=conversations,
        kg=kg, action_gate=action_gate, tracer=tracer, logger=logger,
    )
    mcp = InProcessMCPClient(registry, action_gate=action_gate, logger=logger)
    supervisor = Supervisor(registry, llm, settings)
    checkpointer = build_checkpointer(settings) if settings.agent_engine == "langgraph" else None

    orchestrator = Orchestrator(
        settings=settings, registry=registry, deps=deps, mcp=mcp,
        input_guard=build_input_guardrails(settings), output_guard=build_output_guardrails(settings),
        supervisor=supervisor, conversations=conversations, summarizer=summarizer,
        kg=kg, checkpointer=checkpointer,
    )

    logger.info(
        "platform.ready",
        extra={"event": "ready", "cap_modules": ",".join(m.id for m in registry.modules() if m.enabled)},
    )
    return AppServices(
        settings=settings, registry=registry, deps=deps, mcp=mcp, orchestrator=orchestrator,
        supervisor=supervisor, conversations=conversations, summarizer=summarizer, llm=llm,
        rag=rag, kg=kg, action_gate=action_gate,
        input_guard=orchestrator.input_guard, output_guard=orchestrator.output_guard,
        tracer=tracer, metrics=metrics, logger=logger,
        revocation_store=build_revocation_store(settings),
        user_store=build_default_user_store(),
        ingestion=IngestionService(deps),
        audit=build_audit_logger(settings, logger),
        checkpointer=checkpointer,
    )


async def seed_demo(services: AppServices) -> None:
    """Convention-based dev seeding: call each enabled module's optional
    ``seed.seed_demo(deps)``. Keeps the core decoupled from feature modules."""
    if not services.settings.seed_demo_data:
        return
    for mod in services.registry.modules(include_disabled=False):
        try:
            seed_mod = importlib.import_module(f"app.capabilities.{mod.id}.seed")
        except ModuleNotFoundError:
            continue
        fn = getattr(seed_mod, "seed_demo", None)
        if fn:
            try:
                await fn(services.deps)
                services.logger.info("seed.demo", extra={"event": "seed", "cap_module": mod.id})
            except Exception as exc:  # noqa: BLE001
                services.logger.warning("seed.failed %s: %s", mod.id, exc)


__all__ = ["AppServices", "build_services", "seed_demo"]
