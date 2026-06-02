"""Frozen contracts: the entire stable API surface between the core platform and
capability modules.

Nothing in ``app/core`` (except this file) should be edited to add a feature. A
capability module implements the small set of Protocols / dataclasses declared
here and registers a :class:`CapabilityManifest`; the platform discovers it and
wires it automatically. These types are SemVer'd (see ``CONTRACTS_VERSION``):
arg/return schemas are additive, breaking changes are a major bump.

Design rules encoded here:
  * Tools never raise to the agent -> failures are returned as :class:`ToolError`
    (errors-as-data).
  * Every tool carries ``rbac_role`` (minimum role) and ``side_effecting`` (True
    => the core action gate must approve before ``execute``).
  * Tenant identity lives in :class:`ToolContext`, never in tool args, so a tool
    cannot be tricked into crossing orgs.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Protocol,
    runtime_checkable,
)

from pydantic import BaseModel, Field

# Bump on any change to the interfaces below. Modules declare ``min_core_version``.
CONTRACTS_VERSION = "1.0.0"


# --------------------------------------------------------------------------- #
# Roles & autonomy
# --------------------------------------------------------------------------- #
class Role(str, Enum):
    """Ordered RBAC roles. Higher index => strictly more privilege."""

    VIEWER = "viewer"
    ANALYST = "analyst"
    ADMIN = "admin"


_ROLE_ORDER: dict[str, int] = {Role.VIEWER.value: 0, Role.ANALYST.value: 1, Role.ADMIN.value: 2}


def role_satisfies(held: Iterable[str], required: str) -> bool:
    """True if any held role meets or exceeds ``required`` in the role order."""
    need = _ROLE_ORDER.get(required, 0)
    return any(_ROLE_ORDER.get(str(r), -1) >= need for r in held)


class Autonomy(str, Enum):
    """How much a capability may act. See blueprint §10 (two-layer autonomy)."""

    READ = "read"        # answers / recommends only
    SUGGEST = "suggest"  # drafts an action a human approves (gated)
    AUTO = "auto"        # deterministic policy, promoted only after proven precision


# --------------------------------------------------------------------------- #
# Retrieval primitives
# --------------------------------------------------------------------------- #
class Citation(BaseModel):
    """A grounded reference attached to an answer or tool result."""

    doc_id: str
    source: str = ""                 # e.g. "reports", "easm"
    title: str = ""
    section: str = ""
    snippet: str = ""
    score: float = 0.0
    url: str | None = None
    published_at: str | None = None  # ISO-8601 when known (freshness)


class Chunk(BaseModel):
    """A retrieved unit of text with provenance. ``org_id`` is mandatory and is
    set from the trusted context at ingest/retrieve time, never from user input.
    """

    id: str
    text: str
    score: float = 0.0
    org_id: str
    source: str = ""
    doc_id: str = ""
    title: str = ""
    section: str = ""
    published_at: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_citation(self) -> Citation:
        return Citation(
            doc_id=self.doc_id or self.id,
            source=self.source,
            title=self.title,
            section=self.section,
            snippet=self.text[:280],
            score=self.score,
            published_at=self.published_at,
        )


# --------------------------------------------------------------------------- #
# Tool outcomes (errors-as-data)
# --------------------------------------------------------------------------- #
class ToolResult(BaseModel):
    ok: bool = True
    data: dict[str, Any] = Field(default_factory=dict)
    citations: list[Citation] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)


class ToolError(BaseModel):
    ok: bool = False
    code: str
    message: str
    retriable: bool = False
    details: dict[str, Any] = Field(default_factory=dict)


ToolOutcome = ToolResult | ToolError


# --------------------------------------------------------------------------- #
# Tool context (trusted, per-request)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ToolContext:
    """Carried into every tool/retriever call. Holds the *trusted* identity
    derived from the verified token plus handles to core services. A tool must
    reach for nothing global; everything it needs is here."""

    org_id: str
    user_id: str
    roles: tuple[str, ...]
    trace_id: str
    request_id: str
    deps: CoreDeps
    extra: Mapping[str, Any] = field(default_factory=dict)

    def has_role(self, minimum: str) -> bool:
        return role_satisfies(self.roles, minimum)


# --------------------------------------------------------------------------- #
# Tool (the universal unit of capability)
# --------------------------------------------------------------------------- #
ToolHandler = Callable[[BaseModel, ToolContext], ToolOutcome | Awaitable[ToolOutcome]]


@dataclass(frozen=True)
class Tool:
    """A typed, MCP-exposable function the agent can call.

    ``handler`` may be sync or async and receives a validated args model plus the
    :class:`ToolContext`. RBAC and the action gate are enforced by the *tool
    runner* (the MCP client), not here, so this stays a pure capability wrapper —
    but :meth:`invoke` still guarantees errors-as-data.
    """

    name: str
    description: str
    handler: ToolHandler
    args_schema: type[BaseModel]
    returns_schema: type[BaseModel] | None = None
    side_effecting: bool = False
    rbac_role: str = Role.VIEWER.value
    autonomy: Autonomy = Autonomy.READ
    module_id: str = ""  # stamped by the registry at load time

    async def invoke(self, raw_args: Mapping[str, Any], ctx: ToolContext) -> ToolOutcome:
        try:
            args = self.args_schema.model_validate(dict(raw_args or {}))
        except Exception as exc:  # validation error -> data, not a raise
            return ToolError(code="invalid_args", message=str(exc), details={"tool": self.name})
        try:
            out = self.handler(args, ctx)
            if inspect.isawaitable(out):
                out = await out
            if not isinstance(out, (ToolResult, ToolError)):
                return ToolError(
                    code="bad_tool_return",
                    message=f"tool {self.name} returned {type(out).__name__}, expected ToolResult/ToolError",
                )
            return out
        except Exception as exc:  # never let a tool raise into the agent loop
            return ToolError(
                code="tool_exception",
                message=f"{type(exc).__name__}: {exc}",
                retriable=False,
                details={"tool": self.name},
            )

    def json_schema(self) -> dict[str, Any]:
        """OpenAI/MCP-style tool advertisement (name + description + params)."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.args_schema.model_json_schema(),
            "side_effecting": self.side_effecting,
            "rbac_role": self.rbac_role,
            "module": self.module_id,
        }


def tool(
    name: str,
    description: str,
    args_schema: type[BaseModel],
    *,
    returns_schema: type[BaseModel] | None = None,
    side_effecting: bool = False,
    rbac_role: str = Role.VIEWER.value,
    autonomy: Autonomy = Autonomy.READ,
) -> Callable[[ToolHandler], Tool]:
    """Decorator to declare a :class:`Tool` from a handler function."""

    def decorate(fn: ToolHandler) -> Tool:
        return Tool(
            name=name,
            description=description,
            handler=fn,
            args_schema=args_schema,
            returns_schema=returns_schema,
            side_effecting=side_effecting,
            rbac_role=rbac_role,
            autonomy=autonomy,
        )

    return decorate


# --------------------------------------------------------------------------- #
# Retriever — a corpus binding over the shared RAG pipeline
# --------------------------------------------------------------------------- #
@runtime_checkable
class Retriever(Protocol):
    id: str

    async def retrieve(
        self, query: str, filters: Mapping[str, Any], ctx: ToolContext
    ) -> list[Chunk]: ...


# --------------------------------------------------------------------------- #
# Specialist — a per-module sub-agent.
#
# The supervisor dispatches one specialist PER routed module, in parallel. Each
# specialist is scoped to ITS OWN module's tools — so tool schemas never
# co-locate in one context. It *investigates* (retrieves + calls its tools) and
# returns findings; a single synthesize step joins findings across modules and
# answers. This is the property that lets the platform scale to many modules /
# hundreds of tools without bloating any one agent's context.
# --------------------------------------------------------------------------- #
class SpecialistResult(BaseModel):
    module_id: str
    chunks: list[Chunk] = Field(default_factory=list)   # findings (become cited context)
    events: list[dict[str, Any]] = Field(default_factory=list)  # tool-call trace
    summary: str = ""                                   # optional per-pillar gist
    error: str = ""


@runtime_checkable
class Specialist(Protocol):
    id: str

    async def investigate(self, question: str, ctx: ToolContext) -> SpecialistResult: ...


# Builds a Specialist for one module. Called as ``factory(module, deps, mcp)``.
# ``None`` in a manifest means "use the generic specialist". Typed loosely to
# avoid importing the registry/MCP types here (keeps contracts cycle-free).
SpecialistFactory = Callable[..., Specialist]


# --------------------------------------------------------------------------- #
# Ingestion — how a module's data lands (event-driven; batch = one-shot)
# --------------------------------------------------------------------------- #
class SourceEvent(BaseModel):
    source: str
    org_id: str
    payload: dict[str, Any] = Field(default_factory=dict)


class IngestStats(BaseModel):
    documents: int = 0
    chunks: int = 0
    errors: int = 0


@runtime_checkable
class Sinks(Protocol):
    """Where a connector writes normalized data."""

    async def write_chunks(self, chunks: Sequence[Chunk]) -> None: ...
    async def write_graph(self, org_id: str, nodes: Sequence[dict], edges: Sequence[dict]) -> None: ...


@runtime_checkable
class IngestionConnector(Protocol):
    id: str
    source: str

    async def handle(self, event: SourceEvent, sinks: Sinks) -> IngestStats: ...


# --------------------------------------------------------------------------- #
# Ontology — the KG slice a module contributes
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class NodeType:
    name: str
    keys: tuple[str, ...]


@dataclass(frozen=True)
class EdgeType:
    src: str
    relation: str
    dst: str


@dataclass(frozen=True)
class OntologyContribution:
    nodes: tuple[NodeType, ...] = ()
    edges: tuple[EdgeType, ...] = ()


# --------------------------------------------------------------------------- #
# Action handler — gated side effects (interface now, handlers later)
# --------------------------------------------------------------------------- #
class ActionPreview(BaseModel):
    action_type: str
    summary: str
    args: dict[str, Any] = Field(default_factory=dict)
    reversible: bool = True
    blast_radius: str = "low"


class ActionResult(BaseModel):
    action_type: str
    status: str  # "executed" | "failed" | "skipped"
    detail: str = ""
    audit_ref: str | None = None


@runtime_checkable
class ActionHandler(Protocol):
    action_type: str
    autonomy: Autonomy

    async def preview(self, args: Mapping[str, Any], ctx: ToolContext) -> ActionPreview: ...
    async def execute(self, args: Mapping[str, Any], ctx: ToolContext) -> ActionResult: ...


# --------------------------------------------------------------------------- #
# Routing & manifest (declarative wiring)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RoutingHint:
    intents: tuple[str, ...]
    examples: tuple[str, ...] = ()


@dataclass(frozen=True)
class CapabilityManifest:
    """Everything the platform needs to wire one feature. Read at boot; the
    supervisor's routing, each tenant's visible tools, RBAC and autonomy are all
    *derived* from manifests, never hardcoded in core."""

    id: str
    version: str
    display_name: str
    description: str = ""
    license_tiers: tuple[str, ...] = ("platform",)
    enabled_flag: str = ""          # Settings attribute name; "" => always enabled
    enabled_default: bool = True

    tools: tuple[Tool, ...] = ()
    retrievers: tuple[Retriever, ...] = ()
    specialist: SpecialistFactory | None = None  # None => generic specialist
    system_prompt: str = ""         # path relative to the module dir
    routing_hints: tuple[RoutingHint, ...] = ()
    default_autonomy: Autonomy = Autonomy.READ
    rbac: Mapping[str, str] = field(default_factory=dict)  # tool_name -> min role
    ontology: OntologyContribution | None = None
    ingestion: tuple[IngestionConnector, ...] = ()
    action_handlers: tuple[ActionHandler, ...] = ()

    min_core_version: str = "1.0.0"
    owners: tuple[str, ...] = ()


# --------------------------------------------------------------------------- #
# CoreDeps — the service bundle injected into tools/specialists
# --------------------------------------------------------------------------- #
@dataclass
class CoreDeps:
    """Handles to the frozen core services. Typed as ``Any`` to keep this module
    import-cycle-free; concrete types live in their own packages."""

    settings: Any
    llm: Any            # app.core.llm.lanes.LaneRouter
    rag: Any            # app.core.rag.pipeline.RetrievalPipeline
    registry: Any       # app.core.registry.CapabilityRegistry
    conversations: Any  # session store
    kg: Any             # app.core.memory.kg.KnowledgeGraph (NoOp default)
    action_gate: Any    # app.core.action_gate.gate.ActionGate
    tracer: Any         # observability tracer (NoOp default)
    logger: Any


__all__ = [
    "CONTRACTS_VERSION",
    "Role",
    "role_satisfies",
    "Autonomy",
    "Citation",
    "Chunk",
    "ToolResult",
    "ToolError",
    "ToolOutcome",
    "ToolContext",
    "Tool",
    "ToolHandler",
    "tool",
    "Retriever",
    "Specialist",
    "SpecialistResult",
    "SpecialistFactory",
    "SourceEvent",
    "IngestStats",
    "Sinks",
    "IngestionConnector",
    "NodeType",
    "EdgeType",
    "OntologyContribution",
    "ActionPreview",
    "ActionResult",
    "ActionHandler",
    "RoutingHint",
    "CapabilityManifest",
    "CoreDeps",
]
