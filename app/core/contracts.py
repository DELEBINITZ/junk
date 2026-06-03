"""Frozen CONTRACTS: the entire stable API surface between the core platform and
the capability modules.

================================ MENTAL MODEL ==============================
Think of the system as a games console (the "core") and game cartridges (the
"capability modules": reports, easm, brand, aci, ...). THIS FILE is the shape of
the cartridge slot. The console never changes to add a game; a cartridge just has
to fit the slot. Concretely: nothing in ``app/core`` (except this file) should be
edited to add a feature. A module implements the small set of Protocols /
dataclasses declared here and registers a :class:`CapabilityManifest`; the
platform discovers it and wires it automatically.

These types are SemVer'd (see ``CONTRACTS_VERSION``): adding optional fields is
backward-compatible; removing/renaming is a breaking (major) change.

THREE SAFETY RULES ARE ENCODED HERE — internalize these, they recur everywhere:
  1. Errors-as-data: a tool NEVER raises into the agent. Failures come back as a
     :class:`ToolError` value, so one bad tool can't crash a turn.
  2. Every tool carries ``rbac_role`` (minimum role to call it) and
     ``side_effecting`` (True => the human action gate must approve before it
     runs). The agent can't escalate privilege or fire actions on its own.
  3. Tenant identity lives in :class:`ToolContext`, derived from the verified
     token — NEVER in tool args. A tool cannot be tricked into crossing orgs.
===========================================================================
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

# Bump on ANY change to the interfaces below. Modules can declare the minimum
# core version they require (``CapabilityManifest.min_core_version``).
CONTRACTS_VERSION = "1.0.0"


# --------------------------------------------------------------------------- #
# Roles & autonomy
# --------------------------------------------------------------------------- #
class Role(str, Enum):
    """Ordered RBAC roles. Higher index => strictly more privilege. Inheriting
    from ``str`` too means a Role IS a string ("viewer"), so it serializes and
    compares naturally in JWTs/JSON."""

    VIEWER = "viewer"
    ANALYST = "analyst"
    ADMIN = "admin"


# The privilege ORDER, as numbers, so we can compare roles with >=.
_ROLE_ORDER: dict[str, int] = {Role.VIEWER.value: 0, Role.ANALYST.value: 1, Role.ADMIN.value: 2}


def role_satisfies(held: Iterable[str], required: str) -> bool:
    """True if ANY role the caller holds meets or exceeds ``required``. This one
    function is the whole RBAC check — used by the MCP boundary before every tool
    call and by the API role dependencies. An unknown held role scores -1 (never
    satisfies); an unknown requirement scores 0 (viewer-level)."""
    need = _ROLE_ORDER.get(required, 0)
    return any(_ROLE_ORDER.get(str(r), -1) >= need for r in held)


class Autonomy(str, Enum):
    """How much a capability is allowed to ACT (separate from who can call it).
    See the blueprint's two-layer autonomy model. Most things are READ today;
    SUGGEST drafts an action a human approves; AUTO is reserved for deterministic
    policies promoted only after their precision is proven in production."""

    READ = "read"        # answers / recommends only
    SUGGEST = "suggest"  # drafts an action a human approves (gated)
    AUTO = "auto"        # deterministic policy, promoted only after proven precision


# --------------------------------------------------------------------------- #
# Retrieval primitives
# --------------------------------------------------------------------------- #
class Citation(BaseModel):
    """A grounded reference attached to an answer or tool result. Citations are
    how the UI shows "this claim came from THIS source" and how the output
    guardrail checks the answer is actually backed by evidence."""

    doc_id: str
    source: str = ""                 # which module produced it, e.g. "reports", "easm"
    title: str = ""
    section: str = ""
    snippet: str = ""
    score: float = 0.0
    url: str | None = None
    published_at: str | None = None  # ISO-8601 when known (drives freshness/recency)


class Chunk(BaseModel):
    """A retrieved unit of text with provenance — the atom of RAG evidence that
    flows through the agent as ``context_chunks``. ``org_id`` is MANDATORY and is
    stamped from the trusted context at ingest/retrieve time, never from user
    input: it is the per-piece tenant tag that makes cross-org leakage impossible.
    """

    id: str
    text: str
    score: float = 0.0
    org_id: str                      # required: the owning tenant (isolation)
    source: str = ""
    doc_id: str = ""
    title: str = ""
    section: str = ""
    published_at: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_citation(self) -> Citation:
        """Project a chunk down to a Citation (the public-facing reference). The
        snippet is truncated to keep payloads small. Used by answer_node when it
        maps an answer's [n] marker back to the chunk that filled slot n."""
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
# Tool outcomes (errors-as-data)  <-- SAFETY RULE #1
# --------------------------------------------------------------------------- #
class ToolResult(BaseModel):
    """The SUCCESS shape returned by a tool. ``data`` is structured output,
    ``citations`` are evidence the answer can cite. ``ok=True`` lets callers
    branch on success/failure without try/except."""
    ok: bool = True
    data: dict[str, Any] = Field(default_factory=dict)
    citations: list[Citation] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)


class ToolError(BaseModel):
    """The FAILURE shape — a value, not an exception. ``code`` is machine-readable
    (e.g. "forbidden", "requires_approval", "invalid_args"); ``retriable`` tells
    the caller whether trying again could help. The agent receives this like any
    other result and keeps going."""
    ok: bool = False
    code: str
    message: str
    retriable: bool = False
    details: dict[str, Any] = Field(default_factory=dict)


# A tool call returns EITHER outcome. Callers check ``isinstance(out, ToolResult)``.
ToolOutcome = ToolResult | ToolError


# --------------------------------------------------------------------------- #
# Tool context (trusted, per-request)  <-- SAFETY RULE #3
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ToolContext:
    """Carried into EVERY tool/retriever call. Holds the *trusted* identity
    derived from the verified token, plus handles to core services. ``frozen=True``
    makes it immutable — a tool can't tamper with its own org/roles mid-call.

    Design rule: a tool must reach for nothing global. Everything it legitimately
    needs (who is asking, which org, the service deps) is right here. That keeps
    tools pure, testable, and unable to bypass tenant isolation."""

    org_id: str                  # THE tenant key — set from the token, never from args
    user_id: str
    roles: tuple[str, ...]
    trace_id: str
    request_id: str
    deps: CoreDeps               # the shared service bundle (see bottom of file)
    extra: Mapping[str, Any] = field(default_factory=dict)

    def has_role(self, minimum: str) -> bool:
        return role_satisfies(self.roles, minimum)


# --------------------------------------------------------------------------- #
# Tool (the universal unit of capability)
# --------------------------------------------------------------------------- #
# A tool's handler: receives a VALIDATED args model + the ToolContext, returns an
# outcome (sync or async). This is the function a module author actually writes.
ToolHandler = Callable[[BaseModel, ToolContext], ToolOutcome | Awaitable[ToolOutcome]]


@dataclass(frozen=True)
class Tool:
    """A typed, MCP-exposable function the agent can call — the universal unit of
    capability. Everything a module "can do" is a Tool.

    Note the separation of concerns: RBAC and the action gate are enforced by the
    TOOL RUNNER (the MCP client, mcp/inprocess.py), NOT in here — so this stays a
    clean capability wrapper. What :meth:`invoke` DOES guarantee is errors-as-data
    (safety rule #1): no matter how the handler misbehaves, the agent gets a
    ToolResult or ToolError back, never an exception.
    """

    name: str
    description: str                              # the LLM reads this to decide when to call it
    handler: ToolHandler
    args_schema: type[BaseModel]                  # pydantic model => validation + JSON schema for free
    returns_schema: type[BaseModel] | None = None
    side_effecting: bool = False                  # True => must pass the human action gate (rule #2)
    rbac_role: str = Role.VIEWER.value            # minimum role to call it (rule #2)
    autonomy: Autonomy = Autonomy.READ
    # If False, the heuristic specialist will NOT auto-invoke this tool while
    # gathering (it's reserved for the LLM/planner to call deliberately). Use it for
    # a tool that overlaps a bound retriever (e.g. a RAG "search" tool) so the two
    # don't both fire on every turn. The tool is still advertised to the LLM and
    # callable through the MCP boundary like any other.
    auto_invoke: bool = True
    module_id: str = ""  # stamped by the registry at load time (which module owns this tool)

    async def invoke(self, raw_args: Mapping[str, Any], ctx: ToolContext) -> ToolOutcome:
        """Validate args, run the handler, and GUARANTEE an outcome value. Three
        failure modes are all turned into ToolError instead of raising:
          * args don't match the schema      -> code="invalid_args"
          * handler returns the wrong type   -> code="bad_tool_return"
          * handler raises any exception      -> code="tool_exception"
        This is the concrete implementation of "errors-as-data"."""
        try:
            args = self.args_schema.model_validate(dict(raw_args or {}))
        except Exception as exc:  # validation error -> data, not a raise
            return ToolError(code="invalid_args", message=str(exc), details={"tool": self.name})
        try:
            out = self.handler(args, ctx)
            if inspect.isawaitable(out):          # handler may be sync OR async
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
        """Advertise this tool in OpenAI/MCP function-calling format (name +
        description + JSON-schema params). This is exactly what gets handed to the
        LLM so it knows the tool exists and how to call it."""
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
    auto_invoke: bool = True,
) -> Callable[[ToolHandler], Tool]:
    """Decorator that turns a plain async handler function into a :class:`Tool`.
    This is the ergonomic way module authors declare tools — see any module's
    code, e.g. ``@tool(name="get_threat_actors", ...)`` over an async def."""

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
            auto_invoke=auto_invoke,
        )

    return decorate


# --------------------------------------------------------------------------- #
# Retriever — a corpus binding over the shared RAG pipeline
# --------------------------------------------------------------------------- #
@runtime_checkable
class Retriever(Protocol):
    """A module's binding to a document corpus. ``retrieve`` returns Chunks for a
    query, scoped to the caller's org (via ``ctx``). A Protocol means a module
    can supply ANYTHING with this shape — duck typing, no base class to inherit.
    ``@runtime_checkable`` lets ``isinstance`` checks work against it."""
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
# hundreds of tools without bloating any one agent's context. See specialist.py.
# --------------------------------------------------------------------------- #
class SpecialistResult(BaseModel):
    """What one specialist hands back to the dispatch node. The ``chunks`` become
    cited context; ``events`` are the tool-call trace for observability."""
    module_id: str
    chunks: list[Chunk] = Field(default_factory=list)   # findings (become cited context)
    events: list[dict[str, Any]] = Field(default_factory=list)  # tool-call trace
    summary: str = ""                                   # optional per-pillar gist
    error: str = ""


@runtime_checkable
class Specialist(Protocol):
    """The interface a specialist implements. Just one method: investigate a
    question for one module and return findings. GenericSpecialist (specialist.py)
    is the default; a module can ship a custom one via its manifest."""
    id: str

    async def investigate(self, question: str, ctx: ToolContext) -> SpecialistResult: ...


# Builds a Specialist for one module. Called as ``factory(module, deps, mcp)``.
# ``None`` in a manifest means "use the generic specialist". Typed loosely
# (``Callable[..., Specialist]``) to avoid importing registry/MCP types here,
# which would create an import cycle — contracts must stay dependency-free.
SpecialistFactory = Callable[..., Specialist]


# --------------------------------------------------------------------------- #
# Ingestion — how a module's data lands (event-driven; batch = one-shot)
# --------------------------------------------------------------------------- #
class SourceEvent(BaseModel):
    """An incoming data event for ingestion (e.g. "new report for org X"). The
    chat path doesn't use these — ingestion runs off to the side (external cron /
    event bus) and writes Chunks the retrievers later read."""
    source: str
    org_id: str
    payload: dict[str, Any] = Field(default_factory=dict)


class IngestStats(BaseModel):
    documents: int = 0
    chunks: int = 0
    errors: int = 0


@runtime_checkable
class Sinks(Protocol):
    """Where an ingestion connector WRITES normalized data — the vector store
    (chunks) and the knowledge graph (nodes/edges). Abstracted so connectors
    don't know which concrete backend is wired."""

    async def write_chunks(self, chunks: Sequence[Chunk]) -> None: ...
    async def write_graph(self, org_id: str, nodes: Sequence[dict], edges: Sequence[dict]) -> None: ...


@runtime_checkable
class IngestionConnector(Protocol):
    """Parses a SourceEvent for one source into chunks/graph and writes them to
    the Sinks. A module contributes connectors via its manifest."""
    id: str
    source: str

    async def handle(self, event: SourceEvent, sinks: Sinks) -> IngestStats: ...


# --------------------------------------------------------------------------- #
# Ontology — the KG slice a module contributes
# --------------------------------------------------------------------------- #
# These describe the entity/relationship TYPES a module adds to the shared
# knowledge graph (e.g. easm contributes Asset nodes; aci contributes ThreatActor
# nodes and a "weaponizes" edge). The KG join across modules is a future feature;
# the types are declared now so modules can carry their slice.
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
    """A human-readable description of an action BEFORE it runs — what the
    approval inbox shows a reviewer. ``reversible`` / ``blast_radius`` help the
    reviewer judge risk."""
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
    """Executes a side-effecting action AFTER approval. Two phases: ``preview``
    (describe it for the human) and ``execute`` (do it). The interface ships now;
    concrete handlers come later. This is the other half of the action gate —
    side_effecting tools route here instead of running inline."""
    action_type: str
    autonomy: Autonomy

    async def preview(self, args: Mapping[str, Any], ctx: ToolContext) -> ActionPreview: ...
    async def execute(self, args: Mapping[str, Any], ctx: ToolContext) -> ActionResult: ...


# --------------------------------------------------------------------------- #
# Manifest (declarative wiring)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CapabilityManifest:
    """The CARTRIDGE: everything the platform needs to wire one feature. Read once
    at boot; the supervisor's routing, each tenant's visible tools, RBAC, and
    autonomy are all DERIVED from manifests — never hardcoded in core. Adding a
    module = writing one of these + its tools. That's the extension story.

    Every field maps to a core subsystem:
      tools/retrievers/specialist -> the agent & RAG          description+tools -> supervisor
      rbac/default_autonomy       -> the MCP boundary & gate  ontology          -> the KG
      ingestion/action_handlers   -> ingestion & action gate  enabled_flag      -> deployment

    ROUTING is DYNAMIC: the supervisor/planner decide which module(s) answer a
    question by MEANING — embedding similarity over each module's ``display_name``
    + ``description`` + tool names/descriptions (deterministic, offline) or an LLM
    router. There are no hand-maintained routing keywords: a vague query whose
    intent matches a module is still routed correctly. So a module is routable the
    moment it has a clear ``description`` and well-described tools.
    """

    id: str
    version: str
    display_name: str
    description: str = ""
    license_tiers: tuple[str, ...] = ("platform",)
    enabled_flag: str = ""          # name of a Settings attr; "" => always enabled
    enabled_default: bool = True

    tools: tuple[Tool, ...] = ()
    retrievers: tuple[Retriever, ...] = ()
    specialist: SpecialistFactory | None = None  # None => use the generic specialist
    system_prompt: str = ""         # path (relative to the module dir) to a prompt file
    default_autonomy: Autonomy = Autonomy.READ
    rbac: Mapping[str, str] = field(default_factory=dict)  # tool_name -> min role override
    ontology: OntologyContribution | None = None
    ingestion: tuple[IngestionConnector, ...] = ()
    action_handlers: tuple[ActionHandler, ...] = ()

    min_core_version: str = "1.0.0"   # the registry checks this against CONTRACTS_VERSION
    owners: tuple[str, ...] = ()


# --------------------------------------------------------------------------- #
# CoreDeps — the service bundle injected into tools/specialists
# --------------------------------------------------------------------------- #
@dataclass
class CoreDeps:
    """The bag of shared core services, built ONCE at boot (bootstrap.py) and
    handed to tools/specialists via ToolContext.deps. Typed as ``Any`` on purpose
    to keep this contracts module import-cycle-free — the concrete classes live in
    their own packages and would import back from here. This is hand-rolled
    dependency injection: nothing constructs its own services, everything receives
    them, which is what makes the whole app config-driven and testable."""

    settings: Any
    llm: Any            # app.core.llm.lanes.LaneRouter  (the 3-lane LLM handle)
    rag: Any            # app.core.rag.pipeline.RetrievalPipeline
    registry: Any       # app.core.registry.CapabilityRegistry
    conversations: Any  # session/message store
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
    "CapabilityManifest",
    "CoreDeps",
]
