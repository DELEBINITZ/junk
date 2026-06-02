"""Frozen capability contracts — the stable interfaces every module implements.

This module is the boundary between the platform CORE and drop-in capability
MODULES (reports today; easm/brand/aci later). Modules depend on these types;
the core never imports a module by path. See PRODUCTION_AGENTIC_PLAN.txt §5-§6.

Key distinction:
  * ToolException — RAISED inside a handler for an expected failure.
  * ToolError     — a VALUE returned to the agent (errors-as-data). Tool.run()
                    converts a raised ToolException into a ToolError value so the
                    agent never sees an exception.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Protocol, runtime_checkable

from app.domain import User


CORE_VERSION = "1.0.0"

_ROLE_RANK: dict[str, int] = {"viewer": 1, "analyst": 2, "admin": 3}


def role_allows(user_role: str, min_role: str) -> bool:
    """Coarse role gate: does `user_role` meet the minimum `min_role`?"""

    return _ROLE_RANK.get(user_role, 0) >= _ROLE_RANK.get(min_role, 99)


def _version_tuple(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(part) for part in value.split("."))
    except ValueError:
        return (0,)


class ManifestError(Exception):
    """Raised when a capability manifest is invalid (fail-fast at boot)."""


class ToolException(Exception):
    """Raised inside a tool handler for an expected, user-facing failure.

    Tool.run() catches this and returns a ToolError value instead of letting it
    propagate, so the agent always sees errors-as-data. Handlers that wrap legacy
    code translate the legacy error into this type.
    """

    def __init__(self, message: str, code: int = 400, kind: str = "tool_error"):
        super().__init__(message)
        self.code = code
        self.kind = kind


class Autonomy(str, Enum):
    READ = "read"        # answers / recommends only
    SUGGEST = "suggest"  # drafts an action a human approves
    AUTO = "auto"        # deterministic, proven, policy-bound auto-action


@dataclass(slots=True)
class RoutingHint:
    """How the supervisor/router decides a query belongs to this module."""

    intents: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ToolContext:
    """Trusted, per-call context.

    org_id and user are derived from the verified JWT, NEVER from tool arguments.
    A tool reads everything it needs from here and touches no global state, which
    keeps it tenant-safe, parallel-safe, and unit-testable in isolation.
    """

    org_id: str
    user: User
    trace_id: str
    store: Any = None   # repository handle (DataStore today; abstracted later)
    kg: Any = None      # knowledge-graph handle (later increment)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ToolResult:
    data: dict[str, Any]
    citations: list[str] = field(default_factory=list)
    ok: bool = True


@dataclass(slots=True)
class ToolError:
    """Errors-as-data: returned (not raised) so the planner can adapt."""

    kind: str
    message: str
    code: int = 400
    suggestions: list[str] = field(default_factory=list)
    ok: bool = False


def is_error(value: Any) -> bool:
    return getattr(value, "ok", True) is False


@dataclass
class Tool:
    """A typed, MCP-exposable capability function.

    `input_schema` is a JSON Schema (the same shape MCP `tools/list` returns).
    `handler(args, ctx)` does the work and returns a raw dict; it may raise
    ToolException for expected failures. `run()` enforces the coarse RBAC gate
    and converts any outcome into ToolResult | ToolError.
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[dict[str, Any], ToolContext], dict[str, Any]]
    side_effecting: bool = False
    rbac_role: str = "viewer"
    returns_schema: dict[str, Any] | None = None

    def mcp_definition(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult | ToolError:
        if not role_allows(ctx.user.role, self.rbac_role):
            return ToolError(
                kind="forbidden",
                message=f"Role '{ctx.user.role}' may not call '{self.name}'",
                code=403,
            )
        try:
            data = self.handler(args or {}, ctx)
        except ToolException as exc:
            return ToolError(kind=exc.kind, message=str(exc), code=exc.code)
        except TypeError as exc:
            return ToolError(kind="invalid_arguments", message=str(exc), code=400)
        except Exception as exc:  # never raise to the agent
            return ToolError(kind="internal", message=f"{type(exc).__name__}: {exc}", code=500)
        return ToolResult(data=dict(data) if isinstance(data, dict) else {"result": data})


@runtime_checkable
class Specialist(Protocol):
    """A per-module sub-agent. Most modules use GenericSpecialist instead."""

    id: str

    def build_subgraph(self, deps: Any) -> Any: ...


@dataclass
class GenericSpecialist:
    """The default specialist: the proven plan->tool->ground->answer loop,
    parameterized by a module's tools + prompt.

    Increment 1: a holder that exposes the module's tools/prompt to the router.
    The compiled LangGraph subgraph lands in a later increment.
    """

    id: str
    tools: list[Tool]
    system_prompt: str = ""

    @property
    def tool_names(self) -> list[str]:
        return [tool.name for tool in self.tools]

    def build_subgraph(self, deps: Any) -> Any:  # pragma: no cover - placeholder
        raise NotImplementedError("LangGraph subgraph is wired in a later increment")


@runtime_checkable
class Retriever(Protocol):
    """Binds ONE corpus/collection to the shared hybrid+filter+rerank pipeline.

    This is how 'RAG over reports' generalizes to 'RAG over any corpus' — a new
    data type is a new Retriever + collection, not an agent change.
    """

    id: str

    def retrieve(self, query: str, filters: dict[str, Any], ctx: ToolContext) -> list[Any]: ...


@runtime_checkable
class IngestionConnector(Protocol):
    """Normalizes a source event and writes it to the vector store + KG."""

    id: str
    source: str

    def handle(self, event: Any, sinks: Any) -> Any: ...


@dataclass(slots=True)
class NodeType:
    label: str
    keys: list[str] = field(default_factory=list)


@dataclass(slots=True)
class EdgeType:
    source: str
    relation: str
    target: str


@dataclass(slots=True)
class OntologyContribution:
    """The entity/edge types a module adds to the shared knowledge graph."""

    nodes: list[NodeType] = field(default_factory=list)
    edges: list[EdgeType] = field(default_factory=list)


@runtime_checkable
class ActionHandler(Protocol):
    """A gated side-effect. The approval gate is CORE; modules supply only
    preview() and execute(). Deferred for v1 (read-only) but defined so the
    action layer slots in without core changes.
    """

    action_type: str
    autonomy: Autonomy

    def preview(self, args: dict[str, Any], ctx: ToolContext) -> Any: ...

    def execute(self, args: dict[str, Any], ctx: ToolContext) -> Any: ...


@dataclass
class CapabilityManifest:
    """Declarative wiring for one feature. The platform derives routing, the
    tool registry, RBAC, and autonomy from this — never from hardcoded logic."""

    id: str
    version: str
    display_name: str
    tools: list[Tool] = field(default_factory=list)
    routing_hints: list[RoutingHint] = field(default_factory=list)
    license_tiers: list[str] = field(default_factory=lambda: ["platform"])
    enabled_flag: str = ""
    specialist: Specialist | GenericSpecialist | None = None
    system_prompt: str = ""
    retrievers: list[Any] = field(default_factory=list)
    ingestion: list[Any] = field(default_factory=list)
    ontology: OntologyContribution | None = None
    default_autonomy: Autonomy = Autonomy.READ
    rbac: dict[str, str] = field(default_factory=dict)
    owners: list[str] = field(default_factory=list)
    min_core_version: str = "1.0.0"
    # Deployment composition dial (plan §5.4). False = discovered but NOT loaded
    # unless CAP_<ID>_ENABLED=true. Lets a module live in the repo behind a flag
    # (ship dark -> canary -> promote) without affecting default behavior.
    enabled_by_default: bool = True

    @property
    def tool_names(self) -> list[str]:
        return [tool.name for tool in self.tools]

    def validate(self) -> None:
        if not self.id or not self.id.isidentifier():
            raise ManifestError(f"Invalid module id: {self.id!r}")
        if _version_tuple(self.min_core_version) > _version_tuple(CORE_VERSION):
            raise ManifestError(
                f"Module '{self.id}' needs core >= {self.min_core_version}, have {CORE_VERSION}"
            )
        names = self.tool_names
        if len(names) != len(set(names)):
            raise ManifestError(f"Module '{self.id}' has duplicate tool names")
        for tool_name in self.rbac:
            if tool_name not in names:
                raise ManifestError(
                    f"Module '{self.id}' rbac references unknown tool '{tool_name}'"
                )
        if not self.routing_hints:
            raise ManifestError(f"Module '{self.id}' must declare at least one routing hint")
