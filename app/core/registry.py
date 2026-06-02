"""Capability registry — the heart of the extensibility chassis.

MENTAL MODEL: the console-and-cartridges idea from contracts.py made concrete.
This is the slot reader. At boot it SCANS ``app/capabilities/*/manifest.py``,
validates each cartridge against the core contract, and builds an index of what
the platform can do. From then on the registry is the authority that answers two
different questions:

  * "what is installed?"  -> every loaded module, its tools, retrievers, ontology
    (used to wire the system: routing, ingestion, the KG).
  * "what may THIS caller see?"  -> :meth:`capability_view`, a per-(org, roles)
    slice filtered by entitlement/license AND RBAC. This is the security-relevant
    one: two callers hit the same registry and get DIFFERENT visible tool sets.

Everything is COMPUTED from manifests, never hardcoded — the supervisor's routing
hints, each org's visible tools, RBAC role requirements, and the merged ontology
all derive from what modules declared. Adding a feature = dropping in a module
dir; no core edit. (Blueprint file 15.)
"""

from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass, field
from pathlib import Path

from app.core.contracts import (
    CONTRACTS_VERSION,
    CapabilityManifest,
    OntologyContribution,
    Retriever,
    RoutingHint,
    Tool,
    role_satisfies,
)
from app.core.errors import RegistryError
from app.core.security.context import SecurityContext

# The Python package the registry scans for capability modules. Each subpackage
# with a ``manifest.py`` is treated as one installable feature ("cartridge").
CAPABILITIES_PACKAGE = "app.capabilities"


def _version_tuple(v: str) -> tuple[int, ...]:
    # Parse a "1.2.3" SemVer string into (1, 2, 3) so versions compare numerically
    # (used to check a module's min_core_version against the core's CONTRACTS_VERSION).
    # Anything unparseable degrades to (0,) — i.e. "very old", which never blocks.
    try:
        return tuple(int(x) for x in v.split("."))
    except ValueError:
        return (0,)


@dataclass
class RegisteredModule:
    """A capability module AFTER it has been loaded and validated — the registry's
    in-memory record of one installed feature. ``enabled`` reflects this
    deployment's config flag; a disabled module stays loaded but is hidden from
    every capability view. ``tools`` are keyed by name for fast lookup, and
    ``prompt_text`` is the module's specialist system prompt resolved at load."""

    manifest: CapabilityManifest
    enabled: bool
    tools: dict[str, Tool] = field(default_factory=dict)
    retrievers: dict[str, Retriever] = field(default_factory=dict)
    prompt_text: str = ""

    @property
    def id(self) -> str:
        return self.manifest.id

    def required_role(self, tool_name: str) -> str:
        """The minimum RBAC role needed to call ``tool_name`` in this module. The
        manifest's ``rbac`` map can OVERRIDE a tool's own declared role (e.g. to
        tighten a tool for one tenant tier); otherwise we fall back to the tool's
        ``rbac_role``, and to "viewer" if the tool is unknown. This is the value
        capability_view checks the caller's roles against."""
        t = self.tools.get(tool_name)
        return self.manifest.rbac.get(tool_name, t.rbac_role if t else "viewer")


@dataclass
class CapabilityView:
    """What ONE org/user can see and do — the security-filtered projection of the
    registry for a specific caller. Returned by :meth:`capability_view`; computed
    per (org, roles). Holds only the modules they're entitled to, only the tools
    their roles satisfy, and the routing hints for those modules. The agent and
    API are handed this view, so a caller can never route to or call a capability
    they aren't licensed/authorized for."""

    module_ids: list[str]
    tools: list[Tool]
    routing: list[tuple[str, RoutingHint]]  # (module_id, hint)


class CapabilityRegistry:
    """The live index of installed capabilities. One instance is built at boot
    (in bootstrap.py) and shared for the process lifetime."""

    def __init__(self) -> None:
        self._modules: dict[str, RegisteredModule] = {}      # id -> loaded module
        # The merged knowledge-graph schema, accumulated as modules register their
        # ontology slices. Node names map to their key tuple; edges are flat triples.
        self._ontology_nodes: dict[str, tuple[str, ...]] = {}
        self._ontology_edges: list[tuple[str, str, str]] = []

    # -- registration --------------------------------------------------------
    def register(self, manifest: CapabilityManifest, *, enabled: bool, module_pkg: str = "") -> RegisteredModule:
        """Validate ONE manifest and add it to the index. This is where a cartridge
        is checked against the slot before it's allowed in. The validation gates,
        in order, each raising RegistryError so a bad module fails fast at boot
        rather than misbehaving at request time:
          * the module's required core version must not exceed ours (SemVer);
          * module ids must be unique, and tool names unique within a module;
          * every RBAC override must name a real tool.
        On success it stamps each tool with its owning ``module_id``, resolves the
        specialist prompt, folds in any ontology, and stores a RegisteredModule.
        """
        if _version_tuple(manifest.min_core_version) > _version_tuple(CONTRACTS_VERSION):
            raise RegistryError(
                f"module '{manifest.id}' needs core >= {manifest.min_core_version}, have {CONTRACTS_VERSION}"
            )
        if manifest.id in self._modules:
            raise RegistryError(f"duplicate module id '{manifest.id}'")

        # Re-build each tool as a copy stamped with this module's id. Stamping is
        # what lets later code (the MCP boundary, traces) know which module owns a
        # tool — provenance the manifest author shouldn't have to set by hand.
        tools: dict[str, Tool] = {}
        for t in manifest.tools:
            stamped = Tool(
                name=t.name, description=t.description, handler=t.handler,
                args_schema=t.args_schema, returns_schema=t.returns_schema,
                side_effecting=t.side_effecting, rbac_role=t.rbac_role,
                autonomy=t.autonomy, module_id=manifest.id,
            )
            if stamped.name in tools:
                raise RegistryError(f"duplicate tool '{stamped.name}' in module '{manifest.id}'")
            tools[stamped.name] = stamped

        # RBAC keys must reference real tools — catches a typo'd override that
        # would otherwise silently fail to protect a tool.
        for tname in manifest.rbac:
            if tname not in tools:
                raise RegistryError(f"module '{manifest.id}' rbac references unknown tool '{tname}'")

        retrievers = {r.id: r for r in manifest.retrievers}
        prompt_text = self._load_prompt(manifest, module_pkg)
        if manifest.ontology:
            self._merge_ontology(manifest.id, manifest.ontology)   # contribute KG schema

        mod = RegisteredModule(manifest=manifest, enabled=enabled, tools=tools,
                               retrievers=retrievers, prompt_text=prompt_text)
        self._modules[manifest.id] = mod
        return mod

    def _load_prompt(self, manifest: CapabilityManifest, module_pkg: str) -> str:
        """Resolve a module's specialist system prompt. ``manifest.system_prompt``
        is a path RELATIVE to the module's package dir (kept as an editable file,
        not inlined in code). If unset or unreadable we fall back to a safe generic
        prompt that still encodes the core RAG rule: answer only from cited sources.
        Any IO error degrades to that fallback rather than failing boot."""
        if not manifest.system_prompt:
            return f"You are the {manifest.display_name} specialist. Answer only from retrieved, cited sources."
        try:
            base = Path(importlib.import_module(module_pkg).__file__).parent if module_pkg else Path(".")
            p = base / manifest.system_prompt
            if p.exists():
                return p.read_text(encoding="utf-8")
        except Exception:
            pass
        return f"You are the {manifest.display_name} specialist. Answer only from retrieved, cited sources."

    def _merge_ontology(self, module_id: str, contrib: OntologyContribution) -> None:
        """Fold one module's ontology slice into the shared knowledge-graph schema.
        Different modules contribute different node/edge types (easm adds Asset,
        aci adds ThreatActor, ...); the registry is where those slices MERGE into
        one schema. A node type defined by two modules with DIFFERENT keys is a
        real conflict (the graph couldn't join them) so we raise; edges just
        accumulate. This keeps the cross-module KG coherent."""
        for n in contrib.nodes:
            existing = self._ontology_nodes.get(n.name)
            if existing is not None and existing != n.keys:
                raise RegistryError(
                    f"ontology collision on node '{n.name}' (module '{module_id}'): keys differ"
                )
            self._ontology_nodes[n.name] = n.keys
        for e in contrib.edges:
            self._ontology_edges.append((e.src, e.relation, e.dst))

    # -- discovery -----------------------------------------------------------
    def discover(self, settings) -> CapabilityRegistry:
        """Auto-load every capability module on disk — the "drop in a dir, no core
        edit" mechanism. Walks the subpackages of ``app.capabilities``, and for
        each one imports its ``manifest`` module and reads the ``MANIFEST`` it
        exports. A subpackage is SILENTLY SKIPPED if it has no manifest module or
        no valid MANIFEST, so partial/experimental dirs don't break boot.

        Crucially, whether a discovered module is ENABLED is decided by config:
        the manifest names a settings flag (``enabled_flag``) and we read it off
        ``settings`` (falling back to the manifest's default). That is the bridge
        between "what's installed" (here) and "what's turned on" (config.py).
        Returns ``self`` so callers can chain ``CapabilityRegistry().discover(...)``.
        """
        pkg = importlib.import_module(CAPABILITIES_PACKAGE)
        for info in pkgutil.iter_modules(pkg.__path__):
            # Only real subpackages count as modules; skip files and _private dirs.
            if not info.ispkg or info.name.startswith("_"):
                continue
            module_pkg = f"{CAPABILITIES_PACKAGE}.{info.name}"
            try:
                man_mod = importlib.import_module(f"{module_pkg}.manifest")
            except ModuleNotFoundError:
                continue   # a package without a manifest just isn't a capability
            manifest = getattr(man_mod, "MANIFEST", None)
            if not isinstance(manifest, CapabilityManifest):
                continue   # exported something, but not a valid manifest -> ignore
            # Config-gate: look up this module's flag on Settings; an empty
            # ``enabled_flag`` means "always on" (use the manifest default).
            enabled = (
                getattr(settings, manifest.enabled_flag, manifest.enabled_default)
                if manifest.enabled_flag else manifest.enabled_default
            )
            self.register(manifest, enabled=bool(enabled), module_pkg=module_pkg)
        return self

    # -- accessors -----------------------------------------------------------
    def modules(self, *, include_disabled: bool = True) -> list[RegisteredModule]:
        """All loaded modules. ``include_disabled=False`` returns only the ones
        actually turned on for this deployment (e.g. used by seeding and the
        ready-line)."""
        return [m for m in self._modules.values() if include_disabled or m.enabled]

    def module(self, module_id: str) -> RegisteredModule | None:
        """Look up one module by id (or None). Used heavily by the agent nodes to
        turn a routed module id back into the live module object."""
        return self._modules.get(module_id)

    def find_tool(self, tool_name: str) -> tuple[RegisteredModule, Tool] | None:
        """Locate which module owns a given tool name. The MCP boundary uses this
        to resolve a tool call back to its module (for RBAC + execution)."""
        for m in self._modules.values():
            if tool_name in m.tools:
                return m, m.tools[tool_name]
        return None

    @property
    def ontology(self) -> dict:
        """The merged knowledge-graph schema across all modules (a copy, so callers
        can't mutate the registry's internals)."""
        return {"nodes": dict(self._ontology_nodes), "edges": list(self._ontology_edges)}

    # -- per-org view (entitlement + RBAC, computed not coded) ---------------
    def _entitled(self, module: RegisteredModule, sc: SecurityContext) -> bool:
        """Tenant-level LICENSE check: may this org use this module AT ALL? This is
        coarser than RBAC (which is per-user, per-tool). Today it only requires the
        module be enabled; the comment marks the seam where a real entitlement/
        license store plugs in to gate by tenant tier (``manifest.license_tiers``)
        — e.g. so the ``aci`` module is only visible to orgs that bought it."""
        if not module.enabled:
            return False
        # Default entitlement: every loaded module is visible to every org. Wire a
        # license/entitlement store here to gate by tenant tier (manifest.license_tiers).
        return True

    def capability_view(self, sc: SecurityContext) -> CapabilityView:
        """Build the SECURITY-FILTERED view for one caller — the core multi-tenant
        access decision, computed from manifests (never hardcoded). Two filters,
        applied in this order:

          1. ENTITLEMENT (per org/tenant) — drop whole modules this org isn't
             licensed for (see ``_entitled``);
          2. RBAC (per user) — within an entitled module, include a tool ONLY if
             the caller's roles meet that tool's required role (``role_satisfies``).

        So the returned tool list is exactly "modules this tenant bought ∩ tools
        this user's role permits". The agent and API only ever see this view, which
        is how a low-privilege user in one org cannot route to, see, or invoke a
        capability outside their license/role. Routing hints are carried only for
        the entitled modules, so even the SUPERVISOR can't route to a hidden one.
        """
        module_ids: list[str] = []
        tools: list[Tool] = []
        routing: list[tuple[str, RoutingHint]] = []
        for m in self._modules.values():
            if not self._entitled(m, sc):        # filter 1: org-level license gate
                continue
            module_ids.append(m.id)
            for hint in m.manifest.routing_hints:
                routing.append((m.id, hint))
            for tname, tool in m.tools.items():
                if role_satisfies(sc.roles, m.required_role(tname)):   # filter 2: per-tool RBAC
                    tools.append(tool)
        return CapabilityView(module_ids=module_ids, tools=tools, routing=routing)


__all__ = ["CapabilityRegistry", "RegisteredModule", "CapabilityView"]
