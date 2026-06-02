"""Capability registry — the heart of the extensibility chassis.

At boot it **discovers** capability modules (``app/capabilities/*/manifest.py``),
validates their contracts, and assembles the running system from their
manifests. The supervisor's routing, each org's visible tools, RBAC, and the
merged ontology are all *computed here*, never hardcoded. Adding a feature =
dropping in a module dir; no core edit. (Blueprint file 15.)
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

CAPABILITIES_PACKAGE = "app.capabilities"


def _version_tuple(v: str) -> tuple[int, ...]:
    try:
        return tuple(int(x) for x in v.split("."))
    except ValueError:
        return (0,)


@dataclass
class RegisteredModule:
    manifest: CapabilityManifest
    enabled: bool
    tools: dict[str, Tool] = field(default_factory=dict)
    retrievers: dict[str, Retriever] = field(default_factory=dict)
    prompt_text: str = ""

    @property
    def id(self) -> str:
        return self.manifest.id

    def required_role(self, tool_name: str) -> str:
        t = self.tools.get(tool_name)
        return self.manifest.rbac.get(tool_name, t.rbac_role if t else "viewer")


@dataclass
class CapabilityView:
    """What one org/user can see and do — computed, cached per (org, roles)."""

    module_ids: list[str]
    tools: list[Tool]
    routing: list[tuple[str, RoutingHint]]  # (module_id, hint)


class CapabilityRegistry:
    def __init__(self) -> None:
        self._modules: dict[str, RegisteredModule] = {}
        self._ontology_nodes: dict[str, tuple[str, ...]] = {}
        self._ontology_edges: list[tuple[str, str, str]] = []

    # -- registration --------------------------------------------------------
    def register(self, manifest: CapabilityManifest, *, enabled: bool, module_pkg: str = "") -> RegisteredModule:
        if _version_tuple(manifest.min_core_version) > _version_tuple(CONTRACTS_VERSION):
            raise RegistryError(
                f"module '{manifest.id}' needs core >= {manifest.min_core_version}, have {CONTRACTS_VERSION}"
            )
        if manifest.id in self._modules:
            raise RegistryError(f"duplicate module id '{manifest.id}'")

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

        # RBAC keys must reference real tools.
        for tname in manifest.rbac:
            if tname not in tools:
                raise RegistryError(f"module '{manifest.id}' rbac references unknown tool '{tname}'")

        retrievers = {r.id: r for r in manifest.retrievers}
        prompt_text = self._load_prompt(manifest, module_pkg)
        if manifest.ontology:
            self._merge_ontology(manifest.id, manifest.ontology)

        mod = RegisteredModule(manifest=manifest, enabled=enabled, tools=tools,
                               retrievers=retrievers, prompt_text=prompt_text)
        self._modules[manifest.id] = mod
        return mod

    def _load_prompt(self, manifest: CapabilityManifest, module_pkg: str) -> str:
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
        pkg = importlib.import_module(CAPABILITIES_PACKAGE)
        for info in pkgutil.iter_modules(pkg.__path__):
            if not info.ispkg or info.name.startswith("_"):
                continue
            module_pkg = f"{CAPABILITIES_PACKAGE}.{info.name}"
            try:
                man_mod = importlib.import_module(f"{module_pkg}.manifest")
            except ModuleNotFoundError:
                continue
            manifest = getattr(man_mod, "MANIFEST", None)
            if not isinstance(manifest, CapabilityManifest):
                continue
            enabled = (
                getattr(settings, manifest.enabled_flag, manifest.enabled_default)
                if manifest.enabled_flag else manifest.enabled_default
            )
            self.register(manifest, enabled=bool(enabled), module_pkg=module_pkg)
        return self

    # -- accessors -----------------------------------------------------------
    def modules(self, *, include_disabled: bool = True) -> list[RegisteredModule]:
        return [m for m in self._modules.values() if include_disabled or m.enabled]

    def module(self, module_id: str) -> RegisteredModule | None:
        return self._modules.get(module_id)

    def find_tool(self, tool_name: str) -> tuple[RegisteredModule, Tool] | None:
        for m in self._modules.values():
            if tool_name in m.tools:
                return m, m.tools[tool_name]
        return None

    @property
    def ontology(self) -> dict:
        return {"nodes": dict(self._ontology_nodes), "edges": list(self._ontology_edges)}

    # -- per-org view (entitlement + RBAC, computed not coded) ---------------
    def _entitled(self, module: RegisteredModule, sc: SecurityContext) -> bool:
        if not module.enabled:
            return False
        # Default entitlement: every loaded module is visible to every org. Wire a
        # license/entitlement store here to gate by tenant tier (manifest.license_tiers).
        return True

    def capability_view(self, sc: SecurityContext) -> CapabilityView:
        module_ids: list[str] = []
        tools: list[Tool] = []
        routing: list[tuple[str, RoutingHint]] = []
        for m in self._modules.values():
            if not self._entitled(m, sc):
                continue
            module_ids.append(m.id)
            for hint in m.manifest.routing_hints:
                routing.append((m.id, hint))
            for tname, tool in m.tools.items():
                if role_satisfies(sc.roles, m.required_role(tname)):
                    tools.append(tool)
        return CapabilityView(module_ids=module_ids, tools=tools, routing=routing)


__all__ = ["CapabilityRegistry", "RegisteredModule", "CapabilityView"]
