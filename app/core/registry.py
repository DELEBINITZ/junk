"""Capability registry: discover module manifests at boot, then compute the
per-organization / per-user tool views. No feature is named here — everything is
derived from manifests. Cross-cutting "core" tools (date, past-chat search) are
registered alongside and are available to every module. See plan §5.
"""

from __future__ import annotations

import importlib
import logging
import os
import pkgutil
from typing import Callable

from app.core.contracts import CapabilityManifest, Tool, role_allows
from app.domain import User


logger = logging.getLogger(__name__)

CAPABILITIES_PACKAGE = "app.capabilities"
CORE_MODULE_ID = "core"


class RegistryError(Exception):
    """Raised on a registry-level conflict (e.g. duplicate tool name)."""


class CapabilityRegistry:
    def __init__(self, entitlements: Callable[[str], set[str]] | None = None):
        self.modules: dict[str, CapabilityManifest] = {}
        self.tools: dict[str, tuple[str, Tool]] = {}  # tool_name -> (module_id, Tool)
        self.core_tools: list[Tool] = []
        # Org entitlement hook. Default: every org is entitled to every loaded
        # module (no license store yet). Swap for a real per-org lookup later.
        self._entitlements = entitlements

    # ---- discovery / registration -------------------------------------------
    def discover(self, package: str = CAPABILITIES_PACKAGE) -> "CapabilityRegistry":
        pkg = importlib.import_module(package)
        for info in pkgutil.iter_modules(pkg.__path__):
            if not info.ispkg:
                continue
            module_name = f"{package}.{info.name}.manifest"
            try:
                mod = importlib.import_module(module_name)
            except ModuleNotFoundError:
                # A directory without a manifest (e.g. _template) is not a module.
                logger.info("capability.no_manifest", extra={"capability": info.name})
                continue
            manifest = getattr(mod, "MANIFEST", None)
            if manifest is None:
                logger.warning("capability.manifest_missing_symbol", extra={"capability": info.name})
                continue
            # Deployment-composition gate: a module with enabled_by_default=False
            # is only loaded when CAP_<ID>_ENABLED=true (ship dark -> canary).
            opt_in = os.getenv(f"CAP_{manifest.id.upper()}_ENABLED", "").lower() == "true"
            if not manifest.enabled_by_default and not opt_in:
                logger.info("capability.disabled", extra={"capability": manifest.id})
                continue
            self.register(manifest)
        self._register_core_tools()
        logger.info(
            "registry.discovered",
            extra={"modules": list(self.modules), "tools": len(self.tools)},
        )
        return self

    def register(self, manifest: CapabilityManifest) -> None:
        manifest.validate()
        if manifest.id in self.modules:
            raise RegistryError(f"Duplicate module id: {manifest.id}")
        for tool in manifest.tools:
            # The manifest is the source of truth for the coarse role gate.
            if tool.name in manifest.rbac:
                tool.rbac_role = manifest.rbac[tool.name]
            if tool.name in self.tools:
                other = self.tools[tool.name][0]
                raise RegistryError(
                    f"Tool name collision: '{tool.name}' in '{manifest.id}' and '{other}'"
                )
            self.tools[tool.name] = (manifest.id, tool)
        self.modules[manifest.id] = manifest

    def _register_core_tools(self) -> None:
        # Local import avoids an import-time cycle (core_tools imports contracts
        # + memory, registry imports core_tools only at discovery time).
        from app.core.agent.core_tools import build_core_tools

        for tool in build_core_tools():
            if tool.name in self.tools:
                raise RegistryError(f"Core tool '{tool.name}' collides with a module tool")
            self.core_tools.append(tool)
            self.tools[tool.name] = (CORE_MODULE_ID, tool)

    # ---- per-org / per-user views -------------------------------------------
    def org_entitlements(self, org_id: str) -> set[str]:
        if self._entitlements is not None:
            return self._entitlements(org_id)
        return set(self.modules)  # default-allow until a license store exists

    def _module_enabled(self, manifest: CapabilityManifest) -> bool:
        return True  # enabled_flag wired to a feature-flag service later

    def modules_for_user(self, user: User) -> list[CapabilityManifest]:
        entitled = self.org_entitlements(user.organization_id)
        return [
            m for m in self.modules.values()
            if m.id in entitled and self._module_enabled(m)
        ]

    def tools_for_user(self, user: User) -> list[Tool]:
        out: list[Tool] = [t for t in self.core_tools if role_allows(user.role, t.rbac_role)]
        for manifest in self.modules_for_user(user):
            for tool in manifest.tools:
                if role_allows(user.role, tool.rbac_role):
                    out.append(tool)
        return out

    def module_tools_for_user(self, user: User) -> list[Tool]:
        """Tools that belong to an actual capability module (excludes core)."""

        return [t for t in self.tools_for_user(user) if self.module_of(t.name) != CORE_MODULE_ID]

    def tool(self, name: str) -> Tool | None:
        entry = self.tools.get(name)
        return entry[1] if entry else None

    def module_of(self, tool_name: str) -> str | None:
        entry = self.tools.get(tool_name)
        return entry[0] if entry else None

    def list_tool_definitions(self, user: User) -> list[dict]:
        return [t.mcp_definition() for t in self.tools_for_user(user)]


_registry: CapabilityRegistry | None = None


def get_registry() -> CapabilityRegistry:
    global _registry
    if _registry is None:
        _registry = CapabilityRegistry().discover()
    return _registry


def reset_registry() -> None:
    """Test hook to force re-discovery."""

    global _registry
    _registry = None
