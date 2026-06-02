"""In-process MCP tool runner (v1 transport).

This is where **RBAC and the action gate are enforced** for every tool call,
regardless of which agent invokes it:
  * the caller's roles must satisfy the tool's required role (from the manifest);
  * a ``side_effecting`` tool cannot execute inline — it routes through the action
    gate (human approval), so prompt injection can never fire an action.
Every invocation is logged. Swap this for ``RemoteMCPClient`` to call a module
that has been promoted to its own MCP server — same interface.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from app.core.contracts import ToolContext, ToolError, ToolOutcome, role_satisfies
from app.core.registry import CapabilityRegistry
from app.core.security.context import SecurityContext


class ActionGateProtocol(Protocol):
    async def auto_approves(self, tool, arguments: Mapping[str, Any], ctx: ToolContext) -> bool: ...
    async def request_approval(self, tool, arguments: Mapping[str, Any], ctx: ToolContext) -> str: ...


class MCPClient(Protocol):
    transport: str

    async def list_tools(self, sc: SecurityContext) -> list[dict]: ...
    async def call_tool(self, name: str, arguments: Mapping[str, Any], ctx: ToolContext) -> ToolOutcome: ...


class InProcessMCPClient:
    transport = "inprocess"

    def __init__(
        self,
        registry: CapabilityRegistry,
        *,
        action_gate: ActionGateProtocol | None = None,
        logger: Any = None,
    ) -> None:
        self.registry = registry
        self.action_gate = action_gate
        self.logger = logger

    async def list_tools(self, sc: SecurityContext) -> list[dict]:
        return [t.json_schema() for t in self.registry.capability_view(sc).tools]

    async def call_tool(self, name: str, arguments: Mapping[str, Any], ctx: ToolContext) -> ToolOutcome:
        found = self.registry.find_tool(name)
        if not found:
            return ToolError(code="unknown_tool", message=f"no such tool: {name}")
        module, tool = found
        if not module.enabled:
            return ToolError(code="disabled", message=f"module '{module.id}' is disabled")

        required = module.required_role(name)
        if not role_satisfies(ctx.roles, required):
            self._log("tool_denied", name, ctx, reason="rbac", required=required)
            return ToolError(
                code="forbidden",
                message=f"role '{required}' required to call '{name}'",
                details={"have": list(ctx.roles), "need": required},
            )

        if tool.side_effecting:
            gate = self.action_gate
            approved = bool(gate) and await gate.auto_approves(tool, arguments, ctx)
            if not approved:
                approval_id = ""
                if gate:
                    approval_id = await gate.request_approval(tool, arguments, ctx)
                self._log("tool_gated", name, ctx, approval_id=approval_id)
                return ToolError(
                    code="requires_approval",
                    message=f"'{name}' is a side-effecting action; awaiting human approval",
                    details={"action_type": name, "approval_id": approval_id},
                )

        self._log("tool_call", name, ctx, cap_module=module.id)
        outcome = await tool.invoke(arguments, ctx)
        self._log("tool_result", name, ctx, ok=getattr(outcome, "ok", False))
        return outcome

    def _log(self, event: str, tool: str, ctx: ToolContext, **extra: Any) -> None:
        if self.logger:
            self.logger.info(
                "mcp.%s", event,
                extra={"tool": tool, "org_id": ctx.org_id, "user_id": ctx.user_id,
                       "trace_id": ctx.trace_id, **extra},
            )


__all__ = ["InProcessMCPClient", "MCPClient", "ActionGateProtocol"]
