"""Reports module tools.

Increment 1 wraps the existing, security-reviewed MCP tool functions
(app/mcp_server/tools.py) as chassis Tool objects. The legacy implementations —
RBAC checks, citations, PII redaction — are reused unchanged; only the boundary
is adapted: a raised legacy ToolError becomes a chassis ToolException, which
Tool.run() turns into a ToolError value (errors-as-data). See plan §16.
"""

from __future__ import annotations

from typing import Any, Callable

from app.core.contracts import Tool, ToolContext, ToolException
from app.mcp_server import tools as legacy


def _adapt(name: str) -> Callable[[dict[str, Any], ToolContext], dict[str, Any]]:
    def handler(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
        try:
            return legacy.call_tool(name, args, ctx.user, ctx.store)
        except legacy.ToolError as exc:
            raise ToolException(str(exc), code=exc.code) from exc

    return handler


def _build_tools() -> list[Tool]:
    definitions = {d["name"]: d for d in legacy.tool_definitions()}
    tools: list[Tool] = []
    for name in legacy.TOOL_ORDER:
        definition = definitions[name]
        tools.append(
            Tool(
                name=definition["name"],
                description=definition["description"],
                input_schema=definition["inputSchema"],
                handler=_adapt(name),
                # Every current report tool needs AI/RAG query access, which is
                # analyst+ in the RBAC model (viewers may read but not query).
                rbac_role="analyst",
            )
        )
    return tools


REPORT_TOOLS: list[Tool] = _build_tools()
