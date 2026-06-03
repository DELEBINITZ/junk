"""Test-kit tools — local stubs that MIRROR the `mcp-test-kits` MCP server.

Each tool name + arg schema below is identical to a tool on the upstream server
(github.com/midodimori/mcp-test-kits, capabilities/tools.py), so the MCP boundary
can route a call by name to that server. Two modes, ZERO agent-code difference:

  * TESTKIT_MCP_URL unset  -> these LOCAL bodies run (offline fallback / contract).
  * TESTKIT_MCP_URL set    -> bootstrap builds a FastMCPRemote for module "testkit";
                              the boundary dispatches EXECUTION to the real server and
                              these bodies are bypassed (see inprocess.py step (e)).

All tools are READ-only (viewer), so the heuristic planner may auto-invoke the
ones whose required args it can fill from the question — notably ``generate_uuid``
and ``get_timestamp`` take no required args, so they fire on a plain question and
prove the end-to-end MCP loop with no LLM needed.

The upstream tools return PLAIN scalars; FastMCP wraps a scalar return as
``{"result": <value>}``, which is exactly the shape we return locally too, so the
agent sees an identical ToolResult whether the call ran here or on the server.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from pydantic import BaseModel, Field

from app.core.contracts import Citation, ToolContext, ToolResult, tool


def _result(value, *, label: str, doc_id: str) -> ToolResult:
    """Pack a scalar tool answer the way the remote server's reply decodes to:
    ``data={"result": value}`` plus one Citation so the specialist has grounded
    evidence to fold into the final answer."""
    return ToolResult(
        data={"result": value},
        citations=[Citation(doc_id=doc_id, source="testkit", title=label,
                            snippet=f"{label}: {value}")],
    )


class EchoArgs(BaseModel):
    message: str = Field(description="Text to echo back unchanged.")


@tool(name="echo", description="Return the input message unchanged (connectivity/echo check).",
      args_schema=EchoArgs, rbac_role="viewer")
async def echo(args: EchoArgs, ctx: ToolContext):
    return _result(args.message, label="echo", doc_id="tk-echo")


class AddArgs(BaseModel):
    a: float = Field(description="First addend.")
    b: float = Field(description="Second addend.")


@tool(name="add", description="Add two numbers and return the sum.",
      args_schema=AddArgs, rbac_role="viewer")
async def add(args: AddArgs, ctx: ToolContext):
    return _result(args.a + args.b, label="sum", doc_id="tk-add")


class MultiplyArgs(BaseModel):
    x: float = Field(description="First factor.")
    y: float = Field(description="Second factor.")


@tool(name="multiply", description="Multiply two numbers and return the product.",
      args_schema=MultiplyArgs, rbac_role="viewer")
async def multiply(args: MultiplyArgs, ctx: ToolContext):
    return _result(args.x * args.y, label="product", doc_id="tk-mul")


class ReverseArgs(BaseModel):
    text: str = Field(description="String to reverse.")


@tool(name="reverse_string", description="Reverse the characters of a string.",
      args_schema=ReverseArgs, rbac_role="viewer")
async def reverse_string(args: ReverseArgs, ctx: ToolContext):
    return _result(args.text[::-1], label="reversed", doc_id="tk-rev")


class NoArgs(BaseModel):
    pass


@tool(name="generate_uuid", description="Generate a random UUID (unique identifier).",
      args_schema=NoArgs, rbac_role="viewer")
async def generate_uuid(args: NoArgs, ctx: ToolContext):
    return _result(str(uuid.uuid4()), label="uuid", doc_id="tk-uuid")


class TimestampArgs(BaseModel):
    format: str = Field(default="iso", description="'iso' for ISO-8601, 'unix' for epoch seconds.")


@tool(name="get_timestamp", description="Get the current timestamp (ISO-8601 or Unix epoch).",
      args_schema=TimestampArgs, rbac_role="viewer")
async def get_timestamp(args: TimestampArgs, ctx: ToolContext):
    now = datetime.now(UTC)
    value = int(now.timestamp()) if args.format == "unix" else now.isoformat()
    return _result(value, label="timestamp", doc_id="tk-ts")


# The module's full tool surface, imported by the manifest as ``tools=TOOLS``.
TOOLS = (echo, add, multiply, reverse_string, generate_uuid, get_timestamp)

__all__ = ["TOOLS", "echo", "add", "multiply", "reverse_string", "generate_uuid", "get_timestamp"]
