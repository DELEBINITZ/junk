"""Reference: connect this platform to a REMOTE MCP server (the EASM promotion path).

WHAT THIS PROVES — end to end, against a real server on the wire:
  1. discover tools the remote server exposes  (MCP ``tools/list``)
  2. call one of those tools                    (MCP ``tools/call``)
  3. map the reply back into our ToolResult/ToolError shape

It uses the SAME library and the SAME two calls as the production adapter
``app/core/mcp/fastmcp_client.py`` (class ``FastMCPRemote``). Read the two side by
side — this script is the stripped-down skeleton of that class.

HOW THE REAL EASM WIRING USES THIS (no code change, just config):
    EASM_MCP_URL=http://localhost:3000/mcp   # set this env var (config.py: easm_mcp_url)
  -> bootstrap._build_remote_executors() sees the url, builds FastMCPRemote("…/mcp")
  -> InProcessMCPClient routes every EASM tool call to that FastMCPRemote
  -> RBAC + the action gate STILL run locally first; identity rides in a Bearer
     service token (never in tool args). The remote server re-derives org from it.

RUN IT (server must already be up — see the run command printed by this repo's
helper, or: `cd /tmp/mcp-test-kits/python && uv run mcp-test-kits --transport http --port 3000`):

    uv run --with fastmcp python scripts/mcp_connect_demo.py

The target here is the generic ``mcp-test-kits`` server (tools: echo, add, …). Your
real ``easm-mcp`` would instead expose query_assets / get_exposures / etc. — the
CONNECTION code below is identical regardless of which tools the server ships.
"""

from __future__ import annotations

import asyncio
import os

# Same import the production adapter does lazily in FastMCPRemote._client.
from fastmcp import Client

# Point at any MCP server. In the real platform this value is settings.easm_mcp_url.
SERVER_URL = os.environ.get("EASM_MCP_URL", "http://localhost:3000/mcp")


# --- This is FastMCPRemote._client, minus the Bearer-token branch -----------------
# Production attaches an org-scoped service token here:
#     from fastmcp.client.auth import BearerAuth
#     Client(SERVER_URL, auth=BearerAuth(service_token), timeout=30.0)
# The test-kit server doesn't verify our tokens, so we connect tokenless to demo.
def make_client() -> Client:
    return Client(SERVER_URL, timeout=30.0)


# --- This is FastMCPRemote._to_outcome, simplified to a printable dict -------------
# Mirrors how we recover the server's STRUCTURED output (typed data) with a text
# fallback — the thing that lets a remote tool return the same shape as a local one.
def to_outcome(res) -> dict:
    if getattr(res, "is_error", False):
        text = " ".join(b.text for b in (getattr(res, "content", None) or []) if getattr(b, "text", None))
        return {"ok": False, "error": text or "remote tool error"}
    data = getattr(res, "structured_content", None) or getattr(res, "data", None)
    if data is not None:
        return {"ok": True, "data": data}
    text = " ".join(b.text for b in (getattr(res, "content", None) or []) if getattr(b, "text", None))
    return {"ok": True, "data": {"text": text}}


async def main() -> None:
    print(f"Connecting to MCP server: {SERVER_URL}\n")

    # The fastmcp Client is an async context manager — it opens the transport
    # (Streamable-HTTP here), runs the MCP handshake, and closes on exit. This is
    # exactly the `async with self._client(token) as c:` in FastMCPRemote.
    async with make_client() as client:

        # 1) DISCOVERY — MCP `tools/list`. Same call as FastMCPRemote.list_tools().
        #    A real easm-mcp would return query_assets/get_exposures/... here.
        tools = await client.list_tools()
        print(f"tools/list -> {len(tools)} tools exposed by the server:")
        for t in tools:
            ann = getattr(t, "annotations", None)
            read_only = getattr(ann, "readOnlyHint", None) if ann else None
            print(f"  - {t.name:<20} {(t.description or '').splitlines()[0][:60]}"
                  f"   [read_only_hint={read_only}]")
        print()

        # 2) CALL — MCP `tools/call`. Same call as FastMCPRemote.call_tool().
        #    For EASM this line would be: client.call_tool("get_live_asset_count", {...})
        print("tools/call echo(message='hello from contract-intelligence-poc'):")
        res = await client.call_tool("echo", {"message": "hello from contract-intelligence-poc"})
        print("   ->", to_outcome(res), "\n")

        print("tools/call add(a=7, b=35):")
        res = await client.call_tool("add", {"a": 7, "b": 35})
        print("   ->", to_outcome(res), "\n")

    print("Done. This is precisely the path EASM tool calls take once EASM_MCP_URL is set.")


if __name__ == "__main__":
    asyncio.run(main())
