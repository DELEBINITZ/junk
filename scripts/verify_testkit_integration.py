"""Verify the `testkit` capability is wired in and the agent answers via the MCP.

Run with the test-kit HTTP server up on :3000 and fastmcp available:
    cd /tmp/mcp-test-kits/python && uv run mcp-test-kits --transport http --port 3000   # (separate shell)
    cd <repo> && TESTKIT_MCP_URL=http://localhost:3000/mcp uv run --with fastmcp python scripts/verify_testkit_integration.py
"""

from __future__ import annotations

import asyncio
import os

from app.config import reload_settings
from app.core.bootstrap import build_services, seed_demo
from app.core.contracts import ToolContext
from app.core.security.context import SecurityContext


def sc() -> SecurityContext:
    return SecurityContext(org_id="org_acme", user_id="u-alice", roles=("analyst",), email="a@acme.test")


async def main() -> None:
    url = os.environ.get("TESTKIT_MCP_URL", "")
    print(f"TESTKIT_MCP_URL = {url or '(unset -> local stub mode)'}\n")
    services = build_services(reload_settings())
    await seed_demo(services)
    try:
        # --- 0) module discovered + enabled --------------------------------------
        mod = services.registry.module("testkit")
        print(f"[0] module discovered: id={mod.id!r} enabled={mod.enabled} "
              f"tools={list(mod.tools)}\n")

        # --- A) the LIVE server is reachable through the boundary's executor ------
        # Only the real mcp-test-kits server exposes 8 tools incl. sample_error /
        # long_running_task; the local manifest has 6. Seeing 8 proves the wire.
        executor = services.mcp.remote_executors.get("testkit")
        if executor is not None:
            remote_tools = await executor.list_tools(sc())
            names = sorted(t["name"] for t in remote_tools)
            print(f"[A] remote tools/list via boundary executor -> {len(names)} tools: {names}")
            assert {"sample_error", "long_running_task"} <= set(names), "not the live server!"
            print("    ✓ live mcp-test-kits server is wired into the boundary\n")
        else:
            print("[A] no remote executor (URL unset) — testkit runs local stubs\n")

        # --- B) one tool call through the FULL boundary (RBAC + routing) ----------
        ctx = ToolContext(org_id="org_acme", user_id="u-alice", roles=("analyst",),
                          trace_id="t", request_id="r", deps=services.deps)
        out = await services.mcp.call_tool("generate_uuid", {}, ctx)
        print(f"[B] mcp.call_tool('generate_uuid') -> ok={out.ok} data={out.data}")
        out2 = await services.mcp.call_tool("add", {"a": 7, "b": 35}, ctx)
        print(f"    mcp.call_tool('add', a=7 b=35) -> ok={out2.ok} data={out2.data}\n")

        # --- C) END TO END: the agent answers a question USING the MCP -----------
        for q in ("generate a uuid for me", "what is the current timestamp?"):
            res = await services.orchestrator.run_turn(sc(), question=q)
            print(f"[C] Q: {q}")
            print(f"    route_modules = {res.route_modules}")
            print(f"    answer        = {res.answer.strip()[:200]}")
            print(f"    citations     = {[ (c.get('source'), c.get('title')) for c in res.citations ]}\n")
    finally:
        await services.aclose()


if __name__ == "__main__":
    asyncio.run(main())
