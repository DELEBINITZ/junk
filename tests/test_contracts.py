"""Contract tests every capability module must pass (blueprint file 15 §8).

Run automatically against all registered modules: tool schema validity,
errors-as-data, tenant-context plumbing, gate enforcement for side-effecting
tools, and routing of each module's golden questions.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.core.contracts import ToolError, ToolResult
from tests.conftest import make_sc, tool_ctx

CAPS = Path(__file__).resolve().parent.parent / "app" / "capabilities"


def test_every_tool_has_valid_schema(services):
    for module in services.registry.modules():
        for name, tool in module.tools.items():
            schema = tool.json_schema()
            assert schema["name"] == name
            assert "parameters" in schema
            assert tool.args_schema.model_json_schema()  # must be constructible


@pytest.mark.asyncio
async def test_side_effecting_tools_require_approval(services):
    """No side-effecting tool may execute inline — the gate must intercept it."""
    ctx = tool_ctx(services, make_sc(roles=("admin",)))
    found_any = False
    for module in services.registry.modules(include_disabled=False):
        for name, tool in module.tools.items():
            if tool.side_effecting:
                found_any = True
                out = await services.mcp.call_tool(name, {"asset": "x", "target": "x"}, ctx)
                assert isinstance(out, ToolError) and out.code == "requires_approval", name
    assert found_any, "expected at least one side-effecting tool (easm.trigger_rescan)"


@pytest.mark.asyncio
async def test_rbac_denies_below_min_role(services):
    # viewer cannot call an analyst-gated tool
    viewer = tool_ctx(services, make_sc(roles=("viewer",)))
    out = await services.mcp.call_tool("trigger_rescan", {"asset": "x"}, viewer)
    assert isinstance(out, ToolError) and out.code in ("forbidden", "requires_approval")
    # specifically forbidden for viewer
    assert out.code == "forbidden"


@pytest.mark.asyncio
async def test_tools_return_errors_as_data_not_raise(services):
    ctx = tool_ctx(services, make_sc(roles=("admin",)))
    # bad args should come back as ToolError, never raise
    out = await services.mcp.call_tool("get_report_metadata", {"doc_id": "DOES-NOT-EXIST"}, ctx)
    assert isinstance(out, (ToolResult, ToolError))


@pytest.mark.asyncio
async def test_module_golden_questions_route_correctly(services):
    for golden in CAPS.glob("*/evals/golden.jsonl"):
        for line in golden.read_text().splitlines():
            if not line.strip():
                continue
            case = json.loads(line)
            if "expect_route" not in case:
                continue
            sc = make_sc(org=case.get("org_id", "org_acme"), roles=("admin",))
            rr = await services.supervisor.route(case["question"], sc)
            for expected in case["expect_route"]:
                assert expected in rr.modules, f"{case['id']}: {expected} not in {rr.modules}"
