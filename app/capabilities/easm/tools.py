"""EASM tools.

Query the EASM product backend (asset inventory, vuln scanner) over HTTP, scoped
to ctx.org_id (the trusted org from the JWT — never an org passed in args). If
EASM_API_URL is unset they return errors-as-data (ToolException 503) — never
fabricated assets. When promoted to its own MCP server (easm-mcp) these handlers
call that server via a remote MCP client; the Tool contract is unchanged.
See plan §6.4, §14.
"""

from __future__ import annotations

from typing import Any

import httpx

from app.config import settings
from app.core.contracts import Tool, ToolContext, ToolException


def _easm_get(path: str, params: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    if not settings.easm_api_url:
        raise ToolException("EASM backend is not configured in this environment.", code=503, kind="backend_unavailable")
    headers = {"X-Org-Id": ctx.org_id}  # backend enforces tenancy from this trusted header
    if settings.easm_api_token:
        headers["Authorization"] = f"Bearer {settings.easm_api_token}"
    try:
        response = httpx.get(
            f"{settings.easm_api_url.rstrip('/')}{path}",
            params={k: v for k, v in params.items() if v is not None},
            headers=headers,
            timeout=10.0,
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as exc:
        raise ToolException(f"EASM backend error: {exc}", code=502, kind="backend_error") from exc


def _query_assets(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    return _easm_get("/assets", {
        "q": args.get("query"),
        "asset_type": ",".join(args.get("asset_type", []) or []),
        "limit": args.get("limit", 20),
    }, ctx)


def _get_vulnerabilities(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    return _easm_get("/vulnerabilities", {
        "severity": ",".join(args.get("severity", []) or []),
        "exploited_in_wild": args.get("exploited_in_wild"),
        "limit": args.get("limit", 20),
    }, ctx)


def _get_asset_changes(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    return _easm_get("/asset-changes", {"since": args.get("since"), "severity": args.get("severity")}, ctx)


EASM_TOOLS: list[Tool] = [
    Tool(
        name="query_assets",
        description="Search the organization's internet-facing asset inventory "
        "(domains, IPs, certs, cloud, open ports) with filters.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "asset_type": {"type": "array", "items": {"type": "string"}},
                "filters": {"type": "object"},
                "limit": {"type": "integer", "default": 20},
            },
            "required": ["query"],
        },
        handler=_query_assets,
        rbac_role="analyst",
    ),
    Tool(
        name="get_vulnerabilities",
        description="List vulnerabilities for the org's assets, filterable by "
        "severity and active-exploitation status.",
        input_schema={
            "type": "object",
            "properties": {
                "severity": {"type": "array", "items": {"type": "string"}},
                "exploited_in_wild": {"type": "boolean"},
                "limit": {"type": "integer", "default": 20},
            },
        },
        handler=_get_vulnerabilities,
        rbac_role="analyst",
    ),
    Tool(
        name="get_asset_changes",
        description="Return attack-surface change events (new assets, opened "
        "ports, expiring certs) since a given date.",
        input_schema={
            "type": "object",
            "properties": {
                "since": {"type": "string", "format": "date"},
                "severity": {"type": "string"},
            },
            "required": ["since"],
        },
        handler=_get_asset_changes,
        rbac_role="analyst",
    ),
]
