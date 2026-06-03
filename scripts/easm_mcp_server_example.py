"""Reference: a standalone ``easm-mcp`` FastMCP server — the EASM promotion TARGET.

This is the server your platform talks to once you set ``EASM_MCP_URL``. It is the
same shape as the ``mcp-test-kits`` server you cloned, but it exposes EASM's OWN
tools instead of echo/add. Built on FastMCP because the default remote executor in
bootstrap (``FastMCPRemote``, app/core/mcp/fastmcp_client.py) speaks FastMCP.

THE THREE RULES that make remote EASM a drop-in for in-process EASM:

  1. TOOL NAMES MATCH THE MANIFEST. Each @mcp.tool name below is identical to a
     tool in app/capabilities/easm/tools.py (query_assets, get_exposures,
     get_live_asset_count, trigger_rescan). The boundary routes by name, so names
     must line up 1:1 or the call won't map.

  2. RETURN {data, citations}. FastMCPRemote._to_outcome rebuilds a ToolResult from
     the server's structured output. Return that dict and the agent gets the exact
     same typed object it would from the local tool.

  3. ORG COMES FROM THE TOKEN, NOT THE ARGS. The platform mints a short-lived,
     org-scoped service token (bootstrap._service_token_minters) and sends it as a
     Bearer credential. This server must VERIFY it and re-derive org from the
     claims — never read org from a tool argument. See the verify_token note below;
     it mirrors app/core/mcp/server.py::_sc which does exactly this with decode_token.

RUN IT:
    uv run --with fastmcp python scripts/easm_mcp_server_example.py    # serves :9001/mcp

THEN POINT THE PLATFORM AT IT — no code change, just config:
    EASM_MCP_URL=http://localhost:9001/mcp
(config.py: easm_mcp_url -> bootstrap builds FastMCPRemote -> EASM tool calls go remote.)
"""

from __future__ import annotations

from fastmcp import FastMCP
from pydantic import Field

mcp = FastMCP("easm-mcp")

# Stand-in for the EASM asset DB, org-scoped — identical data to tools.py's _MOCK.
# In production each tool queries your real asset store, scoped by the token's org.
_MOCK: dict[str, dict] = {
    "org_acme": {
        "assets": [
            {"asset_id": "A-1", "host": "admin.acme.test", "type": "host", "tech": "Atlassian Confluence", "exposed": True},
            {"asset_id": "A-2", "host": "shop.acme.test", "type": "host", "tech": "nginx", "exposed": True},
            {"asset_id": "A-4", "host": "api.acme.test", "type": "host", "tech": "Kong", "exposed": True},
        ],
        "exposures": [
            {"exposure_id": "E-1", "asset": "admin.acme.test", "issue": "CVE-2023-22515 critical auth bypass on Confluence", "severity": "critical"},
        ],
    },
}


def _org_from_token() -> str:
    """RULE 3 — where the Bearer service token is verified and org re-derived.

    PRODUCTION (mirror app/core/mcp/server.py::_sc):
        from fastmcp.server.dependencies import get_http_headers
        from app.core.security.jwt import decode_token
        token = get_http_headers().get("authorization", "")[7:]
        claims = decode_token(settings, token, expected_type="access")
        return str(claims["org_id"])          # org ALWAYS from the verified token

    For this offline demo we skip verification and pin a single tenant so the file
    runs with no JWT setup. NEVER ship the line below.
    """
    return "org_acme"


def _cite(doc_id: str, title: str, snippet: str) -> dict:
    # The Citation shape your platform expects inside ToolResult.citations.
    return {"doc_id": doc_id, "source": "easm", "title": title, "snippet": snippet}


@mcp.tool
def query_assets(
    query: str = Field(default="", description="Free-text filter over host/tech/type. Empty = all."),
    limit: int = Field(default=10, ge=1, le=100),
) -> dict:
    """List the organization's external assets (domains, hosts, VPNs) matching a query."""
    org = _org_from_token()
    assets = _MOCK.get(org, {}).get("assets", [])
    if query:
        q = query.lower()
        assets = [a for a in assets if q in f"{a['host']} {a['tech']} {a['type']}".lower()] or assets
    assets = assets[:limit]
    cites = [_cite(a["asset_id"], a["host"],
                   f"Exposed external asset {a['host']} running {a['tech']}.") for a in assets]
    # RULE 2: {data, citations} -> FastMCPRemote rebuilds the same ToolResult.
    return {"data": {"assets": assets, "count": len(assets)}, "citations": cites}


@mcp.tool
def get_exposures(severity: str | None = Field(default=None, description="critical|high|medium|low")) -> dict:
    """Get current external exposures/findings for the organization, optionally by severity."""
    org = _org_from_token()
    exps = _MOCK.get(org, {}).get("exposures", [])
    if severity:
        exps = [e for e in exps if e["severity"] == severity.lower()]
    cites = [_cite(e["exposure_id"], e["asset"], f"{e['asset']}: {e['issue']} ({e['severity']}).") for e in exps]
    return {"data": {"exposures": exps, "count": len(exps)}, "citations": cites}


@mcp.tool
def get_live_asset_count(live_only: bool = True) -> dict:
    """Get the count of the organization's live (internet-exposed) external assets."""
    org = _org_from_token()
    assets = _MOCK.get(org, {}).get("assets", [])
    n = len([a for a in assets if a.get("exposed")]) if live_only else len(assets)
    return {"data": {"live_asset_count": n, "live_only": live_only},
            "citations": [_cite("easm-asset-count", "Live asset count",
                                f"The organization has {n} live internet-exposed external assets.")]}


# Side-effecting tool. The MCP annotation marks it non-read-only; FastMCPRemote
# passes that hint through (read_only_hint/destructive_hint) so your boundary still
# routes it to the human action gate. RBAC (analyst) is enforced LOCALLY before the
# call ever leaves the platform — the remote server never sees a viewer's request.
@mcp.tool(annotations={"destructiveHint": True, "readOnlyHint": False})
def trigger_rescan(asset: str = Field(description="Host or asset_id to rescan.")) -> dict:
    """Request an on-demand external rescan of an asset. Side-effecting: requires approval."""
    # Reached only AFTER local RBAC + the action gate approved it. Enqueue scoped to org.
    return {"data": {"asset": asset, "status": "rescan_queued"}, "citations": []}


if __name__ == "__main__":
    # Same transport the test-kit used (Streamable-HTTP). Endpoint: http://localhost:9001/mcp
    mcp.run(transport="http", host="localhost", port=9001)
