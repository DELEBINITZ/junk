"""EASM agent tools — loads from remote MCP server or uses local stubs.

Production: tools come from real MCP server (configured via EASM_MCP_URL or MCP_SERVERS).
Development: stub tools simulate the EASM API for local testing.
"""

from langchain_core.tools import BaseTool, tool

from security_intel.config import Settings
from security_intel.tools.mcp_loader import load_mcp_tools_for_agent


async def get_easm_tools(settings: Settings) -> list[BaseTool]:
    """Get EASM tools - from remote MCP server if configured, else local stubs."""
    mcp_tools = await load_mcp_tools_for_agent("easm", settings)
    if mcp_tools:
        return mcp_tools
    return _get_stub_tools()


def _get_stub_tools() -> list[BaseTool]:
    """Local stub tools for development before real MCP server is connected."""

    @tool
    async def query_assets(query: str = "", limit: int = 10) -> str:
        """List external assets matching a query. Shows domains, IPs, and services exposed to the internet.

        Args:
            query: Search filter (domain name, IP range, service type). Empty returns all.
            limit: Max results (1-50, default 10).
        """
        return (
            "External Assets:\n"
            "1. api.example.com - 203.0.113.10 - HTTPS/443 - Active\n"
            "2. mail.example.com - 203.0.113.11 - SMTP/25, HTTPS/443 - Active\n"
            "3. vpn.example.com - 203.0.113.12 - OpenVPN/1194 - Active\n"
            "4. legacy.example.com - 203.0.113.13 - HTTP/80 - Stale (no cert)\n"
            "5. dev.example.com - 203.0.113.14 - HTTPS/443 - Active"
        )

    @tool
    async def get_exposures(severity: str = "") -> str:
        """Get current external exposures and findings from attack surface monitoring.

        Args:
            severity: Filter by severity (critical, high, medium, low). Empty returns all.
        """
        return (
            "Exposures Found:\n"
            "1. [CRITICAL] legacy.example.com - Expired TLS certificate (expired 2024-01-15)\n"
            "2. [HIGH] dev.example.com - Open admin panel (/admin) with default credentials\n"
            "3. [MEDIUM] mail.example.com - SPF record misconfiguration\n"
            "4. [LOW] api.example.com - Server version disclosed in headers"
        )

    @tool
    async def get_asset_changes(since: str = "7d") -> str:
        """Get recent attack surface changes (new assets, removed assets, config changes).

        Args:
            since: Time window (e.g., '7d', '30d', '24h'). Default '7d'.
        """
        return (
            "Changes (last 7 days):\n"
            "- NEW: staging.example.com (203.0.113.15) - First seen 2 days ago\n"
            "- CHANGED: api.example.com - New port 8080 opened\n"
            "- REMOVED: old-blog.example.com - No longer resolving"
        )

    @tool
    async def trigger_rescan(asset: str) -> str:
        """Request an immediate rescan of a specific asset. This is a SIDE-EFFECTING action requiring approval.

        Args:
            asset: The asset hostname or IP to rescan.
        """
        return f"Rescan queued for {asset}. Results available in ~5 minutes."

    return [query_assets, get_exposures, get_asset_changes, trigger_rescan]
