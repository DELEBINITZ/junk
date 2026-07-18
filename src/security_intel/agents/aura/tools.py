"""EASM agent tools — loaded from the configured remote MCP server.

Tools come from the real EASM MCP server (configured via EASM_MCP_URL or
MCP_SERVERS). When no server is configured, no tools are returned and the EASM
agent is skipped at build time — we never serve fabricated/stub findings.
"""

from langchain_core.tools import BaseTool

from security_intel.config import Settings
from security_intel.tools.mcp_loader import load_mcp_tools_for_agent
from security_intel.observability.logging import get_logger

logger = get_logger("easm.tools")


async def get_easm_tools(settings: Settings) -> list[BaseTool]:
    """Get EASM tools from the configured MCP server.

    Returns an empty list when no MCP server is configured. The registry then
    skips building the EASM agent rather than exposing placeholder data.
    """
    mcp_tools = await load_mcp_tools_for_agent("easm", settings)
    if not mcp_tools:
        logger.warning(
            "No EASM MCP tools configured (set EASM_MCP_URL or MCP_SERVERS) — "
            "EASM agent will be skipped."
        )
    return mcp_tools
