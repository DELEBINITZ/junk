"""Dynamic MCP tool loading - config-driven, extensible.

Add new MCP servers via MCP_SERVERS env var (JSON):
    MCP_SERVERS='{"easm": {"url": "http://...", "transport": "streamable_http", "api_key": "..."}}'

A new MCP-backed agent needs no orchestrator/router/planner edits — its tools are
discovered here at startup and it is auto-registered from the same config (see
main.py _register_mcp_agents). The router, planner, and derived persona all follow
the enabled agent set automatically.

API note: targets langchain-mcp-adapters 0.3.x. `get_tools()` is async and sessions
are per-call (no persistent context manager needed):
    client = MultiServerMCPClient(mcp_config)
    tools = await client.get_tools(server_name=agent_id)
"""

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from security_intel.config import Settings
from security_intel.observability.logging import get_logger

logger = get_logger("mcp_loader")


async def load_mcp_tools_for_agent(agent_id: str, settings: Settings) -> list[BaseTool]:
    """Load tools from a specific MCP server by agent ID.

    Returns an empty list if no server is configured OR if the server is
    unreachable — a down MCP server logs and skips its agent, never crashes startup.
    """
    servers = settings.mcp_servers_config
    server_config = servers.get(agent_id)

    if not server_config:
        # Fallback: check dedicated env vars (EASM_MCP_URL etc.)
        if agent_id == "easm" and settings.easm_mcp_url:
            server_config = {
                "url": settings.easm_mcp_url,
                "transport": settings.easm_mcp_transport or "streamable_http",
                "api_key": settings.easm_mcp_api_key,
            }
        else:
            return []

    if not server_config.get("url"):
        return []

    mcp_config = _build_mcp_config(agent_id, server_config)

    try:
        client = MultiServerMCPClient(mcp_config)
        return await client.get_tools(server_name=agent_id)
    except Exception as e:  # noqa: BLE001 — unreachable/misconfigured server must not crash startup
        logger.warning(f"MCP tool load failed for '{agent_id}' ({e}); agent will be skipped")
        return []


def mcp_agent_ids(settings: Settings) -> list[str]:
    """Ids of every MCP server declared in config (MCP_SERVERS + the EASM shortcut).

    Used by the startup auto-registration loop so a new MCP-backed agent is picked up
    from config alone — no orchestrator/router/planner edits.
    """
    ids = [aid for aid, cfg in settings.mcp_servers_config.items() if cfg.get("url")]
    if "easm" not in ids and settings.easm_mcp_url:
        ids.append("easm")
    return ids


def _build_mcp_config(agent_id: str, server_config: dict) -> dict:
    """Build langchain-mcp-adapters config for a single server."""
    config = {
        agent_id: {
            "url": server_config["url"],
            # streamable_http is the current MCP transport; sse is deprecated in the spec.
            "transport": server_config.get("transport", "streamable_http"),
        }
    }

    api_key = server_config.get("api_key", "")
    if api_key:
        config[agent_id]["headers"] = {"Authorization": f"Bearer {api_key}"}

    return config
