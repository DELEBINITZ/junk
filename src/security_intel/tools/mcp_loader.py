"""Dynamic MCP tool loading - config-driven, extensible.

Add new MCP servers via MCP_SERVERS env var (JSON):
    MCP_SERVERS='{"easm": {"url": "http://...", "transport": "sse", "api_key": "..."}}'

No code changes needed to add new agents — just config + planner prompt update.
"""

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from security_intel.config import Settings


async def load_mcp_tools_for_agent(agent_id: str, settings: Settings) -> list[BaseTool]:
    """Load tools from a specific MCP server by agent ID.

    Returns empty list if no server configured for this agent.
    """
    servers = settings.mcp_servers_config
    server_config = servers.get(agent_id)

    if not server_config:
        # Fallback: check dedicated env vars (EASM_MCP_URL etc.)
        if agent_id == "easm" and settings.easm_mcp_url:
            server_config = {
                "url": settings.easm_mcp_url,
                "transport": "sse",
                "api_key": settings.easm_mcp_api_key,
            }
        else:
            return []

    mcp_config = _build_mcp_config(agent_id, server_config)

    async with MultiServerMCPClient(mcp_config) as client:
        tools = client.get_tools()

    return tools


async def load_all_mcp_tools(settings: Settings) -> dict[str, list[BaseTool]]:
    """Load tools from ALL configured MCP servers.

    Returns: {agent_id: [tools]}
    """
    servers = settings.mcp_servers_config
    result = {}

    # Add dedicated EASM config if not in mcp_servers
    if "easm" not in servers and settings.easm_mcp_url:
        servers["easm"] = {
            "url": settings.easm_mcp_url,
            "transport": "sse",
            "api_key": settings.easm_mcp_api_key,
        }

    for agent_id, server_config in servers.items():
        if not server_config.get("url"):
            continue
        try:
            tools = await load_mcp_tools_for_agent(agent_id, settings)
            result[agent_id] = tools
        except Exception as e:
            print(f"Warning: Failed to load MCP tools for '{agent_id}': {e}")
            result[agent_id] = []

    return result


async def create_mcp_client(settings: Settings) -> MultiServerMCPClient | None:
    """Create a persistent multi-server MCP client for the app lifespan.

    Connects to ALL configured MCP servers simultaneously.
    Caller must call `await client.__aexit__(None, None, None)` on shutdown.
    """
    servers = settings.mcp_servers_config

    # Add dedicated EASM config
    if "easm" not in servers and settings.easm_mcp_url:
        servers["easm"] = {
            "url": settings.easm_mcp_url,
            "transport": "sse",
            "api_key": settings.easm_mcp_api_key,
        }

    if not servers:
        return None

    mcp_config = {}
    for agent_id, server_config in servers.items():
        if not server_config.get("url"):
            continue
        mcp_config.update(_build_mcp_config(agent_id, server_config))

    if not mcp_config:
        return None

    client = MultiServerMCPClient(mcp_config)
    await client.__aenter__()
    return client


def _build_mcp_config(agent_id: str, server_config: dict) -> dict:
    """Build langchain-mcp-adapters config for a single server."""
    config = {
        agent_id: {
            "url": server_config["url"],
            "transport": server_config.get("transport", "sse"),
        }
    }

    api_key = server_config.get("api_key", "")
    if api_key:
        config[agent_id]["headers"] = {"Authorization": f"Bearer {api_key}"}

    return config
