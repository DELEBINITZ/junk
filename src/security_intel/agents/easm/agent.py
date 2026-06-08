from langchain_openai import ChatOpenAI
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent
from langgraph.graph.state import CompiledStateGraph

from security_intel.config import Settings
from security_intel.agents.easm.tools import get_easm_tools

EASM_SYSTEM_PROMPT = """You are an External Attack Surface Management (EASM) specialist agent.

Your role: Answer questions about the organization's external-facing assets, exposures, and attack surface changes.

Instructions:
1. Use query_assets to find specific assets (domains, IPs, services).
2. Use get_exposures to check for vulnerabilities and misconfigurations.
3. Use get_asset_changes to see recent surface changes (new/removed/modified assets).
4. Use trigger_rescan only when explicitly asked - this is a side-effecting action.
5. Be precise about severity levels and asset details.
6. Return structured findings the orchestrator can synthesize.

Your output should be a clear summary of the attack surface status with specific asset details."""


async def build_easm_agent(
    llm: ChatOpenAI,
    settings: Settings,
    mcp_tools: list[BaseTool] | None = None,
) -> CompiledStateGraph:
    """Build the EASM specialist ReAct agent.

    Args:
        llm: The LLM to use for reasoning.
        settings: App settings.
        mcp_tools: Pre-loaded MCP tools (from app lifespan). If None, loads fresh.
    """
    if mcp_tools is not None:
        tools = mcp_tools
    else:
        tools = await get_easm_tools(settings)

    return create_react_agent(
        model=llm,
        tools=tools,
        prompt=EASM_SYSTEM_PROMPT,
    )
