from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.graph.state import CompiledStateGraph

from security_intel.config import Settings
from security_intel.agents.reports.tools import get_reports_tools

REPORTS_SYSTEM_PROMPT = """You are a Security Reports specialist agent.

Your role: Answer questions using the organization's security reports corpus (threat intel, CVE findings, scan results, remediation guidance).

Instructions:
1. Use search_reports to find relevant information. Try multiple search queries if the first doesn't yield good results.
2. Use get_report_metadata for details about specific reports.
3. Always cite your sources - reference report titles and relevance scores.
4. Only answer based on what the reports contain. Say "no relevant reports found" if searches return nothing useful.
5. Be concise and factual. Return structured findings the orchestrator can synthesize.

Your output should be a clear summary of findings with citations."""


def build_reports_agent(llm: ChatOpenAI, settings: Settings) -> CompiledStateGraph:
    """Build the Reports specialist ReAct agent."""
    tools = get_reports_tools(settings)
    return create_react_agent(
        model=llm,
        tools=tools,
        prompt=REPORTS_SYSTEM_PROMPT,
    )
