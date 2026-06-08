from langchain_core.tools import BaseTool

from security_intel.config import Settings
from security_intel.tools.qdrant_search import (
    build_search_reports_tool,
    build_get_report_metadata_tool,
    build_search_by_filter_tool,
)


def get_reports_tools(settings: Settings) -> list[BaseTool]:
    """Build all tools for the Reports agent."""
    return [
        build_search_reports_tool(settings),
        build_get_report_metadata_tool(settings),
        build_search_by_filter_tool(settings),
    ]
