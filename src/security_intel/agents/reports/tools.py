from langchain_core.tools import BaseTool

from security_intel.config import Settings
from security_intel.tools.qdrant_search import (
    build_search_reports_tool,
    build_get_report_metadata_tool,
    build_search_by_filter_tool,
    build_get_report_content_tool,
)


def get_reports_tools(settings: Settings, enricher=None) -> list[BaseTool]:
    """Build all tools for the Reports agent.

    Args:
        settings: App configuration.
        enricher: Optional QueryEnricher for multi-query fan-out and adaptive retrieval.
    """
    return [
        build_search_reports_tool(settings, enricher=enricher),
        build_get_report_metadata_tool(settings),
        build_search_by_filter_tool(settings),
        build_get_report_content_tool(settings),
    ]
