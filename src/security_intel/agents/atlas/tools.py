from langchain_core.tools import BaseTool

from security_intel.config import Settings
from security_intel.tools.userguide_search import (
    build_get_user_guide_page_tool,
    build_search_user_guide_tool,
)


def get_user_guide_tools(settings: Settings, enricher=None) -> list[BaseTool]:
    """Build the user_guide capability tools for the Atlas agent.

    Args:
        settings: App configuration.
        enricher: Optional QueryEnricher for multi-query fan-out and adaptive retrieval.
    """
    return [
        build_search_user_guide_tool(settings, enricher=enricher),
        build_get_user_guide_page_tool(settings),
    ]
