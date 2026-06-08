"""Test the Strategic Planner agent builds correctly and has expected tools."""

import pytest
from unittest.mock import MagicMock

from security_intel.agents.planner import build_planner, PLANNER_SYSTEM_PROMPT


def test_planner_builds():
    """Planner should build a compiled LangGraph agent."""
    mock_llm = MagicMock()
    mock_llm.bind_tools = MagicMock(return_value=mock_llm)
    planner = build_planner(mock_llm)
    assert planner is not None


def test_planner_system_prompt_contains_agents():
    """System prompt should describe available agents."""
    assert "reports" in PLANNER_SYSTEM_PROMPT
    assert "easm" in PLANNER_SYSTEM_PROMPT
    assert "create_execution_plan" in PLANNER_SYSTEM_PROMPT
