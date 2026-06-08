"""Test configuration loading."""

import os
import pytest

from security_intel.config import Settings


def test_defaults():
    """Settings should load with sane defaults."""
    s = Settings()
    assert s.llm_base_url == "http://localhost:30000/v1"
    assert s.llm_model == "Qwen/Qwen2.5-72B-Instruct"
    assert s.llm_deep_model == "deepseek-ai/DeepSeek-V3"
    assert s.history_window_messages == 20


def test_api_key_list():
    """API keys should split on comma."""
    s = Settings(api_keys="key1, key2, key3")
    assert s.api_key_list == ["key1", "key2", "key3"]


def test_mcp_servers_config():
    """MCP servers JSON should parse correctly."""
    s = Settings(mcp_servers='{"easm": {"url": "http://easm:8000", "transport": "sse"}}')
    config = s.mcp_servers_config
    assert "easm" in config
    assert config["easm"]["url"] == "http://easm:8000"


def test_mcp_servers_empty():
    """Empty MCP servers should return empty dict."""
    s = Settings(mcp_servers="{}")
    assert s.mcp_servers_config == {}
