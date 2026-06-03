"""Test-kit capability — a UTILITY module backed by the `mcp-test-kits` MCP server.

Demonstrates integrating a THIRD-PARTY MCP server (github.com/midodimori/mcp-test-kits)
as a first-class capability: its tools (echo/add/multiply/reverse_string/generate_uuid/
get_timestamp) become tools the agent can route to and call. Local stubs here are the
contract + offline fallback; set TESTKIT_MCP_URL to route execution to the real server.
"""
