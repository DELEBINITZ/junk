"""Test-kit capability MODULE — manifest.

Wires the `mcp-test-kits` MCP server in as a capability. Same fields as
easm/manifest.py; the ONLY thing that makes its tools execute on a remote MCP
server instead of locally is setting ``TESTKIT_MCP_URL`` (bootstrap then builds a
FastMCPRemote for module id "testkit"). No core edit is needed to ADD the module —
the registry discovers this MANIFEST from disk; the ``cap_testkit_enabled`` flag is
the on/off switch.

It is a pure tool-backed module (no ``retrievers``), like EASM, but every tool is
READ-only, so ``default_autonomy`` stays READ and nothing here touches the action
gate.
"""

from __future__ import annotations

from app.capabilities.testkit.tools import TOOLS
from app.core.contracts import Autonomy, CapabilityManifest

MANIFEST = CapabilityManifest(
    id="testkit",
    version="1.0.0",
    display_name="MCP Test Kit",
    description=(
        "Utility tools served by the mcp-test-kits MCP server: echo, add, multiply, "
        "reverse a string, generate a UUID, and get the current timestamp."
    ),
    license_tiers=("platform",),
    enabled_flag="cap_testkit_enabled",
    tools=TOOLS,
    system_prompt="prompts/v1.md",
    # SUPERVISOR routing is DYNAMIC: this module is chosen by the MEANING of the
    # question, scored against the ``description`` above + the tools' names and
    # descriptions (e.g. generate_uuid, get_timestamp, reverse_string). The utility
    # vocabulary lives there, not in curated keywords.
    # All tools are read-only utilities — no side effects, so READ autonomy.
    default_autonomy=Autonomy.READ,
    rbac={
        "echo": "viewer", "add": "viewer", "multiply": "viewer",
        "reverse_string": "viewer", "generate_uuid": "viewer", "get_timestamp": "viewer",
    },
    owners=("team-platform",),
)

__all__ = ["MANIFEST"]
