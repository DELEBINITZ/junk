from langgraph.types import interrupt


SIDE_EFFECTING_TOOLS = {"trigger_rescan"}


async def check_human_approval(tool_name: str, tool_args: dict) -> bool:
    """Gate side-effecting tools with human-in-the-loop approval via LangGraph interrupt.

    When a side-effecting tool is called, this suspends graph execution
    until a human approves or rejects the action.

    Returns True if approved, False if rejected.
    """
    if tool_name not in SIDE_EFFECTING_TOOLS:
        return True

    approval = interrupt(
        value={
            "type": "human_approval_required",
            "tool": tool_name,
            "arguments": tool_args,
            "message": f"Action '{tool_name}' requires human approval before execution.",
        }
    )

    return approval.get("approved", False)


def register_side_effecting_tool(tool_name: str):
    """Register a tool as side-effecting (requires human approval)."""
    SIDE_EFFECTING_TOOLS.add(tool_name)


def is_side_effecting(tool_name: str) -> bool:
    """Check if a tool requires human approval."""
    return tool_name in SIDE_EFFECTING_TOOLS
