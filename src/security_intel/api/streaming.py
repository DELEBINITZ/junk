import json
from typing import AsyncIterator

from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from security_intel.state.schemas import OrchestratorState


async def stream_agent_events(
    orchestrator: CompiledStateGraph,
    input_state: dict,
    config: RunnableConfig,
) -> AsyncIterator[str]:
    """Stream orchestrator events as SSE-formatted strings.

    Uses LangGraph's astream_events (v2) to emit:
    - token: LLM token chunks
    - tool_start: Tool invocation begins
    - tool_end: Tool invocation completes
    - node: Graph node transitions
    - done: Final answer
    """
    try:
        async for event in orchestrator.astream_events(
            input_state, config=config, version="v2"
        ):
            sse_event = _format_event(event)
            if sse_event:
                yield sse_event

        final_state = await orchestrator.aget_state(config)
        if final_state and final_state.values:
            answer = final_state.values.get("final_answer", "")
            if answer:
                yield _sse_format("done", {"answer": answer})

    except Exception as e:
        yield _sse_format("error", {"message": str(e)})


def _format_event(event: dict) -> str | None:
    """Convert a LangGraph event to SSE format."""
    event_type = event.get("event", "")

    if event_type == "on_chat_model_stream":
        chunk = event.get("data", {}).get("chunk")
        if chunk and hasattr(chunk, "content") and chunk.content:
            return _sse_format("token", {"content": chunk.content})

    elif event_type == "on_tool_start":
        return _sse_format("tool_start", {
            "name": event.get("name", ""),
            "input": event.get("data", {}).get("input", {}),
        })

    elif event_type == "on_tool_end":
        output = event.get("data", {}).get("output", "")
        if hasattr(output, "content"):
            output = output.content
        return _sse_format("tool_end", {
            "name": event.get("name", ""),
            "output": str(output)[:500],
        })

    elif event_type == "on_chain_start" and event.get("name", "").endswith("_node"):
        return _sse_format("node", {"name": event["name"]})

    return None


def _sse_format(event_name: str, data: dict) -> str:
    """Format as Server-Sent Event string."""
    return f"event: {event_name}\ndata: {json.dumps(data)}\n\n"
