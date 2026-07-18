"""SSE streaming adapter — bridges LangGraph astream_events to frontend EventSource.

Event contract (matches frontend/index.html):
- token: {"text": "..."} — incremental LLM output
- tool: {"module": "...", "step": "..."} — tool invocation
- status: {"stage": "..."} — node transitions
- plan: {"steps": [...]} — execution plan
- thinking: {"stage": "...", "text": "..."} — agent reasoning
- done: {"answer": "...", "session_id": "...", "citations": [...]} — final response
- error: {"message": "..."} — error
"""

import json
from typing import AsyncIterator

from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from security_intel.observability.logging import get_logger
from security_intel.security.guardrails import StreamingRedactor

logger = get_logger("streaming")


async def stream_agent_events(
    orchestrator: CompiledStateGraph,
    input_state: dict,
    config: RunnableConfig,
) -> AsyncIterator[str]:
    """Stream orchestrator events as SSE-formatted strings."""
    session_id = config["configurable"].get("thread_id", "")
    yield _sse("session", {"session_id": session_id})

    # Only stream LLM tokens from these nodes to the user
    _STREAMABLE_NODES = {"synthesize", "chitchat", "capability_redirect"}

    try:
        final_answer = ""
        current_node = ""
        collected_tokens = []
        redactor = StreamingRedactor()

        async for event in orchestrator.astream_events(input_state, config=config, version="v2"):
            event_type = event.get("event", "")
            # langgraph_node in metadata is the reliable way to identify source node
            event_node = event.get("metadata", {}).get("langgraph_node", "")

            if event_type == "on_chat_model_stream":
                chunk = event.get("data", {}).get("chunk")
                if chunk and hasattr(chunk, "content") and chunk.content:
                    if event_node in _STREAMABLE_NODES:
                        # Redact PII before it reaches the UI (matches final answer).
                        safe = await redactor.feed(chunk.content)
                        if safe:
                            yield _sse("token", {"text": safe})
                            collected_tokens.append(safe)

            elif event_type == "on_tool_start":
                if event_node in ("dispatch", "plan"):
                    tool_name = event.get("name", "")
                    tool_input = event.get("data", {}).get("input", {})
                    yield _sse(
                        "tool",
                        {
                            "module": tool_name,
                            "step": tool_name,
                            "input": _truncate(str(tool_input), 200),
                        },
                    )

            elif event_type == "on_tool_end":
                if event_node in ("dispatch", "plan"):
                    tool_name = event.get("name", "")
                    output = event.get("data", {}).get("output", "")
                    if hasattr(output, "content"):
                        output = output.content
                    if tool_name == "create_execution_plan":
                        yield _sse("plan", {"steps": [], "summary": _truncate(str(output), 300)})
                    else:
                        found = str(output).count("\n") + 1 if output else None
                        yield _sse(
                            "tool",
                            {
                                "module": tool_name,
                                "step": tool_name,
                                "found": found,
                            },
                        )

            elif event_type == "on_chain_start":
                name = event.get("name", "")
                if name in (
                    "context_and_classify",
                    "plan",
                    "dispatch",
                    "synthesize",
                    "capability_redirect",
                    "security_gate",
                    "output_guardrail",
                ):
                    current_node = name
                    yield _sse("status", {"stage": _friendly_stage(name)})

        # Flush any redacted tail held back during streaming.
        tail = await redactor.flush()
        if tail:
            yield _sse("token", {"text": tail})
            collected_tokens.append(tail)

        try:
            final_state = await orchestrator.aget_state(config)
            if final_state and final_state.values:
                vals = final_state.values

                # Security-blocked requests end the graph early with no answer.
                # Surface a generic notice — never the internal block reason.
                if vals.get("blocked"):
                    yield _sse(
                        "done",
                        {
                            "answer": (
                                "I can't help with that request — it was flagged by our "
                                "security policy. Please try rephrasing it as a normal "
                                "question about what I can help you with."
                            ),
                            "session_id": session_id,
                            "citations": [],
                        },
                    )
                    return

                final_answer = vals.get("final_answer", "")
                citations = vals.get("citations", [])
                agents_used = [r["agent_id"] for r in vals.get("agent_results", [])]

                yield _sse(
                    "done",
                    {
                        "answer": final_answer,
                        "session_id": session_id,
                        "citations": citations,
                        "agents_used": agents_used,
                        "is_complex": vals.get("is_complex", False),
                    },
                )
                return
        except ValueError:
            pass

        final_answer = "".join(collected_tokens) if collected_tokens else ""
        yield _sse(
            "done",
            {
                "answer": final_answer,
                "session_id": session_id,
                "citations": [],
            },
        )

    except Exception as e:
        # Log the real error; never leak internals (stack/DB/host info) to the UI.
        logger.error(f"Streaming error: {e}", exc_info=True)
        yield _sse(
            "error",
            {"message": "Something went wrong while generating the response. Please try again."},
        )


def _sse(event_name: str, data: dict) -> str:
    """Format as Server-Sent Event."""
    return f"event: {event_name}\ndata: {json.dumps(data, default=str)}\n\n"


def _friendly_stage(node_name: str) -> str:
    """Convert node name to user-friendly stage description."""
    stages = {
        "security_gate": "Running security checks...",
        "context_and_classify": "Thinking...",
        "plan": "Creating execution plan...",
        "validate_plan": "Validating plan...",
        "dispatch": "Running specialist agents...",
        "synthesize": "Synthesizing answer...",
        "capability_redirect": "Thinking...",
        "output_guardrail": "Finalizing...",
    }
    return stages.get(node_name, node_name)


def _truncate(s: str, max_len: int) -> str:
    return s[:max_len] + "..." if len(s) > max_len else s
