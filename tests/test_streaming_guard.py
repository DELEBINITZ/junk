"""B1 regression: streamed tokens are GUARDED, never raw generation.

The output guardrail (PII redaction / groundedness / injection) runs AFTER the
answer is generated. So answer_node must BUFFER (emit no tokens), and
output_guardrail_node is the only place a generated answer is streamed — after the
guard has redacted/blocked it. These tests lock that ordering: unredacted PII must
never appear in a ``token`` event.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.config import Settings
from app.core.agent.nodes import answer_node, output_guardrail_node
from app.core.agent.state import AgentContext, AgentEvent
from app.core.guardrails.base import Action, GuardrailVerdict
from app.core.guardrails.pipeline import OutputGuardrailPipeline
from app.core.llm.base import NO_CONTEXT_REFUSAL, LLMResponse


class _FakeLLM:
    """Streams/returns a fixed answer. ``stream`` yields it in two chunks so a test
    that wrongly forwards live tokens would surface them."""

    provider = "fake"

    def __init__(self, text: str) -> None:
        self._text = text

    async def stream(self, messages, *, lane=None, **kw):
        mid = len(self._text) // 2
        for part in (self._text[:mid], self._text[mid:]):
            if part:
                yield part

    async def complete(self, messages, *, lane=None, **kw):
        return LLMResponse(text=self._text)


class _EmailRedactor:
    """Minimal output detector: replaces a known PII span with a marker (REDACT)."""

    name = "fake_pii"

    async def check(self, text: str, ctx) -> GuardrailVerdict:
        if "secret@evil.com" in text:
            return GuardrailVerdict(
                detector=self.name, action=Action.REDACT,
                text=text.replace("secret@evil.com", "[REDACTED_EMAIL]"),
                reason="pii",
            )
        return GuardrailVerdict(detector=self.name)


def _ctx(*, llm=None, output_guard=None, emit=None, stream=False) -> AgentContext:
    """An AgentContext with only the fields the answer/guard nodes touch."""
    return AgentContext(
        deps=SimpleNamespace(llm=llm, rag=SimpleNamespace(embedder=None)),
        sc=None,
        tool_ctx=None,
        mcp=None,
        registry=SimpleNamespace(module=lambda _mid: None),   # no per-module prompt
        input_guard=None,
        output_guard=output_guard,
        settings=Settings(_env_file=None),
        emit=emit,
        stream_tokens=stream,
    )


def _collector():
    events: list[AgentEvent] = []

    async def emit(ev: AgentEvent) -> None:
        events.append(ev)

    return events, emit


@pytest.mark.asyncio
async def test_answer_node_buffers_and_emits_no_tokens():
    # Even with streaming on, answer_node must NOT emit token events — the guard
    # has not run yet. It only buffers the generation into state["answer"].
    events, emit = _collector()
    ctx = _ctx(llm=_FakeLLM("Reach me at secret@evil.com please"), emit=emit, stream=True)
    out = await answer_node({"question": "hi", "context_chunks": []}, ctx)
    assert out["answer"] == "Reach me at secret@evil.com please"
    assert not [e for e in events if e.type == "token"]   # nothing streamed pre-guard


@pytest.mark.asyncio
async def test_answer_node_empty_generation_falls_back():
    ctx = _ctx(llm=_FakeLLM("   "), stream=False)
    out = await answer_node({"question": "hi", "context_chunks": []}, ctx)
    assert out["answer"] == NO_CONTEXT_REFUSAL


@pytest.mark.asyncio
async def test_output_guard_streams_only_redacted_text():
    # THE guarantee: tokens emitted by the guard node carry the REDACTED answer,
    # and the raw PII never appears in any token event.
    events, emit = _collector()
    guard = OutputGuardrailPipeline([_EmailRedactor()], groundedness=False)
    ctx = _ctx(output_guard=guard, emit=emit, stream=True)
    state = {"answer": "mail me at secret@evil.com now", "context_chunks": []}

    out = await output_guardrail_node(state, ctx)

    streamed = "".join(e.data["text"] for e in events if e.type == "token")
    assert out["answer"] == "mail me at [REDACTED_EMAIL] now"
    assert streamed == out["answer"]                  # client reconstructs the guarded text exactly
    assert "secret@evil.com" not in streamed          # raw PII never streamed


@pytest.mark.asyncio
async def test_guard_node_does_not_stream_when_not_streaming():
    events, emit = _collector()
    guard = OutputGuardrailPipeline([_EmailRedactor()], groundedness=False)
    ctx = _ctx(output_guard=guard, emit=emit, stream=False)   # stream off
    await output_guardrail_node({"answer": "hello", "context_chunks": []}, ctx)
    assert not [e for e in events if e.type == "token"]
