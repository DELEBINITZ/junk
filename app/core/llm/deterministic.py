"""Deterministic, infra-free LLM (the default provider).

It produces a *grounded* answer by extracting the retrieved-context entries most
relevant to the question and citing them, or an honest refusal when nothing
relevant was retrieved. This makes the whole agent loop — retrieve, ground,
cite, refuse-when-unknown, stream — exercisable and testable with no GPU, keys,
or network, and keeps eval deterministic. Swap to SGLang/OpenAI via config.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator, Sequence

from app.core.llm.base import (
    NO_CONTEXT_REFUSAL,
    ChatMessage,
    Lane,
    LLMResponse,
    LLMToolSpec,
    LLMUsage,
)

_CTX_LINE = re.compile(r"^\s*\[(\d+)\]\s*(.+?)\s*$")
_STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "are",
    "was", "were", "what", "which", "who", "how", "when", "where", "why", "this",
    "that", "with", "from", "by", "at", "as", "it", "be", "do", "does", "did",
    "our", "my", "we", "i", "you", "me", "about", "any", "all", "show", "list",
}


def _tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP and len(w) > 1}


def _summarize(text: str, limit: int = 240) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rsplit(" ", 1)[0] + "…"


class DeterministicLLM:
    provider = "deterministic"

    def __init__(self, settings=None) -> None:
        self.settings = settings

    # -- internals -----------------------------------------------------------
    @staticmethod
    def _question(messages: Sequence[ChatMessage]) -> str:
        for m in reversed(messages):
            if m.role == "user" and m.content.strip():
                return m.content.strip()
        return ""

    @staticmethod
    def _context(messages: Sequence[ChatMessage]) -> list[tuple[int, str]]:
        out: list[tuple[int, str]] = []
        for m in messages:
            for line in m.content.splitlines():
                mt = _CTX_LINE.match(line)
                if mt:
                    out.append((int(mt.group(1)), mt.group(2)))
        return out

    def _answer_text(self, messages: Sequence[ChatMessage]) -> str:
        question = self._question(messages)
        ctx = self._context(messages)
        if not ctx:
            return NO_CONTEXT_REFUSAL
        q = _tokens(question)
        scored = sorted(
            ((len(q & _tokens(text)), idx, text) for idx, text in ctx),
            key=lambda t: (-t[0], t[1]),
        )
        top = [s for s in scored if s[0] > 0][:3] or scored[:1]
        if top and top[0][0] == 0:
            return NO_CONTEXT_REFUSAL
        parts = [f"{_summarize(text)} [{idx}]" for _score, idx, text in top]
        return "Based on the retrieved sources: " + " ".join(parts)

    # -- LLMClient interface -------------------------------------------------
    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        lane: Lane = Lane.STANDARD,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        text = self._answer_text(messages)
        return LLMResponse(
            text=text, model="deterministic", lane=lane,
            usage=LLMUsage(completion_tokens=len(text.split())),
        )

    async def stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        lane: Lane = Lane.STANDARD,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        text = self._answer_text(messages)
        words = text.split(" ")
        for i, w in enumerate(words):
            yield (w if i == 0 else " " + w)

    async def complete_with_tools(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[LLMToolSpec],
        *,
        lane: Lane = Lane.STANDARD,
        tool_choice: str = "auto",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        # Deterministic provider does not drive tool selection; the heuristic
        # router handles planning in this mode. Returns a grounded text answer.
        return await self.complete(messages, lane=lane)

    async def aclose(self) -> None:
        return None


__all__ = ["DeterministicLLM"]
