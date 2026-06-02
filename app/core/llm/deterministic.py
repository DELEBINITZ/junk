"""Deterministic, infra-free LLM (the default provider).

WHY THIS EXISTS — "deterministic stub for zero-infra dev". A real LLM needs a
GPU, a model server or an API key, a network, and it returns *different* text
each run. None of that is acceptable as the default for development, CI, or an
eval gate that must pass/fail reproducibly. So this class is a fake LLMClient
that satisfies the exact same contract (base.py) but uses NO model at all.

HOW IT FAKES AN ANSWER (and why that is enough): a RAG answer is supposed to be
nothing more than the retrieved sources, restated and cited. So this stub simply
reads the ``[n]`` context lines out of the prompt, scores them by word overlap
with the question, and stitches the best few back together WITH their ``[n]``
markers — or returns the honest refusal when nothing overlaps. That is a crude
but faithful imitation of a grounded, citing answer, which is why it lets the
whole agent loop — retrieve, ground, cite, refuse-when-unknown, stream — be
exercised and tested end to end with no infrastructure.

Because the same input always yields the same output, evals are deterministic.
Swap to a real model (SGLang/OpenAI) purely via config (see lanes.build_llm).

Mental model: "an LLM that can only quote its sources, never invent."
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

# Matches a single context line the format "[3] some retrieved text", capturing
# the citation index (3) and the text. This is the inverse of how the dispatch
# node renders chunks as "[n] text", so the stub can recover the numbered
# sources straight out of the prompt it was handed.
_CTX_LINE = re.compile(r"^\s*\[(\d+)\]\s*(.+?)\s*$")
# A stop-word list: common words that carry no topical meaning. Dropping them
# before scoring overlap means the match is driven by the QUESTION'S real
# keywords ("ransomware", "Q3") rather than filler like "the" or "show".
_STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "are",
    "was", "were", "what", "which", "who", "how", "when", "where", "why", "this",
    "that", "with", "from", "by", "at", "as", "it", "be", "do", "does", "did",
    "our", "my", "we", "i", "you", "me", "about", "any", "all", "show", "list",
}


def _tokens(text: str) -> set[str]:
    """Lower-case the text and reduce it to a SET of meaningful word tokens
    (stop-words and 1-char tokens removed). A set, because scoring only cares
    whether a word is present, not how often — overlap is set intersection."""
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP and len(w) > 1}


def _summarize(text: str, limit: int = 240) -> str:
    """Collapse whitespace and hard-cap the length, cutting on a word boundary
    and appending an ellipsis. Keeps each cited snippet short so the synthesized
    answer stays readable and bounded."""
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rsplit(" ", 1)[0] + "…"


class DeterministicLLM:
    """The fake LLMClient. ``provider = "deterministic"`` is the flag other code
    checks to know a real model is NOT wired (e.g. the specialist disables its
    LLM tool planner in that case)."""

    provider = "deterministic"

    def __init__(self, settings=None) -> None:
        self.settings = settings  # accepted for signature parity; not needed to fake answers

    # -- internals -----------------------------------------------------------
    @staticmethod
    def _question(messages: Sequence[ChatMessage]) -> str:
        """Recover the user's actual question: the LAST non-empty ``user``
        message. We scan from the end because earlier user turns are history;
        the final one is what this turn must answer."""
        for m in reversed(messages):
            if m.role == "user" and m.content.strip():
                return m.content.strip()
        return ""

    @staticmethod
    def _context(messages: Sequence[ChatMessage]) -> list[tuple[int, str]]:
        """Pull the numbered evidence back out of the prompt. Every line across
        every message is tested against ``_CTX_LINE``; the matches are the
        ``[n] text`` source entries the dispatch node injected. Returns
        ``(index, text)`` pairs — the index is the citation marker to reuse."""
        out: list[tuple[int, str]] = []
        for m in messages:
            for line in m.content.splitlines():
                mt = _CTX_LINE.match(line)
                if mt:
                    out.append((int(mt.group(1)), mt.group(2)))
        return out

    def _answer_text(self, messages: Sequence[ChatMessage]) -> str:
        """The whole "model": build a grounded, cited answer from the prompt.

        Steps, mirroring what a real grounded LLM is supposed to do:
          1. find the question and the numbered context entries;
          2. if NO context was retrieved -> refuse honestly (rule of RAG);
          3. score each entry by keyword overlap with the question and sort
             best-first (ties broken by the lower index, for stable output);
          4. keep the top 3 entries that actually overlap; if none overlap, fall
             back to the single best entry — but if even that scores zero (the
             context is unrelated to the question) -> refuse rather than quote
             something irrelevant;
          5. stitch the chosen snippets together, each KEEPING its ``[n]`` marker
             so the answer is genuinely cited and answer_node can map markers
             back to source chunks.
        """
        question = self._question(messages)
        ctx = self._context(messages)
        if not ctx:                                   # nothing retrieved -> don't guess
            return NO_CONTEXT_REFUSAL
        q = _tokens(question)
        # (overlap_count, idx, text), sorted by overlap DESC then idx ASC.
        scored = sorted(
            ((len(q & _tokens(text)), idx, text) for idx, text in ctx),
            key=lambda t: (-t[0], t[1]),
        )
        # Prefer entries with real overlap; if there are none, take the top 1.
        top = [s for s in scored if s[0] > 0][:3] or scored[:1]
        if top and top[0][0] == 0:                    # best match still shares no keyword -> refuse
            return NO_CONTEXT_REFUSAL
        parts = [f"{_summarize(text)} [{idx}]" for _score, idx, text in top]
        return "Based on the retrieved sources: " + " ".join(parts)

    # -- LLMClient interface -------------------------------------------------
    # These three methods are the actual contract from base.py. The ``lane`` /
    # ``temperature`` / ``max_tokens`` args are accepted to match the signature
    # but ignored — there is no real model to tune; output depends only on input.
    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        lane: Lane = Lane.STANDARD,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """One-shot generation: compute the grounded answer and wrap it in the
        standard LLMResponse. ``completion_tokens`` is faked as a word count so
        usage accounting downstream still sees a plausible number."""
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
        """Streaming generation: compute the SAME answer, then hand it back one
        word at a time. This imitates a real model's token-by-token stream so the
        streaming code path (SSE to the browser) can be exercised with no server.
        Spaces are re-attached to the front of each word after the first so the
        reassembled text matches ``complete`` exactly."""
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
        # The stub has no model to choose tools, so it NEVER returns tool_calls.
        # That is exactly why the specialist falls back to its deterministic
        # HEURISTIC planner whenever the provider is "deterministic" (see
        # specialist._use_llm_planner): tool selection stays infra-free too. Here
        # we just return a normal grounded text answer.
        return await self.complete(messages, lane=lane)

    async def aclose(self) -> None:
        # Nothing to release — there is no network client or model handle.
        return None


__all__ = ["DeterministicLLM"]
