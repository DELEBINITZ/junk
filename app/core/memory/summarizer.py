"""Rolling conversation summary — bounds context instead of bloating it.

================================ MENTAL MODEL =============================
An LLM has a finite context window, and stuffing an entire long chat into every
prompt is slow, costly, and eventually impossible. The fix is a ROLLING SUMMARY:
once a session grows past a threshold, the older turns are compressed into a
short running summary stored on the session (Session.summary). Each chat turn
then sends only [rolling summary] + [last few raw turns] instead of the whole
history — context size stays bounded no matter how long the conversation gets.
(See nodes.build_answer_messages, which injects this summary as a system message.)

This class produces that summary two ways:
  * LLM summary (prod) — ask the FAST lane (the cheap model; summarizing is not
    the expensive answer step) to MERGE the previous summary with the new turns.
  * extractive fallback — a deterministic, no-LLM summary built by truncating each
    turn. It is the safety net when the LLM call fails AND the only behavior on
    the deterministic provider, so long chats keep working with zero infra.
===========================================================================
"""

from __future__ import annotations

from collections.abc import Sequence

from app.core.llm.base import ChatMessage, Lane
from app.core.memory.conversations import Message

# System prompt for the LLM summarizer. It tells the model to MERGE (not replace)
# the prior summary and to preserve the entities/decisions/open-questions that
# matter for a security-analyst chat — the things later turns will refer back to.
_SYS = (
    "You maintain a concise running summary of a security-analyst chat. Merge the "
    "previous summary with the new turns. Keep entities (assets, CVEs, actors, "
    "domains), decisions, and open questions. Be terse. Output only the summary."
)


class RollingSummarizer:
    """Compresses conversation history into a bounded summary. Stateless apart
    from its LLM handle — the chat service decides WHEN to call it; this just
    knows HOW to fold new turns into the previous summary."""

    def __init__(self, llm) -> None:
        self.llm = llm

    def _extractive(self, prev_summary: str, messages: Sequence[Message]) -> str:
        """Deterministic, LLM-free summary: keep the prior summary, then append a
        one-line, length-capped gist of each user/assistant turn. Crude but always
        available — this is what guarantees the feature works without a model. The
        final ``[-1500:]`` keeps even the fallback summary itself bounded in size."""
        bits: list[str] = []
        if prev_summary:
            bits.append(prev_summary)
        for m in messages:
            if m.role == "user" and m.content.strip():
                bits.append(f"User asked: {m.content.strip()[:160]}")
            elif m.role == "assistant" and m.content.strip():
                bits.append(f"Assistant answered: {m.content.strip()[:160]}")
        text = " | ".join(bits)
        return text[-1500:]   # hard cap so the summary can't grow without bound

    async def summarize(self, prev_summary: str, messages: Sequence[Message]) -> str:
        """Return the updated rolling summary after folding in ``messages``.

        Order of decisions: nothing new -> keep the old summary unchanged; the
        deterministic provider (tests/no-infra) -> use the extractive path; else
        ask the FAST-lane LLM to merge, and if it errors or returns empty, fall
        back to the extractive summary. The summarizer NEVER raises into the chat
        turn — a summarization hiccup must not break answering."""
        if not messages:
            return prev_summary
        if getattr(self.llm, "provider", "") == "deterministic":
            return self._extractive(prev_summary, messages)
        # Render the new turns as plain "role: content" lines and hand the model
        # both the previous summary and the new turns so it can merge them.
        convo = "\n".join(f"{m.role}: {m.content}" for m in messages)
        user = f"PREVIOUS SUMMARY:\n{prev_summary or '(none)'}\n\nNEW TURNS:\n{convo}"
        try:
            resp = await self.llm.complete(
                [ChatMessage(role="system", content=_SYS), ChatMessage(role="user", content=user)],
                lane=Lane.FAST,    # summarizing is cheap work -> the fast/cheap model
            )
            # Empty model output is treated as a failure -> use the deterministic fallback.
            return resp.text.strip() or self._extractive(prev_summary, messages)
        except Exception:
            return self._extractive(prev_summary, messages)


__all__ = ["RollingSummarizer"]
