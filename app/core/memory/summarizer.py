"""Rolling conversation summary — bounds context instead of bloating it.

When a session grows past a threshold, older turns are compressed into a running
summary kept on the session. Uses the fast LLM lane in prod; falls back to a
deterministic extractive summary (and is the only behavior on the deterministic
provider) so long chats work with no infra.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.core.llm.base import ChatMessage, Lane
from app.core.memory.conversations import Message

_SYS = (
    "You maintain a concise running summary of a security-analyst chat. Merge the "
    "previous summary with the new turns. Keep entities (assets, CVEs, actors, "
    "domains), decisions, and open questions. Be terse. Output only the summary."
)


class RollingSummarizer:
    def __init__(self, llm) -> None:
        self.llm = llm

    def _extractive(self, prev_summary: str, messages: Sequence[Message]) -> str:
        bits: list[str] = []
        if prev_summary:
            bits.append(prev_summary)
        for m in messages:
            if m.role == "user" and m.content.strip():
                bits.append(f"User asked: {m.content.strip()[:160]}")
            elif m.role == "assistant" and m.content.strip():
                bits.append(f"Assistant answered: {m.content.strip()[:160]}")
        text = " | ".join(bits)
        return text[-1500:]

    async def summarize(self, prev_summary: str, messages: Sequence[Message]) -> str:
        if not messages:
            return prev_summary
        if getattr(self.llm, "provider", "") == "deterministic":
            return self._extractive(prev_summary, messages)
        convo = "\n".join(f"{m.role}: {m.content}" for m in messages)
        user = f"PREVIOUS SUMMARY:\n{prev_summary or '(none)'}\n\nNEW TURNS:\n{convo}"
        try:
            resp = await self.llm.complete(
                [ChatMessage(role="system", content=_SYS), ChatMessage(role="user", content=user)],
                lane=Lane.FAST,
            )
            return resp.text.strip() or self._extractive(prev_summary, messages)
        except Exception:
            return self._extractive(prev_summary, messages)


__all__ = ["RollingSummarizer"]
