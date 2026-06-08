"""Rolling context summarizer — compresses old turns into a bounded summary.

Uses watermark (summarized_upto) to only process newly-evicted messages.
Falls back to extractive summary if LLM is unavailable.
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from security_intel.memory.conversations import ConversationStore, ChatMessage, ChatSession

HISTORY_WINDOW = 20
SUMMARIZE_THRESHOLD = 10

SUMMARIZE_PROMPT = """You are a conversation summarizer for a security intelligence platform.

Given the previous summary and new conversation turns, produce an updated summary that:
1. Preserves key facts, decisions, and findings mentioned
2. Keeps entity names (assets, CVEs, IPs, domains) intact
3. Notes what questions were asked and answered
4. Stays under 500 words
5. Is written in third person ("The user asked about...")

Previous summary:
{prev_summary}

New turns to incorporate:
{new_turns}

Updated summary:"""


class RollingSummarizer:
    """Watermark-based rolling summarizer for chat sessions."""

    def __init__(self, conversations: ConversationStore, fast_llm: ChatOpenAI):
        self._conversations = conversations
        self._llm = fast_llm

    async def maybe_summarize(self, org_id: str, session: ChatSession) -> None:
        """Summarize if enough messages have fallen out of the live window."""
        target = session.message_count - HISTORY_WINDOW
        if target <= session.summarized_upto:
            return

        evicted = await self._conversations.get_messages(
            org_id, session.id,
            limit=target - session.summarized_upto,
            offset=session.summarized_upto,
        )

        if not evicted:
            return

        new_summary = await self._summarize(session.summary, evicted)

        await self._conversations.update_summary(
            org_id, session.id, new_summary, target
        )

    async def _summarize(self, prev_summary: str, messages: list[ChatMessage]) -> str:
        """Compress turns into updated summary using FAST lane LLM."""
        turns_text = self._format_turns(messages)

        try:
            response = await self._llm.ainvoke([
                SystemMessage(content="You summarize conversations concisely."),
                HumanMessage(content=SUMMARIZE_PROMPT.format(
                    prev_summary=prev_summary or "(no prior summary)",
                    new_turns=turns_text,
                )),
            ])
            return response.content.strip()
        except Exception:
            return self._extractive_fallback(prev_summary, messages)

    @staticmethod
    def _extractive_fallback(prev_summary: str, messages: list[ChatMessage]) -> str:
        """Deterministic fallback when LLM is unavailable."""
        bits = []
        if prev_summary:
            bits.append(prev_summary)
        for m in messages:
            if m.role == "user":
                bits.append(f"User asked: {m.content[:160]}")
            elif m.role == "assistant":
                bits.append(f"Assistant answered: {m.content[:160]}")
        return " | ".join(bits)[-1500:]

    @staticmethod
    def _format_turns(messages: list[ChatMessage]) -> str:
        parts = []
        for m in messages:
            parts.append(f"[{m.role}]: {m.content[:300]}")
        return "\n".join(parts)
