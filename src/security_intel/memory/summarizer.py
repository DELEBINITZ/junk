"""Rolling context summarizer — compresses old turns into a bounded summary.

Uses watermark (summarized_upto) to only process newly-evicted messages.
Falls back to extractive summary if LLM is unavailable.
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from security_intel.memory.conversations import ConversationStore, ChatMessage, ChatSession

# Summary-buffer tuning (1 turn = 1 user + 1 assistant message = 2 messages).
# Keep the last KEEP_RECENT_MESSAGES verbatim after a fold; only fold once the
# unsummarized tail reaches SUMMARIZE_TRIGGER (batches summary updates so we don't
# summarize every turn). The watermark stays contiguous with the verbatim window
# loaded for context, so there is never a gap and never a stale-summary-only case.
KEEP_RECENT_MESSAGES = 4   # ~2 turns kept verbatim after summarizing
SUMMARIZE_TRIGGER = 6      # fold once the tail reaches ~3 turns (folds ~2 msgs at a time)

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
        """Fold older turns into the rolling summary once the tail grows large.

        Only runs when the unsummarized tail reaches SUMMARIZE_TRIGGER, then folds
        everything except the last KEEP_RECENT_MESSAGES — leaving the watermark
        exactly where the context loader resumes verbatim (contiguous, no gap).
        Intended to be called in the background so it never delays a response.
        """
        unsummarized = session.message_count - session.summarized_upto
        if unsummarized < SUMMARIZE_TRIGGER:
            return

        target = session.message_count - KEEP_RECENT_MESSAGES
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
