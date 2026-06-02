"""Answer composition + streaming, shared by the graph's answer node and the
streaming orchestrator path (so both ground/cite identically). See plan §7, §11.

`stream_answer` yields real per-token output from a streaming-capable client
(SGLang) and falls back to word-chunking the deterministic grounded summary.
"""

from __future__ import annotations

from typing import Iterator


REFUSAL_MESSAGE = "I can't help with that request."
NO_ACCESS_MESSAGE = (
    "Your access level does not permit AI report queries. Ask an administrator "
    "for analyst access."
)
INSUFFICIENT_MESSAGE = (
    "I don't have enough information in the reports you're authorized to see to "
    "answer that."
)
SYSTEM_PROMPT = (
    "You are a security-reports assistant. Answer ONLY from the context. Cite "
    "every claim with the provided [ID §section] markers. If the context is "
    "insufficient, say so plainly."
)


def compose_grounded(matches: list[dict], max_top_k: int) -> tuple[str, list[str]]:
    """Build a grounded, cited summary from retrieved matches."""

    if not matches:
        return INSUFFICIENT_MESSAGE, []
    lines = ["Based on the authorized reports:"]
    citations: list[str] = []
    for match in matches[:max_top_k]:
        cite = match.get("citation")
        snippet = (match.get("snippet") or "").strip().replace("\n", " ")[:280]
        lines.append(
            f"- [{match.get('contract_id')} §{match.get('section_number')}] {snippet}"
            + (f" {cite}" if cite else "")
        )
        if cite:
            citations.append(cite)
    return "\n".join(lines), citations


def _user_prompt(query: str, grounded: str) -> str:
    return f"Context:\n{grounded}\n\nQuestion: {query}"


def finalize_answer(client, query: str, grounded: str) -> str:
    """Non-streaming answer. Deterministic client returns the grounded summary;
    a real LLM phrases the answer from the grounded context."""

    if getattr(client, "provider_name", "deterministic") != "deterministic":
        try:
            return client.invoke(system_prompt=SYSTEM_PROMPT, user_prompt=_user_prompt(query, grounded))
        except Exception:
            return grounded
    return grounded


def stream_answer(client, query: str, grounded: str) -> Iterator[str]:
    """Yield answer chunks. Real per-token stream from a streaming client;
    otherwise word-chunk the grounded summary so the SSE UX is identical."""

    if getattr(client, "provider_name", "deterministic") != "deterministic" and hasattr(client, "stream"):
        try:
            yield from client.stream(system_prompt=SYSTEM_PROMPT, user_prompt=_user_prompt(query, grounded))
            return
        except Exception:
            pass
    for word in grounded.split(" "):
        yield word + " "
