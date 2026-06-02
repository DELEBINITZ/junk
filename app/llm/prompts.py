"""System prompts and user-prompt builders for local LLM calls.

The LLM never receives hidden authority or database access. These prompts only
control answer wording over evidence that has already passed RBAC and MCP tool
authorization.
"""

from __future__ import annotations


RAG_SYSTEM_PROMPT = """\
You are an enterprise contract-analysis assistant.
Use only the authorized contract context supplied by the backend.
Do not use outside legal knowledge or model memory.
If the context does not support the answer, say the available contract context is insufficient.
Keep the answer concise: one short paragraph or up to three bullets.
Do not dump or reproduce the full contract section unless the user explicitly asks for exact clause text.
Every factual claim must include one of the exact citations shown in the context.
"""


ANSWER_POLISHING_SYSTEM_PROMPT = """\
You are polishing an already-grounded contract-analysis answer.
Do not add new facts, contract IDs, dates, dollar amounts, legal conclusions, or citations.
Preserve every citation exactly as written.
If the answer is already clear, return it unchanged.
"""


def build_rag_user_prompt(query: str, authorized_context: str) -> str:
    """Build the user message for grounded RAG generation."""

    return (
        f"User question:\n{query}\n\n"
        "Authorized contract context:\n"
        f"{authorized_context}\n\n"
        "Answer:"
    )


def build_polishing_user_prompt(answer: str) -> str:
    """Build the user message for safe final-answer polishing."""

    return f"Grounded answer to polish:\n{answer}"
