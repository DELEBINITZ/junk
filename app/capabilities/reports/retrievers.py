"""Binds the reports corpus to the shared retrieval pipeline.

This is the seam that makes retrieval corpus-agnostic: adding a new corpus
(easm_kb, aci_kb, conversations_kb, ...) means a sibling Retriever pointed at its
collection — the pipeline, agent, and guardrails are untouched. See plan §7.3.
"""

from __future__ import annotations

from app.core.contracts import ToolContext
from app.core.rag.pipeline import RetrievalPipeline


class ReportsRetriever:
    id = "reports_kb"

    def __init__(self):
        self._pipeline = RetrievalPipeline(collection="reports_kb")

    def retrieve(self, query: str, filters: dict | None, ctx: ToolContext) -> list[dict]:
        return self._pipeline.retrieve(query, ctx, top_k=8, filters=filters or {})


reports_retriever = ReportsRetriever()
