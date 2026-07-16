"""Query enrichment engine for RAG optimization.

Strategies:
- DIRECT: Pass-through for precise queries (CVE IDs, exact terms)
- MULTI_QUERY: Generate multiple reformulations, fan-out searches, merge results
- HYDE: Generate hypothetical document, embed that for better semantic matching
- STEP_BACK: Generate broader/abstract query alongside specific one

The enricher classifies query type and selects strategy automatically.
"""

import asyncio
from enum import Enum

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from security_intel.observability.logging import get_logger
from security_intel.prompts.enrichment import (
    STRATEGY_CLASSIFIER_PROMPT,
    MULTI_QUERY_PROMPT,
    HYDE_PROMPT,
    STEP_BACK_PROMPT,
)

logger = get_logger("query_enrichment")


class RetrievalStrategy(Enum):
    DIRECT = "direct"
    MULTI_QUERY = "multi_query"
    HYDE = "hyde"
    STEP_BACK = "step_back"


class QueryEnricher:
    """Enriches queries using adaptive strategy selection.

    Integrates with RAG pipeline to improve retrieval quality by:
    1. Classifying query type
    2. Applying appropriate expansion strategy
    3. Returning multiple query variants for fan-out search
    """

    def __init__(self, llm: ChatOpenAI):
        self._llm = llm

    async def classify_strategy(self, query: str) -> RetrievalStrategy:
        """Classify which retrieval strategy fits this query."""
        try:
            response = await asyncio.wait_for(
                self._llm.ainvoke([
                    HumanMessage(content=STRATEGY_CLASSIFIER_PROMPT.format(query=query))
                ]),
                timeout=5,
            )
            strategy_str = response.content.strip().upper()
            return RetrievalStrategy(strategy_str.lower())
        except (asyncio.TimeoutError, ValueError, Exception) as e:
            logger.debug(f"Strategy classification failed ({e}), defaulting to MULTI_QUERY")
            return RetrievalStrategy.MULTI_QUERY

    async def enrich(self, query: str, strategy: RetrievalStrategy | None = None) -> "EnrichedQuery":
        """Enrich query using the appropriate strategy.

        Returns EnrichedQuery with multiple search variants.
        """
        if strategy is None:
            strategy = await self.classify_strategy(query)

        logger.info(f"Query enrichment strategy: {strategy.value} for '{query[:60]}...'")

        if strategy == RetrievalStrategy.DIRECT:
            return EnrichedQuery(
                original=query,
                variants=[query],
                strategy=strategy,
            )

        if strategy == RetrievalStrategy.MULTI_QUERY:
            variants = await self._generate_multi_query(query)
            return EnrichedQuery(
                original=query,
                variants=[query] + variants,
                strategy=strategy,
            )

        if strategy == RetrievalStrategy.HYDE:
            hyde_doc = await self._generate_hyde(query)
            return EnrichedQuery(
                original=query,
                variants=[query, hyde_doc] if hyde_doc else [query],
                strategy=strategy,
                hyde_document=hyde_doc,
            )

        if strategy == RetrievalStrategy.STEP_BACK:
            abstract = await self._generate_step_back(query)
            variants = [query]
            if abstract:
                variants.append(abstract)
            return EnrichedQuery(
                original=query,
                variants=variants,
                strategy=strategy,
            )

        return EnrichedQuery(original=query, variants=[query], strategy=RetrievalStrategy.DIRECT)

    async def _generate_multi_query(self, query: str, n: int = 3) -> list[str]:
        """Generate N alternative query formulations."""
        try:
            response = await asyncio.wait_for(
                self._llm.ainvoke([
                    HumanMessage(content=MULTI_QUERY_PROMPT.format(query=query, n=n))
                ]),
                timeout=8,
            )
            lines = [l.strip() for l in response.content.strip().split("\n") if l.strip()]
            lines = [l.lstrip("0123456789.-) ") for l in lines]
            return lines[:n]
        except (asyncio.TimeoutError, Exception) as e:
            logger.warning(f"Multi-query generation failed: {e}")
            return []

    async def _generate_hyde(self, query: str) -> str | None:
        """Generate hypothetical document for embedding."""
        try:
            response = await asyncio.wait_for(
                self._llm.ainvoke([
                    HumanMessage(content=HYDE_PROMPT.format(query=query))
                ]),
                timeout=8,
            )
            return response.content.strip()
        except (asyncio.TimeoutError, Exception) as e:
            logger.warning(f"HyDE generation failed: {e}")
            return None

    async def _generate_step_back(self, query: str) -> str | None:
        """Generate abstract/broader version of query."""
        try:
            response = await asyncio.wait_for(
                self._llm.ainvoke([
                    HumanMessage(content=STEP_BACK_PROMPT.format(query=query))
                ]),
                timeout=5,
            )
            return response.content.strip()
        except (asyncio.TimeoutError, Exception) as e:
            logger.warning(f"Step-back generation failed: {e}")
            return None


class EnrichedQuery:
    """Result of query enrichment — carries multiple search variants."""

    def __init__(
        self,
        original: str,
        variants: list[str],
        strategy: RetrievalStrategy,
        hyde_document: str | None = None,
    ):
        self.original = original
        self.variants = variants
        self.strategy = strategy
        self.hyde_document = hyde_document

    @property
    def search_queries(self) -> list[str]:
        """All unique queries to fan-out search over."""
        seen = set()
        unique = []
        for v in self.variants:
            if v not in seen:
                seen.add(v)
                unique.append(v)
        return unique

    def __repr__(self) -> str:
        return f"EnrichedQuery(strategy={self.strategy.value}, variants={len(self.variants)})"
