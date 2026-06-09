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

logger = get_logger("query_enrichment")


class RetrievalStrategy(Enum):
    DIRECT = "direct"
    MULTI_QUERY = "multi_query"
    HYDE = "hyde"
    STEP_BACK = "step_back"


STRATEGY_CLASSIFIER_PROMPT = """Classify this security search query into ONE retrieval strategy.

Strategies:
- DIRECT: Query contains specific identifiers (CVE IDs, exact hostnames, IP addresses, specific tool names, exact threat actor names). No expansion needed.
- MULTI_QUERY: Broad or ambiguous query that would benefit from multiple search angles (e.g., "what ransomware threats are relevant", "show me critical findings").
- HYDE: Vague or high-level query where generating a hypothetical answer document would help match relevant content (e.g., "what should I know about our security posture", "summarize recent threats").
- STEP_BACK: Query that needs both specific and abstract context (e.g., "how does CVE-2024-1234 compare to similar vulnerabilities", "what's the broader impact of this exposure").

Rules:
- If query has CVE-*, IP address, exact domain, or specific identifier → DIRECT
- If query asks about a category/type of thing → MULTI_QUERY
- If query is abstract/summarization-oriented → HYDE
- If query combines specific + general → STEP_BACK
- When unsure → MULTI_QUERY (safest default)

Query: {query}

Respond with ONLY the strategy name (DIRECT, MULTI_QUERY, HYDE, or STEP_BACK):"""


MULTI_QUERY_PROMPT = """You are a security intelligence search optimizer. Given a user's search query, generate {n} alternative formulations that would retrieve different relevant documents from a security reports corpus.

Rules:
- Each reformulation should target a DIFFERENT angle or aspect
- Include synonyms, related terms, and alternative phrasings
- Maintain the security domain context
- Keep each reformulation concise (under 30 words)
- Include the original entities/identifiers in at least one variant

User query: {query}

Generate {n} search queries, one per line. No numbering, no explanations:"""


HYDE_PROMPT = """You are a security intelligence analyst. Given a user's question, write a SHORT passage (3-5 sentences) that would appear in a security report answering this question.

Rules:
- Write as if this IS the document passage (not a response to the user)
- Include technical terms, CVE patterns, severity language typical of security reports
- Be specific but realistic — use plausible details that match the query domain
- Keep it under 100 words

User question: {query}

Hypothetical document passage:"""


STEP_BACK_PROMPT = """Given this specific security query, generate ONE broader/more abstract version that captures the general topic area.

Example:
- Specific: "Is CVE-2024-3400 affecting our Palo Alto firewalls?"
- Abstract: "Palo Alto firewall vulnerabilities and exploitation in enterprise environments"

Example:
- Specific: "What phishing campaigns targeted our finance team last month?"
- Abstract: "Business email compromise and spear phishing tactics against financial departments"

Query: {query}

Abstract version (one line, no explanation):"""


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
