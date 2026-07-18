"""Prompts for the query enrichment engine (RAG retrieval optimization).

Domain-aware: each is .format()-ed with a {domain} label (e.g. "security reports
corpus" or "product user guide") so expansions match whatever corpus the calling
agent searches — a user-guide "how do I export findings" query no longer gets
security-report-flavored variants that miss the docs.

Preserve the {query} / {n} / {domain} placeholders.
"""

STRATEGY_CLASSIFIER_PROMPT = """Classify this search query into ONE retrieval strategy for a {domain}.

Strategies:
- DIRECT: Query contains specific identifiers (IDs, exact names, product feature/page names, version numbers, exact terms). No expansion needed.
- MULTI_QUERY: Broad or ambiguous query that would benefit from multiple search angles (e.g., "how do I configure alerts", "what critical findings are there").
- HYDE: Vague or high-level query where generating a hypothetical answer passage would help match relevant content (e.g., "what should I know about X", "summarize recent activity").
- STEP_BACK: Query that needs both specific and abstract context (e.g., "how does X compare to similar things", "what's the broader impact of this").

Rules:
- If the query has an explicit identifier, exact feature/page name, or version -> DIRECT
- If it asks about a category/type of thing -> MULTI_QUERY
- If it is abstract/summarization-oriented -> HYDE
- If it combines specific + general -> STEP_BACK
- When unsure -> MULTI_QUERY (safest default)

Query: {query}

Respond with ONLY the strategy name (DIRECT, MULTI_QUERY, HYDE, or STEP_BACK):"""


MULTI_QUERY_PROMPT = """You are a search optimizer for a {domain}. Given a user's search query, generate {n} alternative formulations that would retrieve different relevant documents from that corpus.

Rules:
- Each reformulation should target a DIFFERENT angle or aspect
- Include synonyms, related terms, and alternative phrasings the corpus might use
- Stay within the {domain} context and the user's intent — do not drift to unrelated topics
- Keep each reformulation concise (under 30 words)
- Include the original entities/identifiers/feature names in at least one variant

User query: {query}

Generate {n} search queries, one per line. No numbering, no explanations:"""


HYDE_PROMPT = """You are an expert author of a {domain}. Given a user's question, write a SHORT passage (3-5 sentences) that would plausibly appear in that corpus and answer the question.

Rules:
- Write as if this IS the document passage (not a reply to the user)
- Use the terminology, structure, and phrasing typical of a {domain}
- Be specific but realistic — plausible details that match the query
- Keep it under 100 words

User question: {query}

Hypothetical document passage:"""


STEP_BACK_PROMPT = """Given this specific query about a {domain}, generate ONE broader/more abstract version that captures the general topic area. Stay within the {domain}; do not drift to unrelated subjects.

Example (product how-to):
- Specific: "Which button do I click to add a new monitored domain on the dashboard?"
- Abstract: "Adding and configuring monitored assets from the dashboard"

Example (reporting):
- Specific: "How many high-severity items were found last week?"
- Abstract: "Reviewing recent findings by severity over a time window"

Query: {query}

Abstract version (one line, no explanation):"""
