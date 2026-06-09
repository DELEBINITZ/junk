"""Prompts for the query enrichment engine (RAG retrieval optimization).

All are .format()-ed by QueryEnricher — preserve their {query} / {n} placeholders.
"""

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
