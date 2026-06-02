"""Rule-based temporal filter extraction.

THE PROBLEM THIS SOLVES (recency / time-decay filtering in RAG): vector search
ranks purely by semantic similarity, so a question like "what changed last year?"
happily returns a perfectly-similar chunk from three years ago — semantically it
matches, but it's the wrong TIME period. The fix is to parse the time expression
out of the question and turn it into a hard date window, then have the vector DB
DROP anything outside that window BEFORE ranking. (That's distinct from the
recency RE-WEIGHTING in pipeline.py, which softly down-weights old chunks; this is
a hard include/exclude.)

Mental model: anchored to *today*, map phrases like "last year" / "past 90 days" /
"in 2024" / "Q2 2025" to an explicit ``(date_from, date_to)`` pair of RFC3339
timestamps. A fast LLM lane can supplement this in ``router_mode=llm``; these
deterministic rules are the always-on backstop.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta


def _iso_from(d: date) -> str:
    # Start-of-day RFC3339 timestamp — the INCLUSIVE lower bound of the window.
    return f"{d.isoformat()}T00:00:00Z"


def _iso_to(d: date) -> str:
    # End-of-day RFC3339 timestamp — the INCLUSIVE upper bound of the window.
    return f"{d.isoformat()}T23:59:59Z"


def _quarter_range(year: int, q: int) -> tuple[date, date]:
    # Map a fiscal quarter (1-4) to its first and last calendar day. The end is
    # computed as "first day of the month AFTER the quarter, minus one day", which
    # handles month lengths (incl. the year rollover for Q4) without a lookup table.
    start_month = 3 * (q - 1) + 1
    start = date(year, start_month, 1)
    end = date(year + (start_month + 2) // 12, ((start_month + 2) % 12) + 1, 1) - timedelta(days=1)
    return start, end


def extract_time_filters(
    query: str, now: date | None = None
) -> tuple[str | None, str | None]:
    """Return (date_from, date_to) as RFC3339 strings, or (None, None).

    Rules are tried MOST-SPECIFIC first (an explicit "Q2 2025" before a bare
    "2025" before vague "recent"), and the first match wins. ``(None, None)`` means
    the question carried no time expression, so no date filter is applied at all.
    ``now`` is injectable purely so tests can pin "today".
    """
    now = now or datetime.utcnow().date()
    q = query.lower()

    # "Q2 2025" -> that exact quarter.
    m = re.search(r"\bq([1-4])\s*(\d{4})\b", q)
    if m:
        s, e = _quarter_range(int(m.group(2)), int(m.group(1)))
        return _iso_from(s), _iso_to(e)

    # "in 2024" / "for 2024" / a bare "2024" -> the whole calendar year.
    m = re.search(r"\b(?:in|during|for|year)\s+(\d{4})\b", q) or re.search(r"\b(20\d{2})\b", q)
    if m:
        y = int(m.group(1))
        return _iso_from(date(y, 1, 1)), _iso_to(date(y, 12, 31))

    # "since 2022" -> open-ended from that year's start up to today.
    m = re.search(r"\bsince\s+(\d{4})\b", q)
    if m:
        return _iso_from(date(int(m.group(1)), 1, 1)), _iso_to(now)

    # "past/last N days|weeks|months|years" -> a rolling window ending today.
    # Months/years use rough 30/365-day approximations — fine for a search window.
    m = re.search(r"\b(?:past|last)\s+(\d{1,3})\s*(day|week|month|year)s?\b", q)
    if m:
        n = int(m.group(1))
        unit = m.group(2)
        days = {"day": 1, "week": 7, "month": 30, "year": 365}[unit] * n
        return _iso_from(now - timedelta(days=days)), _iso_to(now)

    # The remaining branches handle relative phrases ("last quarter", "this year",
    # "recent") by computing the window from ``now``.
    if "last quarter" in q:
        cq = (now.month - 1) // 3 + 1     # current quarter number (1-4) from the month
        if cq == 1:                       # we're in Q1, so "last quarter" is Q4 of last year
            s, e = _quarter_range(now.year - 1, 4)
        else:
            s, e = _quarter_range(now.year, cq - 1)
        return _iso_from(s), _iso_to(e)

    if "this quarter" in q:
        cq = (now.month - 1) // 3 + 1
        s, _e = _quarter_range(now.year, cq)
        return _iso_from(s), _iso_to(now)     # up to today, not the quarter's end

    if "last year" in q or "previous year" in q:
        y = now.year - 1
        return _iso_from(date(y, 1, 1)), _iso_to(date(y, 12, 31))

    if "this year" in q:
        return _iso_from(date(now.year, 1, 1)), _iso_to(now)

    if "last month" in q:
        # Step back to the first of this month, then one day earlier lands in the
        # previous month; its first day .. last day is the window.
        first_this = now.replace(day=1)
        last_prev = first_this - timedelta(days=1)
        return _iso_from(last_prev.replace(day=1)), _iso_to(last_prev)

    # Vague freshness words -> a fixed 90-day recency window (the catch-all).
    if "recent" in q or "latest" in q or "lately" in q:
        return _iso_from(now - timedelta(days=90)), _iso_to(now)

    # No temporal expression found -> no date filter (search across all time).
    return None, None


__all__ = ["extract_time_filters"]
