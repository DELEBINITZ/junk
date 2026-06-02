"""Rule-based temporal filter extraction.

Anchored to *today*, this turns "last year" / "past 90 days" / "in 2024" / "Q2
2025" into an explicit ``(date_from, date_to)`` window applied at the vector DB
BEFORE ranking — the fix for "last year's data returns 3-year-old chunks". A fast
LLM lane can supplement this in ``router_mode=llm``; the rules are the backstop.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta


def _iso_from(d: date) -> str:
    return f"{d.isoformat()}T00:00:00Z"


def _iso_to(d: date) -> str:
    return f"{d.isoformat()}T23:59:59Z"


def _quarter_range(year: int, q: int) -> tuple[date, date]:
    start_month = 3 * (q - 1) + 1
    start = date(year, start_month, 1)
    end = date(year + (start_month + 2) // 12, ((start_month + 2) % 12) + 1, 1) - timedelta(days=1)
    return start, end


def extract_time_filters(
    query: str, now: date | None = None
) -> tuple[str | None, str | None]:
    """Return (date_from, date_to) as RFC3339 strings, or (None, None)."""
    now = now or datetime.utcnow().date()
    q = query.lower()

    m = re.search(r"\bq([1-4])\s*(\d{4})\b", q)
    if m:
        s, e = _quarter_range(int(m.group(2)), int(m.group(1)))
        return _iso_from(s), _iso_to(e)

    m = re.search(r"\b(?:in|during|for|year)\s+(\d{4})\b", q) or re.search(r"\b(20\d{2})\b", q)
    if m:
        y = int(m.group(1))
        return _iso_from(date(y, 1, 1)), _iso_to(date(y, 12, 31))

    m = re.search(r"\bsince\s+(\d{4})\b", q)
    if m:
        return _iso_from(date(int(m.group(1)), 1, 1)), _iso_to(now)

    m = re.search(r"\b(?:past|last)\s+(\d{1,3})\s*(day|week|month|year)s?\b", q)
    if m:
        n = int(m.group(1))
        unit = m.group(2)
        days = {"day": 1, "week": 7, "month": 30, "year": 365}[unit] * n
        return _iso_from(now - timedelta(days=days)), _iso_to(now)

    if "last quarter" in q:
        cq = (now.month - 1) // 3 + 1
        if cq == 1:
            s, e = _quarter_range(now.year - 1, 4)
        else:
            s, e = _quarter_range(now.year, cq - 1)
        return _iso_from(s), _iso_to(e)

    if "this quarter" in q:
        cq = (now.month - 1) // 3 + 1
        s, _e = _quarter_range(now.year, cq)
        return _iso_from(s), _iso_to(now)

    if "last year" in q or "previous year" in q:
        y = now.year - 1
        return _iso_from(date(y, 1, 1)), _iso_to(date(y, 12, 31))

    if "this year" in q:
        return _iso_from(date(now.year, 1, 1)), _iso_to(now)

    if "last month" in q:
        first_this = now.replace(day=1)
        last_prev = first_this - timedelta(days=1)
        return _iso_from(last_prev.replace(day=1)), _iso_to(last_prev)

    if "recent" in q or "latest" in q or "lately" in q:
        return _iso_from(now - timedelta(days=90)), _iso_to(now)

    return None, None


__all__ = ["extract_time_filters"]
