"""Deterministic answer-quality scorers — no LLM, safe to run inline in prod.

These are the primitives that let us *measure* answer correctness instead of
trusting it: groundedness (is the answer supported by what was retrieved?),
citation coverage (does it point back to its sources?), and — highest-signal for a
security product — unsupported-entity detection (a CVE id or version string in the
answer that appears in NO retrieved source is a hallucination red flag).

Being LLM-free, they run cheaply on every answer (online eval sampling, Phase 4)
and back the citation contract for the multi-agent synthesizer (Phase 3). They are
proxies, not judges: high groundedness ≠ correct, but LOW groundedness or an
unsupported CVE is a reliable *warning*.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict

# CVE ids: CVE-YYYY-NNNN.. (4+ digit sequence). Case-insensitive; normalized upper.
_CVE_RE = re.compile(r"\bCVE-\d{4}-\d{4,7}\b", re.IGNORECASE)
# CVSS-ish / version-ish numbers are noisy; we deliberately scope entity checks to
# CVE ids (unambiguous, high-value) rather than all numbers to avoid false alarms.

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}")

_STOPWORDS = frozenset(
    """the a an and or but if then else for to of in on at by with from as is are was
    were be been being this that these those it its it's you your we our they their he
    she them his her i me my mine ours yours will would can could should may might must
    do does did done have has had having not no yes so than too very just about into over
    under out up down off above below more most less least also which who whom whose what
    when where why how all any both each few many some such only own same other another""".split()
)


def extract_cve_ids(text: str) -> set[str]:
    """All CVE ids in ``text``, normalized to uppercase. Reused by the Phase-3 join."""
    return {m.group(0).upper() for m in _CVE_RE.finditer(text or "")}


def _content_tokens(text: str) -> set[str]:
    return {
        t.lower()
        for t in _WORD_RE.findall(text or "")
        if t.lower() not in _STOPWORDS
    }


def groundedness_score(answer: str, source_texts: list[str]) -> float:
    """Fraction of the answer's content tokens that also appear in the retrieved
    sources. Cheap lexical proxy for "is this answer supported by the corpus?".

    1.0 = every meaningful word is attested in a source; low = lots of novel,
    unsupported content (paraphrase lowers it, so treat as a signal not a verdict).
    Returns 1.0 for an answer with no content tokens (vacuously grounded).
    """
    ans = _content_tokens(answer)
    if not ans:
        return 1.0
    src = set()
    for s in source_texts:
        src |= _content_tokens(s)
    supported = sum(1 for t in ans if t in src)
    return supported / len(ans)


def citation_coverage(answer: str, sources: list[dict]) -> dict:
    """How well the answer points back to its sources.

    ``sources`` items may carry ``title``, ``url`` and/or ``doc_id``. A source is
    "cited" if any of its identifiers appears verbatim in the answer text. Returns
    the coverage fraction plus the cited/uncited identifiers so a caller can require
    e.g. ``coverage >= 0.5`` or "every asserted source is cited".
    """
    if not sources:
        return {"coverage": 1.0, "cited": [], "uncited": [], "total_sources": 0}
    ans = answer or ""
    cited, uncited = [], []
    for s in sources:
        ids = [str(s.get(k)) for k in ("title", "url", "doc_id") if s.get(k)]
        label = ids[0] if ids else "<unlabeled>"
        if any(idv and idv in ans for idv in ids):
            cited.append(label)
        else:
            uncited.append(label)
    total = len(sources)
    return {
        "coverage": len(cited) / total if total else 1.0,
        "cited": cited,
        "uncited": uncited,
        "total_sources": total,
    }


def unsupported_cves(answer: str, source_texts: list[str]) -> list[str]:
    """CVE ids asserted in the answer that appear in NO retrieved source.

    The single highest-signal hallucination check for this product: a fabricated or
    misattributed CVE id is both easy to emit and dangerous to act on.
    """
    in_answer = extract_cve_ids(answer)
    if not in_answer:
        return []
    in_sources: set[str] = set()
    for s in source_texts:
        in_sources |= extract_cve_ids(s)
    return sorted(in_answer - in_sources)


@dataclass
class AnswerScore:
    groundedness: float
    citation_coverage: float
    unsupported_cves: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """No hard red flags. Thresholds are intentionally conservative — this gates
        a *warning*, not the response itself, unless a caller decides otherwise."""
        return not self.flags

    def as_dict(self) -> dict:
        return asdict(self) | {"ok": self.ok}


def score_answer(
    answer: str,
    sources: list[dict],
    *,
    min_groundedness: float = 0.35,
    min_citation_coverage: float = 0.0,
) -> AnswerScore:
    """Bundle the scorers into one report + flags.

    ``sources`` items should carry a ``text`` field (for groundedness / CVE checks)
    and any of ``title``/``url``/``doc_id`` (for citation coverage). Flags fire on:
    an unsupported CVE, groundedness below ``min_groundedness``, or citation coverage
    below ``min_citation_coverage`` (default 0 = off).
    """
    source_texts = [str(s.get("text", "")) for s in sources]
    g = groundedness_score(answer, source_texts)
    cc = citation_coverage(answer, sources)["coverage"]
    bad_cves = unsupported_cves(answer, source_texts)

    flags: list[str] = []
    if bad_cves:
        flags.append(f"unsupported_cves:{','.join(bad_cves)}")
    if g < min_groundedness:
        flags.append(f"low_groundedness:{g:.2f}")
    if cc < min_citation_coverage:
        flags.append(f"low_citation_coverage:{cc:.2f}")

    return AnswerScore(
        groundedness=round(g, 3),
        citation_coverage=round(cc, 3),
        unsupported_cves=bad_cves,
        flags=flags,
    )
