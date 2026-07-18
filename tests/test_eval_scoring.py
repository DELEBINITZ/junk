"""Tests for the deterministic answer-quality scorers (no LLM, CI-safe)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from security_intel.observability.eval_scoring import (  # noqa: E402
    extract_cve_ids,
    groundedness_score,
    citation_coverage,
    unsupported_cves,
    score_answer,
)


def test_extract_cve_ids_normalizes_and_dedups():
    text = "See cve-2024-3400 and CVE-2024-3400 and CVE-2023-12345."
    assert extract_cve_ids(text) == {"CVE-2024-3400", "CVE-2023-12345"}


def test_extract_cve_ids_empty():
    assert extract_cve_ids("no cves here") == set()
    assert extract_cve_ids("") == set()


def test_groundedness_full_and_none():
    src = ["The EASM dashboard lists discovered assets and exposures."]
    assert groundedness_score("discovered assets and exposures", src) == 1.0
    # Novel, unsupported content scores low.
    assert groundedness_score("quantum blockchain synergy paradigm", src) < 0.25


def test_groundedness_empty_answer_is_vacuously_grounded():
    assert groundedness_score("", ["anything"]) == 1.0


def test_citation_coverage_counts_referenced_sources():
    sources = [
        {"title": "EASM Dashboard", "doc_id": "d1"},
        {"title": "Requirements", "doc_id": "d2"},
    ]
    ans = "See the EASM Dashboard page for details."
    cov = citation_coverage(ans, sources)
    assert cov["coverage"] == 0.5
    assert "EASM Dashboard" in cov["cited"]
    assert "Requirements" in cov["uncited"]


def test_citation_coverage_no_sources():
    assert citation_coverage("anything", [])["coverage"] == 1.0


def test_unsupported_cves_flags_hallucinated_id():
    sources = ["Report discusses CVE-2024-3400 in detail."]
    # CVE-2024-9999 is NOT in any source -> hallucination.
    assert unsupported_cves("Both CVE-2024-3400 and CVE-2024-9999 apply.", sources) == [
        "CVE-2024-9999"
    ]


def test_unsupported_cves_none_when_supported():
    sources = ["Report discusses CVE-2024-3400."]
    assert unsupported_cves("CVE-2024-3400 is relevant.", sources) == []


def test_score_answer_clean_is_ok():
    sources = [{"text": "The EASM dashboard shows discovered assets.", "title": "EASM Dashboard"}]
    s = score_answer("The EASM Dashboard shows discovered assets.", sources)
    assert s.ok
    assert s.groundedness > 0.5
    assert s.unsupported_cves == []


def test_score_answer_flags_unsupported_cve():
    sources = [{"text": "General guidance about assets.", "title": "Guide"}]
    s = score_answer("You are affected by CVE-2024-9999.", sources)
    assert not s.ok
    assert any(f.startswith("unsupported_cves:") for f in s.flags)
    assert "CVE-2024-9999" in s.unsupported_cves


def test_score_answer_flags_low_groundedness():
    sources = [{"text": "Cats sit on mats.", "title": "Unrelated"}]
    s = score_answer("Configure the firewall zone policy and NAT rules carefully.", sources)
    assert not s.ok
    assert any(f.startswith("low_groundedness:") for f in s.flags)


def test_score_answer_as_dict_roundtrip():
    s = score_answer("hello", [{"text": "hello world", "title": "T"}])
    d = s.as_dict()
    assert set(d) >= {"groundedness", "citation_coverage", "unsupported_cves", "flags", "ok"}
