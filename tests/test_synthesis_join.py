"""Tests for the deterministic code-side cross-reference join."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from security_intel.agents.synthesis_join import cross_reference  # noqa: E402


def _r(agent_id, findings):
    return {"agent_id": agent_id, "findings": findings}


def test_overlap_found_across_two_agents():
    results = [
        _r("aura", "Exposed host has CVE-2024-3400 and CVE-2024-1111."),
        _r("sentinel", "Recent report covers cve-2024-3400 exploitation in the wild."),
    ]
    xr = cross_reference(results)
    assert xr.has_overlap
    assert xr.overlaps["CVE"] == {"CVE-2024-3400": ["aura", "sentinel"]}
    # single-source CVE is NOT an overlap
    assert "CVE-2024-1111" not in xr.overlaps["CVE"]


def test_no_overlap_when_single_source_only():
    results = [
        _r("aura", "Asset exposes CVE-2024-1111."),
        _r("sentinel", "Report discusses CVE-2024-2222."),
    ]
    xr = cross_reference(results)
    assert not xr.has_overlap
    assert xr.any_entities  # entities exist, just no intersection
    facts = xr.render_facts()
    assert "NO entity appears in more than one source" in facts
    # both single-source entities are surfaced so the model states absence honestly
    assert "CVE-2024-1111" in facts and "CVE-2024-2222" in facts


def test_no_entities_renders_empty_block():
    results = [
        _r("aura", "You have three exposed web servers."),
        _r("sentinel", "A ransomware group is active this quarter."),
    ]
    xr = cross_reference(results)
    assert not xr.any_entities
    assert xr.render_facts() == ""


def test_render_facts_marks_overlap_as_ground_truth():
    results = [
        _r("aura", "CVE-2024-3400 on host a1."),
        _r("sentinel", "CVE-2024-3400 in report R-9."),
    ]
    facts = cross_reference(results).render_facts()
    assert "VERIFIED CROSS-REFERENCE" in facts
    assert "GROUND TRUTH" in facts
    assert "CVE-2024-3400: found by aura, sentinel" in facts


def test_citations_carry_provenance():
    results = [
        _r("aura", "CVE-2024-3400 exposed."),
        _r("sentinel", "CVE-2024-3400 reported."),
    ]
    cites = cross_reference(results).citations()
    assert cites == [
        {"type": "cross_reference", "entity_type": "CVE",
         "entity": "CVE-2024-3400", "agents": ["aura", "sentinel"]}
    ]


def test_case_insensitive_and_deduped_overlap():
    results = [
        _r("aura", "cve-2024-3400 and CVE-2024-3400 twice."),
        _r("sentinel", "CVE-2024-3400 once."),
    ]
    xr = cross_reference(results)
    assert xr.overlaps["CVE"] == {"CVE-2024-3400": ["aura", "sentinel"]}
