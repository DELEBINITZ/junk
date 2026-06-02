"""Guardrail + RAG behavior."""

from __future__ import annotations

import pytest

from app.core.contracts import Chunk
from app.core.guardrails import build_input_guardrails, build_output_guardrails
from tests.conftest import tool_ctx


@pytest.mark.asyncio
async def test_secret_redaction(services):
    ig = build_input_guardrails(services.settings)
    r = await ig.run("here is my key AKIAIOSFODNN7EXAMPLE and api_key=supersecretvalue")
    assert "REDACTED" in r.text and not r.blocked


@pytest.mark.asyncio
async def test_security_topic_allowed(services):
    ig = build_input_guardrails(services.settings)
    r = await ig.run("explain how attackers exploit CVE-2023-22515 and which malware they drop")
    assert not r.blocked


@pytest.mark.asyncio
async def test_asset_domain_not_redacted(services):
    ig = build_input_guardrails(services.settings)
    r = await ig.run("is admin.acme.test exposed?")
    assert "admin.acme.test" in r.text


@pytest.mark.asyncio
async def test_hallucinated_citation_blocked(services):
    og = build_output_guardrails(services.settings)
    r = await og.run("A fabricated breach is confirmed [9].",
                     [Chunk(id="1", text="real source about confluence", org_id="o", doc_id="R1")])
    assert r.blocked


@pytest.mark.asyncio
async def test_pii_leak_redacted_in_output(services):
    og = build_output_guardrails(services.settings)
    r = await og.run("Contact SSN 123-45-6789 [1].",
                     [Chunk(id="1", text="contact info source", org_id="o", doc_id="R1")])
    assert "REDACTED_SSN" in r.text


@pytest.mark.asyncio
async def test_time_filter_2024(services, acme):
    r = await services.deps.rag.retrieve("certificate issues in 2024", collection="reports_kb", ctx=tool_ctx(services, acme))
    assert any(c.doc_id == "R-0900" for c in r)
    assert all((c.published_at or "").startswith("2024") for c in r)


@pytest.mark.asyncio
async def test_retrieval_relevance(services, acme):
    r = await services.deps.rag.retrieve("confluence CVE exposure", collection="reports_kb", ctx=tool_ctx(services, acme))
    assert r and r[0].doc_id in ("R-1001", "R-1004")
