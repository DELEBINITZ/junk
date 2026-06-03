"""Chat-system essentials: small-talk/scope triage + message feedback."""

from __future__ import annotations

import pytest

from tests.conftest import login


# --------------------------------------------------------------------------- #
# small-talk / scope triage
# --------------------------------------------------------------------------- #
def test_triage_classifier():
    from app.core.agent.nodes import _triage_category

    assert _triage_category("hi") == "greeting"
    assert _triage_category("hello, how are you?") == "greeting"
    assert _triage_category("thanks!") == "greeting"
    assert _triage_category("what can you do?") == "help"
    assert _triage_category("who are you") == "identity"
    assert _triage_category("what assets are exposed on our attack surface") == "task"
    # starts with "hi" but is a real question -> NOT a greeting
    assert _triage_category("highlight our biggest risks this quarter") == "task"


@pytest.mark.asyncio
async def test_greeting_gets_direct_reply_no_tools(services, acme):
    r = await services.orchestrator.run_turn(acme, question="hi, how are you?")
    assert r.route_modules == []                 # no routing, no retrieval, no tools
    assert r.citations == []
    assert "assistant" in r.answer.lower()       # friendly steer


@pytest.mark.asyncio
async def test_help_lists_capabilities(services, acme):
    r = await services.orchestrator.run_turn(acme, question="what can you do?")
    assert r.route_modules == []
    assert "Security Reports" in r.answer        # dynamic capability list from the registry


@pytest.mark.asyncio
async def test_real_question_still_runs_agent(services, acme):
    r = await services.orchestrator.run_turn(acme, question="what critical CVE is on our confluence server?")
    assert "CVE-2023-22515" in r.answer
    assert r.route_modules                        # the agent actually ran (not triaged away)


# --------------------------------------------------------------------------- #
# message feedback
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_message_feedback_store(services, acme):
    r = await services.orchestrator.run_turn(acme, question="what are our top risks?")
    assert await services.conversations.set_message_feedback(acme.org_id, r.message_id, 1)
    msgs = await services.conversations.get_messages(acme.org_id, r.session_id)
    assistant = [m for m in msgs if m.role == "assistant"][-1]
    assert assistant.feedback == 1


@pytest.mark.asyncio
async def test_feedback_is_org_scoped(services, acme, globex):
    r = await services.orchestrator.run_turn(acme, question="what are our top risks?")
    # globex cannot rate acme's message
    assert await services.conversations.set_message_feedback(globex.org_id, r.message_id, -1) is False


def test_feedback_api_roundtrip(client):
    tok = login(client)
    h = {"Authorization": f"Bearer {tok}"}
    r = client.post("/v1/chat", json={"message": "what assets are exposed?"}, headers=h)
    assert r.status_code == 200, r.text
    mid, sid = r.json()["message_id"], r.json()["session_id"]

    fb = client.post(f"/v1/messages/{mid}/feedback", json={"value": 1}, headers=h)
    assert fb.status_code == 200 and fb.json()["feedback"] == 1

    sd = client.get(f"/v1/sessions/{sid}", headers=h)
    assistant_msgs = [m for m in sd.json()["messages"] if m["role"] == "assistant"]
    assert assistant_msgs[-1]["feedback"] == 1    # persisted + returned to the UI
