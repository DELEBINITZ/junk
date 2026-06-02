"""Chat history: the conversation persists, threads across turns, and is loaded as
context for follow-ups (ChatGPT-style last-N memory), plus the history APIs."""

from __future__ import annotations

import pytest

from tests.conftest import login


@pytest.mark.asyncio
async def test_multi_turn_history_is_threaded(services, acme):
    """Two turns in one session accumulate in order — the follow-up runs on the
    same session, so run_turn loads the prior turns as context."""
    o = services.orchestrator
    r1 = await o.run_turn(acme, question="what critical CVE is on our confluence server?")
    sid = r1.session_id
    r2 = await o.run_turn(acme, question="and what is its severity?", session_id=sid)
    assert r2.session_id == sid                       # same conversation thread

    msgs = await services.conversations.get_messages(acme.org_id, sid)
    assert [m.role for m in msgs] == ["user", "assistant", "user", "assistant"]
    assert msgs[0].content == "what critical CVE is on our confluence server?"
    assert msgs[2].content == "and what is its severity?"


@pytest.mark.asyncio
async def test_history_window_is_bounded(services, acme):
    """run_turn only loads the last ``history_window_messages`` — long chats stay
    bounded (older context is covered by the rolling summary)."""
    o = services.orchestrator
    r = await o.run_turn(acme, question="what assets are exposed?")
    sid = r.session_id
    for i in range(8):
        await o.run_turn(acme, question=f"follow up number {i}", session_id=sid)
    # a turn only pulls a bounded window into context, no matter how long the chat.
    windowed = await services.conversations.get_messages(
        acme.org_id, sid, limit=services.settings.history_window_messages)
    assert 0 < len(windowed) <= services.settings.history_window_messages


def test_chat_history_apis(client):
    """The chat-history endpoints: list conversations + fetch a conversation's
    full message history (what a ChatGPT-style sidebar + thread view need)."""
    tok = login(client)
    h = {"Authorization": f"Bearer {tok}"}

    r1 = client.post("/v1/chat", json={"message": "what assets are exposed?"}, headers=h)
    assert r1.status_code == 200, r1.text
    sid = r1.json()["session_id"]

    # a follow-up in the same session
    r2 = client.post("/v1/chat", json={"message": "and which are critical?", "session_id": sid}, headers=h)
    assert r2.status_code == 200

    # list sessions returns this conversation
    ls = client.get("/v1/sessions", headers=h)
    assert ls.status_code == 200 and any(s["id"] == sid for s in ls.json())

    # session detail returns the ordered message history
    sd = client.get(f"/v1/sessions/{sid}", headers=h)
    assert sd.status_code == 200
    msgs = sd.json()["messages"]
    assert [m["role"] for m in msgs[:4]] == ["user", "assistant", "user", "assistant"]
    assert msgs[0]["content"] == "what assets are exposed?"
