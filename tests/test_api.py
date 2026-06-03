"""HTTP API smoke tests via TestClient (boots the real app + lifespan)."""

from __future__ import annotations

from tests.conftest import login


def test_health(client):
    assert client.get("/healthz").json()["status"] == "ok"
    assert client.get("/readyz").status_code == 200


def test_login_me(client):
    at = login(client)
    me = client.get("/v1/auth/me", headers={"Authorization": f"Bearer {at}"}).json()
    assert me["org_id"] == "org_acme"


def test_chat_grounded(client):
    at = login(client)
    r = client.post("/v1/chat", json={"message": "What critical CVE is on our Confluence server?"},
                    headers={"Authorization": f"Bearer {at}"})
    assert r.status_code == 200
    body = r.json()
    assert "CVE-2023-22515" in body["answer"] and body["citations"]


def test_auth_required(client):
    assert client.post("/v1/chat", json={"message": "hi"}).status_code == 401


def test_stream_sse(client):
    at = login(client)
    r = client.get("/v1/chat/stream", params={"message": "top risks?", "access_token": at})
    assert r.status_code == 200
    assert "event: token" in r.text and "event: done" in r.text


def test_sessions_lifecycle(client):
    at = login(client)
    H = {"Authorization": f"Bearer {at}"}
    s = client.post("/v1/sessions", json={"title": "T"}, headers=H).json()
    client.post("/v1/chat", json={"message": "what are our top risks?", "session_id": s["id"]}, headers=H)
    detail = client.get(f"/v1/sessions/{s['id']}", headers=H).json()
    assert len(detail["messages"]) >= 2
    assert client.delete(f"/v1/sessions/{s['id']}", headers=H).json()["status"] == "deleted"


def test_capabilities_and_route_preview(client):
    at = login(client)
    H = {"Authorization": f"Bearer {at}"}
    caps = client.get("/v1/capabilities", headers=H).json()
    assert any(m["id"] == "reports" for m in caps["modules"])
    rp = client.post("/v1/route/preview", json={"message": "summarize our top risks"}, headers=H).json()
    assert "reports" in rp["modules"]


def test_rbac_viewer_cannot_ingest(client):
    ct = login(client, "carol@acme.test")
    r = client.post("/v1/ingest", json={"documents": [{"doc_id": "x", "text": "y"}]},
                    headers={"Authorization": f"Bearer {ct}"})
    assert r.status_code == 403
