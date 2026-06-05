"""Remote MCP transport auth (X-API-Key): the client sends the per-server key and
the server enforces it (constant-time) before any identity/tool work. This is the
"is this my trusted core calling?" gate, layered under the signed identity token.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.config import Settings


# --- client sends the header ------------------------------------------------

class _FakeResp:
    def __init__(self, payload):
        self._p = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._p


class _FakeHttp:
    def __init__(self):
        self.calls = []

    async def post(self, url, json=None, headers=None):
        self.calls.append({"url": url, "json": json, "headers": headers})
        return _FakeResp({"jsonrpc": "2.0", "id": json["id"], "result": {"tools": []}})


@pytest.mark.asyncio
async def test_remote_client_sends_api_key_header():
    from app.core.mcp.remote import RemoteMCPClient

    c = RemoteMCPClient("http://srv", token_for_ctx=lambda ctx: "t",
                        token_for_sc=lambda sc: "t", api_key="secret-k")
    c._client = _FakeHttp()
    await c.list_tools(sc=object())
    h = c._client.calls[0]["headers"]
    assert h["Authorization"] == "Bearer t"      # identity token still present
    assert h["X-API-Key"] == "secret-k"          # transport key sent


@pytest.mark.asyncio
async def test_remote_client_omits_api_key_when_unset():
    from app.core.mcp.remote import RemoteMCPClient

    c = RemoteMCPClient("http://srv", token_for_ctx=lambda ctx: "t", token_for_sc=lambda sc: "t")
    c._client = _FakeHttp()
    await c.list_tools(sc=object())
    assert "X-API-Key" not in c._client.calls[0]["headers"]


# --- server enforces the header ---------------------------------------------

def _server(api_key: str):
    from app.core.mcp.server import make_mcp_app

    s = Settings(_env_file=None, jwt_secret="x" * 48)
    return make_mcp_app(SimpleNamespace(), s,
                        SimpleNamespace(action_gate=None, logger=None),
                        audience="easm-mcp", api_key=api_key)


def _post(app, headers=None):
    from fastapi.testclient import TestClient

    body = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
    return TestClient(app).post("/mcp", json=body, headers=headers or {}).json()


def test_server_rejects_missing_api_key():
    pytest.importorskip("fastapi")
    assert _post(_server("secret"))["error"]["message"] == "auth failed"


def test_server_rejects_wrong_api_key():
    pytest.importorskip("fastapi")
    assert _post(_server("secret"), headers={"X-API-Key": "nope"})["error"]["message"] == "auth failed"


def test_server_correct_api_key_passes_transport_then_fails_on_token():
    # Correct key clears LAYER 1; with no bearer token it then fails LAYER 2 — proving
    # the key alone is not sufficient (defense in depth) and that a correct key is not
    # rejected by the transport gate.
    pytest.importorskip("fastapi")
    out = _post(_server("secret"), headers={"X-API-Key": "secret"})
    # still an auth failure, but reached the identity layer (no token supplied)
    assert out["error"]["message"] == "auth failed"
