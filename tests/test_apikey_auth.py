"""API-key auth mode (AUTH_PROVIDER=apikey): key authenticates, identity comes
from the request (body / headers / query). Testing/trusted-gateway only — the
prod guard rejects it."""

from __future__ import annotations

import os

import pytest
from pydantic import ValidationError

from app.config import Settings, reload_settings
from app.core.errors import AuthError
from app.core.security.deps import apikey_context


class FakeReq:
    """Minimal stand-in for a Starlette Request (only .headers/.query_params .get)."""
    def __init__(self, headers=None, query=None):
        self.headers = headers or {}
        self.query_params = query or {}


def _settings(**kw):
    return Settings(_env_file=None, auth_provider="apikey", api_keys=["k1"], **kw)


def test_identity_from_explicit_params():        # the chat-body path
    sc = apikey_context(FakeReq(headers={"x-api-key": "k1"}), _settings(),
                        org_id="org_acme", user_id="u1", roles=["analyst"])
    assert sc.org_id == "org_acme" and sc.user_id == "u1" and "analyst" in sc.roles


def test_identity_from_headers():
    sc = apikey_context(FakeReq(headers={
        "x-api-key": "k1", "x-org-id": "org_acme", "x-user-id": "u2", "x-roles": "viewer,analyst"}), _settings())
    assert sc.org_id == "org_acme" and set(sc.roles) == {"viewer", "analyst"}


def test_identity_from_query():                  # the SSE path
    sc = apikey_context(FakeReq(query={
        "api_key": "k1", "org_id": "org_acme", "user_id": "u3", "roles": "analyst"}), _settings())
    assert sc.org_id == "org_acme" and sc.user_id == "u3"


def test_default_roles_when_absent():
    sc = apikey_context(FakeReq(headers={"x-api-key": "k1", "x-org-id": "o", "x-user-id": "u"}),
                        _settings(apikey_default_roles=["analyst"]))
    assert sc.roles == ("analyst",)


def test_bad_key_rejected():
    with pytest.raises(AuthError):
        apikey_context(FakeReq(headers={"x-api-key": "nope"}), _settings(), org_id="o", user_id="u")


def test_missing_org_user_rejected():
    with pytest.raises(AuthError):
        apikey_context(FakeReq(headers={"x-api-key": "k1"}), _settings())


def test_prod_guard_rejects_apikey():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, environment="prod", auth_provider="apikey",
                 debug=False, seed_demo_data=False, llm_provider="sglang",
                 embedding_provider="tei", retrieval_backend="qdrant",
                 store_backend="postgres", database_url="postgresql://u:p@h:5432/db",
                 cap_easm_enabled=False, cap_brand_enabled=False, cap_aci_enabled=False)


# ---- integration: the chat API in apikey mode ----
@pytest.fixture
def apikey_client():
    from fastapi.testclient import TestClient

    os.environ["AUTH_PROVIDER"] = "apikey"
    os.environ["API_KEYS"] = "testkey"
    reload_settings()
    from app.main import create_app
    try:
        with TestClient(create_app()) as c:
            yield c
    finally:
        os.environ.pop("AUTH_PROVIDER", None)
        os.environ.pop("API_KEYS", None)
        reload_settings()


def test_apikey_chat_with_body_identity(apikey_client):
    r = apikey_client.post(
        "/v1/chat",
        json={"message": "what assets are exposed?", "org_id": "org_acme",
              "user_id": "u-alice", "roles": ["analyst"]},
        headers={"X-API-Key": "testkey"},
    )
    assert r.status_code == 200, r.text
    assert r.json()["session_id"]


def test_apikey_missing_key_is_401(apikey_client):
    r = apikey_client.post("/v1/chat", json={"message": "hi", "org_id": "org_acme", "user_id": "u"})
    assert r.status_code == 401
