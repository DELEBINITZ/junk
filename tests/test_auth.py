"""The simplified auth model: every request needs (1) a gateway API key AND (2) a
verified JWT that supplies identity. No login/password/OIDC/refresh flow.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.config import get_settings
from app.core.security.jwt import create_access_token
from app.main import app
from tests.conftest import _DEV_API_KEY, login

CHAT = {"message": "what are our top risks?"}


def _jwt(org="org_acme", user="u-alice", roles=("admin",)) -> str:
    return create_access_token(get_settings(), sub=user, org_id=org, roles=roles).token


def test_api_key_and_jwt_succeeds(client):
    """Valid API key (from the client fixture) + valid JWT -> 200, identity from JWT."""
    me = client.get("/v1/auth/me", headers={"Authorization": f"Bearer {login(client)}"}).json()
    assert me["org_id"] == "org_acme" and me["id"] == "u-alice"


def test_missing_api_key_is_rejected():
    """A valid JWT but NO API key -> 401: the gateway key gates every request."""
    with TestClient(app) as c:                     # no default x-api-key header
        r = c.get("/v1/auth/me", headers={"Authorization": f"Bearer {_jwt()}"})
        assert r.status_code == 401


def test_wrong_api_key_is_rejected(client):
    r = client.get("/v1/auth/me", headers={"x-api-key": "nope", "Authorization": f"Bearer {_jwt()}"})
    assert r.status_code == 401


def test_api_key_without_jwt_is_rejected(client):
    """API key present (fixture) but NO JWT -> 401: identity must come from a token."""
    assert client.get("/v1/auth/me").status_code == 401


def test_tampered_jwt_is_rejected(client):
    tok = _jwt()
    tampered = tok[:-3] + ("aaa" if not tok.endswith("aaa") else "bbb")
    r = client.get("/v1/auth/me", headers={"Authorization": f"Bearer {tampered}"})
    assert r.status_code == 401


def test_identity_from_token_not_body(client):
    """A viewer JWT cannot escalate by putting another identity in the chat body —
    the body has no identity fields and the tenant/roles come only from the token."""
    viewer = _jwt(org="org_globex", user="u-erin", roles=("viewer",))
    H = {"Authorization": f"Bearer {viewer}"}
    # ingest needs analyst+; this viewer token must be denied regardless of body.
    r = client.post("/v1/ingest", json={"documents": [{"doc_id": "x", "text": "y"}],
                                        "org_id": "org_acme", "roles": ["admin"]}, headers=H)
    assert r.status_code == 403


def test_dev_api_key_constant_matches_default():
    """Guard: the fixture key is the configured default, so the gateway check passes."""
    assert _DEV_API_KEY in set(get_settings().api_keys)
