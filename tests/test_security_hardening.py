"""Security hardening: the MCP service-token trust model.

Service tokens are a DISTINCT type ("service"), so they can't be replayed as user
tokens on the main API; they are bound to one server by ``aud``; and — when an
asymmetric keypair is configured — a server holding only the PUBLIC key cannot mint
tokens (closing the "any promoted server can forge admin tokens for any org" hole).
"""

from __future__ import annotations

import pytest

from app.config import Settings
from app.core.errors import AuthError
from app.core.security.jwt import (
    create_access_token,
    create_service_token,
    decode_service_token,
    decode_token,
)


def _settings(**kw) -> Settings:
    return Settings(_env_file=None, jwt_secret="x" * 48, **kw)


# --- audience binding (HS256 fallback path) ---------------------------------

def test_service_token_rejected_for_wrong_audience():
    s = _settings()
    tok = create_service_token(s, sub="u", org_id="org_acme", roles=("viewer",),
                               audience="easm-mcp", ttl_seconds=60)
    with pytest.raises(AuthError, match="audience"):
        decode_service_token(s, tok, expected_audience="aci-mcp")


def test_service_token_accepted_for_correct_audience():
    s = _settings()
    tok = create_service_token(s, sub="u", org_id="org_acme", roles=("analyst",),
                               audience="easm-mcp", ttl_seconds=60)
    claims = decode_service_token(s, tok, expected_audience="easm-mcp")
    assert claims["org_id"] == "org_acme" and claims["aud"] == "easm-mcp"
    assert claims["type"] == "service"


# --- cross-domain token reuse is blocked both ways --------------------------

def test_service_token_rejected_on_main_api_path():
    # A service token is type="service"; the main API verifies expected_type="access",
    # so it can't be replayed as a user token.
    s = _settings()
    tok = create_service_token(s, sub="u", org_id="org_acme", roles=("admin",),
                               audience="easm-mcp", ttl_seconds=60)
    with pytest.raises(AuthError, match="expected access"):
        decode_token(s, tok, expected_type="access")


def test_user_token_rejected_on_mcp_path():
    # Symmetric guarantee: a user/access token must NOT authenticate an MCP tool call.
    s = _settings()
    issued = create_access_token(s, sub="u", org_id="org_acme", roles=("admin",))
    with pytest.raises(AuthError, match="expected service"):
        decode_service_token(s, issued.token, expected_audience="easm-mcp")


def test_tampered_service_token_error_is_generic():
    s = _settings()
    tok = create_service_token(s, sub="u", org_id="org_acme", roles=("viewer",),
                               audience="easm-mcp", ttl_seconds=60)
    tampered = tok[:-3] + ("aaa" if not tok.endswith("aaa") else "bbb")
    with pytest.raises(AuthError) as ei:
        decode_service_token(s, tampered)
    assert str(ei.value) == "invalid token"     # generic, no PyJWT internals leaked


# --- asymmetric signing: a server holding only the public key can't mint -----

def _rsa_pem_pair():
    pytest.importorskip("cryptography")
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    priv = key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption()).decode()
    pub = key.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo).decode()
    return priv, pub


def test_asymmetric_roundtrip_core_signs_server_verifies():
    priv, pub = _rsa_pem_pair()
    core = _settings(jwt_service_private_key=priv, jwt_service_public_key=pub)
    server = _settings(jwt_service_private_key="", jwt_service_public_key=pub)   # PUBLIC only
    tok = create_service_token(core, sub="u", org_id="org_acme", roles=("admin",),
                               audience="easm-mcp", ttl_seconds=60)
    claims = decode_service_token(server, tok, expected_audience="easm-mcp")
    assert claims["org_id"] == "org_acme" and claims["type"] == "service"


def test_asymmetric_server_cannot_mint():
    # A compromised MCP server holds only the public key. RS256 signing needs the
    # PRIVATE key, so any attempt to mint a token from the server fails outright —
    # it cannot forge identity for another org.
    _priv, pub = _rsa_pem_pair()
    server = _settings(jwt_service_private_key="", jwt_service_public_key=pub)
    with pytest.raises(Exception):     # noqa: B017 - signing with no private key fails
        create_service_token(server, sub="evil", org_id="victim", roles=("admin",),
                             audience="easm-mcp", ttl_seconds=60)
