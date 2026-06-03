"""Security hardening regressions: service-token audience binding + generic auth
errors. These guard the fixes that closed the cross-server token-replay gap and
stopped echoing token internals back to callers.
"""

from __future__ import annotations

import pytest

from app.config import Settings
from app.core.errors import AuthError
from app.core.security.jwt import create_service_token, decode_token


def _settings() -> Settings:
    return Settings(_env_file=None, jwt_secret="x" * 48)


def test_service_token_rejected_for_wrong_audience():
    """A token minted for easm-mcp must NOT verify against the aci-mcp server —
    expected_audience binds the token to one server, blocking replay."""
    s = _settings()
    tok = create_service_token(s, sub="u", org_id="org_acme", roles=("viewer",),
                               audience="easm-mcp", ttl_seconds=60)
    with pytest.raises(AuthError, match="audience"):
        decode_token(s, tok, expected_type="access", expected_audience="aci-mcp")


def test_service_token_accepted_for_correct_audience():
    s = _settings()
    tok = create_service_token(s, sub="u", org_id="org_acme", roles=("analyst",),
                               audience="easm-mcp", ttl_seconds=60)
    claims = decode_token(s, tok, expected_type="access", expected_audience="easm-mcp")
    assert claims["org_id"] == "org_acme" and claims["aud"] == "easm-mcp"


def test_no_audience_check_is_backward_compatible():
    """The main API auth path passes no expected_audience, so a normal token still
    verifies regardless of any aud claim — the new check is opt-in per server."""
    s = _settings()
    tok = create_service_token(s, sub="u", org_id="org_acme", roles=("viewer",),
                               audience="easm-mcp", ttl_seconds=60)
    claims = decode_token(s, tok, expected_type="access")   # no expected_audience
    assert claims["sub"] == "u"


def test_tampered_token_error_is_generic():
    """A bad-signature token raises a GENERIC message — we never echo which check
    failed or any token internals back to the caller."""
    s = _settings()
    tok = create_service_token(s, sub="u", org_id="org_acme", roles=("viewer",),
                               audience="easm-mcp", ttl_seconds=60)
    tampered = tok[:-3] + ("aaa" if not tok.endswith("aaa") else "bbb")
    with pytest.raises(AuthError) as ei:
        decode_token(s, tampered)
    assert str(ei.value) == "invalid token"     # generic, no PyJWT internals leaked
