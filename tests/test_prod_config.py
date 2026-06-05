"""Production guard: ENVIRONMENT=prod fail-closes on insecure/incomplete settings.

The deterministic/in-memory stubs have been removed (a real LLM/embedder/Qdrant/
Postgres is always required), so the guard now checks the remaining footguns: a
missing DB URL, dev secrets, the dev API key, and demo toggles.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.config import Settings

# A complete, secure prod config; individual tests break ONE thing to trigger the guard.
_REAL = dict(
    _env_file=None, environment="prod", debug=False, seed_demo_data=False,
    database_url="postgresql://u:p@h:5432/db",
    jwt_secret="x" * 40, api_keys=["prod-gateway-key-001"],
    # No guard-model endpoints: injection/content-safety run on the main LLM
    # (LLMJudgeGuard), and a real LLM endpoint is already mandatory.
    cap_reports_enabled=True, cap_easm_enabled=False,
    cap_brand_enabled=False, cap_aci_enabled=False,
)


def test_prod_rejects_missing_database_url():
    cfg = {**_REAL, "database_url": ""}
    with pytest.raises(ValidationError):
        Settings(**cfg)


def test_prod_rejects_dev_secrets():
    # dev JWT secret + dev API key must be replaced in prod
    with pytest.raises(ValidationError):
        Settings(_env_file=None, environment="prod", debug=False, seed_demo_data=False,
                 database_url="postgresql://u:p@h:5432/db")


def test_prod_error_lists_what_to_fix():
    try:
        Settings(_env_file=None, environment="prod")
    except Exception as exc:
        msg = str(exc)
        assert "DATABASE_URL" in msg
        assert "JWT_SECRET" in msg
        assert "API_KEYS" in msg
    else:
        pytest.fail("prod did not reject incomplete config")


def test_prod_accepts_full_real_config():
    assert Settings(**_REAL).is_prod


def test_prod_needs_no_guard_model_endpoints():
    # injection/content-safety run on the main LLM (LLMJudgeGuard) — prod boots
    # with guardrails fully on and NO dedicated guard-model URLs in the config.
    s = Settings(**_REAL)
    assert s.is_prod and s.guardrails_enabled
    assert s.injection_detection and s.topic_safety


def test_prod_requires_mcp_url_for_enabled_tool_module():
    # easm enabled in prod with NO server URL would serve mock data -> reject
    cfg = {**_REAL, "cap_easm_enabled": True, "easm_mcp_url": ""}
    with pytest.raises(ValidationError):
        Settings(**cfg)


def test_prod_remote_mcp_requires_api_key():
    # A remote MCP server wired in prod with NO transport API key -> reject (anyone
    # who can reach the server could otherwise drive its tools).
    cfg = {**_REAL, "cap_easm_enabled": True, "easm_mcp_url": "https://easm/mcp"}
    with pytest.raises(ValidationError):
        Settings(**cfg)


def test_prod_remote_mcp_with_api_key_ok():
    cfg = {**_REAL, "cap_easm_enabled": True, "easm_mcp_url": "https://easm/mcp",
           "mcp_api_key": "a-strong-mcp-key"}
    assert Settings(**cfg).is_prod


def test_prod_remote_mcp_per_module_api_key_ok():
    cfg = {**_REAL, "cap_easm_enabled": True, "easm_mcp_url": "https://easm/mcp",
           "mcp_api_keys": {"easm": "per-module-key"}}
    s = Settings(**cfg)
    assert s.mcp_api_key_for("easm") == "per-module-key"


def test_dev_defaults_are_real_providers():
    # No more deterministic default — dev now points at the real (self-hosted) stack.
    s = Settings(_env_file=None, environment="dev")
    assert not s.is_prod
    assert s.llm_provider == "vllm" and s.embedding_provider == "tei"
    assert s.retrieval_backend == "qdrant" and s.store_backend == "postgres"
