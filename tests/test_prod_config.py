"""Production guard: ENVIRONMENT=prod fail-closes on any stub/insecure setting,
so the security product can never silently run on hash-embeddings + in-memory data.
Dev/test (the deterministic path) is unaffected.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.config import Settings


def test_prod_rejects_stub_defaults():
    # all defaults are stubs (deterministic LLM/embeddings, memory stores) -> reject
    with pytest.raises(ValidationError):
        Settings(_env_file=None, environment="prod")


def test_prod_error_lists_what_to_fix():
    try:
        Settings(_env_file=None, environment="prod")
    except Exception as exc:
        msg = str(exc)
        assert "LLM_PROVIDER=deterministic" in msg
        assert "RETRIEVAL_BACKEND=memory" in msg
        assert "STORE_BACKEND=memory" in msg
    else:
        pytest.fail("prod did not reject stub config")


def test_prod_accepts_full_real_config():
    s = Settings(
        _env_file=None, environment="prod", debug=False, seed_demo_data=False,
        llm_provider="sglang", embedding_provider="tei",
        retrieval_backend="qdrant", store_backend="postgres",
        database_url="postgresql://u:p@h:5432/db",
        auth_provider="local", jwt_secret="x" * 40,
        cap_reports_enabled=True, cap_easm_enabled=False,
        cap_brand_enabled=False, cap_aci_enabled=False,
    )
    assert s.is_prod


def test_prod_requires_mcp_url_for_enabled_tool_module():
    # easm enabled in prod with NO server URL would serve mock data -> reject
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None, environment="prod", debug=False, seed_demo_data=False,
            llm_provider="sglang", embedding_provider="tei",
            retrieval_backend="qdrant", store_backend="postgres",
            database_url="postgresql://u:p@h:5432/db",
            auth_provider="local", jwt_secret="x" * 40,
            cap_easm_enabled=True, easm_mcp_url="",
            cap_brand_enabled=False, cap_aci_enabled=False,
        )


def test_dev_defaults_unaffected():
    s = Settings(_env_file=None, environment="dev")
    assert not s.is_prod and s.llm_provider == "deterministic"
