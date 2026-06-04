"""Library-based guardrails wire correctly from config.

PII -> Microsoft Presidio · prompt injection -> Llama Prompt Guard 2 · content
safety -> Llama Guard 3. The hand-rolled regex floors were removed; the only custom
detectors are the ones no library covers (secret redaction, output exfiltration).
"""

from __future__ import annotations

import pytest

from app.config import Settings
from app.core.errors import ConfigError
from app.core.guardrails.detectors import _flag_injection
from app.core.guardrails.pipeline import build_input_guardrails, build_output_guardrails


def test_flag_injection_parsing():
    # robust to the various Prompt Guard response shapes
    assert _flag_injection([{"label": "JAILBREAK", "score": 0.95}], 0.8) is True
    assert _flag_injection([[{"label": "benign", "score": 0.99}]], 0.8) is False
    assert _flag_injection({"label": "INJECTION", "score": 0.6}, 0.8) is False    # below threshold
    assert _flag_injection([{"label": "LABEL_1", "score": 0.9}], 0.8) is True
    assert _flag_injection("garbage", 0.8) is False


def test_input_pipeline_wires_prompt_guard_and_llama_guard():
    s = Settings(_env_file=None, prompt_guard_url="http://pg:8085", llama_guard_url="http://lg:8086/v1")
    names = [d.name for d in build_input_guardrails(s).detectors]
    assert "secret_redactor" in names      # always-on floor (custom — no library covers it inline)
    assert "prompt_injection" in names     # Llama Prompt Guard 2
    assert "llama_guard" in names          # Llama Guard 3


def test_input_pipeline_no_llama_guard_without_url():
    # content safety is added only when its classifier endpoint is configured
    names = [d.name for d in build_input_guardrails(Settings(_env_file=None)).detectors]
    assert "llama_guard" not in names
    assert "secret_redactor" in names      # the floor is always present


def test_prompt_guard_threshold_and_failclosed_wired():
    s = Settings(_env_file=None, prompt_guard_url="http://pg:8085",
                 prompt_guard_threshold=0.7, guardrails_fail_closed=True)
    inj = next(d for d in build_input_guardrails(s).detectors if d.name == "prompt_injection")
    assert inj.threshold == 0.7 and inj.fail_closed is True and inj.model_url == "http://pg:8085"


def test_fail_closed_default_is_true():
    # production-safe default: a classifier outage BLOCKS rather than passes through
    assert Settings(_env_file=None).guardrails_fail_closed is True


def test_pii_build_is_fail_fast_on_missing_presidio():
    """Presidio is THE PII path. With PII redaction on, the build either produces the
    redactor (Presidio installed) or raises a LOUD ConfigError (not installed) — never
    a silent no-op."""
    try:
        import presidio_analyzer  # noqa: F401
        installed = True
    except Exception:
        installed = False
    if installed:
        names = [d.name for d in build_output_guardrails(Settings(_env_file=None)).detectors]
        assert "pii_redactor" in names
    else:
        with pytest.raises(ConfigError):
            build_output_guardrails(Settings(_env_file=None))


def test_pii_disabled_still_builds_exfil_guard():
    # With PII off, the output pipeline still defangs exfiltration vectors.
    names = [d.name for d in build_output_guardrails(Settings(_env_file=None, pii_redaction=False)).detectors]
    assert "output_exfiltration" in names and "pii_redactor" not in names


@pytest.mark.asyncio
async def test_presidio_redacts_when_installed():
    pytest.importorskip("presidio_analyzer")
    from app.core.guardrails.detectors import PresidioPIIRedactor

    d = PresidioPIIRedactor(entities=("PERSON", "EMAIL_ADDRESS"))
    v = await d.check("Contact John Smith at john.smith@acme.com about the report.", None)
    assert v.action.name == "REDACT"
