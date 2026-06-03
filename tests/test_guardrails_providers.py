"""Self-hostable guardrail providers (Presidio / Prompt Guard 2 / Llama Guard 3)
wire correctly from config and degrade gracefully when an optional dep is absent.
"""

from __future__ import annotations

import pytest

from app.config import Settings
from app.core.guardrails.detectors import _flag_injection
from app.core.guardrails.pipeline import build_input_guardrails, build_output_guardrails


def test_flag_injection_parsing():
    # robust to the various classifier response shapes
    assert _flag_injection([{"label": "JAILBREAK", "score": 0.95}], 0.8) is True
    assert _flag_injection([[{"label": "benign", "score": 0.99}]], 0.8) is False
    assert _flag_injection({"label": "INJECTION", "score": 0.6}, 0.8) is False    # below threshold
    assert _flag_injection([{"label": "LABEL_1", "score": 0.9}], 0.8) is True
    assert _flag_injection("garbage", 0.8) is False


def test_input_pipeline_adds_llama_guard_when_configured():
    s = Settings(_env_file=None, llama_guard_url="http://lg:8086/v1")
    names = [d.name for d in build_input_guardrails(s).detectors]
    assert "topic_safety" in names         # regex harm floor still present
    assert "llama_guard" in names          # model layer added ALONGSIDE it


def test_input_pipeline_no_llama_guard_by_default():
    names = [d.name for d in build_input_guardrails(Settings(_env_file=None)).detectors]
    assert "llama_guard" not in names


def test_prompt_guard_threshold_and_failclosed_wired():
    s = Settings(_env_file=None, prompt_guard_url="http://pg:8085",
                 prompt_guard_threshold=0.7, guardrails_fail_closed=True)
    inj = next(d for d in build_input_guardrails(s).detectors if d.name == "prompt_injection")
    assert inj.threshold == 0.7 and inj.fail_closed is True and inj.model_url == "http://pg:8085"


def test_pii_presidio_falls_back_to_regex_without_lib():
    # presidio not installed in this env -> build must still produce a PII detector
    names = [d.name for d in build_output_guardrails(Settings(_env_file=None, pii_provider="presidio")).detectors]
    assert "pii_redactor" in names


def test_pii_regex_default_and_exfil_present():
    names = [d.name for d in build_output_guardrails(Settings(_env_file=None, pii_provider="regex")).detectors]
    assert "pii_redactor" in names and "output_exfiltration" in names


@pytest.mark.asyncio
async def test_presidio_redacts_when_installed():
    pytest.importorskip("presidio_analyzer")
    from app.core.guardrails.detectors import PresidioPIIRedactor

    d = PresidioPIIRedactor(entities=("PERSON", "EMAIL_ADDRESS"))
    v = await d.check("Contact John Smith at john.smith@acme.com about the report.", None)
    assert v.action.name == "REDACT"
    assert "John Smith" not in v.text and "john.smith@acme.com" not in v.text
