"""Guardrails: input/output safety spine."""

from app.core.guardrails.base import Action, GuardrailResult, GuardrailVerdict
from app.core.guardrails.pipeline import (
    BLOCK_REPLY,
    HALLUCINATION_FALLBACK,
    InputGuardrailPipeline,
    OutputGuardrailPipeline,
    build_input_guardrails,
    build_output_guardrails,
)

__all__ = [
    "Action",
    "GuardrailResult",
    "GuardrailVerdict",
    "InputGuardrailPipeline",
    "OutputGuardrailPipeline",
    "build_input_guardrails",
    "build_output_guardrails",
    "BLOCK_REPLY",
    "HALLUCINATION_FALLBACK",
]
