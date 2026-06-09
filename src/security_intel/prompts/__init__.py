"""Centralized prompt templates, organized by agent / pipeline component.

Import prompts from here instead of inlining them in logic, e.g.:
    from security_intel.prompts.reports import REPORTS_SYSTEM_PROMPT
    from security_intel.prompts import SYNTHESIS_PROMPT
"""

from security_intel.prompts.reports import REPORTS_SYSTEM_PROMPT
from security_intel.prompts.easm import EASM_SYSTEM_PROMPT
from security_intel.prompts.orchestrator import (
    ORCHESTRATOR_PERSONA,
    ROUTER_PROMPT,
    CHITCHAT_PROMPT,
    SYNTHESIS_PROMPT,
    SYNTH_FALLBACK_MSG,
)
from security_intel.prompts.enrichment import (
    STRATEGY_CLASSIFIER_PROMPT,
    MULTI_QUERY_PROMPT,
    HYDE_PROMPT,
    STEP_BACK_PROMPT,
)
from security_intel.prompts.planner import PLANNER_SYSTEM_TEMPLATE

__all__ = [
    "REPORTS_SYSTEM_PROMPT",
    "EASM_SYSTEM_PROMPT",
    "ORCHESTRATOR_PERSONA",
    "ROUTER_PROMPT",
    "CHITCHAT_PROMPT",
    "SYNTHESIS_PROMPT",
    "SYNTH_FALLBACK_MSG",
    "STRATEGY_CLASSIFIER_PROMPT",
    "MULTI_QUERY_PROMPT",
    "HYDE_PROMPT",
    "STEP_BACK_PROMPT",
    "PLANNER_SYSTEM_TEMPLATE",
]
