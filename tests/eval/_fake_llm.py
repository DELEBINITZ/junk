"""Fake-LLM scaffolding for running the routing eval WITHOUT a live LLM.

This does NOT measure routing *decision quality* — that's the LLM's job and needs a
real model. It measures the routing *plumbing*: given a scripted decision, does the
real orchestrator graph classify → route → dispatch to the right agent (real Qdrant,
real tool_call execution, real reflection gate) and produce an answer end-to-end.

Used only by `run_eval.py --fake-llm`.
"""

from __future__ import annotations

import json
import re
import sys
import types

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

# ROUTER_PROMPT ends with "Question: <q>\n\nContext from prior conversation:"
_Q_RE = re.compile(r"Question:\s*(.*?)\s*\n\nContext from prior conversation:", re.S)


class FakeRoutingLLM(BaseChatModel):
    """Returns a scripted routing decision for router prompts, canned text otherwise.

    Forces the router's text-parse fallback (with_structured_output raises), so this
    also exercises that resilience path.
    """

    decisions: dict = {}

    @property
    def _llm_type(self) -> str:
        return "fake-routing"

    def bind_tools(self, tools, **kwargs):  # planner/create_react_agent calls this
        return self

    def with_structured_output(self, schema, **kwargs):
        raise NotImplementedError("fake: force text-parse router fallback")

    def _text(self, messages) -> str:
        joined = "\n".join(
            m.content for m in messages if isinstance(getattr(m, "content", None), str)
        )
        if "intelligent gateway" in joined:  # ROUTER_PROMPT signature
            m = _Q_RE.search(joined)
            q = m.group(1).strip() if m else ""
            decision = self.decisions.get(q, {"action": "SIMPLE", "agent": "", "task": q,
                                              "confidence": 0.3})
            return json.dumps(decision)
        # synthesis / chitchat / anything else
        return "Based on the available findings, here is a concise answer."

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        msg = AIMessage(content=self._text(messages))
        return ChatResult(generations=[ChatGeneration(message=msg)])

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        return self._generate(messages, stop=stop, **kwargs)


def build_decisions(cases: list[dict]) -> dict:
    """Golden query -> scripted RouterDecision dict (from each case's class)."""
    out = {}
    for c in cases:
        cls = c.get("class")
        exp = c.get("expect_agents", [])
        if cls == "direct":
            d = {"action": "DIRECT", "response": "Happy to help with security intel and product how-to.",
                 "confidence": 0.95}
        elif cls in ("refuse", "clarify"):
            # REFUSE was retired — an unroutable query now gracefully CLARIFYs
            # (capability redirect), never a cold rejection.
            d = {"action": "CLARIFY", "confidence": 0.3}
        elif cls == "complex":
            d = {"action": "COMPLEX", "confidence": 0.9}
        else:
            d = {"action": "SIMPLE", "agent": exp[0] if exp else "", "task": c["query"],
                 "confidence": 0.95}
        out[c["query"]] = d
    return out


def install_guardrail_stub() -> None:
    """Replace the Presidio-backed guardrails module with a passthrough BEFORE the
    orchestrator lazily imports it — lets the eval run with no Presidio/spaCy."""
    mod = types.ModuleType("security_intel.security.guardrails")

    async def input_guardrail_node(state, config, llm=None, assistant_desc="", **_):
        return {"blocked": False, "block_reason": ""}

    async def output_guardrail_node(state, config, domains="", **_):
        return {}

    mod.input_guardrail_node = input_guardrail_node
    mod.output_guardrail_node = output_guardrail_node
    mod._get_analyzer = lambda: None
    mod._get_anonymizer = lambda: None
    sys.modules["security_intel.security.guardrails"] = mod


def register_light_agents(registry, settings) -> None:
    """Register sentinel + atlas directly (same modes/tools as main) without
    importing the full FastAPI app stack. Aura (EASM) is omitted (needs an MCP server)."""
    from security_intel.agents.registry import AgentSpec
    from security_intel.agents.sentinel.tools import get_reports_tools
    from security_intel.agents.atlas.tools import get_user_guide_tools
    from security_intel.prompts.sentinel import SENTINEL_SYSTEM_PROMPT
    from security_intel.prompts.atlas import ATLAS_SYSTEM_PROMPT

    registry.register(AgentSpec(
        id="sentinel", display_name="Sentinel", domain_label="security reports & threat intelligence",
        description="Searches security reports corpus (threat intel, AI-generated reports).",
        capabilities=["Semantic search over security reports", "Filter by threat type/TLP",
                      "Get report metadata"],
        system_prompt=SENTINEL_SYSTEM_PROMPT, tools=get_reports_tools(settings),
        mode="tool_call", primary_tool="search_reports",
    ))
    registry.register(AgentSpec(
        id="atlas", display_name="Atlas", domain_label="FortiRecon product guidance",
        description="Helps you use the FortiRecon platform: how-to, navigation, dashboards, "
                    "features, and configuration, from the product documentation.",
        capabilities=["Explain dashboards/menus/features", "Step-by-step how-to",
                      "Navigation / where to find things"],
        system_prompt=ATLAS_SYSTEM_PROMPT, tools=get_user_guide_tools(settings),
        # react: matches main.py — the agent can chain search -> full-page fetch.
        mode="react",
    ))
