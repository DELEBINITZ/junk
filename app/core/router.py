"""The orchestrator/router seam.

Detect the relevant module(s) cheaply, then expose ONLY their tools (+ always-on
core tools) to the agent, capped — so the planner's tool list and context stay
BOUNDED as modules grow (the answer to "one agent with all tools confuses the LLM
and blows the context window"). Two scorers ship:

  * KeywordRouteScorer (default, deterministic) — overlap with routing hints.
  * LLMPlanner (ROUTER_MODE=llm) — a fast-lane LLM selects modules + lane.

Both produce the same RouteDecision; the keyword scorer is also the fallback when
the LLM planner returns nothing. See plan §6.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Protocol

from app.core.contracts import CapabilityManifest
from app.core.registry import CORE_MODULE_ID, CapabilityRegistry
from app.domain import User


_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN.findall(text.lower()))


def _extract_json(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    return text[start : end + 1] if start != -1 and end != -1 else "{}"


@dataclass(slots=True)
class RouteDecision:
    module_ids: list[str]
    tool_names: list[str]
    lane: str = "standard"
    scores: dict[str, float] = field(default_factory=dict)


class RouteScorer(Protocol):
    def score(self, query_tokens: set[str], manifest: CapabilityManifest) -> float: ...


class KeywordRouteScorer:
    def score(self, query_tokens: set[str], manifest: CapabilityManifest) -> float:
        hint_tokens: set[str] = set()
        for hint in manifest.routing_hints:
            for phrase in (*hint.intents, *hint.examples):
                hint_tokens |= _tokens(phrase)
        if not hint_tokens or not query_tokens:
            return 0.0
        return len(query_tokens & hint_tokens) / len(query_tokens)


class LLMPlanner:
    """Fast-lane LLM module router. Returns (module_ids, lane). On any failure
    (parse error, unavailable model) returns ([], 'standard') so the Router falls
    back to the keyword scorer."""

    def __init__(self, client):
        self.client = client

    def plan(self, query: str, manifests: list[CapabilityManifest]) -> tuple[list[str], str]:
        catalog = "\n".join(
            f"- {m.id}: {', '.join(m.routing_hints[0].intents[:8]) if m.routing_hints else ''}"
            for m in manifests
        )
        system = (
            "Route the user query to the relevant capability module(s). Respond "
            'with JSON: {"modules": ["<id>", ...], "lane": "fast|standard|deep"}. '
            "Use only module ids from the catalog; pick the deep lane only for "
            "hard multi-step analysis."
        )
        try:
            raw = self.client.invoke(system_prompt=system, user_prompt=f"Catalog:\n{catalog}\n\nQuery: {query}")
            data = json.loads(_extract_json(raw))
            valid_ids = {m.id for m in manifests}
            ids = [i for i in data.get("modules", []) if i in valid_ids]
            lane = data.get("lane", "standard")
            return ids, lane if lane in ("fast", "standard", "deep") else "standard"
        except Exception:
            return [], "standard"


class Router:
    def __init__(
        self,
        registry: CapabilityRegistry,
        scorer: RouteScorer | None = None,
        planner: LLMPlanner | None = None,
        max_tools: int = 12,
        threshold: float = 0.0,
    ):
        self.registry = registry
        self.scorer = scorer or KeywordRouteScorer()
        self.planner = planner
        self.max_tools = max_tools
        self.threshold = threshold

    def route(self, query: str, user: User) -> RouteDecision:
        modules = self.registry.modules_for_user(user)
        qtokens = _tokens(query)
        scores = {m.id: self.scorer.score(qtokens, m) for m in modules}
        lane = "standard"

        if self.planner is not None:
            ids, lane = self.planner.plan(query, modules)
            selected = [m for m in modules if m.id in ids]
        else:
            selected = [m for m in modules if scores.get(m.id, 0.0) > self.threshold]

        if not selected and modules:
            # Fallback: never strand the user — pick the best keyword match.
            selected = [max(modules, key=lambda m: scores.get(m.id, 0.0))]
        selected_ids = {m.id for m in selected}

        tool_names: list[str] = []
        for tool in self.registry.tools_for_user(user):
            module_id = self.registry.module_of(tool.name)
            if module_id == CORE_MODULE_ID or module_id in selected_ids:
                tool_names.append(tool.name)
        return RouteDecision(
            module_ids=[m.id for m in selected],
            tool_names=tool_names[: self.max_tools],
            lane=lane,
            scores=scores,
        )
