"""Dynamic routing: capabilities are chosen by MEANING (module description + tool
descriptions), not by hand-written keyword routing_hints (which have been removed).
With a real LLM the LLM router decides; the deterministic default uses semantic
embedding similarity over each module's profile.
"""

from __future__ import annotations

import pytest

from app.config import Settings
from app.core.agent.supervisor import Supervisor


@pytest.mark.asyncio
async def test_semantic_scores_rank_by_meaning(services):
    """Embedding similarity over each module's profile ranks the relevant module
    high for an attack-surface query — no keyword matching involved."""
    sup = Supervisor(services.registry, services.deps.llm, services.deps.settings,
                     embedder=services.deps.rag.embedder, max_fanout=2)
    scores = await sup._semantic(
        "what is exposed on our external attack surface and open ports",
        ["reports", "easm", "aci", "brand"],
    )
    assert scores                                   # got embedding scores
    top2 = sorted(scores, key=scores.get, reverse=True)[:2]
    assert "easm" in top2                           # matched by meaning, not keywords


@pytest.mark.asyncio
async def test_semantic_strategy_routes_without_keywords(services, acme):
    s = Settings(_env_file=None, routing_strategy="semantic")
    sup = Supervisor(services.registry, services.deps.llm, s,
                     embedder=services.deps.rag.embedder, max_fanout=2)
    r = await sup.route("what is exposed on our external attack surface", acme)
    assert r.mode == "semantic"
    assert "easm" in r.modules


def test_module_profile_uses_description_and_tools(services):
    """A module's routing profile is built from its description + tool descriptions
    (hints are just extra text), so a module is routable even with zero hints."""
    sup = Supervisor(services.registry, services.deps.llm, services.deps.settings,
                     embedder=services.deps.rag.embedder)
    profile = sup._module_profile(services.registry.module("easm")).lower()
    assert "attack surface" in profile          # from description
    assert "query_assets" in profile            # from a tool name
    assert "external assets" in profile         # from a tool description


@pytest.mark.asyncio
async def test_semantic_routes_attack_surface_question(services, acme):
    """A plainly-worded attack-surface question routes to easm by MEANING (semantic
    similarity over description + tools), with no keyword/routing_hints involved."""
    sup = Supervisor(services.registry, services.deps.llm, services.deps.settings,
                     embedder=services.deps.rag.embedder, max_fanout=2)
    r = await sup.route("what assets do we have exposed to the internet?", acme)
    assert r.mode == "semantic"
    assert "easm" in r.modules


@pytest.mark.asyncio
async def test_cross_domain_question_fans_out(services, acme):
    """A genuinely cross-domain question routes to max_fanout specialists by meaning.
    The threat-actor clause reliably surfaces aci; the second slot is the next-best
    semantic match. (On the real LLM-router path this is easm + aci; the offline
    embedder is approximate — see test_contracts routing test.)"""
    sup = Supervisor(services.registry, services.deps.llm, services.deps.settings,
                     embedder=services.deps.rag.embedder, max_fanout=2)
    r = await sup.route("what is our biggest exposure and which threat actor weaponizes it?", acme)
    assert len(r.modules) == 2          # fanned out
    assert "aci" in r.modules           # the adversary clause routed to aci by meaning
