"""Deterministic cross-referencing for multi-agent synthesis.

THE multi-agent correctness risk is the JOIN: "which of our exposed assets (Aura)
appear in recent threat reports (Sentinel)?" is a set-intersection, and LLMs do
set-intersection unreliably — they fuzzy-match, invent overlaps, or miss real ones.

So we do the join HERE, in code, over entities extracted from each agent's findings,
and hand the synthesizer the *computed* result as ground truth. The LLM then only
NARRATES a pre-computed intersection instead of inferring it. This eliminates the
join-hallucination failure mode; `unsupported_cves` (post-synthesis) catches any
entity the model adds anyway.

Entity extractors are a registry so new join keys (asset/domain/IP once Aura returns
them structured) plug in without touching the synthesizer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from security_intel.observability.eval_scoring import extract_cve_ids

# entity_type -> extractor(text) -> set[str] of normalized entities.
# CVE is the concrete, unambiguous, high-value key today; add asset/domain/IP
# extractors here as agents start returning those.
_EXTRACTORS = {"CVE": extract_cve_ids}


@dataclass
class CrossReference:
    # entity_type -> {entity -> [agent_id, ...]} for entities found by >= 2 agents.
    overlaps: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    # entity_type -> {agent_id -> set(entities)} — full per-agent extraction (provenance).
    per_agent: dict[str, dict[str, set]] = field(default_factory=dict)

    @property
    def has_overlap(self) -> bool:
        return any(ents for ents in self.overlaps.values())

    @property
    def any_entities(self) -> bool:
        return any(agent_map for agent_map in self.per_agent.values())

    def citations(self) -> list[dict]:
        """Structured provenance for each computed overlap (goes into state citations)."""
        out = []
        for etype, ents in self.overlaps.items():
            for ent, agents in sorted(ents.items()):
                out.append({
                    "type": "cross_reference",
                    "entity_type": etype,
                    "entity": ent,
                    "agents": agents,
                })
        return out

    def render_facts(self) -> str:
        """A deterministic block for the synthesis prompt. Empty when there is nothing
        computed to assert (no entities anywhere), so single-domain answers are
        unaffected. When entities exist but nothing overlaps, we STILL emit a block so
        the model states the absence instead of inventing a match."""
        if not self.any_entities:
            return ""
        lines = [
            "VERIFIED CROSS-REFERENCE (computed deterministically from the agent "
            "findings — treat as GROUND TRUTH; state EXACTLY these and do NOT infer, "
            "add, or omit any match):"
        ]
        emitted = False
        for etype, ents in self.overlaps.items():
            for ent, agents in sorted(ents.items()):
                lines.append(f"- {etype} {ent}: found by {', '.join(agents)}")
                emitted = True
        if not emitted:
            found = sorted(
                {e for agent_map in self.per_agent.values() for ents in agent_map.values() for e in ents}
            )
            lines.append(
                "- NO entity appears in more than one source. There is NO overlap to "
                f"report. Entities seen (single-source only): {', '.join(found) or 'none'}."
            )
        return "\n".join(lines)


def cross_reference(real_results: list[dict]) -> CrossReference:
    """Extract entities from each agent's findings text and compute cross-source
    overlaps (an entity found by >= 2 distinct agents).

    ``real_results`` are the non-error agent results: dicts with ``agent_id`` and
    ``findings`` (text). Deterministic and side-effect-free.
    """
    per_agent: dict[str, dict[str, set]] = {etype: {} for etype in _EXTRACTORS}
    for r in real_results:
        aid = r.get("agent_id", "?")
        text = r.get("findings", "") or ""
        for etype, extract in _EXTRACTORS.items():
            ents = extract(text)
            if ents:
                # union in case an agent appears twice
                per_agent[etype].setdefault(aid, set()).update(ents)

    overlaps: dict[str, dict[str, list[str]]] = {}
    for etype, agent_map in per_agent.items():
        counts: dict[str, list[str]] = {}
        for aid, ents in agent_map.items():
            for e in ents:
                counts.setdefault(e, []).append(aid)
        overlaps[etype] = {
            e: sorted(set(agents)) for e, agents in counts.items() if len(set(agents)) >= 2
        }
    return CrossReference(overlaps=overlaps, per_agent=per_agent)
