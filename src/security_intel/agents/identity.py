"""System identity — DERIVED from the enabled agent set, not hardcoded.

The orchestrator's persona, scope, and refusal boundaries used to be fixed strings
that said "Security Intelligence Assistant" no matter which agents were active. That
meant a deployment running only the user-guide agent still behaved like a security
tool (and refused legitimate product-usage questions as "out of scope").

`SystemProfile` fixes that: it reads the live registry (the agents that actually
built) plus optional operator overrides, and produces the name / tagline / scope /
domain list that every prompt is parameterized on. Enable a different set of agents
and the whole assistant's personality reshapes — no prompt edits required.
"""

from __future__ import annotations

from dataclasses import dataclass


def _strip_agent_suffix(display_name: str) -> str:
    """'Brand Protection Agent' -> 'Brand Protection' (label, not a role).

    Mainly for auto-generated MCP display names; the core specialists (Atlas,
    Sentinel, Aura) don't carry an "Agent" suffix.
    """
    name = (display_name or "").strip()
    if name.lower().endswith("agent"):
        name = name[: -len("agent")].strip()
    return name or "Assistant"


@dataclass
class SystemProfile:
    """The assistant's derived identity, threaded into every persona-bearing prompt.

    - name:     the MASTER's user-facing name ("FortiRecon Assistant") — never a specialist's.
    - tagline:  one-line role, spoken in first person context ("a friendly guide …").
    - scope:    multi-line bullets of what it can actually help with (from the agents).
    - domains:  short comma list of capability labels, used in scope/boundary sentences.
    - catalog:  the router/planner agent catalog (routing source of truth).
    - single_domain / agent_count: shape hints for wording.
    """

    name: str
    tagline: str
    scope: str
    domains: str
    catalog: str
    single_domain: bool
    agent_count: int

    def assistant_descriptor(self) -> str:
        """One-liner describing the assistant — used by the security classifier so it
        knows what a *legitimate* in-domain request looks like for this deployment."""
        return f"{self.name} — {self.tagline}. It can help with: {self.domains}."


def build_system_profile(registry, settings) -> SystemProfile:
    """Derive the assistant identity from the agents that are actually built.

    Falls back gracefully when no agents are registered (e.g. misconfiguration) so
    the assistant still answers rather than crashing — it simply has an empty scope.
    """
    specs = [registry.get_spec(a) for a in registry.agent_ids]
    specs = [s for s in specs if s]

    # User-facing capability AREA per specialist (NOT the internal agent name). The
    # master advertises what it can do, not who does it — specialist names are internal.
    def _domain(s) -> str:
        return s.domain_label or _strip_agent_suffix(s.display_name)

    labels = [_domain(s) for s in specs]

    scope_lines = []
    for s in specs:
        desc = " ".join((s.description or "").split())  # collapse whitespace/newlines
        scope_lines.append(f"- {_domain(s)}: {desc}" if desc else f"- {_domain(s)}")
    scope = "\n".join(scope_lines) if scope_lines else (
        "- (No capabilities are currently enabled.)"
    )

    domains = ", ".join(labels) if labels else "your configured capabilities"
    single = len(specs) == 1

    # --- Master name --- NEVER derived from a specialist's name (they are internal).
    if settings.assistant_name:
        name = settings.assistant_name
    else:
        name = "Assistant"

    # --- Tagline ---
    if settings.assistant_tagline:
        tagline = settings.assistant_tagline
    elif labels:
        tagline = f"a knowledgeable assistant that helps with {domains}"
    else:
        tagline = "a knowledgeable assistant"

    return SystemProfile(
        name=name,
        tagline=tagline,
        scope=scope,
        domains=domains,
        catalog=registry.build_agent_catalog(),
        single_domain=single,
        agent_count=len(specs),
    )
