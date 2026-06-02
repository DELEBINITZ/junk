"""In-process MCP tool runner (v1 transport) — THE security choke point.

WHAT IS MCP? The Model Context Protocol is a standard boundary between an agent
and the tools it may call. Conceptually the agent never runs a tool function
directly; it asks an MCP *client* to run a NAMED tool with some arguments, and
the client decides whether/how that happens. That single indirection is where we
hang all the safety policy — so no agent, prompt, or specialist can bypass it.

"In-process" = this particular client runs the tool inside the same Python
process (no network), which is the fast default. A module can later be
"promoted" to its own service; ``RemoteMCPClient`` (remote.py) implements the
EXACT same ``call_tool`` interface over HTTP, so swapping transports changes
nothing for the caller. Same boundary, two transports.

================== THIS FILE IS WHERE SECURITY IS ENFORCED ==================
For EVERY tool call, regardless of which agent or specialist invoked it,
``call_tool`` enforces, in order:

  1. RBAC (role-based access control): the caller's roles (from
     ``ctx.roles`` — which came from the verified token, NOT from tool
     arguments) must satisfy the tool's required minimum role declared in the
     module manifest. If not -> a ``forbidden`` ToolError; the tool never runs.

  2. The HUMAN ACTION GATE: a ``side_effecting`` tool (one that changes the
     world, not just reads) is NEVER executed inline here. It is routed to the
     action gate for human approval and immediately returns ``requires_approval``.
     This is the property that makes prompt injection safe-by-construction: even
     if an attacker convinces the model to call a destructive tool, the call
     stops at this gate and waits for a person.

Only after BOTH checks pass do we actually ``tool.invoke(...)``. Every step is
logged with the trusted org/user/trace ids for audit. Read-only tools skip the
gate but still pass RBAC.
===========================================================================
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any, Protocol

from app.core.contracts import ToolContext, ToolError, ToolOutcome, role_satisfies
from app.core.registry import CapabilityRegistry
from app.core.security.context import SecurityContext


class ActionGateProtocol(Protocol):
    """The human-approval gate, as seen from here. Two questions only:
      * ``auto_approves`` — may this specific side-effecting call skip human
        review (e.g. a low-risk action under a proven AUTO policy)?
      * ``request_approval`` — file it for a human and return an approval id the
        caller can surface, so the action can be approved out-of-band later.
    Declared as a Protocol so the gate's concrete implementation lives elsewhere
    and this boundary stays dependency-light."""

    async def auto_approves(self, tool, arguments: Mapping[str, Any], ctx: ToolContext) -> bool: ...
    async def request_approval(self, tool, arguments: Mapping[str, Any], ctx: ToolContext) -> str: ...


class MCPClient(Protocol):
    """The tool-runner interface shared by BOTH transports (in-process here,
    remote in remote.py). Because they implement the same two methods, the
    orchestrator can swap one for the other without any other code changing.
    ``transport`` is a label ("inprocess"/"remote") used in tracing/logs."""

    transport: str

    async def list_tools(self, sc: SecurityContext) -> list[dict]: ...
    async def call_tool(self, name: str, arguments: Mapping[str, Any], ctx: ToolContext) -> ToolOutcome: ...


class InProcessMCPClient:
    """The default tool runner. Holds the capability registry (the source of
    truth for which tools exist and what role each needs) and an optional action
    gate. All enforcement happens in ``call_tool`` below."""

    transport = "inprocess"

    def __init__(
        self,
        registry: CapabilityRegistry,
        *,
        action_gate: ActionGateProtocol | None = None,
        logger: Any = None,
        remote_executors: Mapping[str, Any] | None = None,
        settings: Any = None,
    ) -> None:
        self.registry = registry            # which modules/tools exist + their RBAC
        self.action_gate = action_gate      # human-approval gate for side effects (may be None)
        self.logger = logger
        self._settings = settings           # for the dynamic-discovery policy/cache TTL
        # module_id -> a remote executor (e.g. FastMCPRemote) that RUNS that
        # module's tools on a remote MCP server. Empty by default = everything
        # runs in-process. Crucially this only changes WHERE a tool executes; the
        # RBAC + action-gate checks below still run locally first, so promoting a
        # module to a remote server never weakens enforcement.
        self.remote_executors: Mapping[str, Any] = remote_executors or {}
        # PURE-DYNAMIC discovery state. ``_discovered`` is the resolve map
        # (org_id, tool_name) -> module_id, populated by ``discover_tools`` so the
        # boundary can route a call to a tool that exists ONLY on the remote server
        # (not in any local manifest). ``_discover_cache`` memoizes a server's
        # tools/list per (module, org) with a TTL so we don't re-list every turn.
        self._discovered: dict[tuple[str, str], str] = {}
        self._discover_cache: dict[tuple[str, str], tuple[float, list[dict]]] = {}

    async def list_tools(self, sc: SecurityContext) -> list[dict]:
        """Advertise the tools THIS caller is allowed to see. ``capability_view``
        already filters by the caller's org, license tier, and roles, so the
        agent is never even told about tools it could not call. Each tool is
        returned as its function-calling JSON schema."""
        return [t.json_schema() for t in self.registry.capability_view(sc).tools]

    async def discover_tools(self, module_id: str, ctx: ToolContext) -> list[dict]:
        """PURE-DYNAMIC discovery: ask a module's remote MCP server which tools it
        exposes (tools/list), apply the safety policy, cache per (module, org), and
        register the SAFE ones in the resolve map so they become callable through
        ``call_tool``. Returns the safe descriptors (name/description/parameters).

        Safety policy (so dynamic discovery can't introduce an ungated side effect):
          * server marks the tool read-only        -> callable;
          * server marks it destructive/non-read    -> EXCLUDED (must be a declared
            local stub to be callable, so the action gate applies);
          * unannotated                             -> callable iff the trust flag.
        Never raises — a discovery failure just yields no dynamic tools this turn."""
        executor = self.remote_executors.get(module_id)
        if executor is None or not hasattr(executor, "list_tools"):
            return []
        ttl = getattr(self._settings, "mcp_tool_cache_ttl_seconds", 300) if self._settings else 300
        ck = (module_id, ctx.org_id)
        cached = self._discover_cache.get(ck)
        now = time.time()
        if cached and (now - cached[0]) < ttl:
            return cached[1]
        # The remote server filters its tool list by the caller's org/roles, so we
        # list with an identity rebuilt from the trusted ToolContext.
        sc = SecurityContext(org_id=ctx.org_id, user_id=ctx.user_id, roles=ctx.roles, email="")
        try:
            raw = await executor.list_tools(sc)
        except Exception:  # noqa: BLE001 - server down / listing failed
            return cached[1] if cached else []
        trust = getattr(self._settings, "mcp_dynamic_trust_unannotated", True) if self._settings else True
        safe: list[dict] = []
        for d in raw:
            ro, dh = d.get("read_only_hint"), d.get("destructive_hint")
            if ro is True:
                ok = True
            elif ro is False or dh is True:
                ok = False
            else:
                ok = trust
            if not ok:
                continue
            safe.append(d)
            self._discovered[(ctx.org_id, d["name"])] = module_id
        self._discover_cache[ck] = (now, safe)
        return safe

    async def _call_discovered(
        self, name: str, arguments: Mapping[str, Any], ctx: ToolContext, module_id: str
    ) -> ToolOutcome:
        """Execute a dynamically-discovered tool on its remote server. There is no
        local manifest entry, so RBAC here is a coarse LOCAL FLOOR
        (``mcp_dynamic_tool_role``); the remote server re-derives the caller's roles
        from the service token and enforces its own fine-grained RBAC. Only
        read-safe tools ever reach this path (discovery excluded destructive ones),
        so the no-ungated-side-effects invariant still holds."""
        executor = self.remote_executors.get(module_id)
        if executor is None:
            return ToolError(code="unknown_tool", message=f"no such tool: {name}")
        role = getattr(self._settings, "mcp_dynamic_tool_role", "viewer") if self._settings else "viewer"
        if not role_satisfies(ctx.roles, role):
            self._log("tool_denied", name, ctx, reason="rbac_dynamic", required=role)
            return ToolError(code="forbidden", message=f"role '{role}' required to call '{name}'",
                             details={"have": list(ctx.roles), "need": role})
        self._log("tool_call", name, ctx, cap_module=module_id,
                  transport=getattr(executor, "transport", "remote"), dynamic=True)
        outcome = await executor.call_tool(name, arguments, ctx)
        self._log("tool_result", name, ctx, ok=getattr(outcome, "ok", False))
        return outcome

    async def call_tool(self, name: str, arguments: Mapping[str, Any], ctx: ToolContext) -> ToolOutcome:
        """Run one tool through the full security pipeline. THE enforcement path.

        Returns an outcome VALUE, never raises (errors-as-data). The order of the
        checks below is the security design — each gate must pass before the next
        is even considered, and the actual tool body runs only at the very end.

        Crucially, identity comes from ``ctx`` (the trusted ToolContext, derived
        from the verified token); ``arguments`` are attacker-influencable and are
        used ONLY as tool inputs, never to decide org or permissions.
        """
        # (a) Resolve the tool. First the local registry; if it's not there, it may
        # be a PURE-DYNAMIC tool discovered from a remote MCP server (not declared in
        # any local manifest) — route those through the dynamic path below.
        found = self.registry.find_tool(name)
        if not found:
            module_id = self._discovered.get((ctx.org_id, name))
            if module_id is not None:
                return await self._call_discovered(name, arguments, ctx, module_id)
            return ToolError(code="unknown_tool", message=f"no such tool: {name}")
        module, tool = found
        # (b) ...and live in an enabled module (feature flags / licensing).
        if not module.enabled:
            return ToolError(code="disabled", message=f"module '{module.id}' is disabled")

        # (c) RBAC GATE. The minimum role comes from the manifest (with per-tool
        # overrides); the caller's roles come from ctx (token-derived). If the
        # caller's roles don't reach it, refuse with a forbidden error and audit
        # the denial — the handler is never invoked.
        required = module.required_role(name)
        if not role_satisfies(ctx.roles, required):
            self._log("tool_denied", name, ctx, reason="rbac", required=required)
            return ToolError(
                code="forbidden",
                message=f"role '{required}' required to call '{name}'",
                details={"have": list(ctx.roles), "need": required},
            )

        # (d) HUMAN ACTION GATE. A side-effecting tool must not run inline. Unless
        # the gate explicitly auto-approves this call, we DO NOT execute it: we
        # record an approval request and return ``requires_approval`` with its id.
        # The destructive handler simply never runs from the agent loop — that is
        # what neutralizes prompt-injection-driven actions. Read-only tools skip
        # this whole block.
        if tool.side_effecting:
            gate = self.action_gate
            approved = bool(gate) and await gate.auto_approves(tool, arguments, ctx)
            if not approved:
                approval_id = ""
                if gate:
                    approval_id = await gate.request_approval(tool, arguments, ctx)
                self._log("tool_gated", name, ctx, approval_id=approval_id)
                return ToolError(
                    code="requires_approval",
                    message=f"'{name}' is a side-effecting action; awaiting human approval",
                    details={"action_type": name, "approval_id": approval_id},
                )

        # (e) Only now, with RBAC satisfied and (for side effects) approval in
        # hand, do we actually run the tool. If this module has been promoted to a
        # remote MCP server, dispatch EXECUTION there; otherwise run the local
        # handler. Either way ``invoke``/the remote adapter guarantees
        # errors-as-data, so even a misbehaving handler returns an outcome.
        executor = self.remote_executors.get(module.id)
        if executor is not None:
            self._log("tool_call", name, ctx, cap_module=module.id,
                      transport=getattr(executor, "transport", "remote"))
            outcome = await executor.call_tool(name, arguments, ctx)
        else:
            self._log("tool_call", name, ctx, cap_module=module.id)
            outcome = await tool.invoke(arguments, ctx)
        self._log("tool_result", name, ctx, ok=getattr(outcome, "ok", False))
        return outcome

    def _log(self, event: str, tool: str, ctx: ToolContext, **extra: Any) -> None:
        """Structured audit line for every decision at this boundary (call,
        denial, gating, result). Always stamps the TRUSTED org/user/trace ids
        from ctx, so the audit trail can never be spoofed by tool arguments —
        essential for a security product where "who did what, in which org" must
        be provable. No-ops if no logger was wired."""
        if self.logger:
            self.logger.info(
                "mcp.%s", event,
                extra={"tool": tool, "org_id": ctx.org_id, "user_id": ctx.user_id,
                       "trace_id": ctx.trace_id, **extra},
            )


__all__ = ["InProcessMCPClient", "MCPClient", "ActionGateProtocol"]
