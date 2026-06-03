"""Real-LLM demo of DYNAMIC routing + DEEP REASONING (planner + reflect).

Exercises the authoritative path (router_mode=llm, orchestrator_mode=planner) with
a real OpenAI model, so you can SEE: (1) routing decided by meaning — a question
that shares no keywords with any module still routes correctly; (2) the planner
decomposing a cross-domain question into dependent steps; (3) the reflect gate
looping to fill gaps; (4) a capability answered through the MCP server.

CREDENTIALS — provide either way:
  * export OPENAI_API_KEY=sk-... ; export OPENAI_MODEL=gpt-4o-mini    (preferred)
  * or put OPENAI_API_KEY=... and OPENAI_MODEL=... lines in .env.production
This script reads the environment first, then falls back to parsing .env.production
for ONLY those two keys. It never prints the key.

RUN (optionally start the test-kit MCP server first for the MCP demo):
  cd /tmp/mcp-test-kits/python && uv run mcp-test-kits --transport http --port 3000 &   # optional
  cd <repo> && ./.venv/bin/python scripts/run_real_llm_demo.py
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path


def _load_openai_creds() -> tuple[str, str]:
    """Resolve (api_key, model) from the environment, falling back to .env.production
    for just those two keys. Returns ('', '') if no key is found anywhere."""
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    model = os.environ.get("OPENAI_MODEL", "").strip()
    envf = Path(__file__).resolve().parent.parent / ".env.production"
    if (not key or not model) and envf.exists():
        for line in envf.read_text().splitlines():
            line = line.strip()
            if line.startswith("OPENAI_API_KEY=") and not key:
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
            elif line.startswith("OPENAI_MODEL=") and not model:
                model = line.split("=", 1)[1].strip().strip('"').strip("'")
    return key, model or "gpt-4o-mini"


def _configure_env() -> bool:
    """Point the platform at OpenAI for this process. Keeps embeddings deterministic
    (offline) and does NOT load the production self-hosted MCP URLs. Returns False if
    no API key is available."""
    key, model = _load_openai_creds()
    if not key:
        print("No OPENAI_API_KEY found in the environment or .env.production.")
        print("Add it, e.g.:  export OPENAI_API_KEY=sk-...  export OPENAI_MODEL=gpt-4o-mini")
        return False
    os.environ["LLM_PROVIDER"] = "openai"
    os.environ["OPENAI_API_KEY"] = key
    os.environ["OPENAI_MODEL"] = model
    os.environ["EMBEDDING_PROVIDER"] = "deterministic"   # offline embedder; routing uses the LLM
    os.environ["ROUTER_MODE"] = "llm"                    # dynamic LLM routing
    os.environ["ORCHESTRATOR_MODE"] = "planner"         # deep reasoning: plan -> dispatch -> reflect
    os.environ.setdefault("MAX_REPLANS", "2")           # allow the reflect loop to iterate
    # Route the testkit module's tools to the live MCP server if it's running.
    os.environ.setdefault("TESTKIT_MCP_URL", "http://localhost:3000/mcp")
    print(f"LLM provider=openai model={model} | router=llm | orchestrator=planner\n")
    return True


def sc():
    from app.core.security.context import SecurityContext
    return SecurityContext(org_id="org_acme", user_id="u-alice", roles=("analyst",), email="a@acme.test")


async def main() -> None:
    if not _configure_env():
        return
    from app.config import reload_settings
    from app.core.bootstrap import build_services, seed_demo

    services = build_services(reload_settings())
    await seed_demo(services)
    try:
        # Each turn streams agent events; we capture route/plan/reflect to SHOW the
        # reasoning, not just the final text.
        demos = [
            ("DYNAMIC ROUTING (no keyword overlap with any module)",
             "Could an outsider reach us through anything we've accidentally left "
             "open to the internet?"),
            ("DEEP REASONING (cross-domain, dependent steps + reflect)",
             "What is our single most serious internet-exposed weakness right now, "
             "and is any known threat actor positioned to exploit it?"),
            ("MCP-BACKED CAPABILITY (executes on the mcp-test-kits server)",
             "Generate a UUID for our incident ticket and tell me the current timestamp."),
        ]
        for title, q in demos:
            events: list[dict] = []

            async def emit(ev):
                events.append({"type": ev.type, **(ev.data or {})})

            res = await services.orchestrator.run_turn(sc(), question=q, emit=emit, stream=False)
            route = [e for e in events if e["type"] == "route"]
            plans = [e for e in events if e["type"] == "plan"]
            reflects = [e for e in events if e.get("stage") == "reflecting"]
            print(f"### {title}")
            print(f"Q: {q}")
            print(f"  route_modules : {res.route_modules}")
            if route:
                print(f"  route mode    : {route[0].get('mode')}")
            if plans:
                steps = plans[0].get("steps", [])
                for s in steps:
                    dep = f" (after {s['depends_on']})" if s.get("depends_on") else ""
                    print(f"  plan step     : [{s['domain']}] {s['subq'][:80]}{dep}")
            if reflects:
                print(f"  reflect loops : {len(reflects)} (deep-reasoning gate re-queried for gaps)")
            print(f"  answer        : {res.answer.strip()[:400]}")
            print(f"  citations     : {[(c.get('source'), c.get('title')) for c in res.citations][:6]}\n")
    finally:
        await services.aclose()


if __name__ == "__main__":
    asyncio.run(main())
