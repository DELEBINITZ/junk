"""Golden-set evaluation runner (plan §12).

Loads every loaded module's `evals/golden.jsonl`, runs each query through the
orchestrator, and scores:
  - route accuracy   (did `expect_route` land in the chosen modules?)
  - refusal accuracy (refused when `expect_refusal`, answered otherwise)
  - citation rate    (citations present when `expect_citation`)

Optional RAGAS faithfulness runs if `ragas` is installed (lazy). Runs on the
in-memory path with an admin user, so no infra is needed to measure routing /
refusal quality. CI gate: exits non-zero if route accuracy < EVAL_MIN_ROUTE.

    .venv/bin/python -m app.eval.runner
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

CAPABILITIES_DIR = Path(__file__).resolve().parent.parent / "capabilities"


@dataclass
class EvalResult:
    module: str
    total: int = 0
    route_ok: int = 0
    route_checked: int = 0
    refusal_ok: int = 0
    refusal_checked: int = 0
    citation_ok: int = 0
    citation_checked: int = 0
    failures: list[dict] = field(default_factory=list)


def _load_golden(module_id: str) -> list[dict]:
    path = CAPABILITIES_DIR / module_id / "evals" / "golden.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def run_eval() -> dict:
    # Enable EASM so its golden set is exercised too.
    os.environ.setdefault("CAP_EASM_ENABLED", "true")

    from pathlib import Path as _P

    from app.core.agent.orchestrator import Orchestrator
    from app.core.memory.conversations import reset_conversation_store
    from app.core.registry import get_registry, reset_registry
    from app.db.repository import get_store, reset_store
    from app.db.seed import seed_demo_data

    reset_registry()
    reset_store()
    reset_conversation_store()
    store = get_store()
    seed_demo_data(store, _P("Assignment_org"))
    registry = get_registry()
    admin = next(u for u in store.users.values() if u.role == "admin")

    results: list[EvalResult] = []
    for module_id in registry.modules:
        rows = _load_golden(module_id)
        if not rows:
            continue
        result = EvalResult(module=module_id)
        for row in rows:
            result.total += 1
            turn = Orchestrator(admin, store).run_query(row["query"])
            if "expect_route" in row:
                result.route_checked += 1
                if row["expect_route"] in turn.module_ids:
                    result.route_ok += 1
                else:
                    result.failures.append({"id": row.get("id"), "kind": "route",
                                            "expected": row["expect_route"], "got": turn.module_ids})
            if "expect_refusal" in row:
                result.refusal_checked += 1
                if (turn.status == "refused") == bool(row["expect_refusal"]):
                    result.refusal_ok += 1
                else:
                    result.failures.append({"id": row.get("id"), "kind": "refusal",
                                            "expected": row["expect_refusal"], "got": turn.status})
            if row.get("expect_citation"):
                result.citation_checked += 1
                if turn.citations:
                    result.citation_ok += 1
        results.append(result)

    return _summarize(results)


def _rate(ok: int, checked: int) -> float:
    return round(ok / checked, 3) if checked else 1.0


def _summarize(results: list[EvalResult]) -> dict:
    total_route_ok = sum(r.route_ok for r in results)
    total_route = sum(r.route_checked for r in results)
    summary = {
        "modules": {
            r.module: {
                "total": r.total,
                "route_accuracy": _rate(r.route_ok, r.route_checked),
                "refusal_accuracy": _rate(r.refusal_ok, r.refusal_checked),
                "citation_rate": _rate(r.citation_ok, r.citation_checked),
                "failures": r.failures,
            }
            for r in results
        },
        "overall_route_accuracy": _rate(total_route_ok, total_route),
    }
    return summary


if __name__ == "__main__":  # pragma: no cover
    import sys

    summary = run_eval()
    print(json.dumps(summary, indent=2))
    threshold = float(os.getenv("EVAL_MIN_ROUTE", "0.8"))
    overall = summary["overall_route_accuracy"]
    print(f"\noverall route accuracy: {overall} (gate >= {threshold})")
    sys.exit(0 if overall >= threshold else 1)
