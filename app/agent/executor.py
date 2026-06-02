"""Execution engine for agent plans.

The executor is deliberately boring: validate the tool allowlist, honor
dependencies, call MCP, retry transient failures, and return a trace. That trace
is part of the demo because it proves the agent followed an auditable plan.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from app.agent.planner import AgentPlan, PlanStep
from app.db.repository import DataStore
from app.domain import User
from app.mcp_server.client import InProcessMCPClient, MCPClientError
from app.mcp_server.tools import TOOL_ORDER
from app.observability.logging import safe_extra


logger = logging.getLogger(__name__)


class AgentExecutionError(Exception):
    """Raised when the plan graph is invalid before tool execution."""

    pass


def execute_plan(
    plan: AgentPlan,
    user: User,
    store: DataStore,
    max_retries: int = 3,
    timeout_seconds: int = 60,
    allowed_tools: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Execute a plan through MCP and return step-by-step tool records."""

    results: list[dict[str, Any]] = []
    completed: dict[str, dict[str, Any]] = {}
    allowed_tool_set = set(allowed_tools or TOOL_ORDER)
    mcp_client = InProcessMCPClient(user, store)
    deadline = time.monotonic() + timeout_seconds
    logger.info(
        "agent.execution.start",
        extra=safe_extra(
            intent=plan.intent,
            user_id=user.id,
            organization_id=user.organization_id,
            step_count=len(plan.steps),
            timeout_seconds=timeout_seconds,
        ),
    )

    for step in plan.steps:
        _ensure_dependencies(step, completed)
        record = _execute_step(step, mcp_client, allowed_tool_set, max_retries, deadline)
        results.append(record)
        completed[step.step_id] = record

        if (
            record["status"] == "success"
            and plan.intent == "renewal_action_plan"
            and step.tool == "find_expiring_contracts"
        ):
            # The renewal workflow is data-dependent: first find the contracts,
            # then add clause and risk steps for each returned contract.
            for contract in record["result"].get("contracts", []):
                contract_id = contract["contract_id"]
                clause_step = PlanStep(
                    step_id=f"termination_{contract_id}",
                    tool="extract_clause",
                    args={"contract_id": contract_id, "clause_type": "termination"},
                    depends_on=[step.step_id],
                )
                risk_step = PlanStep(
                    step_id=f"risk_{contract_id}",
                    tool="calculate_risk_score",
                    args={"contract_id": contract_id},
                    depends_on=[step.step_id],
                )
                for dynamic_step in [clause_step, risk_step]:
                    dynamic_record = _execute_step(
                        dynamic_step,
                        mcp_client,
                        allowed_tool_set,
                        max_retries,
                        deadline,
                    )
                    results.append(dynamic_record)
                    completed[dynamic_step.step_id] = dynamic_record

    return results


def _ensure_dependencies(step: PlanStep, completed: dict[str, dict[str, Any]]) -> None:
    """Ensure a step is not executed before its declared dependencies."""

    missing = [dependency for dependency in step.depends_on if dependency not in completed]
    if missing:
        raise AgentExecutionError(f"Missing dependencies for {step.step_id}: {', '.join(missing)}")


def _execute_step(
    step: PlanStep,
    mcp_client: InProcessMCPClient,
    allowed_tools: set[str],
    max_retries: int,
    deadline: float,
) -> dict[str, Any]:
    """Execute one MCP step with retry and timeout accounting."""

    if step.tool not in allowed_tools:
        logger.warning(
            "agent.step.rejected_disallowed_tool",
            extra=safe_extra(step_id=step.step_id, tool=step.tool),
        )
        return {
            "step_id": step.step_id,
            "tool": step.tool,
            "args": step.args,
            "status": "failed",
            "attempts": 0,
            "error": f"Tool is not in the allowed tool list: {step.tool}",
            "result": {},
        }

    last_error = ""
    for attempt in range(1, max_retries + 1):
        if time.monotonic() > deadline:
            last_error = "Agent execution timed out"
            logger.error(
                "agent.step.timeout",
                extra=safe_extra(step_id=step.step_id, tool=step.tool, attempt=attempt),
            )
            break
        try:
            step_started = time.monotonic()
            logger.info(
                "agent.step.start",
                extra=safe_extra(step_id=step.step_id, tool=step.tool, attempt=attempt),
            )
            result = mcp_client.call_tool(step.tool, step.args)
            logger.info(
                "agent.step.success",
                extra=safe_extra(
                    step_id=step.step_id,
                    tool=step.tool,
                    attempt=attempt,
                    duration_ms=round((time.monotonic() - step_started) * 1000, 2),
                ),
            )
            return {
                "step_id": step.step_id,
                "tool": step.tool,
                "args": step.args,
                "status": "success",
                "attempts": attempt,
                "result": result,
            }
        except MCPClientError as exc:
            last_error = str(exc)
            log_method = logger.warning if _is_non_retryable(last_error) else logger.info
            log_method(
                "agent.step.retry_or_fail",
                extra=safe_extra(
                    step_id=step.step_id,
                    tool=step.tool,
                    attempt=attempt,
                    non_retryable=_is_non_retryable(last_error),
                    error=last_error,
                ),
            )
            if _is_non_retryable(last_error):
                break

    logger.error(
        "agent.step.failed",
        extra=safe_extra(step_id=step.step_id, tool=step.tool, attempts=attempt if "attempt" in locals() else 0, error=last_error),
    )
    return {
        "step_id": step.step_id,
        "tool": step.tool,
        "args": step.args,
        "status": "failed",
        "attempts": attempt if "attempt" in locals() else 0,
        "error": last_error,
        "result": {},
    }


def _is_non_retryable(error: str) -> bool:
    """Classify errors where retrying would only repeat a policy/input failure."""

    lowered = error.lower()
    return any(
        phrase in lowered
        for phrase in [
            "access denied",
            "not found",
            "invalid date",
            "unsupported clause_type",
            "organization_id does not match",
            "requires name and arguments",
        ]
    )
