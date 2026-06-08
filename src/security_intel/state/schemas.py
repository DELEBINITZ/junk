from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class PlanStep(TypedDict):
    agent: str
    task: str
    depends_on: list[int]


class ExecutionPlan(TypedDict):
    steps: list[PlanStep]
    synthesis_goal: str


class AgentResult(TypedDict):
    agent_id: str
    findings: str
    citations: list[dict]
    tool_calls: list[dict]


class OrchestratorState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    user_query: str
    org_id: str
    user_id: str
    roles: list[str]
    session_id: str
    is_complex: bool
    plan: ExecutionPlan | None
    agent_results: list[AgentResult]
    final_answer: str
    citations: list[dict]
    blocked: bool
    block_reason: str


class SubAgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    task: str
    org_id: str
    user_id: str
    roles: list[str]
    findings: str
    citations: list[dict]
