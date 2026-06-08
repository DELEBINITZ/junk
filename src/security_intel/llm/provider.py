from enum import Enum

from langchain_openai import ChatOpenAI

from security_intel.config import Settings


class Lane(Enum):
    FAST = "fast"
    STANDARD = "standard"
    DEEP = "deep"


def get_llm(settings: Settings) -> ChatOpenAI:
    """Standard lane LLM (72B) for answer synthesis and reasoning."""
    return ChatOpenAI(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        streaming=True,
    )


def get_fast_llm(settings: Settings) -> ChatOpenAI:
    """Fast lane LLM (7B) for planning, routing, summarization."""
    return ChatOpenAI(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_fast_model,
        temperature=0.0,
        max_tokens=1024,
        streaming=True,
    )


def get_deep_llm(settings: Settings) -> ChatOpenAI:
    """Deep lane LLM (strongest model) for complex multi-hop analysis."""
    return ChatOpenAI(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_deep_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens * 2,
        streaming=True,
    )


class LaneRouter:
    """Routes tasks to the appropriate model tier based on complexity.

    Usage:
        router = LaneRouter(settings)
        llm = router.get(Lane.FAST)  # for summarization
        llm = router.get(Lane.DEEP)  # for complex analysis
        llm = router.for_task("summarize")  # auto-route by task name
    """

    TASK_LANES = {
        "route": Lane.FAST,
        "rewrite": Lane.FAST,
        "summarize": Lane.FAST,
        "plan": Lane.FAST,
        "answer": Lane.STANDARD,
        "synthesize": Lane.STANDARD,
        "deep": Lane.DEEP,
        "analyze": Lane.DEEP,
        "multi_hop": Lane.DEEP,
    }

    def __init__(self, settings: Settings):
        self._models: dict[Lane, ChatOpenAI] = {
            Lane.FAST: get_fast_llm(settings),
            Lane.STANDARD: get_llm(settings),
            Lane.DEEP: get_deep_llm(settings),
        }

    def get(self, lane: Lane) -> ChatOpenAI:
        return self._models[lane]

    def for_task(self, task: str) -> ChatOpenAI:
        lane = self.TASK_LANES.get(task, Lane.STANDARD)
        return self._models[lane]

    @property
    def fast(self) -> ChatOpenAI:
        return self._models[Lane.FAST]

    @property
    def standard(self) -> ChatOpenAI:
        return self._models[Lane.STANDARD]

    @property
    def deep(self) -> ChatOpenAI:
        return self._models[Lane.DEEP]
