"""A tiny LangGraph-shaped state-machine engine (zero dependencies).

Same mental model as LangGraph — a ``StateGraph`` of named async nodes, plain
and conditional edges, ``START``/``END`` sentinels, nodes return partial state
updates that are merged. This is the default engine so the system runs with no
extra deps; ``app/core/agent/langgraph_engine.py`` mirrors the identical node
set onto real LangGraph (with checkpointing) when ``agent_engine=langgraph``.
Token streaming happens inside nodes via ``AgentContext.emit``, not here.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable

START = "__start__"
END = "__end__"

NodeFn = Callable[[dict], Awaitable[dict]]
RouterFn = Callable[[dict], str]


class GraphError(RuntimeError):
    pass


class StateGraph:
    def __init__(self) -> None:
        self._nodes: dict[str, NodeFn] = {}
        self._edges: dict[str, str] = {}
        self._cond: dict[str, tuple[RouterFn, dict[str, str]]] = {}
        self._entry: str | None = None

    def add_node(self, name: str, fn: NodeFn) -> StateGraph:
        if name in (START, END):
            raise GraphError(f"reserved node name: {name}")
        self._nodes[name] = fn
        return self

    def add_edge(self, src: str, dst: str) -> StateGraph:
        self._edges[src] = dst
        return self

    def add_conditional_edges(self, src: str, router: RouterFn, mapping: dict[str, str]) -> StateGraph:
        self._cond[src] = (router, mapping)
        return self

    def set_entry(self, name: str) -> StateGraph:
        self._entry = name
        return self

    def compile(self, *, max_steps: int = 64) -> CompiledGraph:
        if not self._entry:
            raise GraphError("no entry node set")
        return CompiledGraph(self._nodes, self._edges, self._cond, self._entry, max_steps)


class CompiledGraph:
    def __init__(self, nodes, edges, cond, entry, max_steps) -> None:
        self._nodes = nodes
        self._edges = edges
        self._cond = cond
        self._entry = entry
        self._max_steps = max_steps

    def _next(self, node: str, state: dict) -> str:
        if node in self._cond:
            router, mapping = self._cond[node]
            key = router(state)
            if key not in mapping:
                raise GraphError(f"conditional from '{node}' returned unmapped key '{key}'")
            return mapping[key]
        return self._edges.get(node, END)

    async def run(self, state: dict) -> dict:
        node = self._entry
        steps = 0
        while node != END:
            if steps >= self._max_steps:
                raise GraphError(f"max steps ({self._max_steps}) exceeded — cycle?")
            steps += 1
            fn = self._nodes.get(node)
            if fn is None:
                raise GraphError(f"unknown node '{node}'")
            updates = await fn(state) or {}
            state.update(updates)
            node = self._next(node, state)
        return state

    async def astream(self, state: dict) -> AsyncIterator[tuple[str, dict]]:
        node = self._entry
        steps = 0
        while node != END:
            if steps >= self._max_steps:
                raise GraphError(f"max steps ({self._max_steps}) exceeded — cycle?")
            steps += 1
            fn = self._nodes[node]
            updates = await fn(state) or {}
            state.update(updates)
            yield node, state
            node = self._next(node, state)


__all__ = ["StateGraph", "CompiledGraph", "GraphError", "START", "END", "NodeFn"]
