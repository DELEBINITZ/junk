"""A minimal LangGraph-shaped execution engine.

Nodes are `state -> state-updates` functions; edges are a next-node name or a
`state -> name` conditional. This mirrors LangGraph's node/edge/State model
(ChatState in state.py matches its TypedDict), so swapping to
`langgraph.StateGraph` later is mechanical — without taking the dependency now or
shipping a graph we can't run/verify locally. See plan §6.

`run()` executes to END; `stream()` yields (node_name, updates) per step for
observability/streaming.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

State = dict[str, Any]
NodeFn = Callable[[State], dict]
Edge = Any  # str (next node) or Callable[[State], str]

END = "__end__"


@dataclass
class Graph:
    entry: str
    nodes: dict[str, NodeFn] = field(default_factory=dict)
    edges: dict[str, Edge] = field(default_factory=dict)
    max_steps: int = 100

    def _next(self, current: str, state: State) -> str:
        edge = self.edges.get(current, END)
        return edge(state) if callable(edge) else edge

    def run(self, state: State) -> State:
        current = self.entry
        steps = 0
        while current != END:
            steps += 1
            if steps > self.max_steps:
                raise RuntimeError(f"graph exceeded {self.max_steps} steps (cycle?)")
            updates = self.nodes[current](state) or {}
            state.update(updates)
            current = self._next(current, state)
        return state

    def stream(self, state: State) -> Iterator[tuple[str, dict]]:
        current = self.entry
        steps = 0
        while current != END:
            steps += 1
            if steps > self.max_steps:
                raise RuntimeError(f"graph exceeded {self.max_steps} steps (cycle?)")
            updates = self.nodes[current](state) or {}
            state.update(updates)
            yield current, updates
            current = self._next(current, state)
