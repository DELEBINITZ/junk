"""A tiny LangGraph-shaped state-machine engine (zero dependencies).

================================ START HERE ================================
If you are new to "agentic systems" and LangGraph, read THIS FILE FIRST. It is
~100 lines and it teaches the one idea everything else is built on: an agent is
a **state machine** — a graph of steps ("nodes") connected by "edges", walked
one step at a time, carrying a shared bag of data ("state") from start to end.

The mental model (identical to real LangGraph):

    state = {question, history, ...}        a plain dict that flows through
       |
   [input_guardrail]  --- node: an async function (state) -> partial update
       |
    (edge)            --- edge: "after this node, go to that node"
       |
   [route] -> [gather_context] -> [answer] -> [output_guardrail]
       |
     END             --- the run stops; final state is returned

Four primitives, that's the whole vocabulary:
  * NODE  — a named async function that reads the state and returns a dict of
            CHANGES (a *partial* update, not the whole state).
  * EDGE  — a fixed "from A go to B".
  * CONDITIONAL EDGE — "from A, run a little router(state) function; its return
            value is looked up in a mapping to decide where to go". This is how
            an agent branches (e.g. if the guardrail blocked -> jump to END).
  * START / END — sentinel names marking where the walk begins and stops.

Why build our own instead of always using LangGraph? So the system runs with
ZERO extra dependencies by default. ``app/core/agent/engines.py`` mirrors this
EXACT same node set onto the real ``langgraph`` library (which adds durable
"checkpointing" so a run can pause and resume) when you set
``agent_engine=langgraph``. Same nodes, same edges, two engines — that is why
this file and LangGraph feel interchangeable.

Note: token-by-token streaming to the user does NOT happen here. Nodes emit
events themselves via ``AgentContext.emit`` (see state.py). This engine only
decides *which node runs next*.
===========================================================================
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable

# START/END are just reserved string names. Using sentinels (not real nodes)
# keeps the "where do I begin / when am I done" markers impossible to confuse
# with an actual step. Real LangGraph uses the same two concepts.
START = "__start__"
END = "__end__"

# Type aliases that document the two function shapes the graph works with:
#   NodeFn   — an async step:   async def node(state: dict) -> dict (the changes)
#   RouterFn — a sync chooser:  def router(state: dict) -> str (a mapping key)
NodeFn = Callable[[dict], Awaitable[dict]]
RouterFn = Callable[[dict], str]


class GraphError(RuntimeError):
    """Raised for graph wiring/runtime mistakes (bad node name, missing entry,
    an infinite cycle, a conditional router returning an unmapped key)."""
    pass


class StateGraph:
    """The BUILDER. You describe the graph here (add nodes, add edges, pick the
    entry point), then call :meth:`compile` to freeze it into something runnable.

    This builder/compiled split mirrors LangGraph exactly: you *declare* the
    shape once at startup, then *run* the compiled graph many times per request.
    """

    def __init__(self) -> None:
        # name -> the async function to run for that step.
        self._nodes: dict[str, NodeFn] = {}
        # name -> name. The plain "after A always go to B" edges.
        self._edges: dict[str, str] = {}
        # name -> (router_fn, {router_return_value: destination_node}). The
        # branching edges. Only nodes that need to make a decision live here.
        self._cond: dict[str, tuple[RouterFn, dict[str, str]]] = {}
        # which node the walk starts at.
        self._entry: str | None = None

    def add_node(self, name: str, fn: NodeFn) -> StateGraph:
        # Guard against naming a node "__start__"/"__end__" — those are reserved
        # sentinels, not runnable steps. Returning ``self`` enables chaining:
        # g.add_node(...).add_edge(...).set_entry(...).
        if name in (START, END):
            raise GraphError(f"reserved node name: {name}")
        self._nodes[name] = fn
        return self

    def add_edge(self, src: str, dst: str) -> StateGraph:
        # Unconditional hop: when ``src`` finishes, always go to ``dst``.
        self._edges[src] = dst
        return self

    def add_conditional_edges(self, src: str, router: RouterFn, mapping: dict[str, str]) -> StateGraph:
        # Branch point: after ``src`` runs, call ``router(state)``; whatever
        # string it returns is looked up in ``mapping`` to pick the next node.
        # Example used in nodes.py:
        #   router = lambda s: "blocked" if s.get("blocked") else "ok"
        #   mapping = {"blocked": END, "ok": N_ROUTE}
        # => a blocked guardrail short-circuits straight to END.
        self._cond[src] = (router, mapping)
        return self

    def set_entry(self, name: str) -> StateGraph:
        # The first node the walk executes.
        self._entry = name
        return self

    def compile(self, *, max_steps: int = 64) -> CompiledGraph:
        # Freeze the builder into an immutable, runnable graph. ``max_steps`` is
        # a safety fuse: if the graph ever loops forever (a cycle with no exit),
        # the run aborts instead of hanging. Our graph is linear so 64 is plenty.
        if not self._entry:
            raise GraphError("no entry node set")
        return CompiledGraph(self._nodes, self._edges, self._cond, self._entry, max_steps)


class CompiledGraph:
    """The RUNNABLE graph. Holds the frozen wiring and knows how to walk it.
    Created by :meth:`StateGraph.compile`; you only call :meth:`run` (or
    :meth:`astream`) on it."""

    def __init__(self, nodes, edges, cond, entry, max_steps) -> None:
        self._nodes = nodes
        self._edges = edges
        self._cond = cond
        self._entry = entry
        self._max_steps = max_steps

    def _next(self, node: str, state: dict) -> str:
        """Decide the next node after ``node`` finishes.

        Rule of precedence: a conditional edge wins if one exists for this node
        (we run its router and look up the result); otherwise fall back to the
        plain edge; if there is no edge at all, the walk ends (END).
        """
        if node in self._cond:
            router, mapping = self._cond[node]
            key = router(state)          # ask the router which branch to take
            if key not in mapping:       # router returned something we can't route -> bug
                raise GraphError(f"conditional from '{node}' returned unmapped key '{key}'")
            return mapping[key]
        return self._edges.get(node, END)

    async def run(self, state: dict) -> dict:
        """Walk the graph start -> END and return the FINAL accumulated state.

        This is the heart of the engine. Read the loop slowly — it is the entire
        "agent runtime" in seven lines:
        """
        node = self._entry
        steps = 0
        while node != END:                       # keep walking until we reach END
            if steps >= self._max_steps:         # cycle fuse (see compile())
                raise GraphError(f"max steps ({self._max_steps}) exceeded — cycle?")
            steps += 1
            fn = self._nodes.get(node)
            if fn is None:                        # edge pointed at a node we never added
                raise GraphError(f"unknown node '{node}'")
            updates = await fn(state) or {}       # 1. run the step -> get its CHANGES
            state.update(updates)                 # 2. MERGE changes into shared state
            node = self._next(node, state)        # 3. decide where to go next
        return state                              # END reached -> hand back final state

    async def astream(self, state: dict) -> AsyncIterator[tuple[str, dict]]:
        """Same walk as :meth:`run`, but *yields* ``(node_name, state)`` after
        every step. Useful for debugging/observability — you can watch the state
        grow node by node. (The chat service streams tokens a different way, via
        node-level events; this is the graph-level step stream.)"""
        node = self._entry
        steps = 0
        while node != END:
            if steps >= self._max_steps:
                raise GraphError(f"max steps ({self._max_steps}) exceeded — cycle?")
            steps += 1
            fn = self._nodes[node]
            updates = await fn(state) or {}
            state.update(updates)
            yield node, state                     # emit the step, then continue
            node = self._next(node, state)


__all__ = ["StateGraph", "CompiledGraph", "GraphError", "START", "END", "NodeFn"]
