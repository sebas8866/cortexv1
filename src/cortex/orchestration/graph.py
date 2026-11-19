"""A small, explicit state-graph runtime.

Nodes are async functions that take the shared state and return a partial update.
Edges are fixed or routed by a function of the state. Every step is timed and
recorded so a run can be audited after the fact.

The API intentionally mirrors LangGraph's StateGraph (add_node / add_edge /
conditional routing / compile) so workflows can move to LangGraph when they need
its checkpointing and human-in-the-loop interrupts. See docs/architecture.md.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from cortex.db import utcnow

END = "__end__"

State = dict[str, Any]
NodeFn = Callable[[State], Awaitable[State | None]]
Router = Callable[[State], str]


@dataclass(slots=True, frozen=True)
class StepRecord:
    node: str
    started_at: datetime
    duration_ms: float
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "node": self.node,
            "started_at": self.started_at.isoformat(),
            "duration_ms": round(self.duration_ms, 2),
            "error": self.error,
        }


@dataclass(slots=True)
class RunTrace:
    state: State
    steps: list[StepRecord] = field(default_factory=list)

    @property
    def path(self) -> list[str]:
        return [s.node for s in self.steps]


class GraphError(RuntimeError):
    def __init__(self, message: str, trace: RunTrace) -> None:
        super().__init__(message)
        self.trace = trace


class Graph:
    def __init__(self, name: str) -> None:
        self.name = name
        self._nodes: dict[str, NodeFn] = {}
        self._edges: dict[str, str] = {}
        self._routers: dict[str, tuple[Router, frozenset[str]]] = {}
        self._entry: str | None = None

    def add_node(self, name: str, fn: NodeFn) -> Graph:
        if name == END or name in self._nodes:
            raise ValueError(f"invalid or duplicate node name {name!r}")
        self._nodes[name] = fn
        return self

    def add_edge(self, src: str, dst: str) -> Graph:
        self._check_free(src)
        self._edges[src] = dst
        return self

    def add_router(self, src: str, router: Router, targets: Iterable[str]) -> Graph:
        """Route from ``src`` to whichever of ``targets`` the router returns."""
        self._check_free(src)
        self._routers[src] = (router, frozenset(targets))
        return self

    def set_entry(self, name: str) -> Graph:
        self._entry = name
        return self

    def compile(self) -> CompiledGraph:
        if self._entry is None or self._entry not in self._nodes:
            raise ValueError(f"graph {self.name!r}: entry node is not set or unknown")
        known = set(self._nodes) | {END}
        for node in self._nodes:
            if node not in self._edges and node not in self._routers:
                raise ValueError(f"graph {self.name!r}: node {node!r} has no outgoing edge")
        for src, dst in self._edges.items():
            if src not in self._nodes or dst not in known:
                raise ValueError(f"graph {self.name!r}: bad edge {src!r} -> {dst!r}")
        for src, (_, targets) in self._routers.items():
            if src not in self._nodes or not targets <= known:
                raise ValueError(f"graph {self.name!r}: bad router on {src!r}")
        return CompiledGraph(
            self.name, self._entry, dict(self._nodes), dict(self._edges), dict(self._routers)
        )

    def _check_free(self, src: str) -> None:
        if src in self._edges or src in self._routers:
            raise ValueError(f"node {src!r} already has an outgoing edge")


class CompiledGraph:
    def __init__(
        self,
        name: str,
        entry: str,
        nodes: dict[str, NodeFn],
        edges: dict[str, str],
        routers: dict[str, tuple[Router, frozenset[str]]],
    ) -> None:
        self.name = name
        self._entry = entry
        self._nodes = nodes
        self._edges = edges
        self._routers = routers

    async def run(
        self,
        state: State | None = None,
        *,
        max_steps: int = 50,
        on_step: Callable[[StepRecord], None] | None = None,
    ) -> RunTrace:
        trace = RunTrace(state=dict(state or {}))
        current = self._entry
        while current != END:
            if len(trace.steps) >= max_steps:
                raise GraphError(f"{self.name}: exceeded {max_steps} steps (cycle?)", trace)
            started, t0 = utcnow(), time.perf_counter()
            try:
                update = await self._nodes[current](trace.state)
            except Exception as exc:
                record = StepRecord(current, started, _ms(t0), error=f"{type(exc).__name__}: {exc}")
                trace.steps.append(record)
                raise GraphError(f"{self.name}: node {current!r} failed: {exc}", trace) from exc
            if update:
                trace.state.update(update)
            record = StepRecord(current, started, _ms(t0))
            trace.steps.append(record)
            if on_step:
                on_step(record)
            current = self._next(current, trace)
        return trace

    def _next(self, node: str, trace: RunTrace) -> str:
        if node in self._edges:
            return self._edges[node]
        router, targets = self._routers[node]
        target = router(trace.state)
        if target not in targets:
            raise GraphError(f"{self.name}: router on {node!r} returned unknown {target!r}", trace)
        return target


def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000
