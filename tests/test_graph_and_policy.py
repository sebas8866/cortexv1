from __future__ import annotations

import asyncio

import pytest

from cortex.agents import ProposedAction, Risk
from cortex.config import PolicySettings
from cortex.orchestration import END, ActionPolicy, Graph, GraphError
from cortex.orchestration.graph import State


async def _inc(state: State) -> State:
    return {"n": state.get("n", 0) + 1}


def test_router_controls_the_path() -> None:
    graph = (
        Graph("loop")
        .add_node("inc", _inc)
        .set_entry("inc")
        .add_router("inc", lambda s: END if s["n"] >= 3 else "inc", targets=("inc", END))
        .compile()
    )
    trace = asyncio.run(graph.run())
    assert trace.state["n"] == 3
    assert trace.path == ["inc", "inc", "inc"]


def test_step_limit_stops_runaway_cycles() -> None:
    graph = Graph("forever").add_node("inc", _inc).set_entry("inc").add_edge("inc", "inc").compile()
    with pytest.raises(GraphError) as err:
        asyncio.run(graph.run(max_steps=5))
    assert len(err.value.trace.steps) == 5


def test_node_failure_is_recorded_in_trace() -> None:
    async def boom(state: State) -> State:
        raise RuntimeError("upstream API down")

    graph = (
        Graph("g")
        .add_node("ok", _inc)
        .add_node("boom", boom)
        .set_entry("ok")
        .add_edge("ok", "boom")
        .add_edge("boom", END)
        .compile()
    )
    with pytest.raises(GraphError) as err:
        asyncio.run(graph.run())
    assert err.value.trace.path == ["ok", "boom"]
    assert "upstream API down" in (err.value.trace.steps[-1].error or "")


def test_compile_rejects_dangling_nodes() -> None:
    with pytest.raises(ValueError, match="no outgoing edge"):
        Graph("g").add_node("a", _inc).set_entry("a").compile()


def _action(kind: str, risk: Risk, **params: float) -> ProposedAction:
    return ProposedAction(agent="t", kind=kind, target="x", risk=risk, rationale="", params=params)


def test_policy_verdicts() -> None:
    policy = ActionPolicy(
        PolicySettings(
            max_auto_risk="low",
            max_budget_change_pct=25,
            daily_auto_action_limit=1,
            denied_kinds=["delete_campaign"],
        )
    )
    decisions = policy.evaluate(
        [
            _action("delete_campaign", Risk.LOW),
            _action("pause_campaign", Risk.MEDIUM),
            _action("increase_budget", Risk.LOW, change_pct=40),
            _action("increase_budget", Risk.LOW, change_pct=20),
            _action("decrease_budget", Risk.LOW, change_pct=-20),
        ]
    )
    assert [d.verdict for d in decisions] == ["deny", "review", "review", "auto", "review"]
    assert "limit" in decisions[-1].reason
