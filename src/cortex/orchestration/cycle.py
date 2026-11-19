"""The standard agent cycle: sense/decide -> govern -> act/escalate -> remember -> report.

Every agent runs through the same graph, so every agent gets the same guardrails,
audit trail and memory write-back without re-implementing them.

    decide --> govern --(any auto)--> act --> escalate --> remember --> report --> END
                      \\--(none)----------------^
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from cortex.agents.base import Agent, AgentContext, AgentResult
from cortex.llm import BRIEFING_MARKER
from cortex.orchestration.executor import ExecutionResult
from cortex.orchestration.graph import END, CompiledGraph, Graph, State
from cortex.orchestration.policy import PolicyDecision

if TYPE_CHECKING:
    from cortex.runtime.cortex import Cortex

_REPORT_SYSTEM = (
    "You write short operator briefings for an autonomous operations system. Lead with what "
    "changed and what needs a human decision. No filler, no speculation beyond the data."
)


def build_standard_cycle(cortex: Cortex, agent: Agent, ctx: AgentContext) -> CompiledGraph:
    async def decide(state: State) -> State:
        return {"result": await agent.run(ctx, state.get("inputs", {}))}

    async def govern(state: State) -> State:
        result: AgentResult = state["result"]
        recent = cortex.control.count_auto_executions(ctx.scope, hours=24)
        return {"decisions": cortex.policy.evaluate(result.actions, auto_actions_last_24h=recent)}

    def route_after_govern(state: State) -> str:
        return "act" if any(d.verdict == "auto" for d in state["decisions"]) else "escalate"

    async def act(state: State) -> State:
        executions: list[ExecutionResult] = []
        for d in state["decisions"]:
            if d.verdict == "auto":
                result = await cortex.executor.execute(d.action, ctx.scope)
                cortex.control.record_execution(ctx.run_id, ctx.scope, result, approved_by="policy")
                executions.append(result)
        return {"executions": executions}

    async def escalate(state: State) -> State:
        queued = [
            approval
            for d in state["decisions"]
            if d.verdict == "review"
            and (approval := cortex.control.add_approval(ctx.run_id, ctx.scope, d.action, d.reason))
        ]
        return {"approvals": queued}

    async def remember(state: State) -> State:
        result: AgentResult = state["result"]
        decisions: list[PolicyDecision] = state["decisions"]
        top_risk = max((int(a.risk) for a in result.actions), default=0)
        episode = ctx.memory.record_episode(
            ctx.scope,
            agent.name,
            result.summary,
            importance=min(1.0, 0.3 + 0.15 * top_risk),
            run_id=ctx.run_id,
            tags=(agent.name, "cycle"),
            payload={
                "observations": [o.to_dict() for o in result.observations],
                "metrics": result.metrics,
                "notes": result.notes,
                "decisions": [
                    {"action": d.action.to_dict(), "verdict": d.verdict, "reason": d.reason}
                    for d in decisions
                ],
            },
        )
        report = cortex.consolidator.run(ctx.scope)
        return {"episode_id": episode.id, "consolidation": report}

    async def report(state: State) -> State:
        briefing = _render_briefing(agent.name, state)
        text = await ctx.llm.complete(
            system=_REPORT_SYSTEM,
            prompt=f"Turn this run record into a briefing.\n{BRIEFING_MARKER}\n{briefing}",
        )
        return {"report": text}

    graph = (
        Graph(f"cycle:{agent.name}")
        .add_node("decide", decide)
        .add_node("govern", govern)
        .add_node("act", act)
        .add_node("escalate", escalate)
        .add_node("remember", remember)
        .add_node("report", report)
        .set_entry("decide")
        .add_edge("decide", "govern")
        .add_router("govern", route_after_govern, targets=("act", "escalate"))
        .add_edge("act", "escalate")
        .add_edge("escalate", "remember")
        .add_edge("remember", "report")
        .add_edge("report", END)
    )
    return graph.compile()


def _render_briefing(agent_name: str, state: State) -> str:
    result: AgentResult = state["result"]
    executions: list[ExecutionResult] = state.get("executions", [])
    approvals: list[Any] = state.get("approvals", [])
    denied = [d for d in state["decisions"] if d.verdict == "deny"]
    consolidation = state["consolidation"]

    lines = [f"## {agent_name}", result.summary, ""]
    if executions:
        lines.append("Executed automatically:")
        lines += [
            f"- {e.kind} {e.target}: {'ok' if e.ok else 'FAILED'}"
            f"{' (dry run)' if e.dry_run else ''}"
            for e in executions
        ]
    if approvals:
        lines.append("Waiting for approval:")
        lines += [f"- [{a.id}] {a.action.kind} {a.action.target}: {a.reason}" for a in approvals]
    if denied:
        lines.append("Refused by policy:")
        lines += [f"- {d.action.kind} {d.action.target}: {d.reason}" for d in denied]
    if result.notes:
        lines.append("Notes:")
        lines += [f"- {n}" for n in result.notes]
    if consolidation.promoted:
        lines.append("Learned (promoted to long-term memory):")
        lines += [
            f"- {f.subject} {f.predicate} = {f.value} (confidence {f.confidence:.2f})"
            for f in consolidation.promoted
        ]
    return "\n".join(lines)
