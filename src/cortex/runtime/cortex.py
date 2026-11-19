"""The Cortex runtime: wires memory, policy, execution and the control plane together."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from datetime import datetime
from typing import Any

from cortex.agents import AgentContext, create_agent
from cortex.config import Settings
from cortex.db import utcnow
from cortex.llm import LLM, build_llm
from cortex.memory import Consolidator, MemoryStore
from cortex.orchestration.cycle import build_standard_cycle
from cortex.orchestration.executor import ActionExecutor
from cortex.orchestration.graph import GraphError
from cortex.orchestration.policy import ActionPolicy
from cortex.runtime.control import Approval, ControlStore, RunRecord, RunStatus

log = logging.getLogger(__name__)


class Cortex:
    def __init__(
        self,
        settings: Settings,
        *,
        clock: Callable[[], datetime] = utcnow,
        llm: LLM | None = None,
    ) -> None:
        self.settings = settings
        self.clock = clock
        db = settings.db_path
        self.memory = MemoryStore(db, settings=settings.memory, clock=clock)
        self.control = ControlStore(db, clock=clock)
        self.llm = llm or build_llm(settings.llm)
        self.policy = ActionPolicy(settings.policy)
        self.executor = ActionExecutor(self.memory, dry_run=settings.policy.dry_run)
        self.consolidator = Consolidator(self.memory, settings.memory.consolidation)

    async def run_agent(
        self,
        agent_name: str,
        inputs: Mapping[str, Any] | None = None,
        *,
        scope: str | None = None,
    ) -> RunRecord:
        scope = scope or self.settings.default_scope
        agent = create_agent(agent_name, self.settings.agents.get(agent_name, {}))
        run = self.control.create_run(agent_name, scope)
        ctx = AgentContext(
            memory=self.memory, llm=self.llm, scope=scope, run_id=run.id, clock=self.clock
        )
        graph = build_standard_cycle(self, agent, ctx)
        try:
            trace = await graph.run({"inputs": dict(inputs or {})})
        except GraphError as exc:
            log.warning("run %s failed: %s", run.id, exc)
            return self.control.finish_run(
                run.id,
                RunStatus.FAILED,
                summary="",
                trace=[s.as_dict() for s in exc.trace.steps],
                error=str(exc),
            )
        return self.control.finish_run(
            run.id,
            RunStatus.SUCCEEDED,
            summary=trace.state["report"],
            trace=[s.as_dict() for s in trace.steps],
        )

    async def decide(self, approval_id: str, *, approve: bool, by: str, note: str = "") -> Approval:
        """Record a human verdict; execute the action if approved. Verdicts become memory."""
        approval = self.control.decide(approval_id, approve=approve, by=by, note=note)
        action = approval.action
        if approve:
            result = await self.executor.execute(action, approval.scope)
            self.control.record_execution(approval.run_id, approval.scope, result, approved_by=by)
        self.memory.record_episode(
            approval.scope,
            agent="human",
            content=(
                f"{by} {'approved' if approve else 'rejected'} {action.kind} on {action.target}"
                f" proposed by {action.agent}: {action.rationale}. {note}".strip()
            ),
            importance=0.8,  # human verdicts are scarce, high-signal training data
            run_id=approval.run_id,
            tags=("approval", action.agent, action.kind),
            payload={"approval_id": approval.id, "approved": approve},
        )
        return approval

    def close(self) -> None:
        self.memory.close()
        self.control.close()
