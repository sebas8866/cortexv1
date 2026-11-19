"""Control-plane HTTP API.

POST /runs executes a cycle inline and returns the run (good for operators and
demos). POST /tasks enqueues work for the worker fleet (the 24/7 path).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from cortex import __version__
from cortex.agents import available_agents
from cortex.config import Settings, load_settings
from cortex.memory import MemoryKind
from cortex.orchestration.queue import Task, build_queue
from cortex.runtime.control import ApprovalStatus
from cortex.runtime.cortex import Cortex


class RunRequest(BaseModel):
    agent: str
    scope: str | None = None
    inputs: dict[str, Any] = Field(default_factory=dict)


class DecisionRequest(BaseModel):
    approve: bool
    by: str = Field(min_length=1, description="Who made the call; stored in the audit log.")
    note: str = ""


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.cortex = Cortex(settings)
        app.state.queue = build_queue(settings.queue)
        yield
        app.state.cortex.close()

    app = FastAPI(title="Cortex control plane", version=__version__, lifespan=lifespan)

    def cx() -> Cortex:
        cortex: Cortex = app.state.cortex
        return cortex

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "version": __version__,
            "llm": cx().llm.name,
            "dry_run": settings.policy.dry_run,
        }

    @app.get("/agents")
    async def agents() -> dict[str, str]:
        return available_agents()

    @app.post("/runs")
    async def create_run(req: RunRequest) -> dict[str, Any]:
        try:
            record = await cx().run_agent(req.agent, req.inputs, scope=req.scope)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        return record.as_dict()

    @app.get("/runs")
    async def list_runs(limit: int = Query(20, le=200)) -> list[dict[str, Any]]:
        return [r.as_dict() for r in cx().control.list_runs(limit)]

    @app.get("/runs/{run_id}")
    async def get_run(run_id: str) -> dict[str, Any]:
        try:
            return cx().control.get_run(run_id).as_dict()
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.post("/tasks", status_code=202)
    async def enqueue(req: RunRequest) -> dict[str, str]:
        task = Task(req.agent, req.scope or settings.default_scope, req.inputs)
        await app.state.queue.enqueue(task)
        return {"task_id": task.id}

    @app.get("/approvals")
    async def approvals(
        status: ApprovalStatus | None = ApprovalStatus.PENDING,
    ) -> list[dict[str, Any]]:
        return [a.as_dict() for a in cx().control.list_approvals(status)]

    @app.post("/approvals/{approval_id}")
    async def decide(approval_id: str, req: DecisionRequest) -> dict[str, Any]:
        try:
            approval = await cx().decide(approval_id, approve=req.approve, by=req.by, note=req.note)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        return approval.as_dict()

    @app.get("/executions")
    async def executions(limit: int = Query(50, le=500)) -> list[dict[str, Any]]:
        return cx().control.list_executions(limit)

    @app.get("/memory/search")
    async def memory_search(
        q: str, scope: str | None = None, kind: MemoryKind | None = None, limit: int = 10
    ) -> list[dict[str, Any]]:
        hits = cx().memory.recall(
            q, scope or settings.default_scope, kinds=[kind] if kind else None, limit=limit
        )
        return [
            {
                "id": h.id,
                "kind": h.kind.value,
                "scope": h.scope,
                "text": h.text,
                "score": h.score,
                "signals": {
                    "relevance": h.relevance,
                    "recency": h.recency,
                    "importance": h.importance,
                },
            }
            for h in hits
        ]

    @app.get("/memory/facts/{subject}")
    async def facts(subject: str, scope: str | None = None) -> list[dict[str, Any]]:
        return [
            {
                "predicate": f.predicate,
                "value": f.value,
                "confidence": round(f.confidence, 3),
                "evidence": f.evidence_count,
                "sources": f.source_episode_ids,
                "scope": f.scope,
                "updated_at": f.updated_at.isoformat(),
            }
            for f in cx().memory.facts_about(scope or settings.default_scope, subject)
        ]

    @app.get("/memory/stats")
    async def memory_stats(scope: str | None = None) -> dict[str, int]:
        return cx().memory.stats(scope or settings.default_scope)

    return app
