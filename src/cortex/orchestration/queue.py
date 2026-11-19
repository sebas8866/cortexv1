"""Task queues. In-memory for single-process use, Redis for API + worker fleets.

The Redis queue uses the reliable-queue pattern: BLMOVE moves a task onto a
per-queue processing list, and it is only removed on ack. A crashed worker leaves
its task on the processing list where it can be recovered (see recover_stale()).
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

from cortex.config import QueueSettings
from cortex.db import from_iso, new_id, to_iso, utcnow


@dataclass(slots=True)
class Task:
    agent: str
    scope: str
    inputs: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: new_id("task"))
    enqueued_at: datetime = field(default_factory=utcnow)
    attempts: int = 0

    def to_json(self) -> str:
        return json.dumps(
            {
                "id": self.id,
                "agent": self.agent,
                "scope": self.scope,
                "inputs": self.inputs,
                "enqueued_at": to_iso(self.enqueued_at),
                "attempts": self.attempts,
            }
        )

    @classmethod
    def from_json(cls, raw: str | bytes) -> Task:
        data = json.loads(raw)
        return cls(
            id=data["id"],
            agent=data["agent"],
            scope=data["scope"],
            inputs=data.get("inputs", {}),
            enqueued_at=from_iso(data["enqueued_at"]),
            attempts=int(data.get("attempts", 0)),
        )


class TaskQueue(Protocol):
    async def enqueue(self, task: Task) -> None: ...
    async def dequeue(self, timeout: float = 5.0) -> Task | None: ...
    async def ack(self, task: Task) -> None: ...
    async def nack(self, task: Task, *, requeue: bool) -> None: ...


class InMemoryQueue:
    def __init__(self) -> None:
        self._q: asyncio.Queue[Task] = asyncio.Queue()

    async def enqueue(self, task: Task) -> None:
        await self._q.put(task)

    async def dequeue(self, timeout: float = 5.0) -> Task | None:
        try:
            return await asyncio.wait_for(self._q.get(), timeout)
        except TimeoutError:
            return None

    async def ack(self, task: Task) -> None:
        self._q.task_done()

    async def nack(self, task: Task, *, requeue: bool) -> None:
        self._q.task_done()
        if requeue:
            task.attempts += 1
            await self._q.put(task)

    def qsize(self) -> int:
        return self._q.qsize()


class RedisQueue:
    def __init__(self, url: str, name: str) -> None:
        try:
            from redis import asyncio as aioredis
        except ImportError as exc:  # pragma: no cover - optional extra
            raise RuntimeError(
                "install the 'redis' extra: pip install 'cortex-agents[redis]'"
            ) from exc
        self._redis = aioredis.from_url(url, decode_responses=True)
        self._pending = name
        self._processing = f"{name}:processing"
        self._raw: dict[str, str] = {}

    async def enqueue(self, task: Task) -> None:
        await self._redis.lpush(self._pending, task.to_json())

    async def dequeue(self, timeout: float = 5.0) -> Task | None:
        raw = await self._redis.blmove(self._pending, self._processing, timeout, "RIGHT", "LEFT")
        if raw is None:
            return None
        task = Task.from_json(raw)
        self._raw[task.id] = raw
        return task

    async def ack(self, task: Task) -> None:
        if raw := self._raw.pop(task.id, None):
            await self._redis.lrem(self._processing, 1, raw)

    async def nack(self, task: Task, *, requeue: bool) -> None:
        await self.ack(task)
        if requeue:
            task.attempts += 1
            await self.enqueue(task)

    async def recover_stale(self) -> int:
        """Move everything on the processing list back to pending. Run on startup of a
        single worker, or from an operator command when workers are known to be down."""
        moved = 0
        while await self._redis.lmove(self._processing, self._pending, "RIGHT", "LEFT"):
            moved += 1
        return moved


def build_queue(settings: QueueSettings) -> TaskQueue:
    if settings.backend == "redis":
        return RedisQueue(settings.redis_url, settings.name)
    return InMemoryQueue()
