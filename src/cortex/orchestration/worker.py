"""Worker and scheduler: the always-on loop.

The scheduler turns configured intervals into tasks; workers pull tasks and run
the agent cycle. Both are plain asyncio loops so they can share a process (the
`cortex worker` command) or be split across containers with the Redis queue.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Callable, Sequence
from datetime import datetime

from cortex.config import Schedule
from cortex.db import utcnow
from cortex.orchestration.queue import Task, TaskQueue
from cortex.runtime.control import RunRecord, RunStatus
from cortex.runtime.cortex import Cortex

log = logging.getLogger(__name__)


class Worker:
    def __init__(self, cortex: Cortex, queue: TaskQueue, *, max_attempts: int = 3) -> None:
        self.cortex = cortex
        self.queue = queue
        self.max_attempts = max_attempts

    async def run_once(self, timeout: float = 5.0) -> RunRecord | None:
        task = await self.queue.dequeue(timeout)
        if task is None:
            return None
        try:
            record = await self.cortex.run_agent(task.agent, task.inputs, scope=task.scope)
        except Exception:
            log.exception("task %s crashed", task.id)
            await self.queue.nack(task, requeue=task.attempts + 1 < self.max_attempts)
            return None
        if record.status is RunStatus.FAILED and task.attempts + 1 < self.max_attempts:
            await self.queue.nack(task, requeue=True)
        else:
            await self.queue.ack(task)
        log.info("task %s -> run %s (%s)", task.id, record.id, record.status.value)
        return record

    async def run_forever(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            await self.run_once(timeout=1.0)


class Scheduler:
    def __init__(
        self,
        schedules: Sequence[Schedule],
        queue: TaskQueue,
        *,
        default_scope: str,
        clock: Callable[[], datetime] = utcnow,
    ) -> None:
        self.schedules = list(schedules)
        self.queue = queue
        self.default_scope = default_scope
        self.clock = clock
        # Last-fire times are in-process; a restart fires every schedule once (see Limitations).
        self._last: dict[int, datetime] = {}

    async def tick(self) -> list[Task]:
        now = self.clock()
        fired: list[Task] = []
        for i, sched in enumerate(self.schedules):
            last = self._last.get(i)
            if last is not None and now - last < sched.interval:
                continue
            task = Task(sched.agent, sched.scope or self.default_scope, dict(sched.inputs))
            await self.queue.enqueue(task)
            self._last[i] = now
            fired.append(task)
        return fired

    async def run_forever(self, stop: asyncio.Event, tick_seconds: float = 10.0) -> None:
        while not stop.is_set():
            for task in await self.tick():
                log.info("scheduled %s for %s", task.agent, task.scope)
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(stop.wait(), tick_seconds)
