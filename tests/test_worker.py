from __future__ import annotations

import asyncio

from cortex.config import Schedule
from cortex.orchestration.queue import InMemoryQueue
from cortex.orchestration.worker import Scheduler, Worker
from cortex.runtime.control import RunStatus
from cortex.runtime.cortex import Cortex


def test_scheduler_fires_on_interval(clock, campaigns_csv: str) -> None:  # type: ignore[no-untyped-def]
    queue = InMemoryQueue()
    schedule = Schedule(agent="ads_optimizer", every="6h", inputs={"source": campaigns_csv})
    scheduler = Scheduler([schedule], queue, default_scope="acme/ads", clock=clock)

    async def scenario() -> list[int]:
        fired = [len(await scheduler.tick())]  # first tick always fires
        clock.advance(hours=5)
        fired.append(len(await scheduler.tick()))  # inside the interval
        clock.advance(hours=1)
        fired.append(len(await scheduler.tick()))  # interval elapsed
        return fired

    assert asyncio.run(scenario()) == [1, 0, 1]
    assert queue.qsize() == 2


def test_worker_runs_queued_task(cortex: Cortex, campaigns_csv: str) -> None:
    queue = InMemoryQueue()
    worker = Worker(cortex, queue)

    async def scenario() -> tuple[RunStatus | None, RunStatus | None]:
        scheduler = Scheduler(
            [Schedule(agent="ads_optimizer", every="1h", inputs={"source": campaigns_csv})],
            queue,
            default_scope="acme/ads",
            clock=cortex.clock,
        )
        await scheduler.tick()
        ran = await worker.run_once(timeout=1)
        idle = await worker.run_once(timeout=0.05)
        return (ran.status if ran else None), (idle.status if idle else None)

    assert asyncio.run(scenario()) == (RunStatus.SUCCEEDED, None)
    assert queue.qsize() == 0


def test_worker_retries_failed_runs_then_gives_up(cortex: Cortex) -> None:
    queue = InMemoryQueue()
    worker = Worker(cortex, queue, max_attempts=2)

    async def scenario() -> int:
        from cortex.orchestration.queue import Task

        await queue.enqueue(Task("ads_optimizer", "acme/ads", {"source": "missing.csv"}))
        await worker.run_once(timeout=1)  # attempt 1 fails, requeued
        await worker.run_once(timeout=1)  # attempt 2 fails, dropped
        return queue.qsize()

    assert asyncio.run(scenario()) == 0
    assert [r.status for r in cortex.control.list_runs()] == [RunStatus.FAILED] * 2
