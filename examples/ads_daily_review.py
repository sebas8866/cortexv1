"""End-to-end example: two days of autonomous ads review with a human in the loop.

Run from the repo root:

    python examples/ads_daily_review.py

No API keys, no network: the offline LLM writes the briefing and every action is a
dry run. What to look for in the output:

Day 1  winners are scaled automatically (low risk, within the 25% cap); the
       zero-conversion campaign is queued for a human because pausing is medium risk.
Human  approves the pause. The verdict is stored as an episode, the effect as a fact.
Day 2  the same data comes in. Scaled campaigns are held by the cooldown remembered
       from day 1, the paused campaign is skipped, and "winner" tiers seen on both
       days are promoted from observations to long-term facts.
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

from cortex import Cortex, load_settings

ROOT = Path(__file__).resolve().parents[1]


class SimClock:
    """A controllable clock so two 'days' run in a second."""

    def __init__(self, start: datetime) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **kwargs: float) -> None:
        self.now += timedelta(**kwargs)


def banner(text: str) -> None:
    print(f"\n{'=' * 72}\n{text}\n{'=' * 72}")


async def main() -> None:
    workdir = Path(tempfile.mkdtemp(prefix="cortex-demo-"))
    settings = load_settings(ROOT / "config" / "cortex.yaml")
    settings.data_dir = workdir
    settings.llm.provider = "offline"
    clock = SimClock(datetime(2026, 11, 18, 7, 0, tzinfo=UTC))
    cortex = Cortex(settings, clock=clock)
    source = {"source": str(ROOT / "examples" / "data" / "campaigns.csv")}

    try:
        banner("DAY 1 - 07:00 scheduled ads review")
        run = await cortex.run_agent("ads_optimizer", source, scope="acme/ads")
        print(run.summary)

        pending = cortex.control.list_approvals()
        banner(f"HUMAN - {len(pending)} action(s) waiting for approval")
        for approval in pending:
            print(f"approving {approval.action.kind} on {approval.action.target}")
            await cortex.decide(
                approval.id, approve=True, by="ops@acme", note="confirmed dead test"
            )

        clock.advance(hours=24)
        banner("DAY 2 - 07:00 scheduled ads review (same numbers, new memory)")
        run = await cortex.run_agent("ads_optimizer", source, scope="acme/ads")
        print(run.summary)

        banner("MEMORY - what Cortex now believes about campaign c_101")
        for fact in cortex.memory.facts_about("acme/ads", "campaign:c_101"):
            print(
                f"  {fact.predicate:22} {fact.value:34} conf={fact.confidence:.2f} "
                f"evidence={fact.evidence_count}"
            )

        banner("MEMORY - recall('paused campaign approval')")
        for hit in cortex.memory.recall("paused campaign approval", "acme/ads", limit=4):
            print(f"  {hit.score:.3f} [{hit.kind.value}] {hit.text[:90]}")

        print(f"\nmemory stats: {cortex.memory.stats('acme/ads')}")
    finally:
        cortex.close()
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(main())
