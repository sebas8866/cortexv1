"""End-to-end agent cycles against the example data, with a fake clock."""

from __future__ import annotations

import asyncio

from fastapi.testclient import TestClient

from cortex.api import create_app
from cortex.config import Settings
from cortex.runtime.control import RunStatus
from cortex.runtime.cortex import Cortex


def _kinds(cortex: Cortex) -> list[tuple[str, str]]:
    return sorted((e["kind"], e["target"]) for e in cortex.control.list_executions())


def test_ads_two_day_flow_uses_memory(cortex: Cortex, clock, campaigns_csv: str) -> None:  # type: ignore[no-untyped-def]
    inputs = {"source": campaigns_csv}

    day1 = asyncio.run(cortex.run_agent("ads_optimizer", inputs, scope="acme/ads"))
    assert day1.status is RunStatus.SUCCEEDED
    assert [s["node"] for s in day1.trace] == [
        "decide",
        "govern",
        "act",
        "escalate",
        "remember",
        "report",
    ]
    # Winners scaled, loser trimmed: all low risk, so executed by policy.
    assert _kinds(cortex) == [
        ("decrease_budget", "campaign:c_102"),
        ("increase_budget", "campaign:c_101"),
        ("increase_budget", "campaign:c_103"),
    ]
    # Zero-conversion pause is medium risk: waits for a human.
    pending = cortex.control.list_approvals()
    assert [(a.action.kind, a.action.target) for a in pending] == [
        ("pause_campaign", "campaign:c_104")
    ]

    asyncio.run(cortex.decide(pending[0].id, approve=True, by="ops"))
    status = cortex.memory.get_fact("acme/ads", "campaign:c_104", "status")
    assert status is not None and status.value == "paused"

    clock.advance(hours=24)
    day2 = asyncio.run(cortex.run_agent("ads_optimizer", inputs, scope="acme/ads"))
    assert day2.status is RunStatus.SUCCEEDED
    assert len(cortex.control.list_executions()) == 4  # 3 from day 1 + the approved pause
    assert cortex.control.list_approvals() == []
    assert "held" in day2.summary and "paused earlier" in day2.summary

    tier = cortex.memory.get_fact("acme/ads", "campaign:c_101", "performance_tier")
    assert tier is not None and tier.value == "winner" and tier.evidence_count == 1
    assert len(tier.source_episode_ids) == 2

    clock.advance(hours=25)  # cooldown (48h) is over
    asyncio.run(cortex.run_agent("ads_optimizer", inputs, scope="acme/ads"))
    assert len(cortex.control.list_executions()) == 7


def test_security_cycle_blocks_and_alerts(cortex: Cortex, auth_events: str) -> None:
    run = asyncio.run(cortex.run_agent("security_monitor", {"source": auth_events}, scope="acme"))
    assert run.status is RunStatus.SUCCEEDED
    pending = {(a.action.kind, a.action.target) for a in cortex.control.list_approvals()}
    assert ("block_ip", "ip:203.0.113.45") in pending
    executed = {(e["kind"], e["target"]) for e in cortex.control.list_executions()}
    assert ("alert", "user:james") in executed  # low-risk info alerts go out automatically


def test_failed_run_is_recorded(cortex: Cortex) -> None:
    run = asyncio.run(cortex.run_agent("ads_optimizer", {"source": "missing.csv"}))
    assert run.status is RunStatus.FAILED
    assert run.trace[-1]["node"] == "decide" and run.trace[-1]["error"]


def test_api_round_trip(settings: Settings, campaigns_csv: str) -> None:
    with TestClient(create_app(settings)) as client:
        assert client.get("/health").json()["status"] == "ok"
        assert "ads_optimizer" in client.get("/agents").json()

        run = client.post(
            "/runs",
            json={
                "agent": "ads_optimizer",
                "scope": "acme/ads",
                "inputs": {"source": campaigns_csv},
            },
        ).json()
        assert run["status"] == "succeeded"

        pending = client.get("/approvals").json()
        assert len(pending) == 1
        decided = client.post(
            f"/approvals/{pending[0]['id']}", json={"approve": False, "by": "ops"}
        ).json()
        assert decided["status"] == "rejected"
        assert (
            client.post(
                f"/approvals/{pending[0]['id']}", json={"approve": True, "by": "ops"}
            ).status_code
            == 409
        )

        facts = client.get("/memory/facts/campaign:c_101", params={"scope": "acme/ads"}).json()
        assert {f["predicate"] for f in facts} >= {"daily_budget", "last_budget_change"}
        assert client.get(
            "/memory/search", params={"q": "rejected pause", "scope": "acme/ads"}
        ).json()
        assert client.post("/runs", json={"agent": "nope"}).status_code == 404
        # memory backend: the API has no worker, so queued tasks would never run
        assert client.post("/tasks", json={"agent": "research"}).status_code == 503
