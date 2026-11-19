"""Security monitor: brute-force detection and unfamiliar-location logins.

Reads normalized auth events (JSON lines). Memory supplies the baseline: each
user's known login countries are facts that grow as logins are corroborated, so
the agent gets quieter about normal travel patterns over time.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from cortex.agents.base import Agent, AgentContext, AgentResult, ProposedAction, Risk
from cortex.agents.registry import register
from cortex.memory import Observation


class SecurityConfig(BaseModel):
    source: str = "examples/data/auth_events.jsonl"
    failed_login_threshold: int = 5
    window_minutes: int = 10


@dataclass(slots=True, frozen=True)
class AuthEvent:
    ts: datetime
    user: str
    ip: str
    country: str
    outcome: str  # "success" | "failure"


def load_events(path: Path) -> list[AuthEvent]:
    events: list[AuthEvent] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        events.append(
            AuthEvent(
                ts=datetime.fromisoformat(raw["ts"]),
                user=raw["user"],
                ip=raw["ip"],
                country=raw["country"],
                outcome=raw["outcome"],
            )
        )
    return sorted(events, key=lambda e: e.ts)


@register
class SecurityMonitorAgent(Agent):
    name = "security_monitor"
    description = "Detects brute-force bursts and logins from unfamiliar countries."

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        super().__init__(config)
        self.cfg = SecurityConfig.model_validate(self.config)

    async def run(self, ctx: AgentContext, inputs: Mapping[str, Any]) -> AgentResult:
        events = load_events(Path(str(inputs.get("source", self.cfg.source))))
        actions: list[ProposedAction] = []
        observations: list[Observation] = []
        notes: list[str] = []

        hostile_ips = self._brute_force_ips(events)
        for ip, count in hostile_ips.items():
            blocked = ctx.memory.get_fact(ctx.scope, f"ip:{ip}", "status")
            if blocked and blocked.value == "blocked":
                notes.append(f"{ip}: already blocked")
                continue
            actions.append(
                ProposedAction(
                    agent=self.name,
                    kind="block_ip",
                    target=f"ip:{ip}",
                    risk=Risk.MEDIUM,
                    rationale=f"{count} failed logins within {self.cfg.window_minutes} minutes",
                    params={"ip": ip, "failures": count},
                    effects=[Observation(f"ip:{ip}", "status", "blocked", 1.0)],
                )
            )

        alerted: set[tuple[str, str]] = set()  # one unfamiliar-country alert per user+country
        for e in (e for e in events if e.outcome == "success"):
            subject = f"user:{e.user}"
            if e.ip in hostile_ips:
                actions.append(
                    ProposedAction(
                        agent=self.name,
                        kind="alert",
                        target=subject,
                        risk=Risk.CRITICAL,
                        rationale=f"successful login from {e.ip} after a brute-force burst",
                        params={"user": e.user, "ip": e.ip, "severity": "critical"},
                    )
                )
                continue
            known = ctx.memory.get_fact(ctx.scope, subject, f"login_country:{e.country}")
            if known is None and (e.user, e.country) not in alerted:
                alerted.add((e.user, e.country))
                notes.append(f"{e.user}: first login seen from {e.country}")
                actions.append(
                    ProposedAction(
                        agent=self.name,
                        kind="alert",
                        target=subject,
                        risk=Risk.LOW,
                        rationale=f"login from unfamiliar country {e.country} ({e.ip})",
                        params={"user": e.user, "ip": e.ip, "severity": "info"},
                    )
                )
            observations.append(Observation(subject, f"login_country:{e.country}", "seen", 0.5))

        summary = (
            f"Scanned {len(events)} auth events: {len(hostile_ips)} brute-force sources, "
            f"{sum(1 for a in actions if a.kind == 'alert')} alerts."
        )
        return AgentResult(
            agent=self.name,
            summary=summary,
            actions=actions,
            observations=observations,
            notes=notes,
            metrics={"events": len(events), "brute_force_ips": len(hostile_ips)},
        )

    def _brute_force_ips(self, events: list[AuthEvent]) -> dict[str, int]:
        """Sliding window over failures per IP; returns the peak count for IPs over threshold."""
        window = timedelta(minutes=self.cfg.window_minutes)
        failures: dict[str, list[datetime]] = defaultdict(list)
        for e in events:
            if e.outcome == "failure":
                failures[e.ip].append(e.ts)
        flagged: dict[str, int] = {}
        for ip, stamps in failures.items():
            start, peak = 0, 0
            for end, ts in enumerate(stamps):
                while ts - stamps[start] > window:
                    start += 1
                peak = max(peak, end - start + 1)
            if peak >= self.cfg.failed_login_threshold:
                flagged[ip] = peak
        return flagged
