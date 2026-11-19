"""The agent contract.

An agent is a pure decision-maker: given inputs and memory, it returns a summary,
proposed actions and observations. It never executes side effects itself. The
runtime decides (via policy) which proposals run automatically, which wait for a
human, and which are refused.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from typing import Any, ClassVar

from cortex.db import new_id
from cortex.llm import LLM
from cortex.memory import MemoryStore, Observation


class Risk(IntEnum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass(slots=True)
class ProposedAction:
    agent: str
    kind: str  # e.g. "increase_budget", "pause_campaign", "block_ip"
    target: str  # the thing acted on, e.g. "campaign:c_101"
    risk: Risk
    rationale: str
    params: dict[str, Any] = field(default_factory=dict)
    # Facts that become true if this action executes. Written with confidence 1.0.
    effects: list[Observation] = field(default_factory=list)
    id: str = field(default_factory=lambda: new_id("act"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "agent": self.agent,
            "kind": self.kind,
            "target": self.target,
            "risk": self.risk.name.lower(),
            "rationale": self.rationale,
            "params": self.params,
            "effects": [e.to_dict() for e in self.effects],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ProposedAction:
        return cls(
            id=data["id"],
            agent=data["agent"],
            kind=data["kind"],
            target=data["target"],
            risk=Risk[str(data["risk"]).upper()],
            rationale=data["rationale"],
            params=dict(data.get("params", {})),
            effects=[Observation.from_dict(e) for e in data.get("effects", [])],
        )


@dataclass(slots=True)
class AgentResult:
    agent: str
    summary: str
    actions: list[ProposedAction] = field(default_factory=list)
    observations: list[Observation] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    metrics: dict[str, float | int | str] = field(default_factory=dict)


@dataclass(slots=True)
class AgentContext:
    memory: MemoryStore
    llm: LLM
    scope: str
    run_id: str
    clock: Callable[[], datetime]

    def now(self) -> datetime:
        return self.clock()


class Agent(ABC):
    name: ClassVar[str]
    description: ClassVar[str]

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self.config: dict[str, Any] = dict(config or {})

    @abstractmethod
    async def run(self, ctx: AgentContext, inputs: Mapping[str, Any]) -> AgentResult:
        """Sense and decide. Must not perform side effects."""
